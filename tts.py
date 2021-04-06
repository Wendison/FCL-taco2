#!/usr/bin/env python3
# Modified from Espnet

"""FCL-taco2 training / decoding functions."""

import copy
import json
import logging
import math
import os
import time

import chainer
import kaldiio
import numpy as np
import torch

from chainer import training
from chainer.training import extensions

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.dynamic_import import dynamic_import
from batchfy_fcl import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.utils.training.iterators import ShufflingEnabler

import matplotlib

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

from apex import amp

matplotlib.use("Agg")


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator."""

    def __init__(self, model, iterator, target, device):
        """Initilize module.

        Args:
            model (torch.nn.Module): Pytorch model instance.
            iterator (chainer.dataset.Iterator): Iterator for validation.
            target (chainer.Chain): Dummy chain instance.
            device (torch.device): The device to be used in evaluation.

        """
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.device = device

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        """Evaluate over validation iterator."""
        iterator = self._iterators["main"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                if isinstance(batch, tuple):
                    x = tuple(arr.to(self.device) for arr in batch)
                else:
                    x = batch
                    for key in x.keys():
                        if key!='ds_nonzeros':
                            x[key] = x[key].to(self.device)
                            
                    
                observation = {}
                with chainer.reporter.report_scope(observation):
                    # convert to torch tensor
                    if isinstance(x, tuple):
                        self.model(*x)
                    else:
                        self.model(**x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom updater."""

    def __init__(self, model, grad_clip, iterator, optimizer, device, accum_grad=1, use_amp=False, num_batches=None, outdir=None):
        """Initilize module.

        Args:
            model (torch.nn.Module) model: Pytorch model instance.
            grad_clip (float) grad_clip : The gradient clipping value.
            iterator (chainer.dataset.Iterator): Iterator for training.
            optimizer (torch.optim.Optimizer) : Pytorch optimizer instance.
            device (torch.device): The device to be used in training.

        """
        super(CustomUpdater, self).__init__(iterator, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.device = device
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.use_amp = use_amp
        self.num_batches = num_batches
        self.outdir = outdir
        
    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Update model one step."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        if isinstance(batch, tuple):
            x = tuple(arr.to(self.device) for arr in batch)
        else:
            x = batch
            for key in x.keys():
                x[key] = x[key].to(self.device)
        
        # print(x.keys(), x['ds_nonzeros'])
        # compute loss and gradient
        if isinstance(x, tuple):
            loss = self.model(*x).mean() / self.accum_grad
        else:
            loss = self.model(**x).mean() / self.accum_grad
            
        if self.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0

        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip)
        logging.debug("grad norm={}".format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            optimizer.step()
        optimizer.zero_grad()

    def update(self):
        """Run update function."""
        # start_time = time.time()
        self.update_core()
        # consume_time = time.time()-start_time
        # print(f'time for updating once: {consume_time}s')
        if self.forward_count == 0:
            # print('iter:', self.iteration)
            self.iteration += 1
            if self.use_amp and self.iteration % (self.num_batches*10)==0: # save amp-checkpoint every 10 epochs
                optimizer = self.get_optimizer("main")
                model = self.model
                checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()
                        }
                torch.save(checkpoint, f'{self.outdir}/amp_checkpoint_{self.iteration}.pt')
                


class CustomConverter(object):
    """Custom converter."""

    def __init__(self, reduction_factor=1, 
                 use_fe_condition=False, 
                 append_position=False,
                 ):
        """Initilize module."""
        # NOTE: keep as class for future development
        self.reduction_factor = reduction_factor
        self.use_fe_condition = use_fe_condition
        self.append_position = append_position

    def __call__(self, batch, device=torch.device("cpu")):
        """Convert a given batch.

        Args:
            batch (list): List of ndarrays.
            device (torch.device): The device to be send.

        Returns:
            dict: Dict of converted tensors.

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys, spembs, extras, f0, energy = batch[0]
        
        
        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)
        
        # reorganize ys 
        # print(ilens, ilens.shape)
        if extras is not None:
            new_ys = []
            non_zero_lens_mask = []
            ds_nonzeros = []
            if self.append_position:
                position = []
            for ib in range(ilens.shape[0]):
                # reorganize ys: divide ys with different phn/char, remove the phn/char with zero length
                ys_ib = ys[ib]
                ds_ib = extras[ib] # durations for 
                new_ys_ib = []
                non_zero_lens_mask_ib = []
                for it in range(ilens[ib]):
                    start = int(sum(ds_ib[:it]))*self.reduction_factor
                    end = int(sum(ds_ib[:it+1]))*self.reduction_factor
                    if start != end:
                        ys_split = torch.from_numpy(ys_ib[start:end]).float()
                        new_ys_ib.append(ys_split) # l x odim
                        non_zero_lens_mask_ib.append(1) # if length > 0, then mask=1
                        ds_nonzeros.append(int(ds_ib[it]*self.reduction_factor))
                        if self.append_position:
                            position.append(torch.FloatTensor(list(range(end-start)))/(end-start))
                    else:
                        non_zero_lens_mask_ib.append(0) # if length = 0, then mask=0
                
                new_ys.extend(new_ys_ib)
                non_zero_lens_mask.append(torch.tensor(non_zero_lens_mask_ib))


            new_ys = pad_list(new_ys,0).to(device) # #-of-phn x Lmax x odim
            non_zero_lens_mask = pad_list(non_zero_lens_mask, 0)

            
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)
        if self.use_fe_condition:
            new_f0 = pad_list([torch.from_numpy(f00).float() for f00 in f0], 0) # B x Imax x 1
            new_en = pad_list([torch.from_numpy(enn).float() for enn in energy], 0) # B x Imax x 1

        # prepare dict
        new_batch = {
            "xs": xs,
            "ilens": ilens,
            "ys": ys,
            "olens": olens,
        }

        # load speaker embedding
        if spembs is not None:
            spembs = torch.from_numpy(np.array(spembs)).float()
            new_batch["spembs"] = spembs.to(device)

        # load second target
        if extras is not None:
            extras = pad_list([torch.from_numpy(extra).float() for extra in extras], 0)
            new_batch["extras"] = extras.to(device)
            new_batch["new_ys"] = new_ys
            new_batch["non_zero_lens_mask"] = non_zero_lens_mask
            new_batch["ds_nonzeros"] = torch.tensor(ds_nonzeros).to(device) 
            new_batch["output_masks"] = make_non_pad_mask(new_batch["ds_nonzeros"]).to(device) # #-of-phn x new_Lmax
            assert new_batch["new_ys"].shape[1] == new_batch["output_masks"].shape[1]
            if self.append_position:
                position = pad_list(position, 0)
                new_batch['position'] = position
                assert position.shape[0]==new_ys.shape[0]
            if self.use_fe_condition:
                new_batch['f0'] = new_f0
                new_batch['energy'] = new_en
                
        return new_batch


def train(args):
    """Train FCL-taco2 model."""
    set_deterministic_pytorch(args)
    
    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())

    # reverse input and output dimension
    idim = int(valid_json[utts[0]]["output"][0]["shape"][1])
    odim = int(valid_json[utts[0]]["input"][0]["shape"][1])
    logging.info("#input dims: " + str(idim))
    logging.info("#output dims: " + str(odim))

    # get extra input and output dimenstion
    if args.use_speaker_embedding:
        args.spk_embed_dim = int(valid_json[utts[0]]["input"][1]["shape"][0])
    else:
        args.spk_embed_dim = None
    if args.use_second_target:
        args.spc_dim = int(valid_json[utts[0]]["input"][1]["shape"][1])
    else:
        args.spc_dim = None

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to" + model_conf)
        f.write(
            json.dumps(
                (idim, odim, vars(args)), indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logging.info("ARGS: " + key + ": " + str(vars(args)[key]))

    # specify model architecture
    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(idim, odim, args, TTSInterface)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args, args)
    
    # print('tts_wds:', model.base_plot_keys)
    assert isinstance(model, TTSInterface)
    logging.info(model)
    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        # model = torch.nn.DataParallel(model, device_ids=[4,5,6,7])
        if args.batch_size != 0:
            logging.warning(
                "batch size is automatically increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.ngpu)
            )
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # freeze modules, if specified
    if args.freeze_mods:
        if hasattr(model, "module"):
            freeze_mods = ["module." + x for x in args.freeze_mods]
        else:
            freeze_mods = args.freeze_mods

        for mod, param in model.named_parameters():
            if any(mod.startswith(key) for key in freeze_mods):
                logging.info(f"{mod} is frozen not to be updated.")
                param.requires_grad = False

        model_params = filter(lambda x: x.requires_grad, model.parameters())
    else:
        model_params = model.parameters()

    # Setup an optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            model_params, args.lr, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        optimizer = get_std_opt(
            model_params, args.adim, args.transformer_warmup_steps, args.transformer_lr
        )
    elif args.opt == 'lamb':
        kw = dict(lr=0.1, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=1e-6)
        from apex.optimizers import FusedAdam, FusedLAMB
        optimizer = FusedLAMB(model.parameters(), **kw)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)
    
    if args.use_amp:
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    
    if args.amp_checkpoint is not None:
        logging.info("resumed from %s" % args.amp_checkpoint)
        checkpoint = torch.load(args.amp_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        
    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    
    num_batches = len(train_json.keys()) // args.batch_size
    
    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    if use_sortagrad:
        args.batch_sort_key = "input"
        
    print(f'\n\n batch_sort_key: {args.batch_sort_key} \n\n')
    
    # make minibatch list (variable length)
    train_batchset = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        batch_sort_key=args.batch_sort_key,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        swap_io=True,
        iaxis=0,
        oaxis=0,
    )
    valid_batchset = make_batchset(
        valid_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        batch_sort_key=args.batch_sort_key,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        swap_io=True,
        iaxis=0,
        oaxis=0,
    )
    
    
    from io_utils_fcl import LoadInputsAndTargets
    
    load_tr = LoadInputsAndTargets(
        mode="tts",
        use_speaker_embedding=args.use_speaker_embedding,
        use_second_target=args.use_second_target,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
        keep_all_data_on_mem=args.keep_all_data_on_mem,
        pad_eos=args.pad_eos,
    )

    load_cv = LoadInputsAndTargets(
        mode="tts",
        use_speaker_embedding=args.use_speaker_embedding,
        use_second_target=args.use_second_target,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
        keep_all_data_on_mem=args.keep_all_data_on_mem,
        pad_eos=args.pad_eos,
    )

    converter = CustomConverter(reduction_factor=args.reduction_factor,
                                use_fe_condition=args.use_fe_condition,
                                append_position=args.append_position,
                                )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    train_iter = {
        "main": ChainerDataLoader(
            dataset=TransformDataset(
                train_batchset, lambda data: converter([load_tr(data)])
            ),
            batch_size=1,
            num_workers=args.num_iter_processes,
            shuffle=not use_sortagrad,
            collate_fn=lambda x: x[0],
        )
    }
    valid_iter = {
        "main": ChainerDataLoader(
            dataset=TransformDataset(
                valid_batchset, lambda data: converter([load_cv(data)])
            ),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=args.num_iter_processes,
        )
    }

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, device, args.accum_grad, args.use_amp, num_batches, args.outdir
    )
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer)

    # set intervals
    eval_interval = (args.eval_interval_epochs, "epoch")
    save_interval = (args.save_interval_epochs, "epoch")
    report_interval = (args.report_interval_iters, "iteration")

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(
        CustomEvaluator(model, valid_iter, reporter, device), trigger=eval_interval
    )

    # Save snapshot for each epoch
    trainer.extend(torch_snapshot(), trigger=save_interval)

    # Save best models
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger(
            "validation/main/loss", trigger=eval_interval
        ),
    )


    # Make a plot for training and validation values
    if hasattr(model, "module"):
        base_plot_keys = model.module.base_plot_keys
    else:
        base_plot_keys = model.base_plot_keys
    plot_keys = []
    for key in base_plot_keys:
        plot_key = ["main/" + key, "validation/main/" + key]
        trainer.extend(
            extensions.PlotReport(plot_key, "epoch", file_name=key + ".png"),
            trigger=eval_interval,
        )
        plot_keys += plot_key
    trainer.extend(
        extensions.PlotReport(plot_keys, "epoch", file_name="all_loss.png"),
        trigger=eval_interval,
    )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=report_interval))
    report_keys = ["epoch", "iteration", "elapsed_time"] + plot_keys
    trainer.extend(extensions.PrintReport(report_keys), trigger=report_interval)
    trainer.extend(extensions.ProgressBar(), trigger=report_interval)

    set_early_stop(trainer, args)
    # if args.tensorboard_dir is not None and args.tensorboard_dir != "":
    #     writer = SummaryWriter(args.tensorboard_dir)
    #     trainer.extend(TensorboardLogger(writer, att_reporter), trigger=report_interval)

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, "epoch"),
        )

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


@torch.no_grad()
def decode(args):
    # use my own saving ways
    """Decode with FCL-taco2 model."""
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info("args: " + key + ": " + str(vars(args)[key]))

    # define model
    model_class = dynamic_import(train_args.model_module)
    model = model_class(idim, odim, train_args)
    assert isinstance(model, TTSInterface)
    logging.info(model)

    # load trained model parameters
    logging.info("reading model parameters from " + args.model)
    torch_load(args.model, model)
    model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # read json data
    with open(args.json, "rb") as f:
        js = json.load(f)["utts"]

    from io_utils_fcl import LoadInputsAndTargets
    
    load_inputs_and_targets = LoadInputsAndTargets(
        mode="tts",
        load_input=False,
        sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
        pad_eos=args.pad_eos,
    )
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # define writer instances
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=args.out))
    inference_speeds = []
    # start decoding
    for idx, utt_id in enumerate(js.keys()):
        # setup inputs
        batch = [(utt_id, js[utt_id])]
        data = load_inputs_and_targets(batch)
        x = torch.LongTensor(data[0][0]).to(device)
        spemb = None
        if train_args.use_speaker_embedding:
            spemb = torch.FloatTensor(data[1][0]).to(device)

        # decode and write
        start_time = time.time()
        outs = model.inference(x, args, spemb=spemb)
        inference_speed = int(outs.size(0)) / (time.time() - start_time)
        inference_speeds.append(inference_speed)
        logging.info(
            "inference speed = %.1f frames / sec."
            % (inference_speed)
        )
        
        feat_writer[utt_id] = outs.cpu().numpy()
    
    logging.info(
            "average inference speed = %.1f frames / sec."
            % (sum(inference_speeds)/(idx+1))
        )
    avg_infer_speed = sum(inference_speeds)/(idx+1)
    exp_name = args.model.split('/')[-3]
    fp = open(f'{exp_name}.txt','w')
    fp.write(str(avg_infer_speed))
    fp.close()
    # close file object
    feat_writer.close()
 
    
    

