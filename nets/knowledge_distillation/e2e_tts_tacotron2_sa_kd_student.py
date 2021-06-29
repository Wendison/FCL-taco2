#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tacotron2_sa_kd_student related modules."""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask, pad_list
from nets.modules.decoder_sa_kd import Decoder
from nets.modules.encoder_sa_kd import Encoder
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictorLoss

from variance_predictor import VariancePredictor

import argparse


class Tacotron2Loss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False):
        """Initialize Tactoron2 loss module.
        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
        """
        super(Tacotron2Loss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, after_outs, before_outs, ys, olens):
        """Calculate forward propagation.
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
        )

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()

        return l1_loss, mse_loss


class Tacotron2Loss_KD(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False):
        """Initialize Tactoron2 loss module.
        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
        """
        super(Tacotron2Loss_KD, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, after_outs, before_outs, after_outs_teacher, before_outs_teacher, olens):
        """Calculate forward propagation.
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim). (student)
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim). (student)
            after_outs_teacher (Tensor): teacher 
            before_outs_teacher (Tensor): teacher
            olens (LongTensor): Batch of the lengths of each target (B,).
        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(after_outs.device)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)
            after_outs_teacher = after_outs_teacher.masked_select(masks)
            before_outs_teacher = before_outs_teacher.masked_select(masks)

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, after_outs_teacher) + self.l1_criterion(before_outs, before_outs_teacher)
        mse_loss = self.mse_criterion(after_outs, after_outs_teacher) + self.mse_criterion(before_outs, before_outs_teacher)
        
        return l1_loss, mse_loss
    

class Knowledge_loss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False):
        """Initialize Tactoron2 loss module.
        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
        """
        super(Knowledge_loss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
#        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, student, teacher, ilens, typ=None):
        """Calculate forward propagation.
        Args:
            student (list of Tensor): outputs from student
            teacher (list of Tensor): outputs from teacher
            ilens (list): input sequence lengths
            typ (str): type of loss - L1 or L2
        Returns:
            Tensor: loss value.
        """
        # apply mask to remove padded part
        loss = 0.0
        for s, t in zip(student, teacher):
            masks = make_non_pad_mask(ilens).unsqueeze(-1).to(s.device)
            s = s.masked_select(masks)
            t = t.masked_select(masks)
            # calculate loss
            if type is not None:
                tloss = torch.nn.L1Loss(reduction='mean')(s,t)
            else:
                tloss = self.mse_criterion(s,t)
            loss = loss + tloss
            
        return loss


class prosody_criterions(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False):
        """Initialize Tactoron2 loss module.
        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
        """
        super().__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, 
                p_outs: torch.Tensor,
                ps: torch.Tensor,
                ilens: torch.Tensor,
                typ=None,
                ):
        """Calculate forward propagation.
        Args:
            p_outs (Tensor): Batch of outputs of pitch/energy predictor (B, Tmax, 1).
            ps (Tensor): Batch of target token-averaged pitch/energy (B, Tmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            typ: type of loss - L1 or L2
        Returns:
            Tensor: pitch/energy loss value.
        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ps.device)
            p_outs = p_outs.masked_select(masks)
            ps = ps.masked_select(masks)

        # calculate loss
        if typ is not None:
            loss = torch.nn.L1Loss(reduction='mean')(p_outs, ps)
        else:
            loss = self.mse_criterion(p_outs, ps)
        
        return loss
    

def cal_num_params(net):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return parameters


class Tacotron2_sa(TTSInterface, torch.nn.Module):

    @staticmethod
    def add_arguments(parser):
        """Add model-specific arguments to the parser."""
        group = parser.add_argument_group("tacotron 2 model setting")
        # encoder
        group.add_argument(
            "--embed-dim",
            default=512,
            type=int,
            help="Number of dimension of embedding",
        )
        group.add_argument(
            "--elayers", default=1, type=int, help="Number of encoder layers"
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=512,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--econv-layers",
            default=3,
            type=int,
            help="Number of encoder convolution layers",
        )
        group.add_argument(
            "--econv-chans",
            default=512,
            type=int,
            help="Number of encoder convolution channels",
        )
        group.add_argument(
            "--econv-filts",
            default=5,
            type=int,
            help="Filter size of encoder convolution",
        )
        # decoder
        group.add_argument(
            "--dlayers", default=2, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=1024, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--prenet-layers", default=2, type=int, help="Number of prenet layers"
        )
        group.add_argument(
            "--prenet-units",
            default=256,
            type=int,
            help="Number of prenet hidden units",
        )
        group.add_argument(
            "--postnet-layers", default=5, type=int, help="Number of postnet layers"
        )
        group.add_argument(
            "--postnet-chans", default=512, type=int, help="Number of postnet channels"
        )
        group.add_argument(
            "--postnet-filts", default=5, type=int, help="Filter size of postnet"
        )
        group.add_argument(
            "--output-activation",
            default=None,
            type=str,
            nargs="?",
            help="Output activation function",
        )
        # model (parameter) related
        group.add_argument(
            "--use-batch-norm",
            default=True,
            type=strtobool,
            help="Whether to use batch normalization",
        )
        group.add_argument(
            "--use-concate",
            default=True,
            type=strtobool,
            help="Whether to concatenate encoder embedding with decoder outputs",
        )
        group.add_argument(
            "--use-residual",
            default=True,
            type=strtobool,
            help="Whether to use residual connection in conv layer",
        )
        group.add_argument(
            "--dropout-rate", default=0.5, type=float, help="Dropout rate"
        )
        group.add_argument(
            "--zoneout-rate", default=0.1, type=float, help="Zoneout rate"
        )
        group.add_argument(
            "--reduction-factor", default=1, type=int, help="Reduction factor"
        )
        group.add_argument(
            "--spk-embed-dim",
            default=None,
            type=int,
            help="Number of speaker embedding dimensions",
        )
        group.add_argument(
            "--spc-dim", default=None, type=int, help="Number of spectrogram dimensions"
        )
        group.add_argument(
            "--pretrained-model", default=None, type=str, help="Pretrained model path"
        )
        # loss related
        group.add_argument(
            "--use-masking",
            default=False,
            type=strtobool,
            help="Whether to use masking in calculation of loss",
        )
        group.add_argument(
            "--use-weighted-masking",
            default=False,
            type=strtobool,
            help="Whether to use weighted masking in calculation of loss",
        )
        # duration predictor settings
        group.add_argument(
            "--duration-predictor-layers",
            default=2,
            type=int,
            help="Number of layers in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-chans",
            default=384,
            type=int,
            help="Number of channels in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-kernel-size",
            default=3,
            type=int,
            help="Kernel size in duration predictor",
        )
        group.add_argument(
            "--duration-predictor-dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for duration predictor",
        )
        return parser

    def __init__(self, idim, odim, args=None, com_args=None, teacher_args=None):
        """Initialize Tacotron2 module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            args (Namespace, optional):
                - spk_embed_dim (int): Dimension of the speaker embedding.
                - embed_dim (int): Dimension of character embedding.
                - elayers (int): The number of encoder blstm layers.
                - eunits (int): The number of encoder blstm units.
                - econv_layers (int): The number of encoder conv layers.
                - econv_filts (int): The number of encoder conv filter size.
                - econv_chans (int): The number of encoder conv filter channels.
                - dlayers (int): The number of decoder lstm layers.
                - dunits (int): The number of decoder lstm units.
                - prenet_layers (int): The number of prenet layers.
                - prenet_units (int): The number of prenet units.
                - postnet_layers (int): The number of postnet layers.
                - postnet_filts (int): The number of postnet filter size.
                - postnet_chans (int): The number of postnet filter channels.
                - output_activation (int): The name of activation function for outputs.
                - use_batch_norm (bool): Whether to use batch normalization.
                - use_concate (int): Whether to concatenate encoder embedding
                    with decoder lstm outputs.
                - dropout_rate (float): Dropout rate.
                - zoneout_rate (float): Zoneout rate.
                - reduction_factor (int): Reduction factor.
                - spk_embed_dim (int): Number of speaker embedding dimenstions.
                - spc_dim (int): Number of spectrogram embedding dimenstions
                    (only for use_cbhg=True)
                - use_masking (bool):
                    Whether to apply masking for padded part in loss calculation.
                - use_weighted_masking (bool):
                    Whether to apply weighted masking in loss calculation.
                - duration_predictor_layers (int): Number of duration predictor layers.
                - duration_predictor_chans (int): Number of duration predictor channels.
                - duration_predictor_kernel_size (int):
                    Kernel size of duration predictor.
        """
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # fill missing arguments
        args = fill_missing_args(args, self.add_arguments)
        
        args = vars(args)
        if 'use_fe_condition' not in args.keys():
            args['use_fe_condition'] = com_args.use_fe_condition
        if 'append_position' not in args.keys():
            args['append_position'] = com_args.append_position
        if 'distill_output_knowledge' not in args.keys():
            args['distill_output_knowledge'] = com_args.distill_output_knowledge
        if 'distill_encoder_knowledge' not in args.keys():
            args['distill_encoder_knowledge'] = com_args.distill_encoder_knowledge
        if 'distill_decoder_knowledge' not in args.keys():
            args['distill_decoder_knowledge'] = com_args.distill_decoder_knowledge
        if 'distill_prosody_knowledge' not in args.keys():
            args['distill_prosody_knowledge'] = com_args.distill_prosody_knowledge
        if 'is_train' not in args.keys():
            args['is_train'] = com_args.is_train
        if 'share_proj' not in args.keys():
            args['share_proj'] = com_args.share_proj
        
        args = argparse.Namespace(**args)
        
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.embed_dim = args.embed_dim
        self.spk_embed_dim = args.spk_embed_dim
        self.reduction_factor = args.reduction_factor
        self.use_fe_condition = args.use_fe_condition
        self.append_position = args.append_position
        self.distill_output_knowledge = args.distill_output_knowledge
        self.distill_encoder_knowledge = args.distill_encoder_knowledge
        self.distill_decoder_knowledge = args.distill_decoder_knowledge
        self.distill_prosody_knowledge = args.distill_prosody_knowledge
        self.is_train = args.is_train
        self.share_proj = args.share_proj
        
        if self.is_train:
            is_student = True
        else:
            is_student = False
            
        # define activation function for the final output
        if args.output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, args.output_activation):
            self.output_activation_fn = getattr(F, args.output_activation)
        else:
            raise ValueError(
                "there is no such an activation function. (%s)" % args.output_activation
            )

        # set padding idx
        padding_idx = 0
        
        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=args.embed_dim,
            elayers=args.elayers,
            eunits=args.eunits,
            econv_layers=args.econv_layers,
            econv_chans=args.econv_chans,
            econv_filts=args.econv_filts,
            use_batch_norm=args.use_batch_norm,
            use_residual=args.use_residual,
            dropout_rate=args.dropout_rate,
            padding_idx=padding_idx,
            resume=args.encoder_resume,
            is_student=is_student,
            teacher_args=teacher_args,
            share_proj=args.share_proj,
        )
        dec_idim = (
            args.eunits
            if args.spk_embed_dim is None
            else args.eunits + args.spk_embed_dim
        )
        
        self.dec = Decoder(
            idim=dec_idim,
            odim=odim,
            dlayers=args.dlayers,
            dunits=args.dunits,
            prenet_layers=args.prenet_layers,
            prenet_units=args.prenet_units,
            postnet_layers=args.postnet_layers,
            postnet_chans=args.postnet_chans,
            postnet_filts=args.postnet_filts,
            output_activation_fn=self.output_activation_fn,
            use_batch_norm=args.use_batch_norm,
            use_concate=args.use_concate,
            dropout_rate=args.dropout_rate,
            zoneout_rate=args.zoneout_rate,
            reduction_factor=args.reduction_factor,
            use_fe_condition=args.use_fe_condition,
            append_position=args.append_position,
            is_student=is_student,
            teacher_args=teacher_args,
            share_proj=args.share_proj,
        )
        
        self.duration_predictor = DurationPredictor(
            idim=dec_idim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
        )
        reduction = 'none' if args.use_weighted_masking else 'mean'
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        
        #-------------- picth/energy predictor definition ---------------#
        if self.use_fe_condition:
            output_dim=1
            # pitch prediction
            pitch_predictor_layers=2
            pitch_predictor_chans=384
            pitch_predictor_kernel_size=3
            pitch_predictor_dropout_rate=0.5
            pitch_embed_kernel_size=9
            pitch_embed_dropout_rate=0.5
            self.stop_gradient_from_pitch_predictor=False
            self.pitch_predictor = VariancePredictor(
                    idim=dec_idim,
                    n_layers=pitch_predictor_layers,
                    n_chans=pitch_predictor_chans,
                    kernel_size=pitch_predictor_kernel_size,
                    dropout_rate=pitch_predictor_dropout_rate,
                    output_dim=output_dim,
                    )
            self.pitch_embed = torch.nn.Sequential(
                    torch.nn.Conv1d(
                            in_channels=1,
                            out_channels=dec_idim,
                            kernel_size=pitch_embed_kernel_size,
                            padding=(pitch_embed_kernel_size-1)//2,
                            ),
                            torch.nn.Dropout(pitch_embed_dropout_rate),
                            )
            # energy prediction
            energy_predictor_layers=2
            energy_predictor_chans=384
            energy_predictor_kernel_size=3
            energy_predictor_dropout_rate=0.5
            energy_embed_kernel_size=9
            energy_embed_dropout_rate=0.5
            self.stop_gradient_from_energy_predictor=False
            self.energy_predictor = VariancePredictor(
                    idim=dec_idim,
                    n_layers=energy_predictor_layers,
                    n_chans=energy_predictor_chans,
                    kernel_size=energy_predictor_kernel_size,
                    dropout_rate=energy_predictor_dropout_rate,
                    output_dim=output_dim,
                    )
            self.energy_embed = torch.nn.Sequential(
                    torch.nn.Conv1d(
                            in_channels=1,
                            out_channels=dec_idim,
                            kernel_size=energy_embed_kernel_size,
                            padding=(energy_embed_kernel_size-1)//2,
                            ),
                            torch.nn.Dropout(energy_embed_dropout_rate),
                            )
            
            self.pemb_proj = torch.nn.Linear(args.eunits, teacher_args.eunits, bias=False)
            self.eemb_proj = torch.nn.Linear(args.eunits, teacher_args.eunits, bias=False)
            # define criterions
            self.prosody_criterion = prosody_criterions(
                    use_masking=args.use_masking, use_weighted_masking=args.use_weighted_masking)
           
            
        self.taco2_loss = Tacotron2Loss(
            use_masking=args.use_masking,
            use_weighted_masking=args.use_weighted_masking,
        )
        self.taco2_knowledge_loss = Tacotron2Loss_KD(
            use_masking=args.use_masking,
            use_weighted_masking=args.use_weighted_masking,
        )
        self.knowledge_loss = Knowledge_loss( 
            use_masking=args.use_masking,
            use_weighted_masking=args.use_weighted_masking,
        )
        # load pretrained model
        if args.pretrained_model is not None:
            self.load_pretrained_model(args.pretrained_model)
            
        print('\n############## number of network parameters ##############\n')
        none_params = 0
        if self.share_proj:
            for net in [self.enc.embed_proj, self.enc.convs_proj, self.enc.blstm_proj, self.pemb_proj, self.eemb_proj,
                        self.dec.prenet_proj, self.dec.lstm_proj, self.dec.post_proj]:
                none_params += cal_num_params(net)
        else:
            for net in [self.enc.embed_proj, self.enc.convs_proj, self.enc.blstm_proj, self.pemb_proj, self.eemb_proj,
                        self.dec.prenet_proj, self.dec.lstm0_proj, self.dec.lstm1_proj, self.dec.post0_proj, self.dec.post1_proj,
                        self.dec.post2_proj, self.dec.post3_proj]:
                none_params += cal_num_params(net)
                
        parameters = filter(lambda p: p.requires_grad, self.enc.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for Encoder: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.dec.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for Decoder: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.duration_predictor.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for duration_predictor: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.pitch_predictor.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for pitch_predictor: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.energy_predictor.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for energy_predictor: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.pitch_embed.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for pitch_embed: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.energy_embed.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for energy_embed: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for whole network: %.5fM' % parameters)
        print('Trainable Parameters for whole network: %.5fM' % (parameters-none_params))
        
        print('\n##########################################################\n')
        

    def forward(
        self, xs, ilens, ys, olens, spembs=None, extras=None, new_ys=None, non_zero_lens_mask=None, ds_nonzeros=None, output_masks=None,
        position=None, f0=None, energy=None, teacher_knowledge=None, *args, **kwargs
    ):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
            spembs (Tensor, optional):
                Batch of speaker embedding vectors (B, spk_embed_dim).
            extras (Tensor, optional):
                Batch of groundtruth spectrograms (B, Lmax, spc_dim).
            new_ys (Tensor): reorganized mel-spectrograms
            non_zero_lens_masks (Tensor)
            ds_nonzeros (Tensor)
            output_masks (Tensor)
            position (Tenor): position values for each phoneme
            f0 (Tensor): pitch
            energy (Tensor)
        Returns:
            Tensor: Loss value.
        """
        after_outs_know, before_outs_know, enc_teacher_know, dec_teacher_know, prosody_teacher_know = teacher_knowledge
        # remove unnecessary padded part (for multi-gpus)
        max_in = max(ilens)
        max_out = max(olens)
        if max_in != xs.shape[1]:
            xs = xs[:, :max_in]
        if max_out != ys.shape[1]:
            ys = ys[:, :max_out]

        # calculate FCL-taco2-enc outputs
        hs, hlens, enc_distill_items = self.enc(xs, ilens)
        
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
            
        # duration predictor loss cal
        ds = extras.squeeze(-1)
        d_masks = make_pad_mask(ilens).to(xs.device)
        d_outs = self.duration_predictor(hs, d_masks) # (B, Tmax)
        duration_masks = make_non_pad_mask(ilens).to(ys.device)
#        d_outs = d_outs.masked_select(duration_masks)
        duration_loss = self.duration_criterion(d_outs.masked_select(duration_masks), ds.masked_select(duration_masks))
        d_outs = d_outs.unsqueeze(-1)
        
        if self.use_fe_condition:
            expand_hs = hs
            fe_masks = d_masks
            if self.stop_gradient_from_pitch_predictor:
                p_outs = self.pitch_predictor(expand_hs.detach(), fe_masks.unsqueeze(-1))
            else:
                p_outs = self.pitch_predictor(expand_hs, fe_masks.unsqueeze(-1)) # B x Tmax x 1
            
            if self.stop_gradient_from_energy_predictor:
                e_outs = self.energy_predictor(expand_hs.detach(), fe_masks.unsqueeze(-1))
            else:
                e_outs = self.energy_predictor(expand_hs, fe_masks.unsqueeze(-1)) # B x Tmax x 1
                
            pitch_loss = self.prosody_criterion(p_outs,f0,ilens)
            energy_loss = self.prosody_criterion(e_outs,energy,ilens)
            p_embs = self.pitch_embed(f0.transpose(1,2)).transpose(1,2)
            e_embs = self.energy_embed(energy.transpose(1,2)).transpose(1,2)
        else:
            p_embs = None
            e_embs = None
        
        ylens = olens
        after_outs, before_outs, dec_distill_items = self.dec(hs, hlens, ds, ys, ylens,
                                                               new_ys, non_zero_lens_mask,
                                                               ds_nonzeros, output_masks, position,
                                                               p_embs, e_embs)
        
        p_embs = self.pemb_proj(p_embs)
        e_embs = self.eemb_proj(e_embs)
        prosody_distill_items = [d_outs, p_outs, e_outs, p_embs, e_embs]
        
        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            
        # caluculate taco2 loss
        l1_loss, mse_loss = self.taco2_loss(
            after_outs, before_outs, ys, olens
        )
        loss = l1_loss + mse_loss + duration_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"mse_loss": mse_loss.item()},
            {"dur_loss": duration_loss.item()},
        ]

        if self.use_fe_condition:
            prosody_weight = 1.0
            loss = loss + prosody_weight * (pitch_loss + energy_loss)
            report_keys += [
                    {'pitch_loss': pitch_loss.item()},
                    {'energy_loss': energy_loss.item()},
                    ]
        
        if self.distill_output_knowledge:
            output_l1_loss, output_mse_loss = self.taco2_knowledge_loss(after_outs, before_outs, after_outs_know, before_outs_know, olens)
            loss = loss + output_l1_loss + output_mse_loss
            report_keys += [{'output_l1_loss': output_l1_loss.items()}]
            report_keys += [{'output_mse_loss': output_mse_loss.items()}]
            
        if self.distill_encoder_knowledge:
            encoder_loss = self.knowledge_loss(enc_distill_items, enc_teacher_know, ilens)
            loss = loss + encoder_loss
            report_keys += [{'encoder_loss': encoder_loss.items()}]
        
        if self.distill_decoder_knowledge:
            decoder_loss = self.knowledge_loss(dec_distill_items, dec_teacher_know, olens)
            loss = loss + decoder_loss
            report_keys += [{'decoder_loss': decoder_loss.items()}]
            
        if self.distill_prosody_knowledge:
            prosody_loss = self.knowledge_loss(prosody_distill_items, prosody_teacher_know, ilens)
            loss = loss + prosody_loss
            report_keys += [{'prosody_loss': prosody_loss.items()}]
            
        report_keys += [{"loss": loss.item()}]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, spemb=None, dur=None, f0=None, energy=None,
                  utt_id=None, y=None, *args, **kwargs):
        """Generate the sequence of features given the sequences of characters.
        Args:
            x (Tensor): Input sequence of characters (T,).
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).
        Returns:
            Tensor: Output sequence of features (L, odim).
        """
        # inference
        h = self.enc.inference(x) # Tmax x h-dim
        
        if self.spk_embed_dim is not None:
            spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        
        ilens = torch.LongTensor([h.shape[0]]).to(h.device)
        d_masks = make_pad_mask(ilens).to(h.device)
        if dur is not None:
            d_outs = dur.reshape(-1).long()
        else:
            d_outs = self.duration_predictor.inference(h.unsqueeze(0), d_masks) # B x Tmax
            d_outs = d_outs.squeeze(0).long()
        
        if self.use_fe_condition:
            if f0 is not None:
                p_outs = f0.unsqueeze(0)
                e_outs = energy.unsqueeze(0)
            else:
                expand_hs = h.unsqueeze(0)
                fe_masks = d_masks
                p_outs = self.pitch_predictor(expand_hs, fe_masks.unsqueeze(-1))
                e_outs = self.energy_predictor(expand_hs, fe_masks.unsqueeze(-1))
            p_embs = self.pitch_embed(p_outs.transpose(1,2)).transpose(1,2).squeeze(0)
            e_embs = self.energy_embed(e_outs.transpose(1,2)).transpose(1,2).squeeze(0)
        else:
            p_outs = None
            e_outs = None
            p_embs = None
            e_embs = None
            
        if self.append_position:
            position = []
            for iid in range(d_outs.shape[0]):
                if d_outs[iid] != 0:
                    position.append(torch.FloatTensor(list(range(d_outs[iid].long())))/d_outs[iid])
            position = pad_list(position, 0)
            position = position.to(h.device)
        else:
            position = None
            
        outs = self.dec.inference(
            h,
            d_outs,
            position,
            p_embs,
            e_embs,
        )
        
        return outs

    @property
    def base_plot_keys(self):
        """Return base key names to plot during training.
        keys should match what `chainer.reporter` reports.
        If you add the key `loss`, the reporter will report `main/loss`
        and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss`
        and `validation/main/loss` values.
        Returns:
            list: List of strings which are base keys to plot during training.
        """
        plot_keys = ["loss", "l1_loss", "mse_loss", "dur_loss"]
    
        if self.use_fe_condition:
            plot_keys += ["pitch_loss", "energy_loss"]
        
        if self.distill_output_knowledge:
            plot_keys += ['output_l1_loss', 'output_mse_loss']
        
        if self.distill_encoder_knowledge:
            plot_keys += ['encoder_loss']
        
        if self.distill_decoder_knowledge:
            plot_keys += ['decoder_loss']
            
        if self.use_fe_condition and self.distill_prosody_knowledge:
            plot_keys += ['prosody_loss']
            
        return plot_keys