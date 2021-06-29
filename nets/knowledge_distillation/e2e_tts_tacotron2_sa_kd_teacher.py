#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tacotron2_sa_kd_teacher related modules."""

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

    def __init__(self, idim, odim, args=None, com_args=None):
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
        
        args = argparse.Namespace(**args)
        
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.embed_dim = args.embed_dim
        self.spk_embed_dim = args.spk_embed_dim
        self.reduction_factor = args.reduction_factor
        self.use_fe_condition = args.use_fe_condition
        self.append_position = args.append_position
        
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
            is_student=False,
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
            is_student=False,
        )
        
        self.duration_predictor = DurationPredictor(
            idim=dec_idim,
            n_layers=args.duration_predictor_layers,
            n_chans=args.duration_predictor_chans,
            kernel_size=args.duration_predictor_kernel_size,
            dropout_rate=args.duration_predictor_dropout_rate,
        )
#        reduction = 'none' if args.use_weighted_masking else 'mean'
#        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        
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
#            # define criterions
#            self.prosody_criterion = prosody_criterions(
#                    use_masking=args.use_masking, use_weighted_masking=args.use_weighted_masking)
           
            
        self.taco2_loss = Tacotron2Loss(
            use_masking=args.use_masking,
            use_weighted_masking=args.use_weighted_masking,
        )
        
        # load pretrained model
        if args.pretrained_model is not None:
            self.load_pretrained_model(args.pretrained_model)
            
        print('\n############## number of network parameters ##############\n')
              
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
        
        print('\n##########################################################\n')
        

    def forward(
        self, xs, ilens, ys, olens, spembs=None, extras=None, new_ys=None, non_zero_lens_mask=None, ds_nonzeros=None, output_masks=None,
        position=None, f0=None, energy=None, *args, **kwargs
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
        d_outs = d_outs.unsqueeze(-1) # (B, Tmax, 1)
        
#        duration_masks = make_non_pad_mask(ilens).to(ys.device)
#        d_outs = d_outs.masked_select(duration_masks)
#        duration_loss = self.duration_criterion(d_outs, ds.masked_select(duration_masks))
        
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
                
#            pitch_loss = self.prosody_criterion(p_outs,f0,ilens)
#            energy_loss = self.prosody_criterion(e_outs,energy,ilens)
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
        
        prosody_distill_items = [d_outs, p_outs, e_outs, p_embs, e_embs]
        
        enc_distill_items = self.detach_items(enc_distill_items)
        dec_distill_items = self.detach_items(dec_distill_items)
        prosody_distill_items = self.detach_items(prosody_distill_items)

        return after_outs, before_outs, enc_distill_items, dec_distill_items, prosody_distill_items


    def detach_items(self, items):
        items = [it.detach() for it in items]
        return items