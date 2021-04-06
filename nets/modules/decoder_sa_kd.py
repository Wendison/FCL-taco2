#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tacotron2_sa_KD decoder related modules."""

import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.nets_utils import pad_list

import time 
from joblib import Parallel, delayed
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

import numpy as np

def decoder_init(m):
    """Initialize decoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class ZoneOutCell(torch.nn.Module):
    """ZoneOut Cell module.

    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.

    Examples:
        >>> lstm = torch.nn.LSTMCell(16, 32)
        >>> lstm = ZoneOutCell(lstm, 0.5)

    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305

    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch

    """

    def __init__(self, cell, zoneout_rate=0.1):
        """Initialize zone out cell module.

        Args:
            cell (torch.nn.Module): Pytorch recurrent cell module
                e.g. `torch.nn.Module.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.

        """
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
        if zoneout_rate > 1.0 or zoneout_rate < 0.0:
            raise ValueError(
                "zoneout probability must be in the range from 0.0 to 1.0."
            )

    def forward(self, inputs, hidden):
        """Calculate forward propagation.

        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).

        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).

        """
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class Prenet(torch.nn.Module):
    """Prenet module for decoder of Spectrogram prediction network.

    This is a module of Prenet in the decoder of Spectrogram prediction network,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Prenet preforms nonlinear conversion
    of inputs before input to auto-regressive lstm,
    which helps to learn diagonal attentions.

    Note:
        This module alway applies dropout even in evaluation.
        See the detail in `Natural TTS Synthesis by
        Conditioning WaveNet on Mel Spectrogram Predictions`_.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        """Initialize prenet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of prenet layers.
            n_units (int, optional): The number of prenet units.

        """
        super(Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.prenet = torch.nn.ModuleList()
        for layer in six.moves.range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet += [
                torch.nn.Sequential(torch.nn.Linear(n_inputs, n_units), torch.nn.ReLU())
            ]
        # print('number of params of Prenet:')
        # print(self.num_params())
        
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
            
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., idim).

        Returns:
            Tensor: Batch of output tensors (B, ..., odim).

        """
        for i in six.moves.range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x


class Block_Prenet(torch.nn.Module):
    """Block-Prenet module of decoder for processing the adjacent block mel-spectrogram.
    """

    def __init__(self, idim, n_layers=2, n_chans=256, dropout_rate=0.5):
        """Initialize prenet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of convolutional layers.
            n_chans (int, optional): The number of prenet layers.

        """
        super(Block_Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.block_prenet = torch.nn.ModuleList()
        kernel_size = 3
        for layer in six.moves.range(n_layers):
            n_inputs = idim if layer == 0 else n_chans
            self.block_prenet += [
                torch.nn.Sequential(torch.nn.Conv1d(n_inputs, n_chans, kernel_size, 1, (kernel_size-1)//2), torch.nn.ReLU())
            ]
            # self.block_prenet += [
            #     torch.nn.Sequential(torch.nn.Linear(n_inputs, n_chans), torch.nn.ReLU())
            # ]
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        # print('number of params of Block_Prenet:')
        # print(self.num_params())
        
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
            
    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, idim, tmp_L).

        Returns:
            Tensor: Batch of output tensors (B, odim, 1).

        """
        # print('\n')
        # print(x.shape)
        # x = x.transpose(1,2)
        #(B, idim, tmp_L) -> (B, nchans, tmp_L) -> (B, nchans, tmp_L) -> (B, nchans, 1)
        for i in six.moves.range(len(self.block_prenet)):
            x = F.dropout(self.block_prenet[i](x), self.dropout_rate)
            # print(x.shape)
        # x = x.transpose(1,2)
        x = F.dropout(self.max_pool(x), self.dropout_rate)
        # print(x.shape)
        # print('\n')
        return x
    

class Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        odim,
        n_layers=5,
        n_chans=512,
        n_filts=5,
        dropout_rate=0.5,
        use_batch_norm=True,
    ):
        """Initialize postnet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..

        """
        super(Postnet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in six.moves.range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
            else:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        else:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
    
        # print('number of params of PostNet:')
        # print(self.num_params())
        
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).

        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).

        """
        # len(postnet) = 5
        xs_conv0 = self.postnet[0](xs)
        xs_conv1 = self.postnet[1](xs_conv0)
        xs_conv2 = self.postnet[2](xs_conv1)
        xs_conv3 = self.postnet[3](xs_conv2)
        xs_conv4 = self.postnet[4](xs_conv3)
        # for i in six.moves.range(len(self.postnet)):
        #     xs = self.postnet[i](xs)
        return xs_conv0, xs_conv1, xs_conv2, xs_conv3, xs_conv4


def cal_num_params(net):
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return parameters


class Decoder(torch.nn.Module):
    """Decoder module of Spectrogram prediction network.

    This is a module of decoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The decoder generates the sequence of
    features from the sequence of the hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        odim,
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        output_activation_fn=None,
        use_batch_norm=True,
        use_concate=True,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=1,
        use_fe_condition=True,
        append_position=False,
        is_student=True,
        teacher_args=None,
        share_proj=False,
    ):
        """Initialize Tacotron2 decoder module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            dlayers (int, optional): The number of decoder lstm layers.
            dunits (int, optional): The number of decoder lstm units.
            prenet_layers (int, optional): The number of prenet layers.
            prenet_units (int, optional): The number of prenet units.
            postnet_layers (int, optional): The number of postnet layers.
            postnet_filts (int, optional): The number of postnet filter size.
            postnet_chans (int, optional): The number of postnet filter channels.
            output_activation_fn (torch.nn.Module, optional):
                Activation function for outputs.
            
            use_batch_norm (bool, optional): Whether to use batch normalization.
            use_concate (bool, optional): Whether to concatenate encoder embedding
                with decoder lstm outputs.
            dropout_rate (float, optional): Dropout rate.
            zoneout_rate (float, optional): Zoneout rate.
            reduction_factor (int, optional): Reduction factor.

        """
        super(Decoder, self).__init__()

        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.output_activation_fn = output_activation_fn
        self.use_concate = use_concate
        self.reduction_factor = reduction_factor
        self.use_fe_condition = use_fe_condition
        self.append_position = append_position
        self.is_student = is_student

        # define lstm network
        prenet_units = prenet_units if prenet_layers != 0 else odim
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(dlayers):
            iunits = dunits
            if layer == 0:
                iunits = idim + prenet_units
                if append_position:
                    iunits += 1
            lstm = torch.nn.LSTMCell(iunits, dunits)
            if zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, zoneout_rate)
            self.lstm += [lstm]
        
        # define prenet
        if prenet_layers > 0:
            self.prenet = Prenet(
                idim=odim,
                n_layers=prenet_layers,
                n_units=prenet_units,
                dropout_rate=dropout_rate,
            )
        else:
            self.prenet = None
    
    
        if postnet_layers > 0:
            self.postnet = Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
            )
        else:
            self.postnet = None
        
        #print('postnet:', self.postnet)
        
        # define projection layers
        iunits = idim + dunits if use_concate else dunits
        self.feat_out = torch.nn.Linear(iunits, odim * reduction_factor, bias=False)
       
        self.share_proj = share_proj
        if self.is_student:
            self.prenet_proj = torch.nn.Linear(prenet_units, teacher_args.prenet_units, bias=False)
            if self.share_proj:
                self.lstm_proj = torch.nn.Linear(dunits, teacher_args.dunits, bias=False)
                self.post_proj = torch.nn.Linear(postnet_chans, teacher_args.postnet_chans, bias=False)
            else:
                self.lstm1_proj = torch.nn.Linear(dunits, teacher_args.dunits, bias=False)
                self.post0_proj = torch.nn.Linear(postnet_chans, teacher_args.postnet_chans, bias=False)
                self.post1_proj = torch.nn.Linear(postnet_chans, teacher_args.postnet_chans, bias=False)
                self.post2_proj = torch.nn.Linear(postnet_chans, teacher_args.postnet_chans, bias=False)
                self.post3_proj = torch.nn.Linear(postnet_chans, teacher_args.postnet_chans, bias=False)
        
        # initialize
        self.apply(decoder_init)
        
        parameters = filter(lambda p: p.requires_grad, self.lstm.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for decoder-lstm: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.feat_out.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for decoder-feat_out: %.5fM' % parameters)
        
        parameters = filter(lambda p: p.requires_grad, self.prenet.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters for decoder-prenet: %.5fM' % parameters)
        
        if postnet_layers > 0:
            parameters = filter(lambda p: p.requires_grad, self.postnet.parameters())
            parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
            print('Trainable Parameters for decoder-postnet: %.5fM' % parameters)
        
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
    
    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs
    
    
    def forward(self, hs, hlens, ds, ys, ylens, new_ys, non_zero_lens_mask, ds_nonzeros, output_masks=None, position=None,
                p_embs=None, e_embs=None): # sort by input(text) length?
        """Calculate forward propagation.

        Args:
            
            hs (Tensor): Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (LongTensor): Batch of lengths of each input batch (B,).
            ds (LongTensor): Batch of lengths of each phn/char for each text sample (B, max(hlens)).
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).
            ylens (LongTensor): Batch of lengths of each output batch (B,).
            

        Returns:
            Tensor: Batch of output tensors after postnet (B, Lmax, odim).
            Tensor: Batch of output tensors before postnet (B, Lmax, odim).
            Tensor: Batch of logits of stop prediction (B, Lmax).
            Tensor: Batch of attention weights (B, Lmax, Tmax).

        Note:
            This computation is performed in teacher-forcing manner.

        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1 :: self.reduction_factor] # for each consecutive two frames, only use the second frame as condition 
        
        hlens = list(map(int, hlens))
        
        if self.use_fe_condition:
            hs = hs + p_embs + e_embs
        
        # reorganize hs & ys 
        # now = time.time()
        
        
        nhlens = [sum(nzlm) for nzlm in non_zero_lens_mask] # length of phns/chars in each utt
        assert len(nhlens) == hs.shape[0]
       
        hs = hs[non_zero_lens_mask.eq(1)].unsqueeze(1)
    
        assert hs.shape[0] == len(ds_nonzeros)
        ys = new_ys # new_B x new_Lmax x odim
        
        # length list should be list of int
        hlens = list(map(int, hlens))
        ylens = list(map(int, ylens))
        
        # initialize hidden states of decoder
        c_list = [[self._zero_state(hs)]]
        z_list = [[self._zero_state(hs)]]
        for _ in six.moves.range(1, len(self.lstm)):
            c_list.append([self._zero_state(hs)])
            z_list.append([self._zero_state(hs)])
        
        prev_out = hs.new_zeros(hs.size(0), self.odim)
        
        # loop for an output sequence
        # now = time.time()
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1 :: self.reduction_factor] # for each consecutive two frames, only use the second frame as condition 
        
        outs = [] 
        prenet_outs = []
        lstms0 = []
        lstms1 = []
        for itt, y in enumerate(ys.transpose(0, 1)): # y: new_B x odim, total num of loop is new_Lmax
            att_c = hs.squeeze(1) # new_B x idim
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out # new_B x prenet_units
            
            # save prenet_out for distillation
            prenet_outs.append(prenet_out)
            
            base_cat = [att_c, prenet_out]
            if self.append_position:
                base_cat.append(position[:,itt].reshape(-1,1))
            # if self.use_fe_condition:
            #     base_cat.append(fe)
            # if self.use_block_context:
            #     base_cat.append(block_context)
            xs = torch.cat(base_cat, dim=1) 

            tmp_z, tmp_c = self.lstm[0](xs, (z_list[0][-1], c_list[0][-1]))
            z_list[0].append(tmp_z) # each: new_B x lstm_hidden_dim
            c_list[0].append(tmp_c)
            
            for i in six.moves.range(1, len(self.lstm)):
                tmp_z,tmp_c = self.lstm[i](
                    z_list[i - 1][-1], (z_list[i][-1], c_list[i][-1])
                )
                z_list[i].append(tmp_z)
                c_list[i].append(tmp_c)
            feat_cat = [z_list[-1][-1], att_c]

            zcs = (
                torch.cat(feat_cat, dim=1)
                if self.use_concate
                else z_list[-1]
            )  # zcs: new_B x (dim)
            cur_outs = self.feat_out(zcs).view(hs.size(0), self.odim, -1)
            outs += [cur_outs] # each: new_B x odim x reduction_factor
            prev_out = y  # teacher forcing
        
        prenet_outs = torch.stack(prenet_outs, dim=1) # new_B x new_Lmax x prenet_units
        lstms0 = torch.stack(z_list[0][1:], dim=1) # new_B x new_Lmax x lstm_hidden_dim
        lstms1 = torch.stack(z_list[1][1:], dim=1) # new_B x new_Lmax x lstm_hidden_dim
        #print('prenet_outs.shape:', prenet_outs.shape, 'lstm.shape:', lstms0.shape, lstms1.shape)
        # now = time.time()
        cat_outs = torch.cat(outs, dim=2)  # (new_B, odim, new_Lmax*reduction_factor)
        
        cat_outs = cat_outs.transpose(1,2)[output_masks] # 
        prenet_outs = prenet_outs[output_masks]
        lstms0 = lstms0[output_masks]
        lstms1 = lstms1[output_masks]
        #print('after out_masks, cat_outs.shape:', cat_outs.shape)
        #print('after out_masks, prenet_outs.shape:', prenet_outs.shape, 'lstm.shape:', lstms0.shape, lstms1.shape)
        tmp_outs = []
        tmp_prenet_outs = []
        tmp_lstms0 = []
        tmp_lstms1 = []
        for ih in range(len(ylens)):
            start = int(sum(ylens[:ih]))
            end = int(sum(ylens[:ih+1]))
            tmp_outs.append(cat_outs[start:end,:])
            tmp_prenet_outs.append(prenet_outs[start:end,:])
            tmp_lstms0.append(lstms0[start:end,:])
            tmp_lstms1.append(lstms1[start:end,:])
            
        before_outs = pad_list(tmp_outs,0).transpose(1,2) # (B, odim, Lmax)
        prenet_outs = pad_list(tmp_prenet_outs,0) # B x Lmax x prenet_units
        lstms0 = pad_list(tmp_lstms0,0) # B x Lmax x lstm_hidden_dim
        lstms1 = pad_list(tmp_lstms1,0) # B x Lmax x lstm_hidden_dim
        #print('after pad_list, prenet_outs.shape:', prenet_outs.shape, 'lstm.shape:', lstms0.shape, lstms1.shape)
        if self.is_student:
            prenet_outs = self.prenet_proj(prenet_outs)
            if self.share_proj:
                lstms0 = self.lstm_proj(lstms0)
                lstms1 = self.lstm_proj(lstms1)
            else:
                lstms0 = self.lstm0_proj(lstms0)
                lstms1 = self.lstm1_proj(lstms1)

        if self.reduction_factor > 1:
            before_outs = before_outs.view(
                before_outs.size(0), self.odim, -1
            )  # (B, odim, Lmax)

        if self.postnet is not None:
            ys_conv0, ys_conv1, ys_conv2, ys_conv3, ys_conv4 = self.postnet(before_outs)  # (B, odim, Lmax)
            after_outs = before_outs + ys_conv4
            #print('ys_conv.shape:', ys_conv0.shape, ys_conv1.shape, ys_conv2.shape, ys_conv3.shape, after_outs.shape)
            if self.is_student:
                if self.share_proj:
                    ys_conv0 = self.post_proj(ys_conv0.transpose(1,2))
                    ys_conv1 = self.post_proj(ys_conv1.transpose(1,2))
                    ys_conv2 = self.post_proj(ys_conv2.transpose(1,2))
                    ys_conv3 = self.post_proj(ys_conv3.transpose(1,2))
                else:
                    ys_conv0 = self.post0_proj(ys_conv0.transpose(1,2))
                    ys_conv1 = self.post1_proj(ys_conv1.transpose(1,2))
                    ys_conv2 = self.post2_proj(ys_conv2.transpose(1,2))
                    ys_conv3 = self.post3_proj(ys_conv3.transpose(1,2))
            else:
                ys_conv0 = ys_conv0.transpose(1,2)
                ys_conv1 = ys_conv1.transpose(1,2)
                ys_conv2 = ys_conv2.transpose(1,2)
                ys_conv3 = ys_conv3.transpose(1,2)
            ys_conv4 = ys_conv4.transpose(1,2)
        else:
            after_outs = before_outs
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        
        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)
        
        distill_items = [prenet_outs, lstms0, lstms1, ys_conv0, ys_conv1, ys_conv2, ys_conv3, ys_conv4]
        
        return after_outs, before_outs, distill_items


    def inference(
        self,
        h,
        ds,
        position,
        p_embs=None,
        e_embs=None,
    ):
        """Generate the sequence of features given the sequences of characters.

        Args:
            h (Tensor): Input sequence of encoder hidden states (T, C).

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        Note:
            This computation is performed in auto-regressive manner.

        .. _`Deep Voice 3`: https://arxiv.org/abs/1710.07654

        """
        # setup
        assert len(h.size()) == 2
        if self.use_fe_condition:
            h = h + p_embs + e_embs
            
        ds_nonzeros = ds.view(-1)[ds.view(-1).ne(0)] * self.reduction_factor

        hs = h # (# of phn, idim)
        assert ds_nonzeros.shape[0] == hs.shape[0]
        

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in six.moves.range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        
        prev_out = hs.new_zeros(hs.shape[0], self.odim)
        
        outs = []
        max_out_length = max(ds)
        for im in range(max_out_length):
            att_c = hs
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            base_cat = [att_c, prenet_out] 
            if self.append_position:
                base_cat.append(position[:,im].reshape(-1,1))
           
            xs = torch.cat(base_cat, dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in six.moves.range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            feat_cat = [z_list[-1], att_c]
            zcs = (
                torch.cat(feat_cat, dim=1)
                if self.use_concate
                else z_list[-1]
            ) # new_B x (dim)
            cur_outs = self.feat_out(zcs).view(hs.shape[0], self.odim, -1)
            outs += [cur_outs]  # each: new_B x odim x reduction_factor
            # probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (#-of-phn, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (#-of-phn, odim)
            
        
        cat_outs = torch.cat(outs, dim=2)  # (new_B, odim, reduction_factor*max(ds))
        outs_ih = cat_outs # end-start x odim x new_Lmax*reduction_factor
        ds_ih = ds # duration for each phn/char
        new_outs_ih = []
        fn = 0
        for io in range(hs.shape[0]):
            if ds_ih[io] !=0:
                new_outs_ih.append(outs_ih[fn,:,:int(self.reduction_factor*ds_ih[io])])
                fn += 1
        new_outs_ih = torch.cat(new_outs_ih,-1).transpose(0,1) # L (length of ys for ih utt) x odim    
        before_outs = new_outs_ih.unsqueeze(0).transpose(1,2) # 1 x odim x L
        _, _, _, _, ys_conv4 = self.postnet(before_outs)
        outs = before_outs + ys_conv4  # (1, odim, L)
        outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
        
        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        
        return outs
