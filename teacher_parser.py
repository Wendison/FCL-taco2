#!/usr/bin/env python3

"""FCL-taco2-T model parser"""

import logging
import os
import random
import subprocess
import sys

import configargparse
import numpy as np

from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES


# NOTE: you need this func to generate our sphinx doc
def get_teacher_parser():
    """Get parser of training arguments."""
    parser = configargparse.ArgumentParser(
        description="Train a new text-to-speech (TTS) model on one CPU, "
        "one or multiple GPUs",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # general configuration
    parser.add("--config", default='conf/train_pytorch_tacotron2.sa.teacher.yaml', is_config_file=True, help="config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="second config file path that overwrites the settings in `--config`.",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="third config file path that overwrites "
        "the settings in `--config` and `--config2`.",
    )

    parser.add_argument(
        "--ngpu",
        default=None,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--backend",
        default="pytorch_distill",
        type=str,
        choices=["chainer", "pytorch", "pytorch_distill"],
        help="Backend library",
    )
    #parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument("--seed", default=137, type=int, help="Random seed")
    parser.add_argument(
        "--resume",
        "-r",
        default="",
        type=str,
        nargs="?",
        help="Resume the training from snapshot",
    )
    parser.add_argument(
        "--minibatches",
        "-N",
        type=int,
        default="0",
        help="Process only N minibatches (for debug)",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        nargs="?",
        help="Tensorboard log directory path",
    )
    parser.add_argument(
        "--eval-interval-epochs", default=1, type=int, help="Evaluation interval epochs"
    )
    parser.add_argument(
        "--save-interval-epochs", default=1, type=int, help="Save interval epochs"
    )
    parser.add_argument(
        "--report-interval-iters",
        default=100,
        type=int,
        help="Report interval iterations",
    )
    # task related
    parser.add_argument(
        "--train-json", type=str, required=False, help="Filename of training json"
    )
    parser.add_argument(
        "--valid-json", type=str, required=False, help="Filename of validation json"
    )
    # network architecture
    parser.add_argument(
        "--model-module",
        type=str,
        default="nets.knowledge_distillation.e2e_tts_tacotron2_sa_kd_teacher:Tacotron2",
        help="model defined module",
    )
    # minibatch related
    parser.add_argument(
        "--sortagrad",
        default=0,
        type=int,
        nargs="?",
        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",
    )
    parser.add_argument(
        "--batch-sort-key",
        default="shuffle",
        type=str,
        choices=["shuffle", "output", "input"],
        nargs="?",
        help='Batch sorting key. "shuffle" only work with --batch-count "seq".',
    )
    parser.add_argument(
        "--batch-count",
        default="auto",
        choices=BATCH_COUNT_CHOICES,
        help="How to count batch_size. "
        "The default (auto) will find how to count by args.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch-seqs",
        "-b",
        default=0,
        type=int,
        help="Maximum seqs in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-bins",
        default=0,
        type=int,
        help="Maximum bins in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-in",
        default=0,
        type=int,
        help="Maximum input frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-out",
        default=0,
        type=int,
        help="Maximum output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-inout",
        default=0,
        type=int,
        help="Maximum input+output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--maxlen-in",
        "--batch-seq-maxlen-in",
        default=100,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the input sequence length > ML.",
    )
    parser.add_argument(
        "--maxlen-out",
        "--batch-seq-maxlen-out",
        default=200,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the output sequence length > ML",
    )
    parser.add_argument(
        "--num-iter-processes",
        default=0,
        type=int,
        help="Number of processes of iterator",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    parser.add_argument(
        "--use-speaker-embedding",
        default=False,
        type=strtobool,
        help="Whether to use speaker embedding",
    )
    parser.add_argument(
        "--use-second-target",
        default=False,
        type=strtobool,
        help="Whether to use second target",
    )
    # optimization related
    parser.add_argument(
        "--opt", default="adam", type=str, choices=["adam", "noam"], help="Optimizer"
    )
    parser.add_argument(
        "--accum-grad", default=1, type=int, help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate for optimizer"
    )
    parser.add_argument("--eps", default=1e-6, type=float, help="Epsilon for optimizer")
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        help="Weight decay coefficient for optimizer",
    )
    parser.add_argument(
        "--epochs", "-e", default=30, type=int, help="Number of maximum epochs"
    )
    parser.add_argument(
        "--early-stop-criterion",
        default="validation/main/loss",
        type=str,
        nargs="?",
        help="Value to monitor to trigger an early stopping of the training",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs to wait "
        "without improvement before stopping the training",
    )
    parser.add_argument(
        "--grad-clip", default=1, type=float, help="Gradient norm threshold to clip"
    )
    parser.add_argument(
        "--num-save-attention",
        default=0,
        type=int,
        help="Number of samples of attention to be saved",
    )
    parser.add_argument(
        "--keep-all-data-on-mem",
        default=False,
        type=strtobool,
        help="Whether to keep all data on memory",
    )
    # finetuning related
    parser.add_argument(
        "--enc-init",
        default=None,
        type=str,
        help="Pre-trained TTS model path to initialize encoder.",
    )
    parser.add_argument(
        "--enc-init-mods",
        default="enc.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--dec-init",
        default=None,
        type=str,
        help="Pre-trained TTS model path to initialize decoder.",
    )
    parser.add_argument(
        "--dec-init-mods",
        default="dec.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of decoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--freeze-mods",
        default=None,
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of modules to freeze (not to train), separated by a comma.",
    )
    parser.add_argument(
        "--use-fe-condition",
        default=True,
        type=strtobool,
        help="Whether to cat f0/energy condition",
    )
    parser.add_argument(
        "--pad-eos",
        default=True,
        type=strtobool,
        help="Whether to pad eos at the end of input text",
    )
    parser.add_argument(
        "--append-position",
        default=True,
        type=strtobool,
        help="Whether to append pos to encoder-outputs",
    )
    
    parser.add_argument(
        "--use-amp",
        default=True,
        type=strtobool,
        help="whether use transformer encoder",
    )
    parser.add_argument(
        "--amp-checkpoint",
        default='exp/teacher/results/amp_checkpoint_36800.pt',
        type=str,
        help="amp checkpoint for initialization",
    )
    parser.add_argument(
        "--use-postnet",
        default=True,
        type=strtobool,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--distill-output-knowledge",
        default=True,
        type=strtobool,
        help="whether to distill output knowledge",
    )
    parser.add_argument(
        "--distill-encoder-knowledge",
        default=True,
        type=strtobool,
        help="whether to distill encoder knowledge",
    )
    parser.add_argument(
        "--distill-decoder-knowledge",
        default=True,
        type=strtobool,
        help="whether to distill decode knowledge",
    )
    parser.add_argument(
        "--distill-prosody-knowledge",
        default=True,
        type=strtobool,
        help="whether to distill prosody knowledge",
    )
    
    return parser
