#!/usr/bin/env python3
# Modified from Espnet

"""TTS decoding script."""

import configargparse
import logging
import os
import platform
import subprocess
import sys

from espnet.utils.cli_utils import strtobool


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get parser of decoding arguments."""
    parser = configargparse.ArgumentParser(
        description="Synthesize speech from text using a TTS model on one CPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
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

    parser.add_argument("--ngpu", default=0, type=int, help="Number of GPUs")
    parser.add_argument(
        "--backend",
        default="pytorch",
        type=str,
        #choices=["chainer", "pytorch", "pytorch_wds", "pytorch_tf_wds", "pytorch_8", "pytorch_tf_wds_extHC"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--out", type=str, required=False, help="Output filename")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    # task related
    parser.add_argument(
        "--json", type=str, required=True, help="Filename of train label data (json)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    # decoding related
    parser.add_argument(
        "--maxlenratio", type=float, default=5, help="Maximum length ratio in decoding"
    )
    parser.add_argument(
        "--minlenratio", type=float, default=0, help="Minimum length ratio in decoding"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold value in decoding"
    )
    parser.add_argument(
        "--use-att-constraint",
        type=strtobool,
        default=False,
        help="Whether to use the attention constraint",
    )
    parser.add_argument(
        "--backward-window",
        type=int,
        default=1,
        help="Backward window size in the attention constraint",
    )
    parser.add_argument(
        "--forward-window",
        type=int,
        default=3,
        help="Forward window size in the attention constraint",
    )
    parser.add_argument(
        "--fastspeech-alpha",
        type=float,
        default=1.0,
        help="Alpha to change the speed for FastSpeech",
    )
    # save related
    parser.add_argument(
        "--save-durations",
        default=False,
        type=strtobool,
        help="Whether to save durations converted from attentions",
    )
    parser.add_argument(
        "--save-focus-rates",
        default=False,
        type=strtobool,
        help="Whether to save focus rates of attentions",
    )
    parser.add_argument(
        "--save-root",
        default=False,
        type=str,
        help="features save root",
    )
    parser.add_argument(
        "--use-fe-condition",
        default=True,
        type=str,
        help="whether to use position encoding",
    )
    parser.add_argument(
        "--pad-eos",
        default=True,
        type=strtobool,
        help="Whether to pad eos at the end of input text",
    )
    parser.add_argument(
        "--append-position",
        default=False,
        type=strtobool,
        help="Whether to pad eos at the end of input text",
    )
    parser.add_argument(
        "--use-amp",
        default=False,
        type=strtobool,
        help="whether use transformer encoder",
    )
    parser.add_argument(
        "--amp-checkpoint",
        default=None,
        type=str,
        help="amp checkpoint for initialization",
    )
    parser.add_argument(
        "--encoder-resume",
        default=None,
        type=str,
        help="path for encoder",
    )
    parser.add_argument(
        "--teacher-config",
        default=None,
        type=str,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--distill-output-knowledge",
        default=True,
        type=strtobool,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--distill-encoder-knowledge",
        default=True,
        type=strtobool,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--distill-decoder-knowledge",
        default=True,
        type=strtobool,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--distill-prosody-knowledge",
        default=True,
        type=strtobool,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--is-train",
        default=True,
        type=strtobool,
        help="whether to use GST to preserve style",
    )
    parser.add_argument(
        "--share-proj",
        default=False,
        type=strtobool,
        help="whether to share proj-layers for student-model",
    )
    parser.add_argument(
        "--perform-KD",
        default=False,
        type=strtobool,
        help="whether to perform knowldge distillation",
    )
    parser.add_argument(
        "--test-teacher",
        default=True,
        type=strtobool,
        help="whether to test teacher model, True is teacher, False is student",
    )
    return parser


def main(args):
    """Run deocding."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == "2":
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(
                    ["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]
                ).strip()
                logging.info("CLSP: use gpu" + cvd)
                os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = (
                    subprocess.check_output(
                        ["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]
                    )
                    .decode()
                    .strip()
                )
                logging.info("CLSP: use gpu" + cvd)
                os.environ["CUDA_VISIBLE_DEVICES"] = cvd

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # extract
    logging.info("backend = " + args.backend)
   
    if args.test_teacher:
        from tts import decode
        decode(args)
    else:
        from tts_distill import decode
        from teacher_parser import get_teacher_parser
        from espnet.utils.dynamic_import import dynamic_import
        from espnet.nets.tts_interface import TTSInterface
        teacher_parser = get_teacher_parser()
        teacher_args, _ = teacher_parser.parse_known_args([])
        teacher_model_class = dynamic_import(teacher_args.model_module)
        assert issubclass(teacher_model_class, TTSInterface)
        teacher_model_class.add_arguments(teacher_parser)
        teacher_args = teacher_parser.parse_args([])
        decode(args, teacher_args)


if __name__ == "__main__":
    main(sys.argv[1:])
