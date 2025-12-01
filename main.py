import json
import argparse
from trainer import train
import copy


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args1 = copy.deepcopy(args)
    args.update(param) # Add parameters from json
    
    args.update(args1)
    cli_overrides = {k: v for k, v in args1.items() if v is not None}
    args.update(cli_overrides)
    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default=None,
                        help='Directory to store checkpoints for long runs (defaults to config value).')
    parser.add_argument('--resume-checkpoint', dest='resume_checkpoint', type=str, default=None,
                        help='Path to a checkpoint pkl file to resume training from a previous Kaggle session.')
    parser.add_argument('--save-checkpoints', dest='save_checkpoints', action='store_true',
                        help='Enable saving checkpoints after each task (overrides config).')
    parser.add_argument('--disable-save-checkpoints', dest='save_checkpoints', action='store_false',
                        help='Disable checkpoint saving (overrides config).')
    parser.add_argument('--imagenetr-root', dest='imagenetr_root', type=str, default=None,
                        help='Root directory of the ImageNet-R dataset (train/test subfolders).')
    parser.set_defaults(save_checkpoints=None)
    return parser

if __name__ == '__main__':
    main()
