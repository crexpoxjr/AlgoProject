"""Train models with the shared baseline protocol and checkpoint format.

Example: python train.py --data fi2010 --seeds 42 43 44
Defaults to the proposed model. Use --models to train any combination.
"""
import argparse

from config import DEFAULT_CONFIG
from experiments.train_baselines import fit
from models import MODEL_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', choices=['synthetic', 'fi2010'], default='synthetic')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=['cnn_lin_attn'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[DEFAULT_CONFIG['seed']])
    parser.add_argument('--epochs', type=int, help='override epochs, e.g. for a smoke test')
    parser.add_argument('--ckpt-dir', default='checkpoints')
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    for name in args.models:
        for seed in args.seeds:
            cfg = dict(DEFAULT_CONFIG, seed=seed)
            if args.epochs is not None:
                cfg['epochs'] = args.epochs
            print(f'== {name} seed {seed} data {args.data} ==')
            f1, epoch = fit(name, cfg, args.data, args.ckpt_dir, args.out_dir)
            print(f'Best validation macro-F1 {f1:.4f} at epoch {epoch}')


if __name__ == '__main__':
    main()
