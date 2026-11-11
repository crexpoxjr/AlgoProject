"""Shared reproducible synthetic-input benchmark runner."""
import argparse
import csv
import json
import os
import platform
import statistics
import time

import torch

from config import DEFAULT_CONFIG
from experiments.common import set_seed
from models import build_model
from models.common import count_parameters

MODELS = ('cnn_std_attn', 'cnn_lin_attn')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run(kind):
    p = argparse.ArgumentParser(description=f'Synthetic whole-model inference {kind} benchmark')
    p.add_argument('--lengths', nargs='+', type=int, default=[50, 100, 200, 500, 1000, 2000])
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--output-dir', default='results/benchmarks')
    args = p.parse_args()
    if min(args.lengths) < 1 or args.batch_size < 1 or args.warmup < 0 or args.repeats < 1:
        p.error('lengths, batch size, and repeats must be positive; warmup must be nonnegative')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if kind == 'memory' and device.type != 'cuda':
        p.error('GPU memory measurement requires CUDA; run this benchmark on a GPU')
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG, seed=args.seed, use_sdpa=False)
    metadata = {'kind': kind, 'input': 'synthetic random float32 tensors',
                'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'python': platform.python_version(), 'pytorch': torch.__version__,
                'cuda_runtime': torch.version.cuda, 'config': cfg, 'arguments': vars(args),
                'timing': 'whole-model eval inference, synchronized per repetition; batch latency',
                'memory': 'peak allocated bytes includes resident model and input; incremental peak is above their allocated baseline'}
    rows = []
    for name in args.models:
        for length in args.lengths:
            set_seed(args.seed)
            row = {'model': name, 'sequence_length': length, 'batch_size': args.batch_size,
                   'parameters': None, 'status': 'ok', 'median_ms': None, 'mean_ms': None,
                   'peak_allocated_mib': None, 'incremental_peak_mib': None, 'error': ''}
            model = x = None
            try:
                model = build_model(name, cfg).to(device).eval()
                row['parameters'] = count_parameters(model)
                x = torch.randn(args.batch_size, length, cfg['in_features'], device=device)
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        model(x)
                    synchronize(device)
                    if kind == 'latency':
                        samples = []
                        for _ in range(args.repeats):
                            synchronize(device)
                            start = time.perf_counter()
                            model(x)
                            synchronize(device)
                            samples.append((time.perf_counter() - start) * 1000)
                        row['median_ms'] = statistics.median(samples)
                        row['mean_ms'] = statistics.mean(samples)
                    else:
                        baseline = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        for _ in range(args.repeats):
                            model(x)
                        synchronize(device)
                        peak = torch.cuda.max_memory_allocated(device)
                        row['peak_allocated_mib'] = peak / 2**20
                        row['incremental_peak_mib'] = (peak - baseline) / 2**20
            except torch.cuda.OutOfMemoryError as exc:
                row['status'], row['error'] = 'out_of_memory', str(exc).split('\n')[0]
            finally:
                del model, x
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
    csv_path = os.path.join(args.output_dir, f'{kind}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.output_dir, f'{kind}_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Wrote {csv_path}')

