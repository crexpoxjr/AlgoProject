#!/bin/bash
# Submit with site-specific options, for example:
# sbatch -A YOUR_ALLOCATION -p YOUR_GPU_QUEUE --gres=gpu:1 -t 02:00:00 benchmarks/tacc_job.sh
# Load the required site modules and activate your environment before submitting,
# or add the appropriate setup for your cluster here.
set -euo pipefail
mkdir -p results/benchmarks
nvidia-smi | tee results/benchmarks/nvidia-smi.txt
python -c 'import sys, torch; print(sys.version); print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
python train.py --data fi2010 --seeds 42 43 44
python -m benchmarks.latency --device cuda
python -m benchmarks.memory --device cuda
