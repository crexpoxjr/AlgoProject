"""Measure peak CUDA tensor allocation during inference; see --help."""
from benchmarks.common import run

if __name__ == '__main__':
    run('memory')
