# AlgoProject

> Person 1 additions are included in this version. See [PERSON1_README.md](PERSON1_README.md) for the hybrid model, training, benchmarks, paper drafts, and remaining TACC/FI-2010 work. The original Person 2 integration notes below contain some historical references to files not yet being present.

**Efficient Stock Price Movement Prediction Using a Hybrid CNN–Linear Attention Network**

A repository for the CSCE 4110 Algorithms class project. We predict the *direction* of the
next mid-price move (**Up / Stationary / Down**) from the last `T = 100` snapshots of a
10-level limit order book (FI-2010 dataset, 40 features per snapshot), and ask whether
replacing softmax attention (`O(T²)`) with linear attention (`O(T)`) saves runtime and memory
without hurting predictive quality.

| Person | Area | Main files |
|---|---|---|
| Person 1 | Proposed model, training loop, TACC efficiency benchmarks | `models/linear_attention.py`, `models/hybrid_model.py`, `train.py`, `benchmarks/` |
| **Person 2** | **Baselines, comparative evaluation, related work** | **`models/cnn_baseline.py`, `models/standard_attention.py`, `experiments/`** |
| Person 3 | FI-2010 data pipeline, shared metrics, IEEE paper integration | `data/fi2010_loader.py`, `data/preprocessing.py`, `evaluation/`, `paper/main.tex` |

---

## Contents

1. [Setup](#1-setup)
2. [Repository layout](#2-repository-layout)
3. [The team-wide contract](#3-the-team-wide-contract)
4. [Person 2: baseline models](#4-person-2-baseline-models)
5. [Person 2: running the experiments](#5-person-2-running-the-experiments)
6. [Outputs and where they go in the paper](#6-outputs-and-where-they-go-in-the-paper)
7. [Integration points for Person 1 and Person 3](#7-integration-points-for-person-1-and-person-3)
8. [Testing](#8-testing)
9. [Reproducibility checklist](#9-reproducibility-checklist)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Setup

Make sure you have Python 3.10+ installed. Run these commands from the repository root. They
create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

On TACC (Vista / Lonestar6), load the site's Python/CUDA modules first, then create the venv the
same way. All scripts automatically use the GPU when `torch.cuda.is_available()`.

**Always run scripts as modules from the repo root** (`python -m experiments.compare_models`, not
`python experiments/compare_models.py`), so that imports like `from models import ...` resolve.

---

## 2. Repository layout

```
AlgoProject/
├── config.py                     # SINGLE source of truth for all hyperparameters (shared)
├── data/
│   ├── synthetic.py              # LOB-shaped fake data + get_dataloaders(source, config) switch
│   └── fi2010_loader.py          # (Person 3) real FI-2010 loaders
├── models/
│   ├── __init__.py               # build_model(name, config) registry for all three models
│   ├── common.py                 # shared CNN backbone, pos. encoding, encoder block, head
│   ├── cnn_baseline.py           # Baseline 1: CNN only                    (Person 2)
│   ├── standard_attention.py     # Baseline 2: CNN + softmax attention     (Person 2)
│   ├── linear_attention.py       # (Person 1)
│   └── hybrid_model.py           # (Person 1) proposed CNN + linear attention
├── experiments/                  # (Person 2)
│   ├── common.py                 # seeding, checkpoint I/O, predict(), metrics fallback
│   ├── train_baselines.py        # trains models under the shared protocol
│   ├── compare_models.py         # Experiment 1 + 3: results table (.csv + LaTeX)
│   └── confusion_matrices.py     # confusion-matrix figures + per-class report
├── tests/
│   └── test_baselines.py         # shape / gradient / variable-length / ablation checks
├── paper/
│   ├── sections/
│   │   ├── related_work.tex      # Section II              (Person 2)
│   │   ├── baselines.tex         # Section IV, baselines   (Person 2)
│   │   └── predictive_results.tex# Section VII, predictive (Person 2)
│   └── references_person2.bib    # BibTeX for the above, merge into references.bib
├── checkpoints/                  # created by training (git-ignored)
├── results/                      # created by experiments (tables, predictions, histories)
└── figures/                      # created by experiments (PDF figures for the paper)
```

---

## 3. The team-wide contract

All three models must be interchangeable in every script. The agreed interface is:

```python
# Input
X.shape == (batch_size, sequence_length, 40)   # float32
# Output
logits.shape == (batch_size, 3)                # raw scores, no softmax
# Labels
y in {0: Up, 1: Stationary, 2: Down}           # int64; FI-2010's 1/2/3 mapped to 0/1/2
```

and every model is created the same way:

```python
from config import DEFAULT_CONFIG
from models import build_model

model = build_model("cnn", DEFAULT_CONFIG)            # or "cnn_std_attn", "cnn_lin_attn"
```

`build_model` passes the **entire** config dict as keyword arguments, so every model constructor
must accept `**kwargs` and ignore keys it does not use.

### Shared configuration (`config.py`)

| Key | Value | Meaning |
|---|---|---|
| `in_features` | 40 | 10 levels × (ask price, ask vol, bid price, bid vol) |
| `seq_len` | 100 | history window `T` |
| `num_classes` | 3 | Up / Stationary / Down |
| `horizon_k` | 10 | FI-2010 label horizon (10, 20, 30, 50 or 100 events) |
| `d_model` | 64 | CNN channels = attention width |
| `cnn_kernel_size`, `cnn_layers` | 3, 2 | CNN backbone |
| `n_heads`, `n_attn_layers`, `ffn_mult` | 4, 1, 2 | attention encoder |
| `dropout` | 0.1 | everywhere |
| `use_pos_encoding` | True | sinusoidal, works for any `T` |
| `seed` | 42 | default seed (paper uses 42, 43, 44) |
| `batch_size`, `lr`, `weight_decay` | 64, 1e-3, 1e-4 | AdamW |
| `epochs`, `early_stop_patience` | 30, 5 | early stopping on val macro-F1 |
| `use_class_weights` | True | inverse-frequency CE weights from the train split |

**Only change hyperparameters in `config.py`**, and only if the whole team agrees. Otherwise
the comparison stops being fair. The checkpoint stores the config it was trained with, so old
checkpoints remain loadable after a change.

---

## 4. Person 2: baseline models

### Why there is a `models/common.py`

Experiment 3 (architecture ablation) only means something if the three models differ in
**exactly one place**. So the CNN front end, positional encoding, Transformer encoder block,
pooling and classifier live in `models/common.py`, and each model only chooses which
sequence-mixing module to use:

```
              ┌────────────────────────── shared (models/common.py) ──────────────────────────┐
 X (B,T,40) → │ CNNBackbone → [PosEnc → EncoderBlock( ATTENTION ) × n_attn_layers] → Head │ → logits (B,3)
              └───────────────────────────────────────────────────────────────────────────────┘
 ATTENTION =  none (block removed)  → CNNBaseline             models/cnn_baseline.py
              softmax               → CNNStandardAttention    models/standard_attention.py
              linear                → HybridCNNLinearAttention models/hybrid_model.py (Person 1)
```

| Component | Details |
|---|---|
| `CNNBackbone` | 2 × [Conv1d(k=3, "same" padding) → BatchNorm → GELU → Dropout], `(B,T,40) → (B,T,64)`. Padding keeps `T`, so attention sees the full history length. |
| `SinusoidalPositionalEncoding` | Fixed sin/cos, computed on the fly, so it works for any `T` (50 … 2000+). |
| `EncoderBlock(attention)` | Pre-norm residual block: `x += Drop(Attn(LN(x)))`, `x += Drop(FFN(LN(x)))`, FFN width `2·d_model`. |
| `ClassificationHead` | Mean-pool over time → LayerNorm → Dropout → Linear(64→3). |

### Baseline 1: `CNNBaseline` (`models/cnn_baseline.py`)

The backbone feeds the head directly, with no positional encoding and no attention. It measures
how much the CNN alone can do. **20,675 parameters** with the default config.

### Baseline 2: `CNNStandardAttention` (`models/standard_attention.py`)

Backbone → positional encoding → one encoder block with `SoftmaxSelfAttention` → head.
**54,147 parameters** (the attention block adds 33,472).

`SoftmaxSelfAttention` is written out explicitly:

```python
scores = (q @ k.transpose(-2, -1)) / sqrt(d_head)   # (B, H, T, T)  ← quadratic in T
attn   = scores.softmax(dim=-1)
out    = attn @ v
```

so the `O(T²)` time and memory cost is real and measurable in Experiment 2.

> **Important for Person 1's efficiency benchmarks:** keep `use_sdpa=False` (the default).
> `torch.nn.functional.scaled_dot_product_attention` can dispatch to FlashAttention or
> memory-efficient kernels that never store the `T×T` matrix, which would hide the quadratic
> memory cost and make the comparison with plain-PyTorch linear attention unfair. If you want an
> "optimized softmax" row, build with `build_model("cnn_std_attn", {**cfg, "use_sdpa": True})`
> and label it separately. A unit test checks that both paths give the same output.

Optional: setting `module.store_attn = True` on a `SoftmaxSelfAttention` saves the last attention
map in `module.last_attn` for attention-heatmap figures.

---

## 5. Person 2: running the experiments

The pipeline has three steps. Every script takes `--data synthetic` (works today) or
`--data fi2010` (once Person 3's loader is merged).

### Step 0: sanity-check the models

```bash
python -m tests.test_baselines
```

### Step 1: train

```bash
# quick smoke test on synthetic data (≈30 s on CPU)
python -m experiments.train_baselines --data synthetic --epochs 3

# real runs: both baselines, three seeds
python -m experiments.train_baselines --data fi2010 --seeds 42 43 44

# the proposed model can be trained with the same protocol once hybrid_model.py exists
python -m experiments.train_baselines --data fi2010 --models cnn_lin_attn --seeds 42 43 44
```

Training protocol (the same for every model; it is written up in `paper/sections/baselines.tex`):

* AdamW, `lr=1e-3`, `weight_decay=1e-4`, batch size 64, up to 30 epochs.
* Cross-entropy with inverse-frequency class weights computed on the **training split only**.
* After each epoch, evaluate on the validation split. Save the checkpoint with the best
  **validation macro-F1**, and stop after 5 epochs without improvement.
* Python, NumPy and PyTorch seeds are set **before** building the model and data loaders, so the
  shuffle order is reproducible too. cuDNN is set to deterministic mode.

It produces `checkpoints/<model>_k<horizon>_seed<seed>.pt` and `results/history_<model>_seed<seed>.json`
(per-epoch train loss and validation metrics, which is useful for learning-curve plots).

### Step 2: compare models (Experiment 1 + Experiment 3)

```bash
python -m experiments.compare_models --data fi2010 --seeds 42 43 44
```

This loads every checkpoint it can find and evaluates each one **once** on the **same held-out
test split**. Missing checkpoints (for example, Person 1's model before it is trained) are skipped
with a message. It writes:

* `results/classification_results.csv`: one row per (model, seed) with accuracy, macro
  precision / recall / F1, parameter count and best epoch.
* `results/classification_results.tex`: an IEEE-ready table with mean ± std over seeds. The
  best macro-F1 is in bold. If run on synthetic data, the caption says
  **"SYNTHETIC DATA (do not report)"**.
* `results/predictions_<model>_seed<seed>.npz`: `y_true` / `y_pred`, which Step 3 reuses.

Use `--models cnn cnn_std_attn` to restrict which models are evaluated.

### Step 3: confusion matrices

```bash
python -m experiments.confusion_matrices --seed 42
```

This reads the saved predictions (so the figures always match the table) and writes:

* `figures/confusion_<model>.pdf`: one figure per model.
* `figures/confusion_all.pdf`: a 1×3 panel for the paper (use `figure*` to span both IEEE columns).
* `results/per_class_report.txt`: per-class precision / recall / F1 for the error analysis.

Cells show row-normalized values (per-class recall on the diagonal), with raw counts in brackets.

### Running on TACC

A minimal Slurm script, to be adapted to the allocation and queue names:

```bash
#!/bin/bash
#SBATCH -J baselines
#SBATCH -p gpu-a100-small        # Lonestar6; use the GPU queue on Vista instead
#SBATCH -N 1 -n 1 -t 02:00:00
#SBATCH -o logs/baselines.%j.out
module load python3 cuda
source .venv/bin/activate
nvidia-smi                       # record the GPU model for the paper
python -m experiments.train_baselines --data fi2010 --seeds 42 43 44
python -m experiments.compare_models  --data fi2010 --seeds 42 43 44
python -m experiments.confusion_matrices --seed 42
```

---

## 6. Outputs and where they go in the paper

| Paper section | Source | How to update |
|---|---|---|
| II. Related Work | `paper/sections/related_work.tex` | hand-written; cites keys in `paper/references_person2.bib` |
| IV. Implementation (baselines) | `paper/sections/baselines.tex` | hand-written; re-check the parameter counts with `python -m tests.test_baselines` if `config.py` changes |
| VII. Predictive results: table | `results/classification_results.tex` | **generated**, never edit numbers by hand |
| VII. Predictive results: figure | `figures/confusion_all.pdf` | **generated** |
| VII. Predictive results: text | `paper/sections/predictive_results.tex` | fill the `TODO` sentences from the real FI-2010 numbers |

Person 3: `\input{sections/related_work}`, `\input{sections/baselines}` and
`\input{sections/predictive_results}` from `main.tex`, and merge `references_person2.bib` into
`references.bib`. The `.bib` file marks entries with `% CHECK` where details still need checking
against the publisher's page.

**Never report synthetic-data numbers.** Synthetic labels are a function of random volume
imbalance and only show that the training loop learns. They say nothing about markets.

---

## 7. Integration points for Person 1 and Person 3

These are the only places where Person 2's code depends on teammates' code. Each one fails with
a clear error message, not silently.

### Person 1: `models/hybrid_model.py`

`build_model("cnn_lin_attn", cfg)` imports `HybridCNNLinearAttention` from
`models/hybrid_model.py` and calls it with `**cfg`. For a fair ablation, build it from the shared
blocks and swap in only the attention:

```python
import torch.nn as nn
from models.common import CNNBackbone, SinusoidalPositionalEncoding, EncoderBlock, ClassificationHead
from models.linear_attention import LinearSelfAttention   # forward: (B,T,d) -> (B,T,d)

class HybridCNNLinearAttention(nn.Module):
    def __init__(self, in_features=40, d_model=64, cnn_kernel_size=3, cnn_layers=2, n_heads=4,
                 n_attn_layers=1, ffn_mult=2, num_classes=3, dropout=0.1,
                 use_pos_encoding=True, **_unused):
        super().__init__()
        self.backbone = CNNBackbone(in_features, d_model, cnn_kernel_size, cnn_layers, dropout)
        self.pos = SinusoidalPositionalEncoding(d_model) if use_pos_encoding else nn.Identity()
        self.encoder = nn.Sequential(*[
            EncoderBlock(LinearSelfAttention(d_model, n_heads), d_model, ffn_mult, dropout)
            for _ in range(n_attn_layers)])
        self.head = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, x):
        return self.head(self.encoder(self.pos(self.backbone(x))))
```

Once that file exists, `tests/test_baselines.py` automatically includes the model in its
shape/gradient/variable-length checks, and `compare_models.py` adds it to the table.

**Checkpoint format.** If Person 1's `train.py` saves checkpoints, use the same format and path,
so `compare_models.py` can load them (see `experiments/common.py`):

```python
{"model_name": "cnn_lin_attn", "config": cfg, "state_dict": model.state_dict(),
 "epoch": best_epoch, "val_metrics": {"accuracy": ..., "macro_f1": ...}}
# saved to checkpoints/<model_name>_k<horizon_k>_seed<seed>.pt
```

`experiments.common.save_checkpoint(...)` and `checkpoint_path(...)` already do this. Use
the same training protocol as `experiments/train_baselines.py` (or call its `fit()` function) so
all three rows in the table were trained identically.

### Person 3: `data/fi2010_loader.py`

`data.synthetic.get_dataloaders("fi2010", config)` calls
`data.fi2010_loader.get_dataloaders(config)`, which must return
`(train_loader, val_loader, test_loader)`:

* chronological train / validation / test split (no shuffling across time **before** splitting);
* batches of `(X, y)` with `X: (B, config["seq_len"], 40)` float32 and `y: (B,)` int64 in
  `{0: Up, 1: Stationary, 2: Down}`, using the `config["horizon_k"]` label column;
* the **test loader must use `shuffle=False`**;
* optional but faster: expose labels as `train_loader.dataset.y`, so the class weights are
  computed without iterating over all inputs.

### Person 3: `evaluation/metrics.py`

`experiments.common.compute_metrics` uses `evaluation.metrics.compute_metrics(y_true, y_pred)` when
it exists, so every number in the paper comes from one definition. Until then it falls back to
scikit-learn. The shared function must return a dict with at least the keys `accuracy`,
`macro_precision`, `macro_recall` and `macro_f1` (floats in [0, 1]).

---

## 8. Testing

```bash
python -m tests.test_baselines          # no extra dependencies
pytest -q tests/test_baselines.py       # if pytest is installed
```

| Test | Checks |
|---|---|
| `test_output_shape_contract` | every available model maps `(B,100,40) → (B,3)` |
| `test_backward_and_finite` | loss is finite and **every** parameter receives a gradient |
| `test_variable_sequence_lengths` | the same model instance works for `T = 50, 100, 200, 500` (Experiment 2) |
| `test_sdpa_matches_explicit_attention` | explicit `O(T²)` attention equals PyTorch's fused kernel (≤1e-5) |
| `test_ablation_differs_only_in_attention` | prints parameter counts; the attention block is the only difference |

---

## 9. Reproducibility checklist

For the paper (Section IV / V), record:

- [ ] Final `config.py` values (the table in §3) and the FI-2010 horizon `k` used
- [ ] Seeds (default: 42, 43, 44) and the fact that results are mean ± std over them
- [ ] GPU model (`nvidia-smi`), TACC system, PyTorch / CUDA versions (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
- [ ] Best epoch per run (in `results/classification_results.csv`)
- [ ] That checkpoints were selected by **validation** macro-F1 and evaluated once on test

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'models'` | Run from the repo root with `python -m ...`. |
| `ImportError: models/hybrid_model.py (Person 1) is not in the repo yet` | Expected until Person 1 merges; the other models still run. |
| `ImportError: data/fi2010_loader.py (Person 3) is not in the repo yet` | Use `--data synthetic` until the loader is merged. |
| `[skip] <model> seed <s>: no checkpoint` | Train that model and seed first (Step 1). |
| `No predictions found` in `confusion_matrices` | Run `compare_models` first, using the same seed. |
| CUDA OOM for `cnn_std_attn` at large `T` | Expected: this is the quadratic memory cost Experiment 2 measures. Lower the batch size for long-`T` benchmarks. |
| Accuracy is high but macro-F1 is low | The model is predicting mostly "Stationary"; check `use_class_weights=True` and the confusion matrix. |
