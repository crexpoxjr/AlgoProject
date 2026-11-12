"""FI-2010 window loader for the team's existing training interface.

This file supports the 149-row FI-2010 format in the uploaded project.
Set FI2010_DAILY_DIR to the directory containing the three DeepLOB DecPre
test-fold files to use day 8 for training, day 9 for validation and day 10 for
testing. Set FI2010_SPLIT=full10 to train on days 1–7, validate on day 8, and
test on days 9–10. Place Train_Dst_NoAuction_DecPre_CF_7.txt in the same data
directory for full10 mode. Use the same settings for training and evaluation.

The default split is a provisional chronological split of the provided file,
not an official benchmark split. Verify source label meanings, horizon units,
normalization, chronological ordering and stock/day boundaries before reporting
results. Set fi2010_segments when verified boundaries are available.

Optional config entries:
  fi2010_path: input path (default data/Train_Dst_NoAuction_ZScore_CF_1.txt)
  fi2010_segments: list of [start, end) observation ranges in the input file.
      Windows never cross these ranges. Defaults to treating the entire input as
      one segment, which is NOT evidence that it contains just one stock/day.
  fi2010_purge: observations removed at the end of each split (default 100).
      Confirm this covers the full future-label span in the source data.
  fi2010_split_fractions: train and validation fractions (default 5/7, 1/7).
  fi2010_label_mapping: raw-to-project mapping (default {1:0, 2:1, 3:2}).
      Project class order is 0 Up, 1 Stationary, 2 Down. VERIFY raw meanings.

Horizon keys 10/20/30/50/100 correspond to the five supplied label rows in
order. These are the project's tick-time convention; some sources describe
horizons in sampled observations as 1/2/3/5/10. Confirm for your data release.
"""
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HORIZON_ROWS = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}
DEFAULT_PATH = Path(__file__).resolve().parent / "Train_Dst_NoAuction_ZScore_CF_1.txt"
# The archive packs five stock blocks into each daily file. Offsets were
# inferred from the four large discontinuities in best bid/ask prices.
DAILY_STOCK_EDGES = {
    7: (0, 2647, 11309, 19900, 33129, 55478),
    8: (0, 1873, 11144, 21180, 34060, 52172),
    9: (0, 1888, 7016, 12738, 18559, 31937),
}
# The seven-day archive file is ordered by stock, then day. We matched all
# five horizon-label rows against the individual day files to verify the cuts.
TRAIN_STOCK_EDGES = (0, 23336, 68210, 106759, 161434, 254750)
TRAIN_SESSION_EDGES = (
    0, 3454, 8533, 12436, 15242, 18272, 20535, 23336,
    29654, 35776, 41187, 48179, 54907, 61150, 68210,
    73132, 79097, 83124, 88466, 94412, 100019, 106759,
    117478, 124190, 129376, 140195, 146573, 153580, 161434,
    175533, 190052, 200060, 211124, 223827, 241859, 254750,
)


class FI2010WindowDataset(Dataset):
    """Windows are sliced on demand; end indices identify the target time."""

    def __init__(self, features, labels, ends, seq_len):
        self.features = features
        self.labels = labels
        self.ends = np.asarray(ends, dtype=np.int64)
        self.seq_len = seq_len
        # The existing training script uses dataset.y to compute class weights.
        self.y = torch.from_numpy(labels[self.ends].copy())

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        end = int(self.ends[index])
        window = self.features[end - self.seq_len + 1:end + 1]
        return torch.from_numpy(window), self.y[index]


def _segments(value, n):
    if value is None:
        warnings.warn(
            "Stock/day boundaries are unknown: treating the entire file as one "
            "segment. Set fi2010_segments after verifying the source metadata.",
            stacklevel=2,
        )
        return [(0, n)]
    segments = []
    for pair in value:
        if len(pair) != 2:
            raise ValueError("Each fi2010_segments entry must be [start, end).")
        start, end = pair
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= n:
            raise ValueError(f"Invalid segment {pair!r} for {n} observations.")
        if segments and start < segments[-1][1]:
            raise ValueError("Segments must be sorted and non-overlapping.")
        segments.append((start, end))
    if not segments:
        raise ValueError("fi2010_segments must not be empty.")
    return segments


def _read_fi2010(path, horizon, mapping):
    if not path.is_file():
        raise FileNotFoundError(f"FI-2010 file not found: {path}")
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != 149:
        raise ValueError(f"Expected 149 rows (features and labels), got {raw.shape}.")
    features = np.ascontiguousarray(raw[:40].T)
    source_labels = raw[HORIZON_ROWS[horizon]]
    if not np.isfinite(features).all() or not np.isfinite(source_labels).all():
        raise ValueError("Non-finite values found in features or selected labels.")
    if not np.isin(source_labels, [1, 2, 3]).all():
        raise ValueError("Selected label row contains values outside 1, 2, 3.")
    labels = np.empty(len(source_labels), dtype=np.int64)
    for source, target in mapping.items():
        labels[source_labels == source] = target
    return features, labels


def _deeplob_datasets(directory, horizon, mapping, seq_len, split):
    daily = {
        fold: (f"Test_Dst_NoAuction_DecPre_CF_{fold}.txt", edges, edges)
        for fold, edges in DAILY_STOCK_EDGES.items()
    }
    if split == "full10":
        training = ("Train_Dst_NoAuction_DecPre_CF_7.txt", TRAIN_STOCK_EDGES, TRAIN_SESSION_EDGES)
        files_by_split = (
            ("train", "days 1–7", (training,)),
            ("validation", "day 8", (daily[7],)),
            ("test", "days 9–10", (daily[8], daily[9])),
        )
    else:
        files_by_split = (
            ("train", "day 8", (daily[7],)),
            ("validation", "day 9", (daily[8],)),
            ("test", "day 10", (daily[9],)),
        )

    datasets = []
    # Each snapshot represents ten order events. Keep the full forecast
    # horizon within each stock's session as well as the input history.
    tail = horizon // 10
    for name, days, files in files_by_split:
        feature_blocks, label_blocks, end_blocks = [], [], []
        offset = 0
        for filename, stock_edges, session_edges in files:
            path = directory / filename
            features, labels = _read_fi2010(path, horizon, mapping)
            best_price_changes = np.abs(np.diff(features[:, [0, 2]], axis=0)).max(axis=1)
            breaks = np.flatnonzero(best_price_changes > 0.015) + 1
            if len(labels) != stock_edges[-1] or not np.array_equal(breaks, stock_edges[1:-1]):
                raise ValueError(f"{path} does not match the expected DeepLOB FI-2010 file.")
            end_blocks.extend(
                np.arange(offset + start + seq_len - 1, offset + end - tail, dtype=np.int64)
                for start, end in zip(session_edges[:-1], session_edges[1:])
            )
            feature_blocks.append(features)
            label_blocks.append(labels)
            offset += len(labels)

        ends = np.concatenate(end_blocks)
        if len(ends) == 0:
            raise ValueError(f"No {name} windows for seq_len={seq_len}.")
        features = feature_blocks[0] if len(files) == 1 else np.concatenate(feature_blocks)
        labels = label_blocks[0] if len(files) == 1 else np.concatenate(label_blocks)
        if name == "train":
            center = features.mean(axis=0, dtype=np.float64)
            scale = features.std(axis=0, dtype=np.float64)
            scale[scale == 0] = 1
        features = np.ascontiguousarray((features - center) / scale, dtype=np.float32)
        datasets.append(FI2010WindowDataset(features, labels, ends, seq_len))
        counts = np.bincount(labels[ends], minlength=3)
        print(f"  {name} ({days}): {len(ends)} windows; class counts {counts.tolist()} (0/1/2)")
    print(f"  FI-2010 {split} split: {directory}; horizon={horizon}; train-only standardization")
    return datasets


def get_dataloaders(config):
    """Return (train, validation, test) loaders for the project's scripts."""
    seq_len = int(config["seq_len"])
    batch_size = int(config["batch_size"])
    horizon = config["horizon_k"]
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be positive.")
    if horizon not in HORIZON_ROWS:
        raise ValueError(f"Unsupported horizon {horizon!r}; choose {list(HORIZON_ROWS)}.")
    if int(config.get("in_features", 40)) != 40:
        raise ValueError("This loader expects the 40 LOB features.")
    mapping = config.get("fi2010_label_mapping", {1: 0, 2: 1, 3: 2})
    if set(mapping) != {1, 2, 3} or set(mapping.values()) != {0, 1, 2}:
        raise ValueError("fi2010_label_mapping must map 1,2,3 bijectively to 0,1,2.")

    daily_directory = os.environ.get("FI2010_DAILY_DIR")
    split = os.environ.get("FI2010_SPLIT", "three-day")
    if split not in ("three-day", "full10"):
        raise ValueError("FI2010_SPLIT must be 'three-day' or 'full10'.")
    if split == "full10" and not daily_directory:
        raise ValueError("Set FI2010_DAILY_DIR to the DeepLOB data directory for full10 mode.")
    if daily_directory:
        datasets = _deeplob_datasets(Path(daily_directory).expanduser(), horizon, mapping, seq_len, split)
    else:
        datasets = _single_file_datasets(config, horizon, mapping, seq_len)

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    return (DataLoader(datasets[0], batch_size=batch_size, shuffle=True, generator=generator),
            DataLoader(datasets[1], batch_size=batch_size, shuffle=False),
            DataLoader(datasets[2], batch_size=batch_size, shuffle=False))


def _single_file_datasets(config, horizon, mapping, seq_len):
    path = Path(config.get("fi2010_path", DEFAULT_PATH)).expanduser()
    features, labels = _read_fi2010(path, horizon, mapping)

    n = len(labels)
    train_frac, val_frac = config.get("fi2010_split_fractions", (5 / 7, 1 / 7))
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Split fractions must be positive and sum to less than 1.")
    purge = int(config.get("fi2010_purge", 100))
    if purge < 0:
        raise ValueError("fi2010_purge must not be negative.")
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    # Purge target times at each split's end to reduce future-label overlap.
    # No input window crosses a split boundary.
    ranges = [(0, a - purge), (a, b - purge), (b, n)]
    segments = _segments(config.get("fi2010_segments"), n)
    datasets = []
    for name, (split_start, split_end) in zip(("train", "validation", "test"), ranges):
        pieces = []
        for segment_start, segment_end in segments:
            start = max(split_start, segment_start)
            end = min(split_end, segment_end)
            if end - start >= seq_len:
                pieces.append(np.arange(start + seq_len - 1, end, dtype=np.int64))
        if not pieces:
            raise ValueError(f"No {name} windows: check file size, segments, split, and purge.")
        ds = FI2010WindowDataset(features, labels, np.concatenate(pieces), seq_len)
        counts = np.bincount(labels[ds.ends], minlength=3)
        print(f"  {name}: {len(ds)} windows; class counts {counts.tolist()} (0/1/2)")
        datasets.append(ds)

    print(f"  FI-2010: {path}; horizon={horizon}; purge={purge}; "
          f"mapping={mapping}; split boundaries={a},{b}")
    return datasets
