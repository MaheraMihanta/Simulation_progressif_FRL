from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose trajectory-tracking logs and quantify oscillations.",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing tracking_log.csv")
    parser.add_argument("--skip", type=float, default=1.0, help="Seconds ignored for post-transient metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for run_dir in args.runs:
        csv_path = run_dir / "tracking_log.csv"
        if not csv_path.exists():
            print(f"{run_dir}: missing tracking_log.csv")
            continue
        report(run_dir, csv_path, args.skip)
    return 0


def report(run_dir: Path, csv_path: Path, skip_seconds: float) -> None:
    df = pd.read_csv(csv_path)
    dof = len([column for column in df.columns if column.startswith("q_ref")])
    error = df[[f"error{i}" for i in range(1, dof + 1)]].to_numpy(dtype=float)
    correction = df[[f"correction{i}" for i in range(1, dof + 1)]].to_numpy(dtype=float)
    time = df["time"].to_numpy(dtype=float)
    post = time >= skip_seconds
    if not np.any(post):
        post = np.ones_like(time, dtype=bool)

    error_norm = np.linalg.norm(error, axis=1)
    print(f"\n{run_dir}")
    print(f"  controller: {df['controller'].iloc[0] if 'controller' in df else 'unknown'}")
    print(f"  samples: {len(df)}, duration: {time[-1]:.3f} s")
    print(f"  initial error norm: {np.linalg.norm(error[0]):.6f} rad")
    print(f"  final error norm: {np.linalg.norm(error[-1]):.6f} rad")
    print(f"  max abs joint error: {np.max(np.abs(error)):.6f} rad")
    print(f"  post-{skip_seconds:g}s mean error norm: {np.mean(error_norm[post]):.6f} rad")
    print(f"  post-{skip_seconds:g}s error norm std: {np.std(error_norm[post]):.6f} rad")
    print(f"  correction sign-flip ratio: {_sign_flip_ratio(correction):.6f}")
    print(f"  high-frequency error index: {_high_frequency_index(error):.6f}")
    print(f"  per-joint RMSE: {np.array2string(np.sqrt(np.mean(error**2, axis=0)), precision=6)}")


def _sign_flip_ratio(correction: np.ndarray) -> float:
    if correction.shape[0] < 2:
        return 0.0
    signs = np.sign(correction)
    active = (signs[1:] != 0.0) & (signs[:-1] != 0.0)
    if not np.any(active):
        return 0.0
    flips = (signs[1:] != signs[:-1]) & active
    return float(np.count_nonzero(flips) / np.count_nonzero(active))


def _high_frequency_index(error: np.ndarray) -> float:
    if error.shape[0] < 3:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(error, n=2, axis=0) ** 2)))


if __name__ == "__main__":
    raise SystemExit(main())
