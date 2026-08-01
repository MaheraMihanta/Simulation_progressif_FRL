from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to save")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_plots(
    output_dir: Path,
    time: np.ndarray,
    q: np.ndarray,
    q_ref: np.ndarray,
    error_norm: np.ndarray,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    dof = q.shape[1]

    fig, axes = plt.subplots(dof, 1, figsize=(10, 1.8 * dof), sharex=True)
    if dof == 1:
        axes = [axes]
    for idx, axis in enumerate(axes):
        axis.plot(time, q_ref[:, idx], label=f"q{idx + 1} ref", linewidth=1.4)
        axis.plot(time, q[:, idx], label=f"q{idx + 1}", linewidth=1.1)
        axis.set_ylabel("rad")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    tracking_path = output_dir / "joint_tracking.png"
    fig.savefig(tracking_path, dpi=160)
    plt.close(fig)
    paths.append(tracking_path)

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time, error_norm, color="#b23a48", linewidth=1.5)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("||q_ref - q||")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    error_path = output_dir / "error_norm.png"
    fig.savefig(error_path, dpi=160)
    plt.close(fig)
    paths.append(error_path)
    return paths
