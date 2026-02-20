#!/usr/bin/env python3
import argparse
import os
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


def _maybe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sample_rows(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    n_total = x.shape[0]
    n = min(n, n_total)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randint(0, n_total, (n,), generator=g)
    return x.index_select(0, idx)


def _load_hips_from_urdf(urdf_path: str) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    hips: dict[str, np.ndarray] = {}
    for j in root.findall("joint"):
        name = j.attrib.get("name", "")
        if not name.endswith("_hip_joint"):
            continue
        origin = j.find("origin")
        xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        x, y, z = (float(v) for v in xyz.split())
        hips[name[:2]] = np.array([x, y, z], dtype=np.float64)  # FL/FR/RL/RR
    if set(hips.keys()) != {"FL", "FR", "RL", "RR"}:
        raise RuntimeError(f"expected hip joints for FL/FR/RL/RR, got {sorted(hips.keys())}")
    return hips


def _to_4x3(x12: torch.Tensor) -> torch.Tensor:
    if x12.ndim != 2 or x12.shape[1] != 12:
        raise ValueError(f"expected (N,12), got {tuple(x12.shape)}")
    return x12.view(-1, 4, 3)


def _plot_leg_hist2d(ax, pts: np.ndarray, x_idx: int, y_idx: int, title: str, bins: int = 240) -> None:
    x = pts[:, x_idx]
    y = pts[:, y_idx]
    ax.hist2d(x, y, bins=bins, norm=LogNorm())
    ax.set_title(title)
    ax.set_xlabel(["x", "y", "z"][x_idx])
    ax.set_ylabel(["x", "y", "z"][y_idx])
    ax.set_aspect("equal", adjustable="box")


def _draw_body_xy(ax, hips: dict[str, np.ndarray]) -> None:
    # simple rectangle through hip projections (XY), plus hip markers.
    corners = np.stack([hips["FL"], hips["FR"], hips["RR"], hips["RL"]], axis=0)
    xs = corners[:, 0]
    ys = corners[:, 1]
    ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], "k--", lw=1.0, label="hip rectangle")
    for k, v in hips.items():
        ax.scatter([v[0]], [v[1]], c="k", s=18)
        ax.text(v[0], v[1], f" {k}_hip", fontsize=8, va="center", ha="left")


def _draw_body_xz(ax, hips: dict[str, np.ndarray]) -> None:
    # show hip points projected to XZ (y ignored)
    for k, v in hips.items():
        ax.scatter([v[0]], [v[2]], c="k", s=18)
        ax.text(v[0], v[2], f" {k}_hip", fontsize=8, va="center", ha="left")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot fk_positions_valid distribution over Go2 schematic (using URDF hip locations).")
    ap.add_argument("--data-path", default="fk_positions_valid.pt")
    ap.add_argument("--urdf", default="/home/kdg/IsaacLab/outputs/go2_urdf/urdf/go2.urdf")
    ap.add_argument("--out-dir", default="sota/dataset_stats/fk_positions_valid_go2")
    ap.add_argument("--sample", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _maybe_mkdir(args.out_dir)

    x = torch.load(args.data_path, map_location="cpu")
    if not torch.is_tensor(x) or x.ndim != 2 or x.shape[1] != 12:
        raise TypeError(f"expected (N,12) tensor at {args.data_path}, got {type(x)} shape={getattr(x,'shape',None)}")

    hips = _load_hips_from_urdf(args.urdf)

    xs = _sample_rows(x, args.sample, args.seed)
    x4 = _to_4x3(xs).numpy()  # (N,4,3)
    legs = ["FL", "FR", "RL", "RR"]
    leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}

    # XY (top) 2x2
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), dpi=160)
    for i, leg in enumerate(legs):
        ax = axs[i // 2, i % 2]
        pts = x4[:, leg_idx[leg], :]
        _plot_leg_hist2d(ax, pts, 0, 1, f"{leg} foot (x,y) density")
        _draw_body_xy(ax, hips)
        # mean marker
        m = pts.mean(axis=0)
        ax.scatter([m[0]], [m[1]], c="w", s=60, marker="x", linewidths=2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "go2_top_xy_density.png"))
    plt.close(fig)

    # XZ (side) 2x2
    fig, axs = plt.subplots(2, 2, figsize=(10, 9), dpi=160)
    for i, leg in enumerate(legs):
        ax = axs[i // 2, i % 2]
        pts = x4[:, leg_idx[leg], :]
        _plot_leg_hist2d(ax, pts, 0, 2, f"{leg} foot (x,z) density")
        _draw_body_xz(ax, hips)
        m = pts.mean(axis=0)
        ax.scatter([m[0]], [m[2]], c="w", s=60, marker="x", linewidths=2.0)
        ax.invert_yaxis()  # z down for readability (optional)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "go2_side_xz_density.png"))
    plt.close(fig)

    print(f"[OK] wrote {os.path.join(args.out_dir, 'go2_top_xy_density.png')}")
    print(f"[OK] wrote {os.path.join(args.out_dir, 'go2_side_xz_density.png')}")


if __name__ == "__main__":
    main()

