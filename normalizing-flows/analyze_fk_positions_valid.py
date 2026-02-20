#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class Summary:
    data_path: str
    created_at: str
    n: int
    shape: list[int]
    dtype: str
    per_dim_min: list[float]
    per_dim_max: list[float]
    per_dim_mean: list[float]
    per_dim_std: list[float]
    approx_quantiles: dict[str, list[float]]
    per_leg: dict[str, dict[str, dict[str, float]]]
    approx_corr_12x12: list[list[float]]


def _to_4x3(x12: torch.Tensor) -> torch.Tensor:
    if x12.ndim != 2 or x12.shape[1] != 12:
        raise ValueError(f"expected (N,12), got {tuple(x12.shape)}")
    return x12.view(-1, 4, 3)


def _tensor_to_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().to(torch.float64).tolist()]


def _maybe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sample_rows(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    n_total = x.shape[0]
    n = min(n, n_total)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randint(0, n_total, (n,), generator=g)
    return x.index_select(0, idx)


def _approx_quantiles(x_samp: torch.Tensor, qs: list[float]) -> dict[str, list[float]]:
    # torch.quantile can be slow; for 12 dims + up to ~1e6 samples it's okay.
    q = torch.tensor(qs, dtype=torch.float64)
    x64 = x_samp.to(torch.float64)
    out = torch.quantile(x64, q, dim=0)  # (len(qs), 12)
    return {f"{qq:.3f}": _tensor_to_list(out[i]) for i, qq in enumerate(qs)}


def _corr_12x12(x_samp: torch.Tensor) -> list[list[float]]:
    x = x_samp.to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    std = torch.sqrt(torch.diag(cov)).clamp_min(1e-12)
    corr = cov / (std[:, None] * std[None, :])
    corr = corr.clamp(-1.0, 1.0)
    return corr.cpu().tolist()


def _plot_2d_hist(xy: np.ndarray, out_png: str, title: str) -> None:
    # xy: (N,2)
    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    h = ax.hist2d(xy[:, 0], xy[:, 1], bins=220, norm=matplotlib.colors.LogNorm())
    fig.colorbar(h[3], ax=ax, label="count (log)")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _plot_1d_hist(vals: np.ndarray, out_png: str, title: str, xlabel: str) -> None:
    fig = plt.figure(figsize=(6, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(vals, bins=240, log=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze fk_positions_valid.pt distribution (foot positions).")
    ap.add_argument("--data-path", default="fk_positions_valid.pt")
    ap.add_argument("--out-dir", default="sota/dataset_stats/fk_positions_valid")
    ap.add_argument("--sample", type=int, default=300_000, help="rows sampled for plots/quantiles/corr")
    ap.add_argument("--corr-sample", type=int, default=200_000, help="rows sampled for 12x12 correlation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _maybe_mkdir(args.out_dir)

    x = torch.load(args.data_path, map_location="cpu")
    if not torch.is_tensor(x):
        raise TypeError(f"expected torch.Tensor, got {type(x)}")
    if x.ndim != 2 or x.shape[1] != 12:
        raise ValueError(f"expected (N,12), got {tuple(x.shape)}")

    n = int(x.shape[0])

    per_dim_min = x.amin(dim=0)
    per_dim_max = x.amax(dim=0)
    per_dim_mean = x.mean(dim=0)
    per_dim_std = x.std(dim=0, unbiased=False)

    x_samp = _sample_rows(x, args.sample, args.seed)
    qs = [0.001, 0.010, 0.050, 0.500, 0.950, 0.990, 0.999]
    approx_q = _approx_quantiles(x_samp, qs)

    x_corr = _sample_rows(x, args.corr_sample, args.seed + 123)
    corr = _corr_12x12(x_corr)

    # Per-leg summary on full data (cheap; just view)
    x4 = _to_4x3(x)
    legs = ["FL", "FR", "RL", "RR"]
    axes = ["x", "y", "z"]
    per_leg: dict[str, dict[str, dict[str, float]]] = {}
    for li, leg in enumerate(legs):
        per_leg[leg] = {}
        for ai, axname in enumerate(axes):
            v = x4[:, li, ai]
            per_leg[leg][axname] = {
                "min": float(v.min().item()),
                "max": float(v.max().item()),
                "mean": float(v.mean().item()),
                "std": float(v.std(unbiased=False).item()),
            }

    summary = Summary(
        data_path=args.data_path,
        created_at=datetime.now().isoformat(timespec="seconds"),
        n=n,
        shape=list(x.shape),
        dtype=str(x.dtype),
        per_dim_min=_tensor_to_list(per_dim_min),
        per_dim_max=_tensor_to_list(per_dim_max),
        per_dim_mean=_tensor_to_list(per_dim_mean),
        per_dim_std=_tensor_to_list(per_dim_std),
        approx_quantiles=approx_q,
        per_leg=per_leg,
        approx_corr_12x12=corr,
    )

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    # Plots (sampled)
    x4s = _to_4x3(x_samp).numpy()
    for li, leg in enumerate(legs):
        # 2D hist for (x,y), (x,z), (y,z)
        _plot_2d_hist(x4s[:, li, [0, 1]], os.path.join(args.out_dir, f"{leg}_xy_hist2d.png"), f"{leg} (x,y) hist2d")
        _plot_2d_hist(x4s[:, li, [0, 2]], os.path.join(args.out_dir, f"{leg}_xz_hist2d.png"), f"{leg} (x,z) hist2d")
        _plot_2d_hist(x4s[:, li, [1, 2]], os.path.join(args.out_dir, f"{leg}_yz_hist2d.png"), f"{leg} (y,z) hist2d")

        # 1D hists
        _plot_1d_hist(x4s[:, li, 0], os.path.join(args.out_dir, f"{leg}_x_hist.png"), f"{leg} x hist", "x")
        _plot_1d_hist(x4s[:, li, 1], os.path.join(args.out_dir, f"{leg}_y_hist.png"), f"{leg} y hist", "y")
        _plot_1d_hist(x4s[:, li, 2], os.path.join(args.out_dir, f"{leg}_z_hist.png"), f"{leg} z hist", "z")

        r_xy = np.sqrt(x4s[:, li, 0] ** 2 + x4s[:, li, 1] ** 2)
        r_xyz = np.sqrt(x4s[:, li, 0] ** 2 + x4s[:, li, 1] ** 2 + x4s[:, li, 2] ** 2)
        _plot_1d_hist(r_xy, os.path.join(args.out_dir, f"{leg}_rxy_hist.png"), f"{leg} r_xy hist", "r_xy")
        _plot_1d_hist(r_xyz, os.path.join(args.out_dir, f"{leg}_rxyz_hist.png"), f"{leg} r_xyz hist", "r_xyz")

    print(f"[OK] wrote: {os.path.join(args.out_dir, 'summary.json')}")
    print(f"[OK] wrote plots under: {args.out_dir}")


if __name__ == "__main__":
    main()
