#!/usr/bin/env python3
"""
Analyze a Go2 foot-position dataset (N,12) and optionally apply a boolean validity mask (N,).

Outputs:
  - JSON summary with means/quantiles
  - Interactive 3D HTML scatter (Plotly, embedded JS) showing per-leg clouds + means + hip offsets
  - IK solution for (optionally symmetrized) mean pose and delta vs Unitree Go2 default joint pos
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


_LEG_NAMES = ("FL", "FR", "RL", "RR")


def _to_4x3(x12: torch.Tensor) -> torch.Tensor:
    if x12.ndim != 2 or x12.shape[1] != 12:
        raise ValueError(f"expected (N,12), got {tuple(x12.shape)}")
    return x12.view(-1, 4, 3)


def _symmetrize_mean_4x3(m: torch.Tensor) -> torch.Tensor:
    """
    Enforce left-right symmetry:
      - x, z: average within (FL,FR) and within (RL,RR)
      - y: mirrored magnitude within (FL,FR) and within (RL,RR)
    """
    if m.shape != (4, 3):
        raise ValueError(f"expected (4,3) mean, got {tuple(m.shape)}")

    out = m.clone()

    # Front pair
    front_x = 0.5 * (m[0, 0] + m[1, 0])
    front_z = 0.5 * (m[0, 2] + m[1, 2])
    front_y_mag = 0.5 * (m[0, 1].abs() + m[1, 1].abs())
    out[0] = torch.tensor([front_x, +front_y_mag, front_z], dtype=m.dtype)
    out[1] = torch.tensor([front_x, -front_y_mag, front_z], dtype=m.dtype)

    # Rear pair
    rear_x = 0.5 * (m[2, 0] + m[3, 0])
    rear_z = 0.5 * (m[2, 2] + m[3, 2])
    rear_y_mag = 0.5 * (m[2, 1].abs() + m[3, 1].abs())
    out[2] = torch.tensor([rear_x, +rear_y_mag, rear_z], dtype=m.dtype)
    out[3] = torch.tensor([rear_x, -rear_y_mag, rear_z], dtype=m.dtype)

    return out


def _sample_rows(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    n_total = int(x.shape[0])
    n = min(int(n), n_total)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randint(0, n_total, (n,), generator=g)
    return x.index_select(0, idx)


def _resolve_paths(
    data_path: Path | None,
    mask_path: Path | None,
    isaaclab_kin_dir: Path,
) -> tuple[Path, Path | None]:
    """
    Returns (positions_path, mask_path_or_none).
    If the provided data_path points to a boolean mask, tries to auto-resolve the corresponding
    positions file from IsaacLab's kinematics folder.
    """
    if data_path is None:
        # Prefer local float dataset if present
        if Path("fk_positions_valid_mix.pt").exists():
            return Path("fk_positions_valid_mix.pt"), mask_path
        # Otherwise fall back to IsaacLab kinematics outputs
        cand = isaaclab_kin_dir / "fk_positions_valid_mix.pt"
        if cand.exists():
            return cand, mask_path or (isaaclab_kin_dir / "fk_valid_new_mix.pt")
        raise FileNotFoundError("No --data-path provided and could not auto-find fk_positions_valid_mix.pt")

    if not data_path.exists():
        raise FileNotFoundError(f"--data-path not found: {data_path}")

    obj = torch.load(str(data_path), map_location="cpu")
    if isinstance(obj, torch.Tensor) and obj.dtype == torch.bool and obj.ndim == 1:
        # Looks like a validity mask. Try to locate the associated positions tensor.
        cand = data_path.with_name("fk_positions_valid_mix.pt")
        if cand.exists():
            return cand, data_path
        cand = isaaclab_kin_dir / "fk_positions_valid_mix.pt"
        if cand.exists():
            return cand, data_path
        raise ValueError(
            f"{data_path} looks like a boolean mask (shape={tuple(obj.shape)}). "
            "Please pass the foot positions tensor (N,12) via --data-path, "
            "or place fk_positions_valid_mix.pt alongside the mask."
        )

    return data_path, mask_path


@dataclass
class Summary:
    positions_path: str
    mask_path: str | None
    num_total: int
    num_valid: int
    num_invalid: int
    mean_raw_4x3: list[list[float]]
    mean_sym_4x3: list[list[float]]
    per_leg_quantiles_raw: dict[str, dict[str, list[float]]]
    ik_mean_sym_joint_ordered: list[float] | None
    go2_default_joint_ordered: list[float]
    delta_vs_default: list[float] | None


def _go2_default_joint_ordered() -> torch.Tensor:
    """
    Default joint positions implied by UNITREE_GO2_CFG in isaaclab_assets/robots/unitree.py:
      - L_hip: +0.1 (FL, RL)
      - R_hip: -0.1 (FR, RR)
      - Front thigh: 0.8 (FL, FR)
      - Rear thigh: 1.0 (RL, RR)
      - Calf: -1.5 (all)

    Ordering matches Go2Solver: [hip(FL,FR,RL,RR), thigh(FL,FR,RL,RR), calf(FL,FR,RL,RR)].
    """
    hip = torch.tensor([0.1, -0.1, 0.1, -0.1], dtype=torch.float32)
    thigh = torch.tensor([0.8, 0.8, 1.0, 1.0], dtype=torch.float32)
    calf = torch.tensor([-1.5, -1.5, -1.5, -1.5], dtype=torch.float32)
    return torch.cat([hip, thigh, calf], dim=0)


def _load_go2_solver(isaaclab_kin_dir: Path, device: str) -> Any:
    # Make sure FK_IK_Solver.py import works.
    sys.path.insert(0, str(isaaclab_kin_dir))
    from FK_IK_Solver import Go2Solver  # type: ignore

    return Go2Solver(device=device)


def _write_html_3d(
    out_path: Path,
    x_valid: torch.Tensor,
    mean_raw: torch.Tensor,
    mean_sym: torch.Tensor,
    sample: int,
    seed: int,
    point_size: float,
    title: str,
) -> None:
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("plotly is not available in this environment")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample for speed
    xs = _sample_rows(x_valid, sample, seed)
    x4 = _to_4x3(xs).numpy()  # (N,4,3)

    fig = go.Figure()

    colors = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}
    leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}

    for leg in _LEG_NAMES:
        pts = x4[:, leg_idx[leg], :]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"{leg} feet (samples)",
                marker=dict(size=point_size, color=colors[leg], opacity=0.30),
            )
        )

        mr = mean_raw[leg_idx[leg]].cpu().numpy()
        ms = mean_sym[leg_idx[leg]].cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=[mr[0]],
                y=[mr[1]],
                z=[mr[2]],
                mode="markers+text",
                name=f"{leg} mean(raw)",
                text=[f"{leg}_mean_raw"],
                textposition="top center",
                marker=dict(size=6.5, color=colors[leg], symbol="x"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[ms[0]],
                y=[ms[1]],
                z=[ms[2]],
                mode="markers+text",
                name=f"{leg} mean(sym)",
                text=[f"{leg}_mean_sym"],
                textposition="top center",
                marker=dict(size=6.5, color=colors[leg], symbol="cross"),
            )
        )

    # Hip points from Go2Solver constants (base frame)
    hip_offsets = np.array(
        [
            [0.1934, 0.0465, 0.0],  # FL
            [0.1934, -0.0465, 0.0],  # FR
            [-0.1934, 0.0465, 0.0],  # RL
            [-0.1934, -0.0465, 0.0],  # RR
        ],
        dtype=np.float64,
    )
    fig.add_trace(
        go.Scatter3d(
            x=hip_offsets[:, 0],
            y=hip_offsets[:, 1],
            z=hip_offsets[:, 2],
            mode="markers+text",
            name="hip offsets",
            text=["FL_hip", "FR_hip", "RL_hip", "RR_hip"],
            textposition="top center",
            marker=dict(size=6, color="black"),
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=-1.7, z=1.1)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Go2 foot dataset + mean IK + Go2 default delta.")
    ap.add_argument("--data-path", type=str, default=None, help="Path to (N,12) foot positions OR a bool mask (N,).")
    ap.add_argument("--mask-path", type=str, default=None, help="Optional bool mask (N,) for valid samples.")
    ap.add_argument(
        "--isaaclab-kin-dir",
        type=str,
        default="/home/kdg/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/kinematics",
        help="IsaacLab kinematics directory containing FK_IK_Solver.py and dataset outputs.",
    )
    ap.add_argument("--out-dir", type=str, default="sota/dataset_stats/fk_valid_new_mix_go2")
    ap.add_argument("--html-name", type=str, default="go2_foot_distribution_3d_html.html")
    ap.add_argument("--sample", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point-size", type=float, default=1.6)
    ap.add_argument("--no-symmetrize", action="store_true", help="If set, use raw mean for IK instead of sym mean.")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    isaaclab_kin_dir = Path(args.isaaclab_kin_dir)
    data_path = Path(args.data_path) if args.data_path else None
    mask_path = Path(args.mask_path) if args.mask_path else None

    positions_path, mask_path = _resolve_paths(data_path, mask_path, isaaclab_kin_dir)

    x_all = torch.load(str(positions_path), map_location="cpu")
    if not torch.is_tensor(x_all) or x_all.ndim != 2 or x_all.shape[1] != 12:
        raise TypeError(f"expected (N,12) tensor at {positions_path}, got {type(x_all)} shape={getattr(x_all,'shape',None)}")

    num_total = int(x_all.shape[0])
    if mask_path is not None:
        mask = torch.load(str(mask_path), map_location="cpu")
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1 or int(mask.shape[0]) != num_total:
            raise TypeError(f"expected bool mask shape ({num_total},), got {type(mask)} {getattr(mask,'dtype',None)} {getattr(mask,'shape',None)}")
        x_valid = x_all[mask]
        num_valid = int(mask.sum().item())
        num_invalid = num_total - num_valid
    else:
        x_valid = x_all
        num_valid = num_total
        num_invalid = 0

    # Means
    x4 = _to_4x3(x_valid)
    mean_raw = x4.mean(dim=0)  # (4,3)
    mean_sym = _symmetrize_mean_4x3(mean_raw)

    # Quantiles (sampled, per leg / axis)
    q_sample = min(400_000, int(x_valid.shape[0]))
    xs_q = _sample_rows(x_valid, q_sample, args.seed)
    xq4 = _to_4x3(xs_q)
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    per_leg_quantiles: dict[str, dict[str, list[float]]] = {}
    for li, leg in enumerate(_LEG_NAMES):
        pts = xq4[:, li, :]  # (M,3)
        qv = torch.quantile(pts, torch.tensor(quantiles), dim=0)  # (Q,3)
        per_leg_quantiles[leg] = {
            "quantiles": [float(q) for q in quantiles],
            "x": [float(v) for v in qv[:, 0].tolist()],
            "y": [float(v) for v in qv[:, 1].tolist()],
            "z": [float(v) for v in qv[:, 2].tolist()],
        }

    # IK for mean pose
    solver = _load_go2_solver(isaaclab_kin_dir=isaaclab_kin_dir, device=args.device)
    mean_for_ik = mean_raw if args.no_symmetrize else mean_sym
    mean_for_ik_b = mean_for_ik.unsqueeze(0).to(args.device)
    ik = solver.go2_ik_new(mean_for_ik_b).squeeze(0).detach().cpu()  # (12,)

    default_joint = _go2_default_joint_ordered()
    delta = (ik - default_joint).tolist()

    # Write HTML plot
    html_path = out_dir / args.html_name
    if _PLOTLY_AVAILABLE:
        _write_html_3d(
            out_path=html_path,
            x_valid=x_valid,
            mean_raw=mean_raw,
            mean_sym=mean_sym,
            sample=args.sample,
            seed=args.seed,
            point_size=args.point_size,
            title=f"Go2 foot distribution: {positions_path.name} (valid={num_valid}/{num_total})",
        )

    summary = Summary(
        positions_path=str(positions_path),
        mask_path=str(mask_path) if mask_path is not None else None,
        num_total=num_total,
        num_valid=num_valid,
        num_invalid=num_invalid,
        mean_raw_4x3=mean_raw.cpu().tolist(),
        mean_sym_4x3=mean_sym.cpu().tolist(),
        per_leg_quantiles_raw=per_leg_quantiles,
        ik_mean_sym_joint_ordered=ik.tolist(),
        go2_default_joint_ordered=default_joint.tolist(),
        delta_vs_default=delta,
    )

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    # Also write a short human-readable report
    report_lines = []
    report_lines.append(f"positions_path: {positions_path}")
    report_lines.append(f"mask_path: {mask_path}")
    report_lines.append(f"num_total: {num_total}")
    report_lines.append(f"num_valid: {num_valid}")
    report_lines.append(f"num_invalid: {num_invalid}")
    report_lines.append("")
    report_lines.append("mean_raw_4x3 (FL,FR,RL,RR):")
    report_lines.extend([f"  {_LEG_NAMES[i]}: {mean_raw[i].tolist()}" for i in range(4)])
    report_lines.append("")
    report_lines.append("mean_sym_4x3 (FL,FR,RL,RR):")
    report_lines.extend([f"  {_LEG_NAMES[i]}: {mean_sym[i].tolist()}" for i in range(4)])
    report_lines.append("")
    report_lines.append("IK(mean_sym) joint_ordered [hip(FL,FR,RL,RR), thigh(...), calf(...)]:")
    report_lines.append("  " + " ".join(f"{v:+.6f}" for v in ik.tolist()))
    report_lines.append("Go2 default joint_ordered:")
    report_lines.append("  " + " ".join(f"{v:+.6f}" for v in default_joint.tolist()))
    report_lines.append("delta = IK - default:")
    report_lines.append("  " + " ".join(f"{v:+.6f}" for v in delta))
    if _PLOTLY_AVAILABLE:
        report_lines.append("")
        report_lines.append(f"3D html: {html_path}")
    report_lines.append(f"json: {json_path}")
    (out_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()

