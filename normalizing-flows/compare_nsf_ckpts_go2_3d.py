#!/usr/bin/env python3
import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from run_nsf_12_stand import build_model


@dataclass(frozen=True)
class CkptBundle:
    path: str
    stem: str
    args: dict
    model: torch.nn.Module
    mean: torch.Tensor  # (1,12)
    std: torch.Tensor  # (1,12)
    q0_loc: torch.Tensor  # (1,12)
    q0_scale: torch.Tensor  # (1,12)


def _maybe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


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


def _ckpt_to_bundle(path: str, device: torch.device) -> CkptBundle:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"expected a checkpoint dict with key 'model', got {type(ckpt)} at {path}")

    saved_args = ckpt.get("args", {}) or {}
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if mean is None or std is None:
        raise ValueError(f"checkpoint is missing mean/std: {path}")

    dim = int(saved_args.get("dim", 12))
    model = build_model(
        dim=dim,
        num_layers=int(saved_args.get("num_layers", 8)),
        hidden=int(saved_args.get("hidden", 128)),
        num_blocks=int(saved_args.get("num_blocks", 2)),
        num_bins=int(saved_args.get("num_bins", 8)),
        tail_bound=float(saved_args.get("tail_bound", 3.0)),
        trainable_base=bool(saved_args.get("trainable_base", True)),
        actnorm=bool(saved_args.get("actnorm", False)),
        mixing=str(saved_args.get("mixing", "none")),
        mixing_use_lu=bool(saved_args.get("mixing_use_lu", True)),
        mixing_seed=int(saved_args.get("mixing_seed", 0)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean = torch.as_tensor(mean, device=device, dtype=torch.float32).view(1, -1)
    std = torch.as_tensor(std, device=device, dtype=torch.float32).view(1, -1).clamp_min(1e-6)

    q0_loc = model.q0.loc.detach().to(dtype=torch.float32)
    q0_scale = torch.exp(model.q0.log_scale.detach().to(dtype=torch.float32))

    p = Path(path)
    return CkptBundle(
        path=path,
        stem=p.stem,
        args=saved_args,
        model=model,
        mean=mean,
        std=std,
        q0_loc=q0_loc.reshape(1, -1),
        q0_scale=q0_scale.reshape(1, -1),
    )


def _sample_z(
    bundle: CkptBundle,
    num_samples: int,
    *,
    sample_std: float,
    sigma_bound: float | None,
    seed: int,
) -> torch.Tensor:
    device = next(bundle.model.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn((num_samples, bundle.q0_loc.numel()), generator=g, device=device, dtype=torch.float32)
    if sigma_bound is not None:
        eps = torch.clamp(eps, -float(sigma_bound), float(sigma_bound))
    z = bundle.q0_loc.to(device) + bundle.q0_scale.to(device) * float(sample_std) * eps
    return z


@torch.no_grad()
def _decode_to_raw(bundle: CkptBundle, z: torch.Tensor) -> torch.Tensor:
    x_std = bundle.model(z)
    x_raw = x_std * bundle.std.to(device=x_std.device, dtype=x_std.dtype) + bundle.mean.to(device=x_std.device, dtype=x_std.dtype)
    return x_raw.to(dtype=torch.float32)


def _to_4x3(x12: torch.Tensor) -> torch.Tensor:
    if x12.ndim != 2 or x12.shape[1] != 12:
        raise ValueError(f"expected (N,12), got {tuple(x12.shape)}")
    return x12.view(-1, 4, 3)


def _downsample_np(arr: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(arr) <= max_points:
        return arr
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(arr), size=int(max_points), replace=False)
    return arr[idx]


def _add_scene(fig, scene_id: str, *, title: str, hips: dict[str, np.ndarray]):
    import plotly.graph_objects as go

    hip_pts = np.stack([hips["FL"], hips["FR"], hips["RR"], hips["RL"]], axis=0)
    fig.add_trace(
        go.Scatter3d(
            x=hip_pts[:, 0],
            y=hip_pts[:, 1],
            z=hip_pts[:, 2],
            mode="markers+text",
            name="hip joints",
            text=["FL_hip", "FR_hip", "RR_hip", "RL_hip"],
            textposition="top center",
            marker=dict(size=6, color="black"),
            showlegend=False,
        ),
        row=1,
        col=1 if scene_id == "scene1" else 2,
    )
    rect = np.vstack([hip_pts, hip_pts[:1]])
    fig.add_trace(
        go.Scatter3d(
            x=rect[:, 0],
            y=rect[:, 1],
            z=rect[:, 2],
            mode="lines",
            name="hip rectangle",
            line=dict(color="black", width=4),
            showlegend=False,
        ),
        row=1,
        col=1 if scene_id == "scene1" else 2,
    )

    fig.update_layout(
        **{
            scene_id: dict(
                xaxis_title="x (m)",
                yaxis_title="y (m)",
                zaxis_title="z (m)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.7, y=-1.7, z=1.1)),
            )
        }
    )
    fig.add_annotation(
        text=title,
        x=0.23 if scene_id == "scene1" else 0.77,
        y=1.05,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )


def _plot_two_ckpts(
    a: CkptBundle,
    b: CkptBundle,
    *,
    x_a_raw: torch.Tensor,
    x_b_raw: torch.Tensor,
    center_a_raw: torch.Tensor,
    center_b_raw: torch.Tensor,
    out_html: str,
    urdf: str,
    max_points: int,
    seed: int,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise SystemExit("plotly가 설치되어 있지 않습니다. `pip install plotly` 후 다시 실행하세요.") from e

    hips = _load_hips_from_urdf(urdf)

    legs = ["FL", "FR", "RL", "RR"]
    leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    colors = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}

    xa = _to_4x3(x_a_raw).cpu().numpy()
    xb = _to_4x3(x_b_raw).cpu().numpy()
    ca = _to_4x3(center_a_raw).cpu().numpy()[0]
    cb = _to_4x3(center_b_raw).cpu().numpy()[0]
    ma = _to_4x3(a.mean).cpu().numpy()[0]
    mb = _to_4x3(b.mean).cpu().numpy()[0]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.02,
    )

    def add_ckpt(col: int, *, x: np.ndarray, bundle: CkptBundle, mean_raw: np.ndarray, center_raw: np.ndarray):
        for leg in legs:
            pts = x[:, leg_idx[leg], :]
            pts = _downsample_np(pts, max_points=max_points // 4, seed=seed + col + leg_idx[leg])
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    name=f"{bundle.stem} {leg} samples",
                    marker=dict(size=1.6, color=colors[leg], opacity=0.25),
                    legendgroup=f"{bundle.stem}_{leg}",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

            samp_mean = pts.mean(axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=[samp_mean[0]],
                    y=[samp_mean[1]],
                    z=[samp_mean[2]],
                    mode="markers",
                    name=f"{bundle.stem} {leg} sample-mean",
                    marker=dict(size=7, color=colors[leg], symbol="x", opacity=0.95),
                    legendgroup=f"{bundle.stem}_{leg}",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[mean_raw[leg_idx[leg], 0]],
                    y=[mean_raw[leg_idx[leg], 1]],
                    z=[mean_raw[leg_idx[leg], 2]],
                    mode="markers",
                    name=f"{bundle.stem} mean (train)",
                    marker=dict(size=6, color=colors[leg], symbol="diamond", opacity=0.95),
                    legendgroup=f"{bundle.stem}_mean",
                    showlegend=(col == 1 and leg == "FL"),
                ),
                row=1,
                col=col,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[center_raw[leg_idx[leg], 0]],
                    y=[center_raw[leg_idx[leg], 1]],
                    z=[center_raw[leg_idx[leg], 2]],
                    mode="markers",
                    name=f"{bundle.stem} decode(z=loc)",
                    marker=dict(size=6, color=colors[leg], symbol="circle-open", opacity=0.95),
                    legendgroup=f"{bundle.stem}_center",
                    showlegend=(col == 1 and leg == "FL"),
                ),
                row=1,
                col=col,
            )

    add_ckpt(1, x=xa, bundle=a, mean_raw=ma, center_raw=ca)
    add_ckpt(2, x=xb, bundle=b, mean_raw=mb, center_raw=cb)

    _add_scene(fig, "scene1", title=a.stem, hips=hips)
    _add_scene(fig, "scene2", title=b.stem, hips=hips)

    fig.update_layout(
        title=(
            "NSF decoder samples (Go2 foot tips) — "
            "markers: sample-mean(x) / train-mean(◆) / decode(z=loc)(○)"
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    out_path = Path(out_html)
    _maybe_mkdir(str(out_path.parent))
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[OK] wrote {out_path}")

    for bundle in (a, b):
        loc = bundle.q0_loc.reshape(-1).cpu()
        scale = bundle.q0_scale.reshape(-1).cpu()
        print(
            f"[INFO] {bundle.stem} q0: loc mean={float(loc.mean()):.4f} std(mean scale)={float(scale.mean()):.4f} "
            f"min_scale={float(scale.min()):.4f} max_scale={float(scale.max()):.4f}"
        )


def _select_plane(pts_xyz: np.ndarray, plane: str) -> tuple[np.ndarray, str, str]:
    plane = (plane or "").strip().lower()
    if plane == "xy":
        return pts_xyz[:, [0, 1]], "x (m)", "y (m)"
    if plane == "xz":
        return pts_xyz[:, [0, 2]], "x (m)", "z (m)"
    if plane == "yz":
        return pts_xyz[:, [1, 2]], "y (m)", "z (m)"
    raise ValueError(f"unknown plane={plane!r} (expected one of: xy, xz, yz)")


def _quantile_range(x: np.ndarray, q: float) -> tuple[float, float]:
    q = float(q)
    if not (0.5 < q < 1.0):
        raise ValueError(f"quantile must be in (0.5,1), got {q}")
    lo_q = (1.0 - q) / 2.0
    hi_q = 1.0 - lo_q
    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    if not (lo < hi):
        eps = 1e-6
        lo -= eps
        hi += eps
    return lo, hi


def _plot_density_heatmaps(
    a: CkptBundle,
    b: CkptBundle,
    *,
    x_a_raw: torch.Tensor,
    x_b_raw: torch.Tensor,
    center_a_raw: torch.Tensor,
    center_b_raw: torch.Tensor,
    center_cloud_a_raw: torch.Tensor,
    center_cloud_b_raw: torch.Tensor,
    out_html: str,
    plane: str,
    bins: int,
    quantile: float,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise SystemExit("plotly가 설치되어 있지 않습니다. `pip install plotly` 후 다시 실행하세요.") from e

    legs = ["FL", "FR", "RL", "RR"]
    leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    colors = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}

    xa = _to_4x3(x_a_raw).cpu().numpy()
    xb = _to_4x3(x_b_raw).cpu().numpy()
    ca = _to_4x3(center_a_raw).cpu().numpy()[0]
    cb = _to_4x3(center_b_raw).cpu().numpy()[0]
    ma = _to_4x3(a.mean).cpu().numpy()[0]
    mb = _to_4x3(b.mean).cpu().numpy()[0]
    cla = _to_4x3(center_cloud_a_raw).cpu().numpy()
    clb = _to_4x3(center_cloud_b_raw).cpu().numpy()

    fig = make_subplots(
        rows=4,
        cols=2,
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
    )

    for r, leg in enumerate(legs, start=1):
        pts_a = xa[:, leg_idx[leg], :]
        pts_b = xb[:, leg_idx[leg], :]
        pts_a_2d, x_label, y_label = _select_plane(pts_a, plane)
        pts_b_2d, _, _ = _select_plane(pts_b, plane)
        all_x = np.concatenate([pts_a_2d[:, 0], pts_b_2d[:, 0]], axis=0)
        all_y = np.concatenate([pts_a_2d[:, 1], pts_b_2d[:, 1]], axis=0)
        x_lo, x_hi = _quantile_range(all_x, quantile)
        y_lo, y_hi = _quantile_range(all_y, quantile)

        def add_panel(col: int, pts_2d: np.ndarray, mean_raw: np.ndarray, center_raw: np.ndarray, center_cloud: np.ndarray, title: str):
            H, x_edges, y_edges = np.histogram2d(
                pts_2d[:, 0],
                pts_2d[:, 1],
                bins=int(bins),
                range=[[x_lo, x_hi], [y_lo, y_hi]],
            )
            z = np.log10(H.T + 1.0)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

            fig.add_trace(
                go.Heatmap(
                    x=x_centers,
                    y=y_centers,
                    z=z,
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=float(np.max(z)) if np.max(z) > 0 else 1.0,
                    showscale=(r == 1 and col == 2),
                    colorbar=dict(title="log10(count+1)") if (r == 1 and col == 2) else None,
                ),
                row=r,
                col=col,
            )

            # Center-cloud: samples near Gaussian center (small sigma), shown as faint gray dots.
            cloud_2d, _, _ = _select_plane(center_cloud, plane)
            if cloud_2d.shape[0] > 5000:
                cloud_2d = _downsample_np(cloud_2d, 5000, seed=0)
            fig.add_trace(
                go.Scatter(
                    x=cloud_2d[:, 0],
                    y=cloud_2d[:, 1],
                    mode="markers",
                    marker=dict(size=2, color="rgba(120,120,120,0.25)"),
                    name=f"{title} near-center samples",
                    showlegend=(r == 1 and col == 1),
                ),
                row=r,
                col=col,
            )

            samp_mean = pts_2d.mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=[samp_mean[0]],
                    y=[samp_mean[1]],
                    mode="markers",
                    marker=dict(size=10, color=colors[leg], symbol="x"),
                    name=f"{title} sample-mean",
                    showlegend=(r == 1 and col == 1),
                ),
                row=r,
                col=col,
            )

            mean_2d, _, _ = _select_plane(mean_raw[None, :], plane)
            center_2d, _, _ = _select_plane(center_raw[None, :], plane)
            fig.add_trace(
                go.Scatter(
                    x=[mean_2d[0, 0]],
                    y=[mean_2d[0, 1]],
                    mode="markers",
                    marker=dict(size=9, color=colors[leg], symbol="diamond"),
                    name=f"{title} train-mean",
                    showlegend=(r == 1 and col == 1),
                ),
                row=r,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[center_2d[0, 0]],
                    y=[center_2d[0, 1]],
                    mode="markers",
                    marker=dict(size=9, color=colors[leg], symbol="circle-open"),
                    name=f"{title} decode(z=loc)",
                    showlegend=(r == 1 and col == 1),
                ),
                row=r,
                col=col,
            )

            fig.update_xaxes(title_text=x_label if r == 4 else "", row=r, col=col)
            fig.update_yaxes(title_text=y_label if col == 1 else "", row=r, col=col)

        add_panel(1, pts_a_2d, mean_raw=ma[leg_idx[leg]], center_raw=ca[leg_idx[leg]], center_cloud=cla[:, leg_idx[leg], :], title=a.stem)
        add_panel(2, pts_b_2d, mean_raw=mb[leg_idx[leg]], center_raw=cb[leg_idx[leg]], center_cloud=clb[:, leg_idx[leg], :], title=b.stem)

        fig.add_annotation(
            text=leg,
            x=-0.04,
            y=1.02 - (r - 1) * 0.245,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, color=colors[leg]),
        )

    fig.add_annotation(text=a.stem, x=0.25, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    fig.add_annotation(text=b.stem, x=0.75, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig.update_layout(
        title=f"Density heatmaps ({plane.upper()} plane) — heat: log10(count+1), markers: sample-mean(x) / train-mean(◆) / decode(z=loc)(○)",
        margin=dict(l=20, r=20, t=90, b=30),
        height=1200,
        legend=dict(itemsizing="constant", orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )

    out_path = Path(out_html)
    _maybe_mkdir(str(out_path.parent))
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[OK] wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare two nsf12_stand checkpoints by sampling & 3D plotting (Go2 hips).")
    p.add_argument("--ckpt-a", type=str, required=True, help="Checkpoint A (dict with key 'model')")
    p.add_argument("--ckpt-b", type=str, required=True, help="Checkpoint B (dict with key 'model')")
    p.add_argument("--num-samples", type=int, default=120_000, help="How many latent samples to decode per ckpt")
    p.add_argument("--sample-std", type=float, default=1.0, help="Scale on base Gaussian (z = loc + scale*sample_std*eps)")
    p.add_argument("--sigma-bound", type=float, default=0.0, help="If >0, clamp eps to [-k, k] before scaling (approx trunc)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-points", type=int, default=120_000, help="Max total scatter points per ckpt (roughly split by legs)")
    p.add_argument("--urdf", type=str, default="/home/kdg/IsaacLab/outputs/go2_urdf/urdf/go2.urdf")
    p.add_argument("--out", type=str, default="sota/ckpt_compare/compare_ckpts_go2_3d.html")
    p.add_argument(
        "--density-out",
        type=str,
        default="",
        help="If set, also write a 2D density-heatmap HTML to this path (or auto-suffix if empty and --density is enabled).",
    )
    p.add_argument("--density", action=argparse.BooleanOptionalAction, default=True, help="Also generate 2D density heatmaps.")
    p.add_argument("--density-plane", type=str, default="xy", choices=["xy", "xz", "yz"], help="2D plane for density heatmaps.")
    p.add_argument("--density-bins", type=int, default=80, help="Bins per axis for density heatmaps.")
    p.add_argument("--density-quantile", type=float, default=0.999, help="Quantile range for axis limits (robust to outliers).")
    p.add_argument("--center-samples", type=int, default=8000, help="How many near-center samples to overlay (per ckpt).")
    p.add_argument("--center-sigma", type=float, default=0.5, help="Clamp eps to [-k,k] for near-center sampling overlay.")
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    a = _ckpt_to_bundle(args.ckpt_a, device=device)
    b = _ckpt_to_bundle(args.ckpt_b, device=device)

    sigma_bound = float(args.sigma_bound) if float(args.sigma_bound) > 0 else None
    z_a = _sample_z(a, int(args.num_samples), sample_std=float(args.sample_std), sigma_bound=sigma_bound, seed=int(args.seed))
    z_b = _sample_z(b, int(args.num_samples), sample_std=float(args.sample_std), sigma_bound=sigma_bound, seed=int(args.seed) + 1)

    x_a_raw = _decode_to_raw(a, z_a)
    x_b_raw = _decode_to_raw(b, z_b)

    # "Gaussian center" under the flow: decode the base mean z=loc.
    center_a_raw = _decode_to_raw(a, a.q0_loc.to(device))
    center_b_raw = _decode_to_raw(b, b.q0_loc.to(device))

    _plot_two_ckpts(
        a,
        b,
        x_a_raw=x_a_raw,
        x_b_raw=x_b_raw,
        center_a_raw=center_a_raw,
        center_b_raw=center_b_raw,
        out_html=str(args.out),
        urdf=str(args.urdf),
        max_points=int(args.max_points),
        seed=int(args.seed),
    )

    if bool(args.density):
        density_out = str(args.density_out or "").strip()
        if not density_out:
            base = Path(args.out)
            density_out = str(base.with_name(f"{base.stem}__density_{args.density_plane}.html"))

        center_sigma = float(args.center_sigma)
        center_n = int(args.center_samples)
        if center_n > 0:
            zc_a = _sample_z(a, center_n, sample_std=1.0, sigma_bound=center_sigma, seed=int(args.seed) + 100)
            zc_b = _sample_z(b, center_n, sample_std=1.0, sigma_bound=center_sigma, seed=int(args.seed) + 200)
            center_cloud_a_raw = _decode_to_raw(a, zc_a)
            center_cloud_b_raw = _decode_to_raw(b, zc_b)
        else:
            center_cloud_a_raw = x_a_raw[:0]
            center_cloud_b_raw = x_b_raw[:0]

        _plot_density_heatmaps(
            a,
            b,
            x_a_raw=x_a_raw,
            x_b_raw=x_b_raw,
            center_a_raw=center_a_raw,
            center_b_raw=center_b_raw,
            center_cloud_a_raw=center_cloud_a_raw,
            center_cloud_b_raw=center_cloud_b_raw,
            out_html=density_out,
            plane=str(args.density_plane),
            bins=int(args.density_bins),
            quantile=float(args.density_quantile),
        )


if __name__ == "__main__":
    main()
