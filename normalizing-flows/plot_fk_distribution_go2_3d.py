#!/usr/bin/env python3
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import plotly.graph_objects as go
import torch


def _maybe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sample_rows(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    n_total = x.shape[0]
    n = min(n, n_total)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randint(0, n_total, (n,), generator=g)
    return x.index_select(0, idx)


def _to_4x3(x12: torch.Tensor) -> torch.Tensor:
    if x12.ndim != 2 or x12.shape[1] != 12:
        raise ValueError(f"expected (N,12), got {tuple(x12.shape)}")
    return x12.view(-1, 4, 3)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive 3D plot of fk_positions_valid distribution over Go2 hip frame.")
    ap.add_argument("--data-path", default="fk_positions_valid.pt")
    ap.add_argument("--urdf", default="/home/kdg/IsaacLab/outputs/go2_urdf/urdf/go2.urdf")
    ap.add_argument("--out", default="sota/dataset_stats/fk_positions_valid_go2/go2_foot_distribution_3d.html")
    ap.add_argument("--sample", type=int, default=120_000, help="rows sampled for the 3D plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point-size", type=float, default=1.6)
    args = ap.parse_args()

    _maybe_mkdir(os.path.dirname(args.out))

    x = torch.load(args.data_path, map_location="cpu")
    if not torch.is_tensor(x) or x.ndim != 2 or x.shape[1] != 12:
        raise TypeError(f"expected (N,12) tensor at {args.data_path}, got {type(x)} shape={getattr(x,'shape',None)}")

    hips = _load_hips_from_urdf(args.urdf)

    xs = _sample_rows(x, args.sample, args.seed)
    x4 = _to_4x3(xs).numpy()  # (N,4,3)
    legs = ["FL", "FR", "RL", "RR"]
    leg_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    colors = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}

    fig = go.Figure()

    # Foot clouds + means
    for leg in legs:
        pts = x4[:, leg_idx[leg], :]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"{leg} feet (samples)",
                marker=dict(size=args.point_size, color=colors[leg], opacity=0.35),
            )
        )
        m = pts.mean(axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[m[0]],
                y=[m[1]],
                z=[m[2]],
                mode="markers+text",
                name=f"{leg} mean",
                text=[f"{leg}_mean"],
                textposition="top center",
                marker=dict(size=6.5, color=colors[leg], symbol="x"),
            )
        )

    # Hip points and a simple body rectangle (in hip plane z=0)
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
        )
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
        )
    )

    # Layout: z downward looks more like robotics convention in many viewers
    fig.update_layout(
        title="Go2 FK reachable foot distribution (sampled) + hip frame",
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

    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()

