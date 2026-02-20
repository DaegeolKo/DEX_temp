import argparse
import os
import sys
from typing import Sequence

import torch


LEG_NAMES = ["FL", "FR", "RL", "RR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate *nearby* joint-angle samples (not global random), measure FK foot-tip variation for one leg, "
            "then interpolate (내분) between two FK foot points and run IK to check joint-limit validity."
        )
    )
    parser.add_argument("--num-samples", type=int, default=512, help="Number of nearby joint samples to generate.")
    parser.add_argument(
        "--leg",
        type=str,
        default="FL",
        help="Which leg to analyze: FL/FR/RL/RR or 0/1/2/3.",
    )
    parser.add_argument(
        "--center",
        type=str,
        default="default",
        choices=["default", "mid"],
        help="Center joint pose strategy: 'default' (standing-like) or 'mid' (joint-limit midpoints).",
    )
    parser.add_argument(
        "--center-joints",
        type=float,
        nargs=12,
        default=None,
        metavar=("H_FL", "H_FR", "H_RL", "H_RR", "T_FL", "T_FR", "T_RL", "T_RR", "C_FL", "C_FR", "C_RL", "C_RR"),
        help="Override center joint angles (12 floats, joint-wise order hip(4), thigh(4), calf(4)).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs=3,
        default=(0.03, 0.03, 0.03),
        metavar=("HIP_STD", "THIGH_STD", "CALF_STD"),
        help="Gaussian noise std (rad) applied to (hip, thigh, calf) of the chosen leg.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="If set, overrides --noise-std and uses (joint_range * noise_scale) per joint (unitless).",
    )
    parser.add_argument("--interp-steps", type=int, default=21, help="Number of interpolation points (including ends).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="torch device override (e.g., cuda, cpu).")
    parser.add_argument("--eps", type=float, default=1e-3, help="Tolerance for joint-limit check (radians).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for saved tensors (default: ./outputs_near under this script folder).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save matplotlib visualizations (3D foot scatter + metrics) into out_dir.",
    )
    parser.add_argument("--plot-dpi", type=int, default=160, help="DPI for saved plots (only with --plot).")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving tensor outputs (.pt). Plots may still be saved if --plot is set.",
    )
    return parser.parse_args()


def resolve_leg_index(leg: str) -> int:
    leg = leg.strip()
    if leg.isdigit():
        idx = int(leg)
        if idx < 0 or idx > 3:
            raise ValueError("--leg must be 0..3 or one of FL/FR/RL/RR.")
        return idx
    leg_up = leg.upper()
    if leg_up not in LEG_NAMES:
        raise ValueError("--leg must be 0..3 or one of FL/FR/RL/RR.")
    return LEG_NAMES.index(leg_up)


def get_go2_limits(device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-leg limits for Go2 from USD.

    Returns:
        min_limits, max_limits shaped (3, 4) for (hip, thigh, calf) x (FL, FR, RL, RR).
    """
    hip_min = torch.tensor([-1.0472, -1.0472, -1.0472, -1.0472], device=device, dtype=dtype)
    hip_max = torch.tensor([1.0472, 1.0472, 1.0472, 1.0472], device=device, dtype=dtype)
    thigh_min = torch.tensor([-1.5708, -1.5708, -0.5236, -0.5236], device=device, dtype=dtype)
    thigh_max = torch.tensor([3.4907, 3.4907, 4.5379, 4.5379], device=device, dtype=dtype)
    calf_min = torch.tensor([-2.7227, -2.7227, -2.7227, -2.7227], device=device, dtype=dtype)
    calf_max = torch.tensor([-0.8378, -0.8378, -0.8378, -0.8378], device=device, dtype=dtype)
    min_limits = torch.stack([hip_min, thigh_min, calf_min], dim=0)
    max_limits = torch.stack([hip_max, thigh_max, calf_max], dim=0)
    return min_limits, max_limits


def check_joint_limits_jointwise(
    joint_angles: torch.Tensor,
    min_limits: torch.Tensor,
    max_limits: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Check joint limits for joint-wise ordered angles.

    Args:
        joint_angles: Tensor shaped (N, 12) in order hip(4), thigh(4), calf(4).
        min_limits/max_limits: shaped (3, 4).
        eps: tolerance in radians.

    Returns:
        valid_mask: Tensor shaped (N,) True if all 12 joints are within limits.
    """
    hip = joint_angles[:, 0:4]
    thigh = joint_angles[:, 4:8]
    calf = joint_angles[:, 8:12]

    hip_min, thigh_min, calf_min = min_limits[0], min_limits[1], min_limits[2]
    hip_max, thigh_max, calf_max = max_limits[0], max_limits[1], max_limits[2]

    valid_hip = (hip >= hip_min.unsqueeze(0) - eps) & (hip <= hip_max.unsqueeze(0) + eps)
    valid_thigh = (thigh >= thigh_min.unsqueeze(0) - eps) & (thigh <= thigh_max.unsqueeze(0) + eps)
    valid_calf = (calf >= calf_min.unsqueeze(0) - eps) & (calf <= calf_max.unsqueeze(0) + eps)
    return valid_hip.all(dim=1) & valid_thigh.all(dim=1) & valid_calf.all(dim=1)


def default_center_pose(device: str, dtype: torch.dtype) -> torch.Tensor:
    """A 'standing-like' joint pose (joint-wise order hip(4), thigh(4), calf(4)).

    This matches the values used in `go2_env.py` comments as a reasonable natural configuration.
    """
    return torch.tensor(
        [[0.1000, -0.1000, 0.1050, -0.1050, 0.8909, 0.8909, 1.1620, 1.1620, -1.6790, -1.6790, -1.4150, -1.4150]],
        device=device,
        dtype=dtype,
    )


def clamp_jointwise(q: torch.Tensor, min_limits: torch.Tensor, max_limits: torch.Tensor) -> torch.Tensor:
    min_flat = min_limits.reshape(-1)
    max_flat = max_limits.reshape(-1)
    return torch.maximum(torch.minimum(q, max_flat), min_flat)


def compute_noise_std_for_leg(
    noise_std: Sequence[float],
    noise_scale: float | None,
    min_limits: torch.Tensor,
    max_limits: torch.Tensor,
    leg_idx: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if noise_scale is None:
        std = torch.tensor(noise_std, device=device, dtype=dtype)
        if std.numel() != 3:
            raise ValueError("--noise-std must provide 3 values: hip/thigh/calf.")
        return std
    joint_range = (max_limits - min_limits)[:, leg_idx]  # (3,)
    return joint_range * float(noise_scale)


def format_vec3(x: torch.Tensor) -> str:
    vals = x.detach().cpu().tolist()
    return f"({vals[0]:+.4f}, {vals[1]:+.4f}, {vals[2]:+.4f})"


def save_plots(
    *,
    out_dir: str,
    leg_name: str,
    foot_positions: torch.Tensor,
    foot_a: torch.Tensor,
    foot_b: torch.Tensor,
    foot_interp: torch.Tensor,
    t: torch.Tensor,
    foot_dist: torch.Tensor,
    q_leg_delta_norm: torch.Tensor,
    fk_err: torch.Tensor,
    valid_mask: torch.Tensor,
    valid_mask_leg_only: torch.Tensor,
    nan_mask: torch.Tensor,
    farthest_idx: int,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib import failed ({exc}); skipping plots.")
        return

    try:
        import numpy as np
    except Exception as exc:
        print(f"[plot] numpy import failed ({exc}); skipping plots.")
        return

    def to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    near_np = to_np(foot_positions)  # (N,3)
    a_np = to_np(foot_a).reshape(1, 3)
    b_np = to_np(foot_b).reshape(1, 3)
    interp_np = to_np(foot_interp)  # (M,3)
    t_np = to_np(t).reshape(-1)

    dist_np = to_np(foot_dist).reshape(-1)
    dtheta_np = to_np(q_leg_delta_norm).reshape(-1)
    fk_err_np = to_np(fk_err).reshape(-1)

    valid_all_np = to_np(valid_mask).astype(bool)
    valid_leg_np = to_np(valid_mask_leg_only).astype(bool)
    nan_np = to_np(nan_mask).astype(bool)

    def set_axes_equal_3d(ax, xyz: np.ndarray) -> None:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        z_min, z_max = float(z.min()), float(z.max())
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range <= 0:
            max_range = 1e-3
        cx, cy, cz = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0
        half = max_range / 2.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)

    # --- Plot 1: 3D foot scatter + interpolation segment ---
    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(near_np[:, 0], near_np[:, 1], near_np[:, 2], s=14, alpha=0.55, label="near FK samples")
    ax.scatter(a_np[:, 0], a_np[:, 1], a_np[:, 2], s=90, c="tab:red", label="center (sample[0])")
    ax.scatter(b_np[:, 0], b_np[:, 1], b_np[:, 2], s=90, c="tab:green", label=f"farthest (idx={farthest_idx})")

    ax.plot(interp_np[:, 0], interp_np[:, 1], interp_np[:, 2], color="black", linewidth=1.6, label="interpolation segment")

    interp_valid = interp_np[valid_all_np]
    interp_invalid = interp_np[(~valid_all_np) & (~nan_np)]
    interp_nan = interp_np[nan_np]

    if len(interp_valid) > 0:
        ax.scatter(interp_valid[:, 0], interp_valid[:, 1], interp_valid[:, 2], s=26, c="tab:blue", label="IK valid (all legs)")
    if len(interp_invalid) > 0:
        ax.scatter(
            interp_invalid[:, 0],
            interp_invalid[:, 1],
            interp_invalid[:, 2],
            s=40,
            c="tab:orange",
            marker="x",
            label="IK invalid (limit)",
        )
    if len(interp_nan) > 0:
        ax.scatter(interp_nan[:, 0], interp_nan[:, 1], interp_nan[:, 2], s=40, c="k", marker="x", label="IK NaN")

    xyz_all = np.concatenate([near_np, interp_np], axis=0)
    set_axes_equal_3d(ax, xyz_all)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Go2 {leg_name} foot positions (near FK + interpolation/IK)")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig_path_3d = os.path.join(out_dir, f"near_fk_ik_3d_{leg_name}.png")
    fig.savefig(fig_path_3d, dpi=dpi)
    plt.close(fig)

    # --- Plot 2: metrics (distance, sensitivity, IK consistency) ---
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.6), sharex=False)

    # (a) FK foot displacement from center
    axes[0].plot(dist_np, color="tab:blue", linewidth=1.3)
    axes[0].scatter([0, farthest_idx], [dist_np[0], dist_np[farthest_idx]], c=["tab:red", "tab:green"], s=50)
    axes[0].set_ylabel("||Δp|| [m]")
    axes[0].set_title(f"{leg_name}: FK foot displacement from center")
    axes[0].grid(True, alpha=0.3)

    # (b) Local sensitivity: ||Δθ|| vs ||Δp||
    axes[1].scatter(dtheta_np, dist_np, s=18, alpha=0.7, color="tab:purple")
    axes[1].set_xlabel("||Δθ|| [rad]")
    axes[1].set_ylabel("||Δp|| [m]")
    axes[1].set_title(f"{leg_name}: local sensitivity (joint delta vs foot delta)")
    axes[1].grid(True, alpha=0.3)

    # (c) IK -> FK consistency along interpolation
    axes[2].plot(t_np, fk_err_np, color="black", linewidth=1.2, label="FK(IK(p)) error")
    axes[2].scatter(t_np[valid_all_np], fk_err_np[valid_all_np], s=24, color="tab:blue", label="valid (all legs)")
    axes[2].scatter(t_np[(~valid_all_np) & (~nan_np)], fk_err_np[(~valid_all_np) & (~nan_np)], s=28, color="tab:orange", marker="x", label="invalid (limit)")
    if nan_np.any():
        axes[2].scatter(t_np[nan_np], fk_err_np[nan_np], s=28, color="k", marker="x", label="NaN")
    if valid_leg_np.sum() != valid_all_np.sum():
        axes[2].scatter(
            t_np[valid_leg_np & (~valid_all_np)],
            fk_err_np[valid_leg_np & (~valid_all_np)],
            s=40,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.5,
            label=f"valid ({leg_name}) only",
        )
    axes[2].set_xlabel("t (interpolation)")
    axes[2].set_ylabel("||FK(IK(p)) - p|| [m]")
    axes[2].set_title(f"{leg_name}: interpolation IK consistency + limit validity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig_path_metrics = os.path.join(out_dir, f"near_fk_ik_metrics_{leg_name}.png")
    fig.savefig(fig_path_metrics, dpi=dpi)
    plt.close(fig)

    print(f"[plot] saved: {fig_path_3d}")
    print(f"[plot] saved: {fig_path_metrics}")


def main() -> None:
    args = parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    leg_idx = resolve_leg_index(args.leg)
    leg_name = LEG_NAMES[leg_idx]

    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # Local import convenience (so running from repo root still works)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    from FK_IK_Solver import Go2Solver  # noqa: E402

    solver = Go2Solver(device=device)

    min_limits, max_limits = get_go2_limits(device=device, dtype=dtype)

    if args.center_joints is not None:
        if len(args.center_joints) != 12:
            raise ValueError("--center-joints requires exactly 12 values.")
        q_center = torch.tensor([args.center_joints], device=device, dtype=dtype)
    elif args.center == "default":
        q_center = default_center_pose(device=device, dtype=dtype)
    else:
        q_center = ((min_limits + max_limits) / 2.0).reshape(1, 12)

    q_center = clamp_jointwise(q_center, min_limits=min_limits, max_limits=max_limits)

    # --- 1) Nearby joint sampling (only the chosen leg is perturbed) ---
    num_samples = int(args.num_samples)
    if num_samples < 2:
        raise ValueError("--num-samples must be >= 2.")

    std_leg = compute_noise_std_for_leg(
        noise_std=args.noise_std,
        noise_scale=args.noise_scale,
        min_limits=min_limits,
        max_limits=max_limits,
        leg_idx=leg_idx,
        device=device,
        dtype=dtype,
    )  # (3,)

    q_samples = q_center.repeat(num_samples, 1)
    noise = torch.randn(num_samples, 3, device=device, dtype=dtype) * std_leg.unsqueeze(0)
    noise[0] = 0.0  # keep sample[0] exactly at the center pose

    q_samples[:, leg_idx] += noise[:, 0]  # hip
    q_samples[:, leg_idx + 4] += noise[:, 1]  # thigh
    q_samples[:, leg_idx + 8] += noise[:, 2]  # calf
    q_samples = clamp_jointwise(q_samples, min_limits=min_limits, max_limits=max_limits)

    # --- 2) FK: measure how much the foot moves for small joint deltas ---
    with torch.no_grad():
        feet_flat = solver.go2_fk_new(q_samples)  # (N, 12)
        feet = feet_flat.view(num_samples, 4, 3)  # (N, 4, 3)

    foot_positions = feet[:, leg_idx, :]  # (N, 3)
    foot_center = foot_positions[0]
    foot_deltas = foot_positions - foot_center
    foot_dist = torch.linalg.norm(foot_deltas, dim=1)

    q_leg = torch.stack([q_samples[:, leg_idx], q_samples[:, leg_idx + 4], q_samples[:, leg_idx + 8]], dim=1)
    q_leg_center = q_leg[0]
    q_leg_delta = q_leg - q_leg_center
    q_leg_delta_norm = torch.linalg.norm(q_leg_delta, dim=1)

    farthest_idx = int(torch.argmax(foot_dist).item())

    print("=== Near Joint Sampling + FK Variation ===")
    print(f"device: {device}")
    print(f"leg: {leg_name} (index {leg_idx})")
    print(f"samples: {num_samples} (sample[0] = center pose)")
    print(f"center joint angles ({leg_name}) [hip, thigh, calf] rad: {q_leg_center.detach().cpu().tolist()}")
    print(f"noise std ({leg_name}) [hip, thigh, calf] rad: {std_leg.detach().cpu().tolist()}")
    print(f"delta-theta norm: mean={q_leg_delta_norm.mean().item():.6f}, max={q_leg_delta_norm.max().item():.6f}")
    print(f"FK foot center {leg_name}: {format_vec3(foot_center)} m")
    print(f"FK foot dist: mean={foot_dist.mean().item():.6f}, max={foot_dist.max().item():.6f} (at idx={farthest_idx})")

    # --- 3) Interpolate foot points and run IK + joint-limit checks ---
    a_idx = 0
    b_idx = farthest_idx
    foot_a = foot_positions[a_idx]
    foot_b = foot_positions[b_idx]

    t = torch.linspace(0.0, 1.0, int(args.interp_steps), device=device, dtype=dtype)
    foot_interp = (1.0 - t).unsqueeze(1) * foot_a.unsqueeze(0) + t.unsqueeze(1) * foot_b.unsqueeze(0)  # (M, 3)

    # Keep other legs fixed at the center FK, only move the chosen leg along the segment.
    feet_targets = feet[0].unsqueeze(0).repeat(foot_interp.size(0), 1, 1)
    feet_targets[:, leg_idx, :] = foot_interp

    with torch.no_grad():
        ik_angles = solver.go2_ik_new(feet_targets)  # (M, 12)

    nan_mask = torch.isnan(ik_angles).any(dim=1)

    eps = float(args.eps)
    all_within_limits = check_joint_limits_jointwise(ik_angles, min_limits=min_limits, max_limits=max_limits, eps=eps)
    valid_mask = (~nan_mask) & all_within_limits

    # For convenience, also compute the chosen-leg-only limit status.
    hip_min, hip_max = min_limits[0, leg_idx], max_limits[0, leg_idx]
    thigh_min, thigh_max = min_limits[1, leg_idx], max_limits[1, leg_idx]
    calf_min, calf_max = min_limits[2, leg_idx], max_limits[2, leg_idx]
    hip = ik_angles[:, leg_idx]
    thigh = ik_angles[:, leg_idx + 4]
    calf = ik_angles[:, leg_idx + 8]
    leg_within_limits = (
        (hip >= hip_min - eps)
        & (hip <= hip_max + eps)
        & (thigh >= thigh_min - eps)
        & (thigh <= thigh_max + eps)
        & (calf >= calf_min - eps)
        & (calf <= calf_max + eps)
    )
    valid_mask_leg_only = (~nan_mask) & leg_within_limits

    # Optional: FK the IK result back to measure end-effector consistency (for the chosen leg).
    with torch.no_grad():
        fk_from_ik = solver.go2_fk_new(ik_angles).view(-1, 4, 3)
    fk_err = torch.linalg.norm(fk_from_ik[:, leg_idx, :] - foot_interp, dim=1)

    valid_fk_err = fk_err[valid_mask]
    mean_fk_err = valid_fk_err.mean().item() if valid_fk_err.numel() > 0 else float("nan")
    max_fk_err = valid_fk_err.max().item() if valid_fk_err.numel() > 0 else float("nan")

    print("\n=== Foot Interpolation (내분) + IK Limit Check ===")
    print(f"segment endpoints: idx {a_idx} -> {b_idx}")
    print(f"  p_a {leg_name}: {format_vec3(foot_a)} m")
    print(f"  p_b {leg_name}: {format_vec3(foot_b)} m")
    print(f"interp steps: {foot_interp.size(0)}")
    print(f"IK NaN: {int(nan_mask.sum().item())}/{foot_interp.size(0)}")
    print(f"IK within joint limits (all 4 legs): {int(valid_mask.sum().item())}/{foot_interp.size(0)}")
    print(f"IK within joint limits ({leg_name} only): {int(valid_mask_leg_only.sum().item())}/{foot_interp.size(0)}")
    print(f"FK( IK(.) ) error ({leg_name}, valid only): mean={mean_fk_err:.6e}, max={max_fk_err:.6e} m")

    if valid_mask.sum().item() != foot_interp.size(0):
        bad_idx = (~valid_mask).nonzero(as_tuple=False).flatten()
        if bad_idx.numel() > 0:
            show = bad_idx[: min(5, bad_idx.numel())].detach().cpu().tolist()
            print(f"first invalid interp indices (up to 5): {show}")

    out_dir = args.out_dir or os.path.join(this_dir, "outputs_near")

    if args.plot or (not args.no_save):
        os.makedirs(out_dir, exist_ok=True)

    if args.plot:
        save_plots(
            out_dir=out_dir,
            leg_name=leg_name,
            foot_positions=foot_positions,
            foot_a=foot_a,
            foot_b=foot_b,
            foot_interp=foot_interp,
            t=t,
            foot_dist=foot_dist,
            q_leg_delta_norm=q_leg_delta_norm,
            fk_err=fk_err,
            valid_mask=valid_mask,
            valid_mask_leg_only=valid_mask_leg_only,
            nan_mask=nan_mask,
            farthest_idx=farthest_idx,
            dpi=int(args.plot_dpi),
        )

    if args.no_save:
        return

    torch.save(q_center.detach().cpu(), os.path.join(out_dir, "center_joint_angles.pt"))
    torch.save(q_samples.detach().cpu(), os.path.join(out_dir, "near_joint_angles.pt"))
    torch.save(feet.detach().cpu(), os.path.join(out_dir, "near_feet_fk.pt"))
    torch.save(feet_targets.detach().cpu(), os.path.join(out_dir, "interp_feet_targets.pt"))
    torch.save(ik_angles.detach().cpu(), os.path.join(out_dir, "interp_ik_angles.pt"))
    torch.save(valid_mask.detach().cpu(), os.path.join(out_dir, "interp_valid_mask.pt"))
    torch.save(fk_err.detach().cpu(), os.path.join(out_dir, "interp_fk_error.pt"))

    meta = {
        "device": device,
        "leg": leg_name,
        "leg_idx": leg_idx,
        "num_samples": num_samples,
        "seed": int(args.seed),
        "center_strategy": args.center,
        "noise_std_rad": std_leg.detach().cpu().tolist(),
        "interp_steps": int(args.interp_steps),
        "endpoint_indices": [int(a_idx), int(b_idx)],
    }
    torch.save(meta, os.path.join(out_dir, "meta.pt"))

    print(f"\nSaved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
