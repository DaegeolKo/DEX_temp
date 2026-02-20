import argparse
import base64
import io
import os
import webbrowser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import plotly.graph_objs as go
    import plotly.offline as pyo

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from FK_IK_Solver import Go2Solver


def parse_args():
    parser = argparse.ArgumentParser(description="IK limit check for Go2 feet positions.")
    parser.add_argument(
        "--foot-pos",
        type=str,
        default=None,
        help="Path to torch tensor with foot positions shaped (N, 4, 3) or (N, 12).",
    )
    parser.add_argument(
        "--joint-path",
        type=str,
        default="joint_angles_100k_ordered.pt",
        help="Path to torch tensor with joint angles shaped (N, 12). Used when --foot-pos is not provided.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save reports and outputs.")
    parser.add_argument("--device", type=str, default=None, help="torch device override (e.g., cuda, cpu).")
    parser.add_argument("--eps", type=float, default=1e-3, help="Tolerance for joint limit check (radians).")
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=60000,
        help="Max total points in scatter plot (after per-leg downsampling).",
    )
    parser.add_argument(
        "--focus-leg",
        type=str,
        default=None,
        choices=["FL", "FR", "RL", "RR"],
        help="If set, generate an extra report focusing on a single leg (FL/FR/RL/RR).",
    )
    parser.add_argument(
        "--focus-kind",
        type=str,
        default=None,
        choices=["valid", "invalid", "nan", "limit"],
        help="Kind of points to plot for --focus-leg. Defaults to 'valid'.",
    )
    parser.add_argument(
        "--focus-only",
        action="store_true",
        default=False,
        help="If set, only generate the focus report (skips the full reports).",
    )
    return parser.parse_args()


def get_limits(device: str):
    """Per-leg limits for Go2 from USD."""
    hip_min = torch.tensor([-1.0472, -1.0472, -1.0472, -1.0472], device=device, dtype=torch.float32)
    hip_max = torch.tensor([1.0472, 1.0472, 1.0472, 1.0472], device=device, dtype=torch.float32)
    thigh_min = torch.tensor([-1.5708, -1.5708, -0.5236, -0.5236], device=device, dtype=torch.float32)  # FL, FR, RL, RR
    thigh_max = torch.tensor([3.4907, 3.4907, 4.5379, 4.5379], device=device, dtype=torch.float32)
    calf_min = torch.tensor([-2.7227, -2.7227, -2.7227, -2.7227], device=device, dtype=torch.float32)
    calf_max = torch.tensor([-0.8378, -0.8378, -0.8378, -0.8378], device=device, dtype=torch.float32)
    return (hip_min, hip_max), (thigh_min, thigh_max), (calf_min, calf_max)


def load_joint_data(path: str, device: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found.")
    data = torch.load(path, map_location=device)
    if data.dim() != 2 or data.size(1) != 12:
        raise ValueError(f"Expected joint data shape (N, 12), got {tuple(data.shape)}")
    return data.to(device)


def load_foot_positions(path: str, device: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found.")
    data = torch.load(path, map_location=device)
    if data.dim() == 3 and data.shape[1:] == (4, 3):
        pass
    elif data.dim() == 2:
        if data.shape == (4, 3):
            data = data.unsqueeze(0)
        elif data.size(1) == 12:
            data = data.view(-1, 4, 3)
        else:
            raise ValueError("Expected foot positions shape (N, 4, 3) or (N, 12) flattened.")
    else:
        raise ValueError("Expected foot positions shape (N, 4, 3) or (N, 12) flattened.")
    return data.to(device)


def main():
    args = parse_args()

    matplotlib.use("Agg")  # render to buffer for HTML
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    EPS = args.eps  # tolerance for limit check (rad)
    out_dir = args.out_dir or os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    if args.focus_only and args.focus_leg is None:
        raise ValueError("--focus-only requires --focus-leg (FL/FR/RL/RR).")

    solver = Go2Solver(device=device)
    (hip_min, hip_max), (thigh_min, thigh_max), (calf_min, calf_max) = get_limits(device)

    if args.foot_pos:
        foot_positions = load_foot_positions(args.foot_pos, device)
        num_samples = foot_positions.size(0)
        foot_positions_flat = foot_positions.view(num_samples, -1)
        joint_angles = None
    else:
        joint_angles = load_joint_data(args.joint_path, device)
        # Quick sanity check: joint angles should roughly lie within USD joint limits.
        # If they don't at all, it's often because a foot-position tensor was passed as --joint-path.
        with torch.no_grad():
            hip_ok = (joint_angles[:, 0:4] >= hip_min - EPS) & (joint_angles[:, 0:4] <= hip_max + EPS)
            thigh_ok = (joint_angles[:, 4:8] >= thigh_min - EPS) & (joint_angles[:, 4:8] <= thigh_max + EPS)
            calf_ok = (joint_angles[:, 8:12] >= calf_min - EPS) & (joint_angles[:, 8:12] <= calf_max + EPS)
            any_calf_ok = calf_ok.any().item()
            if not any_calf_ok:
                print(
                    "[WARN] No calf angles in --joint-path fall within expected Go2 calf limits.\n"
                    "       If your tensor stores foot positions (meters, shape (N,4,3) or (N,12)), use --foot-pos instead."
                )
        num_samples = joint_angles.size(0)
        with torch.no_grad():
            foot_positions_flat = solver.go2_fk_new(joint_angles)
        foot_positions = foot_positions_flat.view(num_samples, 4, 3)

    with torch.no_grad():
        ik_angles = solver.go2_ik_new(foot_positions)

    nan_mask = torch.isnan(ik_angles).any(dim=1)

    hip_angles = ik_angles[:, 0:4]
    thigh_angles = ik_angles[:, 4:8]
    calf_angles = ik_angles[:, 8:12]

    valid_hip = (hip_angles >= hip_min - EPS) & (hip_angles <= hip_max + EPS)
    valid_thigh = (thigh_angles >= thigh_min - EPS) & (thigh_angles <= thigh_max + EPS)
    valid_calf = (calf_angles >= calf_min - EPS) & (calf_angles <= calf_max + EPS)

    all_valid_mask = valid_hip.all(dim=1) & valid_thigh.all(dim=1) & valid_calf.all(dim=1) & (~nan_mask)

    num_nan = nan_mask.sum().item()
    num_valid = all_valid_mask.sum().item()
    num_invalid = num_samples - num_valid - num_nan

    # Save FK outputs and a filtered set that stays within limits
    fk_all_path = os.path.join(out_dir, "fk_positions_from_verify.pt")
    fk_valid_path = os.path.join(out_dir, "fk_positions_valid.pt")
    valid_mask_path = os.path.join(out_dir, "fk_valid_mask.pt")
    valid_joint_path = os.path.join(out_dir, "joint_angles_valid.pt") if joint_angles is not None else None

    foot_positions_cpu = foot_positions_flat.detach().cpu()
    joint_angles_cpu = joint_angles.detach().cpu() if joint_angles is not None else None
    valid_mask_cpu = all_valid_mask.detach().cpu()

    torch.save(foot_positions_cpu, fk_all_path)
    torch.save(valid_mask_cpu, valid_mask_path)

    valid_fk_cpu = foot_positions_cpu[valid_mask_cpu]
    torch.save(valid_fk_cpu, fk_valid_path)

    if joint_angles_cpu is not None:
        valid_joint_cpu = joint_angles_cpu[valid_mask_cpu]
        torch.save(valid_joint_cpu, valid_joint_path)

    print(f"Saved foot positions for all {num_samples} samples -> {fk_all_path}")
    print(f"Saved valid foot positions ({num_valid}) -> {fk_valid_path}")
    print(f"Saved validity mask -> {valid_mask_path}")
    if joint_angles_cpu is not None:
        print(f"Saved valid joint angles -> {valid_joint_path}")

    # Violation magnitudes (radians) per joint type
    hip_violation = torch.relu(hip_angles - (hip_max + EPS)) + torch.relu((hip_min - EPS) - hip_angles)
    thigh_violation = torch.relu(thigh_angles - (thigh_max + EPS)) + torch.relu((thigh_min - EPS) - thigh_angles)
    calf_violation = torch.relu(calf_angles - (calf_max + EPS)) + torch.relu((calf_min - EPS) - calf_angles)

    violation_counts = {
        "hip": (hip_violation > 0).sum().item(),
        "thigh": (thigh_violation > 0).sum().item(),
        "calf": (calf_violation > 0).sum().item(),
    }

    total_violations = hip_violation + thigh_violation + calf_violation
    nonzero_viols = total_violations[total_violations > 0]

    print("=== IK Joint Limit Check ===")
    print(f"Samples: {num_samples}")
    print(f"Valid (all joints within limits): {num_valid} ({num_valid/num_samples*100:.2f}%)")
    print(f"Invalid (limit violations): {num_invalid} ({num_invalid/num_samples*100:.2f}%)")
    print(f"NaN IK results: {num_nan}")
    print(f"Violation counts per joint type: {violation_counts}")
    if nonzero_viols.numel() > 0:
        print(
            f"Violation magnitude (rad): mean={nonzero_viols.mean().item():.4f}, "
            f"max={nonzero_viols.max().item():.4f}"
        )
    else:
        print("No violations detected.")

    # Debug: show a few invalid samples (if any)
    invalid_mask = (~all_valid_mask) & (~nan_mask)
    invalid_indices = invalid_mask.nonzero(as_tuple=False).flatten()
    if invalid_indices.numel() > 0:
        leg_names = ["FL", "FR", "RL", "RR"]
        print(f"\nFirst {min(5, invalid_indices.numel())} invalid samples (angles/viols in deg, viols also rad):")
        for idx in invalid_indices[:5]:
            angles = ik_angles[idx]
            viols = torch.stack([hip_violation[idx], thigh_violation[idx], calf_violation[idx]], dim=0)
            print(f"- sample {idx.item()}:")
            for leg in range(4):
                vals_rad = angles[[leg, leg + 4, leg + 8]]
                vals_deg = (vals_rad * 180.0 / torch.pi).cpu().numpy()
                v_rad = viols[:, leg]
                v_deg = (v_rad * 180.0 / torch.pi).cpu().numpy()
                print(
                    f"   {leg_names[leg]}: "
                    f"hip {vals_deg[0]:.3f}° ({v_deg[0]:.3f}° viol, {v_rad[0]:.4f} rad), "
                    f"thigh {vals_deg[1]:.3f}° ({v_deg[1]:.3f}° viol, {v_rad[1]:.4f} rad), "
                    f"calf {vals_deg[2]:.3f}° ({v_deg[2]:.3f}° viol, {v_rad[2]:.4f} rad)"
                )

    # 3D scatter of FK positions: valid vs invalid, leg-separated
    leg_names = ["FL", "FR", "RL", "RR"]
    leg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 4 distinct colors

    MAX_SCATTER_POINTS = args.max_scatter_points  # total budget for all legs/status

    def downsample(arr, max_points):
        if len(arr) <= max_points:
            return arr
        idx = np.random.choice(len(arr), size=max_points, replace=False)
        return arr[idx]

    # Per-leg failure masks (to visualize only the *problematic* foot tips)
    # Reshape IK back to (N, 4, 3) => (leg, [hip, thigh, calf]) so we can locate which leg failed.
    ik_leg_grouped = ik_angles.reshape(num_samples, 3, 4).transpose(1, 2).contiguous()  # (N, 4, 3)
    leg_nan_mask = torch.isnan(ik_leg_grouped).any(dim=2)  # (N, 4)

    per_leg_limit_ok = valid_hip & valid_thigh & valid_calf  # (N, 4) comparisons w/ NaN are False
    leg_limit_violation_mask = (~leg_nan_mask) & (~per_leg_limit_ok)  # (N, 4)
    leg_problem_mask = leg_nan_mask | leg_limit_violation_mask  # (N, 4)

    # Collect per-leg, per-status positions
    valid_positions_by_leg = []
    invalid_positions_by_leg = []
    for leg in range(4):
        v = foot_positions[all_valid_mask][:, leg, :].cpu().numpy()
        inv = foot_positions[invalid_mask][:, leg, :].cpu().numpy()
        valid_positions_by_leg.append(v)
        invalid_positions_by_leg.append(inv)

    # Downsample per group to keep total points under budget
    per_group_budget = max(1, MAX_SCATTER_POINTS // (4 * 2))
    valid_positions_by_leg = [downsample(v, per_group_budget) for v in valid_positions_by_leg]
    invalid_positions_by_leg = [downsample(v, per_group_budget) for v in invalid_positions_by_leg]

    report_path = os.path.join(out_dir, "verify_report.html")
    report_problems_path = os.path.join(out_dir, "verify_report_problems_only.html")

    # Collect per-leg *problem-only* foot positions (only the legs that actually failed)
    nan_positions_by_leg = []
    limit_violation_positions_by_leg = []
    for leg in range(4):
        nan_pos = foot_positions[leg_nan_mask[:, leg], leg, :].cpu().numpy()
        lim_pos = foot_positions[leg_limit_violation_mask[:, leg], leg, :].cpu().numpy()
        nan_positions_by_leg.append(nan_pos)
        limit_violation_positions_by_leg.append(lim_pos)

    # Downsample per group to keep total points under budget (4 legs * 2 problem types)
    per_problem_group_budget = max(1, MAX_SCATTER_POINTS // (4 * 2))
    nan_positions_by_leg = [downsample(v, per_problem_group_budget) for v in nan_positions_by_leg]
    limit_violation_positions_by_leg = [downsample(v, per_problem_group_budget) for v in limit_violation_positions_by_leg]

    # Optional: Focus report for a single leg and a single category, using the full point budget.
    focus_leg = args.focus_leg
    focus_kind = args.focus_kind or "valid"
    focus_report_path = None
    if focus_leg is not None:
        leg_to_idx = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
        leg_idx = leg_to_idx[focus_leg]
        if focus_kind == "valid":
            focus_pos = foot_positions[all_valid_mask][:, leg_idx, :].cpu().numpy()
            focus_name = f"{focus_leg} valid"
            focus_marker = dict(size=2, color=leg_colors[leg_idx], opacity=0.6)
        elif focus_kind == "invalid":
            focus_pos = foot_positions[invalid_mask][:, leg_idx, :].cpu().numpy()
            focus_name = f"{focus_leg} invalid"
            focus_marker = dict(size=3, color=leg_colors[leg_idx], opacity=0.9, symbol="x")
        elif focus_kind == "nan":
            focus_pos = foot_positions[leg_nan_mask[:, leg_idx], leg_idx, :].cpu().numpy()
            focus_name = f"{focus_leg} IK-NaN"
            focus_marker = dict(size=3, color=leg_colors[leg_idx], opacity=0.9, symbol="cross")
        elif focus_kind == "limit":
            focus_pos = foot_positions[leg_limit_violation_mask[:, leg_idx], leg_idx, :].cpu().numpy()
            focus_name = f"{focus_leg} limit-violation"
            focus_marker = dict(size=3, color=leg_colors[leg_idx], opacity=0.9, symbol="x")
        else:
            raise ValueError(f"Unsupported focus kind: {focus_kind}")

        focus_pos = downsample(focus_pos, MAX_SCATTER_POINTS)
        focus_report_path = os.path.join(out_dir, f"verify_report_focus_{focus_leg}_{focus_kind}.html")

    if _PLOTLY_AVAILABLE:
        if focus_report_path is not None:
            focus_trace = go.Scatter3d(
                x=focus_pos[:, 0],
                y=focus_pos[:, 1],
                z=focus_pos[:, 2],
                mode="markers",
                marker=focus_marker,
                name=focus_name,
            )
            focus_layout = go.Layout(
                title=f"Focus: {focus_name} (showing up to {MAX_SCATTER_POINTS} points) — samples {num_samples}",
                scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
                legend=dict(x=0, y=1),
                margin=dict(l=0, r=0, b=0, t=40),
            )
            fig_focus = go.Figure(data=[focus_trace], layout=focus_layout)
            pyo.plot(fig_focus, filename=focus_report_path, auto_open=True, include_plotlyjs=True)

        if args.focus_only and focus_report_path is not None:
            return

        traces = []
        for leg in range(4):
            v = valid_positions_by_leg[leg]
            inv = invalid_positions_by_leg[leg]
            if len(v) > 0:
                traces.append(
                    go.Scatter3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        mode="markers",
                        marker=dict(size=2, color=leg_colors[leg], opacity=0.35),
                        name=f"{leg_names[leg]} valid",
                    )
                )
            if len(inv) > 0:
                traces.append(
                    go.Scatter3d(
                        x=inv[:, 0],
                        y=inv[:, 1],
                        z=inv[:, 2],
                        mode="markers",
                        marker=dict(size=3, color=leg_colors[leg], opacity=0.9, symbol="x"),
                        name=f"{leg_names[leg]} invalid",
                    )
                )
        layout = go.Layout(
            title=f"Foot Positions (valid vs invalid IK) — samples {num_samples}, valid {num_valid}, invalid {num_invalid}, NaN {num_nan}",
            scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
            legend=dict(x=0, y=1),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig = go.Figure(data=traces, layout=layout)
        pyo.plot(fig, filename=report_path, auto_open=True, include_plotlyjs=True)

        # --- Problems-only report: plot only the legs that are actually problematic (NaN or limit violation) ---
        problem_traces = []
        for leg in range(4):
            nan_pos = nan_positions_by_leg[leg]
            lim_pos = limit_violation_positions_by_leg[leg]
            if len(lim_pos) > 0:
                problem_traces.append(
                    go.Scatter3d(
                        x=lim_pos[:, 0],
                        y=lim_pos[:, 1],
                        z=lim_pos[:, 2],
                        mode="markers",
                        marker=dict(size=4, color=leg_colors[leg], opacity=0.9, symbol="x"),
                        name=f"{leg_names[leg]} limit-violation",
                    )
                )
            if len(nan_pos) > 0:
                problem_traces.append(
                    go.Scatter3d(
                        x=nan_pos[:, 0],
                        y=nan_pos[:, 1],
                        z=nan_pos[:, 2],
                        mode="markers",
                        marker=dict(size=5, color=leg_colors[leg], opacity=0.95, symbol="cross"),
                        name=f"{leg_names[leg]} IK-NaN (unreachable)",
                    )
                )
        problem_layout = go.Layout(
            title=(
                "Problematic Foot Tips Only "
                f"(NaN legs: {int(leg_nan_mask.sum().item())}, limit-violating legs: {int(leg_limit_violation_mask.sum().item())}) "
                f"— samples {num_samples}"
            ),
            scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
            legend=dict(x=0, y=1),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig_prob = go.Figure(data=problem_traces, layout=problem_layout)
        pyo.plot(fig_prob, filename=report_problems_path, auto_open=True, include_plotlyjs=True)
    else:
        # Fallback: static PNG embedded in HTML
        if focus_report_path is not None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                focus_pos[:, 0],
                focus_pos[:, 1],
                focus_pos[:, 2],
                c=leg_colors[leg_idx],
                s=4,
                alpha=0.8,
                marker="o" if focus_kind == "valid" else "x",
                label=focus_name,
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.set_title(f"Focus: {focus_name}")
            ax.legend(loc="best")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            scatter_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            with open(focus_report_path, "w") as f:
                f.write("<html><head><title>IK Focus Plot</title></head><body>")
                f.write(f"<h2>Focus: {focus_name}</h2>")
                f.write(f"<p>Samples: {num_samples}, plotted up to: {MAX_SCATTER_POINTS}</p>")
                f.write(f"<img src='data:image/png;base64,{scatter_b64}' style='max-width: 900px;'/>")
                f.write("</body></html>")
            webbrowser.open(f"file://{focus_report_path}")

        if args.focus_only and focus_report_path is not None:
            return

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for leg in range(4):
            v = valid_positions_by_leg[leg]
            inv = invalid_positions_by_leg[leg]
            if len(v) > 0:
                ax.scatter(v[:, 0], v[:, 1], v[:, 2],
                           c=leg_colors[leg], s=2, alpha=0.35, label=f"{leg_names[leg]} valid")
            if len(inv) > 0:
                ax.scatter(inv[:, 0], inv[:, 1], inv[:, 2],
                           c=leg_colors[leg], s=6, alpha=0.9, marker="x", label=f"{leg_names[leg]} invalid")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Foot Positions (valid vs invalid IK)")
        ax.legend(loc="best")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        scatter_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        with open(report_path, "w") as f:
            f.write("<html><head><title>IK Limit Check</title></head><body>")
            f.write(f"<h2>IK Joint Limit Check</h2><p>Samples: {num_samples}, Valid: {num_valid}, Invalid: {num_invalid}, NaN: {num_nan}</p>")
            f.write("<h3>Foot Positions (valid vs invalid IK)</h3>")
            f.write(f"<img src='data:image/png;base64,{scatter_b64}' style='max-width: 900px;'/>")
            f.write("</body></html>")
        webbrowser.open(f"file://{report_path}")

        # Problems-only static plot
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for leg in range(4):
            lim_pos = limit_violation_positions_by_leg[leg]
            nan_pos = nan_positions_by_leg[leg]
            if len(lim_pos) > 0:
                ax.scatter(
                    lim_pos[:, 0],
                    lim_pos[:, 1],
                    lim_pos[:, 2],
                    c=leg_colors[leg],
                    s=10,
                    alpha=0.9,
                    marker="x",
                    label=f"{leg_names[leg]} limit-violation",
                )
            if len(nan_pos) > 0:
                ax.scatter(
                    nan_pos[:, 0],
                    nan_pos[:, 1],
                    nan_pos[:, 2],
                    c=leg_colors[leg],
                    s=14,
                    alpha=0.95,
                    marker="+",
                    label=f"{leg_names[leg]} IK-NaN",
                )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Problematic Foot Tips Only (NaN / limit-violation)")
        ax.legend(loc="best")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        scatter_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        with open(report_problems_path, "w") as f:
            f.write("<html><head><title>IK Problem Foot Tips</title></head><body>")
            f.write("<h2>Problematic Foot Tips Only</h2>")
            f.write(
                f"<p>Samples: {num_samples}, "
                f"Leg-NaN (unreachable): {int(leg_nan_mask.sum().item())}, "
                f"Leg-limit-violation: {int(leg_limit_violation_mask.sum().item())}</p>"
            )
            f.write("<h3>Only the failed legs are plotted</h3>")
            f.write(f"<img src='data:image/png;base64,{scatter_b64}' style='max-width: 900px;'/>")
            f.write("</body></html>")
        webbrowser.open(f"file://{report_problems_path}")


if __name__ == "__main__":
    main()
