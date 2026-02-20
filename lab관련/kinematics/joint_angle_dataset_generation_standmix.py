import argparse

import numpy as np
import torch


def _effective_limits(min_limits: torch.Tensor, max_limits: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrink joint limits conservatively around the center by `scale` (e.g. 0.98)."""
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    margin = (1.0 - scale) * (max_limits - min_limits) / 2.0
    return min_limits + margin, max_limits - margin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Go2 joint-angle dataset with a stand-centered mixture distribution.\n"
        "Output joint order: [hip(4), thigh(4), calf(4)] matching FK_IK_Solver.Go2Solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_samples", type=int, default=10_000_000)
    parser.add_argument("--p_stand", type=float, default=0.7, help="Mixture prob for stand-centered samples.")
    parser.add_argument("--scale", type=float, default=0.98, help="Conservative scaling factor for joint limits.")
    parser.add_argument("--sigma_ratio_hip", type=float, default=0.15)
    parser.add_argument("--sigma_ratio_thigh", type=float, default=0.20)
    parser.add_argument("--sigma_ratio_calf", type=float, default=0.15)
    parser.add_argument(
        "--q0",
        type=float,
        nargs=12,
        default=None,
        help="Stand pose mean in joint-wise order: hip(4) thigh(4) calf(4). If omitted, uses Unitree default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for sampling. Saved tensor is always moved to CPU.",
    )
    parser.add_argument("--output_pt", type=str, default="joint_angles_standmix_100man_ordered.pt")
    parser.add_argument("--output_txt", type=str, default="", help="If set, also save a .txt file.")
    args = parser.parse_args()

    if not (0.0 <= args.p_stand <= 1.0):
        raise ValueError(f"p_stand must be in [0,1], got {args.p_stand}")
    if args.num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {args.num_samples}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Go2 joint limits from USD (per-leg)
    hip_min = torch.tensor([-1.0472, -1.0472, -1.0472, -1.0472], device=device, dtype=torch.float32)
    hip_max = torch.tensor([1.0472, 1.0472, 1.0472, 1.0472], device=device, dtype=torch.float32)
    thigh_min = torch.tensor([-1.5708, -1.5708, -0.5236, -0.5236], device=device, dtype=torch.float32)  # FL, FR, RL, RR
    thigh_max = torch.tensor([3.4907, 3.4907, 4.5379, 4.5379], device=device, dtype=torch.float32)
    calf_min = torch.tensor([-2.7227, -2.7227, -2.7227, -2.7227], device=device, dtype=torch.float32)
    calf_max = torch.tensor([-0.8378, -0.8378, -0.8378, -0.8378], device=device, dtype=torch.float32)

    min_limits = torch.stack([hip_min, thigh_min, calf_min], dim=0)  # (3, 4)
    max_limits = torch.stack([hip_max, thigh_max, calf_max], dim=0)  # (3, 4)
    min_eff, max_eff = _effective_limits(min_limits, max_limits, scale=float(args.scale))
    range_eff = max_eff - min_eff

    # Unitree default stand pose (used in UNITREE_GO2_CFG.init_state.joint_pos).
    if args.q0 is None:
        q0_jointwise = torch.tensor(
            [
                [0.1, -0.1, 0.1, -0.1],  # hip   (FL, FR, RL, RR)
                [0.8, 0.8, 1.0, 1.0],  # thigh
                [-1.5, -1.5, -1.5, -1.5],  # calf
            ],
            device=device,
            dtype=torch.float32,
        )
    else:
        q0 = torch.tensor(args.q0, device=device, dtype=torch.float32).view(3, 4)
        q0_jointwise = q0

    # Std as a fraction of joint range (per-leg, per-joint).
    std_jointwise = torch.stack(
        [
            range_eff[0] * float(args.sigma_ratio_hip),
            range_eff[1] * float(args.sigma_ratio_thigh),
            range_eff[2] * float(args.sigma_ratio_calf),
        ],
        dim=0,
    ).clamp_min(1e-6)  # (3, 4)

    # Mixture sampling.
    n = int(args.num_samples)
    mask = torch.rand(n, device=device) < float(args.p_stand)
    n_stand = int(mask.sum().item())
    n_uniform = n - n_stand

    joint_angles = torch.empty(n, 3, 4, device=device, dtype=torch.float32)
    if n_stand > 0:
        noise = torch.randn(n_stand, 3, 4, device=device, dtype=torch.float32) * std_jointwise
        stand_samples = q0_jointwise.unsqueeze(0) + noise
        stand_samples = torch.clamp(stand_samples, min_eff, max_eff)
        joint_angles[mask] = stand_samples
    if n_uniform > 0:
        uniform_raw = torch.rand(n_uniform, 3, 4, device=device, dtype=torch.float32)
        uniform_samples = uniform_raw * range_eff + min_eff
        joint_angles[~mask] = uniform_samples

    joint_angles_data = joint_angles.view(n, 12).cpu()
    torch.save(joint_angles_data, args.output_pt)
    if args.output_txt:
        np.savetxt(args.output_txt, joint_angles_data.numpy(), fmt="%.6f")

    print("[OK] Generated joint angles (joint-wise order).")
    print(f"  - num_samples: {n}")
    print(f"  - p_stand: {float(args.p_stand):.3f} (stand={n_stand}, uniform={n_uniform})")
    print(f"  - scale: {float(args.scale):.3f}")
    print(f"  - output_pt: {args.output_pt}")
    if args.output_txt:
        print(f"  - output_txt: {args.output_txt}")


if __name__ == "__main__":
    main()

