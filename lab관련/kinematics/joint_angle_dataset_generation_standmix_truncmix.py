import argparse

import math
import numpy as np
import torch


def _effective_limits(min_limits: torch.Tensor, max_limits: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrink joint limits conservatively around the center by `scale` (e.g. 0.98)."""
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    margin = (1.0 - scale) * (max_limits - min_limits) / 2.0
    return min_limits + margin, max_limits - margin


def _sample_truncated_normal(
    n: int,
    q0_jointwise: torch.Tensor,
    std_jointwise: torch.Tensor,
    min_eff: torch.Tensor,
    max_eff: torch.Tensor,
    *,
    device: str,
    chunk: int,
    oversample: float,
    max_tries: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Sample q ~ N(q0, std) with rejection outside [min_eff, max_eff] (per-joint bounds).
    Shapes:
      q0_jointwise/std_jointwise/min_eff/max_eff: (3,4)
    Returns:
      samples: (n,3,4)
      stats: acceptance info
    """
    if n <= 0:
        return torch.empty((0, 3, 4), device=device, dtype=torch.float32), {"draws": 0.0, "accept": 0.0, "acc_rate": 1.0}

    out = torch.empty((n, 3, 4), device=device, dtype=torch.float32)
    filled = 0
    draws_total = 0
    accepted_total = 0

    while filled < n:
        take = min(chunk, n - filled)
        buf = []
        need = take
        tries = 0

        while need > 0:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    f"truncated-normal sampling failed to fill a chunk after {max_tries} tries "
                    f"(need={need}, chunk={take}). Try reducing std or increasing scale/limits."
                )

            n_draw = int(math.ceil(need * oversample))
            eps = torch.randn((n_draw, 3, 4), device=device, dtype=torch.float32)
            q = q0_jointwise.unsqueeze(0) + eps * std_jointwise.unsqueeze(0)
            ok = ((q >= min_eff) & (q <= max_eff)).all(dim=(1, 2))
            acc = q[ok]

            draws_total += n_draw
            accepted_total += int(acc.shape[0])

            if acc.numel() == 0:
                oversample = min(oversample * 1.5, 20.0)
                continue

            buf.append(acc)
            need -= int(acc.shape[0])

        q_take = torch.cat(buf, dim=0)[:take]
        out[filled : filled + take] = q_take
        filled += take

    acc_rate = float(n) / float(draws_total)
    return out, {"draws": float(draws_total), "accept": float(n), "acc_rate": acc_rate}


def _sample_mixture_trunc(
    n: int,
    *,
    q0_jointwise: torch.Tensor,
    std_small: torch.Tensor,
    std_large: torch.Tensor,
    p_small: float,
    min_eff: torch.Tensor,
    max_eff: torch.Tensor,
    device: str,
    chunk: int,
    oversample: float,
    max_tries: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Two-component mixture around q0:
      with prob p_small: N(q0, std_small) truncated
      else:            N(q0, std_large) truncated
    Returns samples (n,3,4).
    """
    if not (0.0 <= p_small <= 1.0):
        raise ValueError(f"p_small must be in [0,1], got {p_small}")

    # We generate each component separately for the full n to keep the implementation simple and stable,
    # then interleave by concatenation (order doesn't matter for training).
    n_small = int(round(n * p_small))
    n_large = n - n_small

    small, st_small = _sample_truncated_normal(
        n_small,
        q0_jointwise,
        std_small,
        min_eff,
        max_eff,
        device=device,
        chunk=chunk,
        oversample=oversample,
        max_tries=max_tries,
    )
    large, st_large = _sample_truncated_normal(
        n_large,
        q0_jointwise,
        std_large,
        min_eff,
        max_eff,
        device=device,
        chunk=chunk,
        oversample=oversample,
        max_tries=max_tries,
    )

    out = torch.cat([small, large], dim=0)
    stats = {
        "n": float(n),
        "n_small": float(n_small),
        "n_large": float(n_large),
        "acc_small": float(st_small["acc_rate"]),
        "acc_large": float(st_large["acc_rate"]),
        "draws_total": float(st_small["draws"] + st_large["draws"]),
        "acc_rate_total": float(n) / float(st_small["draws"] + st_large["draws"]),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Go2 joint-angle dataset with a stand-centered truncated-Gaussian mixture.\n"
        "This avoids clamp-induced bias by rejection sampling (truncated normal).\n"
        "Output joint order: [hip(4), thigh(4), calf(4)] matching FK_IK_Solver.Go2Solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_samples", type=int, default=10_000_000)
    parser.add_argument("--p_small", type=float, default=0.97, help="Mixture prob for the small-std component.")
    parser.add_argument("--scale", type=float, default=0.98, help="Conservative scaling factor for joint limits.")

    parser.add_argument("--sigma_ratio_hip_small", type=float, default=0.10)
    parser.add_argument("--sigma_ratio_thigh_small", type=float, default=0.12)
    parser.add_argument("--sigma_ratio_calf_small", type=float, default=0.10)
    parser.add_argument("--sigma_ratio_hip_large", type=float, default=0.20)
    parser.add_argument("--sigma_ratio_thigh_large", type=float, default=0.25)
    parser.add_argument("--sigma_ratio_calf_large", type=float, default=0.20)

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
    parser.add_argument("--chunk", type=int, default=500_000, help="Chunk size for truncated sampling.")
    parser.add_argument("--oversample", type=float, default=1.3, help="Initial oversampling factor for rejection.")
    parser.add_argument("--max_tries", type=int, default=200, help="Max rejection iterations per chunk.")

    parser.add_argument("--output_pt", type=str, default="joint_angles_standmix_truncmix_100man_ordered.pt")
    parser.add_argument("--output_txt", type=str, default="", help="If set, also save a .txt file.")
    args = parser.parse_args()

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
        q0_jointwise = torch.tensor(args.q0, device=device, dtype=torch.float32).view(3, 4)

    std_small = torch.stack(
        [
            range_eff[0] * float(args.sigma_ratio_hip_small),
            range_eff[1] * float(args.sigma_ratio_thigh_small),
            range_eff[2] * float(args.sigma_ratio_calf_small),
        ],
        dim=0,
    ).clamp_min(1e-6)
    std_large = torch.stack(
        [
            range_eff[0] * float(args.sigma_ratio_hip_large),
            range_eff[1] * float(args.sigma_ratio_thigh_large),
            range_eff[2] * float(args.sigma_ratio_calf_large),
        ],
        dim=0,
    ).clamp_min(1e-6)

    joint_angles, st = _sample_mixture_trunc(
        int(args.num_samples),
        q0_jointwise=q0_jointwise,
        std_small=std_small,
        std_large=std_large,
        p_small=float(args.p_small),
        min_eff=min_eff,
        max_eff=max_eff,
        device=device,
        chunk=int(args.chunk),
        oversample=float(args.oversample),
        max_tries=int(args.max_tries),
    )

    joint_angles_data = joint_angles.view(int(args.num_samples), 12).cpu()
    torch.save(joint_angles_data, args.output_pt)
    if args.output_txt:
        np.savetxt(args.output_txt, joint_angles_data.numpy(), fmt="%.6f")

    print("[OK] Generated joint angles (truncated Gaussian mixture, joint-wise order).")
    print(f"  - num_samples: {int(args.num_samples)}")
    print(f"  - p_small: {float(args.p_small):.3f} (small={int(st['n_small'])}, large={int(st['n_large'])})")
    print(f"  - scale: {float(args.scale):.3f}")
    print(f"  - acceptance: total={st['acc_rate_total']:.3f} small={st['acc_small']:.3f} large={st['acc_large']:.3f}")
    if st["acc_large"] < 0.15:
        print(
            "  [WARN] large component acceptance is low; generation can be slow.\n"
            "         Consider increasing --p_small or reducing --sigma_ratio_*_large."
        )
    print(f"  - output_pt: {args.output_pt}")
    if args.output_txt:
        print(f"  - output_txt: {args.output_txt}")


if __name__ == "__main__":
    main()
