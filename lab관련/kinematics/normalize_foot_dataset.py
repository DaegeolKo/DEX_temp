import argparse
import math
from pathlib import Path
from typing import Tuple

import torch

from FK_IK_Solver import Go2Solver


def unitree_default_joint_pos(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return Unitree Go2 default joint pose in solver order: [FL, FR, RL, RR] x (hip, thigh, calf)."""
    # Patterns from UNITREE_GO2_CFG.init_state.joint_pos
    fl = (0.1, 0.8, -1.5)
    fr = (-0.1, 0.8, -1.5)
    rl = (0.1, 1.0, -1.5)
    rr = (-0.1, 1.0, -1.5)
    vals = fl + fr + rl + rr
    return torch.tensor(vals, device=device, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize foot positions in foot_all_space_dataset.pt relative to standing pose."
    )
    parser.add_argument(
        "--joint-path",
        type=Path,
        default=Path("source/isaaclab_tasks/isaaclab_tasks/direct/kinematics/foot_all_space_dataset.pt"),
        help="Input torch file with joint angles shaped (N,12).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("source/isaaclab_tasks/isaaclab_tasks/direct/kinematics/foot_all_space_dataset_normalized.pt"),
        help="Output torch file to save normalized foot positions shaped (N,4,3).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=500_000, help="Batch size per FK pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    print(f"[INFO] Loading joint angles from {args.joint_path}")
    joint_angles = torch.load(args.joint_path, map_location="cpu")
    if not isinstance(joint_angles, torch.Tensor) or joint_angles.ndim != 2 or joint_angles.shape[1] != 12:
        raise ValueError("Expected a torch.Tensor of shape (N,12)")
    total = joint_angles.shape[0]
    print(f"[INFO] Dataset size: {total} samples")

    solver = Go2Solver(device=device)
    q0 = unitree_default_joint_pos(device, dtype)
    default_foot = solver.go2_fk_new(q0.unsqueeze(0)).reshape(1, 4, 3)  # (1,4,3)

    out_list = []
    batch = args.batch
    for start in range(0, total, batch):
        end = min(start + batch, total)
        chunk = joint_angles[start:end].to(device=device, dtype=dtype, non_blocking=True)
        foot = solver.go2_fk_new(chunk).reshape(-1, 4, 3)
        foot_rel = (foot - default_foot).cpu()
        out_list.append(foot_rel)
        print(f"[INFO] Processed {end}/{total}")

    foot_all = torch.cat(out_list, dim=0)
    meta = {
        "joint_path": str(args.joint_path),
        "default_joint_pos": q0.cpu(),
        "device_used": str(device),
        "batch": batch,
        "note": "foot positions relative to Unitree standing pose (base frame)",
    }
    torch.save({"foot_pos_rel": foot_all, "meta": meta}, args.out_path)
    print(f"[INFO] Saved normalized foot positions to {args.out_path}")


if __name__ == "__main__":
    main()
