import argparse
import sys
from pathlib import Path

import torch


DEFAULT_SOLVER_DIR = "/home/kdg/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/kinematics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize absolute Go2 foot positions relative to standing pose."
    )
    parser.add_argument(
        "--foot-pos-path",
        type=Path,
        required=True,
        help="Input torch file with absolute foot positions, shape (N,12) or (N,4,3).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Output torch file path.",
    )
    parser.add_argument(
        "--solver-dir",
        type=str,
        default=DEFAULT_SOLVER_DIR,
        help="Directory containing FK_IK_Solver.py",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for FK default-foot computation.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=500_000,
        help="Batch size for normalization pass.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only process first N samples (for quick test).",
    )
    parser.add_argument(
        "--save-shape",
        type=str,
        default="4x3",
        choices=["4x3", "12"],
        help="Saved tensor shape: (N,4,3) or flattened (N,12).",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="tensor",
        choices=["tensor", "dict"],
        help="Output format. 'tensor' is compatible with verify.py and run_nsf_12_stand.py.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional path to save metadata dict separately when --save-format tensor.",
    )
    parser.add_argument(
        "--joint-layout",
        type=str,
        default="grouped",
        choices=["grouped", "interleaved"],
        help="Layout for default standing joint pose.",
    )
    return parser.parse_args()


def import_solver(solver_dir: str):
    if solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
    from FK_IK_Solver import Go2Solver  # type: ignore

    return Go2Solver


def default_joint_pose(device: torch.device, dtype: torch.dtype, layout: str) -> torch.Tensor:
    # Unitree default standing pose values.
    # grouped    : [hip4, thigh4, calf4]
    # interleaved: [FL(hip,thigh,calf), FR(...), RL(...), RR(...)]
    if layout == "grouped":
        vals = [
            0.1,
            -0.1,
            0.1,
            -0.1,  # hip (FL,FR,RL,RR)
            0.8,
            0.8,
            1.0,
            1.0,  # thigh
            -1.5,
            -1.5,
            -1.5,
            -1.5,  # calf
        ]
    else:
        vals = [
            0.1,
            0.8,
            -1.5,  # FL
            -0.1,
            0.8,
            -1.5,  # FR
            0.1,
            1.0,
            -1.5,  # RL
            -0.1,
            1.0,
            -1.5,  # RR
        ]
    return torch.tensor(vals, device=device, dtype=dtype)


def _extract_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("foot_pos_abs", "foot_pos", "foot_positions", "data", "x"):
            value = obj.get(key)
            if torch.is_tensor(value):
                return value
        # If only rel exists, still allow but warn-like error for clarity.
        if torch.is_tensor(obj.get("foot_pos_rel")):
            raise ValueError(
                "Input appears to contain 'foot_pos_rel' (already relative). "
                "Please provide absolute foot positions."
            )
    raise ValueError("Could not find tensor in input file. Expected tensor or dict containing foot positions.")


def load_foot_abs(path: Path, max_samples: int) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    raw = torch.load(path, map_location="cpu")
    x = _extract_tensor(raw).to(dtype=torch.float32)
    if x.dim() == 2 and x.size(1) == 12:
        x = x.view(-1, 4, 3)
    elif x.dim() == 3 and x.shape[1:] == (4, 3):
        pass
    else:
        raise ValueError(f"Expected shape (N,12) or (N,4,3), got {tuple(x.shape)}")
    if max_samples > 0:
        x = x[: max_samples]
    return x


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    print(f"[INFO] loading absolute foot positions from {args.foot_pos_path}")
    foot_abs = load_foot_abs(args.foot_pos_path, args.max_samples)
    total = int(foot_abs.shape[0])
    print(f"[INFO] loaded shape: {tuple(foot_abs.shape)}")

    Go2Solver = import_solver(args.solver_dir)
    solver = Go2Solver(device=str(device))

    q0 = default_joint_pose(device, dtype, args.joint_layout)
    foot_default = solver.go2_fk_new(q0.unsqueeze(0)).reshape(1, 4, 3).cpu().to(dtype=torch.float32)

    out_chunks = []
    batch = int(args.batch)
    if batch <= 0:
        raise ValueError("--batch must be > 0")
    for start in range(0, total, batch):
        end = min(start + batch, total)
        chunk = foot_abs[start:end]
        out_chunks.append(chunk - foot_default)
        if end == total or end % (5 * batch) == 0:
            print(f"[INFO] processed {end}/{total}")
    foot_rel = torch.cat(out_chunks, dim=0)

    # Quick reconstruction sanity check.
    rec = foot_rel + foot_default
    err = (rec - foot_abs).abs()
    print(
        f"[INFO] recon error | mean_abs={err.mean().item():.6e}, "
        f"max_abs={err.max().item():.6e}"
    )

    if args.save_shape == "12":
        tensor_out = foot_rel.view(-1, 12)
    else:
        tensor_out = foot_rel

    meta = {
        "src_path": str(args.foot_pos_path),
        "src_shape": tuple(foot_abs.shape),
        "save_shape": args.save_shape,
        "default_joint_pos": q0.cpu(),
        "default_foot_abs": foot_default.squeeze(0),
        "joint_layout": args.joint_layout,
        "note": "foot_rel = foot_abs - foot_default (standing pose, base frame)",
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_format == "dict":
        torch.save({"foot_pos_rel": tensor_out, "meta": meta}, args.out_path)
    else:
        torch.save(tensor_out, args.out_path)

    if args.meta_path is not None:
        args.meta_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(meta, args.meta_path)
        print(f"[INFO] saved meta: {args.meta_path}")

    print(f"[INFO] saved ({args.save_format}): {args.out_path}")


if __name__ == "__main__":
    main()
