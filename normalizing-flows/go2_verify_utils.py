from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


DEFAULT_IK_SOLVER_DIR = "/home/kdg/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/kinematics"


def import_go2_solver(ik_solver_dir: str):
    ik_solver_dir = (ik_solver_dir or "").strip()
    if not ik_solver_dir:
        raise ValueError("--verify-solver-dir 가 비어 있습니다.")
    if ik_solver_dir not in sys.path:
        sys.path.insert(0, ik_solver_dir)
    try:
        from FK_IK_Solver import Go2Solver  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Go2Solver import 실패: {e}\n"
            f"  verify-solver-dir={ik_solver_dir}\n"
            "  예상 파일: FK_IK_Solver.py (Go2Solver 클래스 포함)"
        ) from e
    return Go2Solver


def _go2_limits(device: torch.device, dtype: torch.dtype):
    """Per-leg limits for Go2 from USD (IsaacLab verify.py와 동일)."""
    hip_min = torch.tensor([-1.0472, -1.0472, -1.0472, -1.0472], device=device, dtype=dtype)
    hip_max = torch.tensor([1.0472, 1.0472, 1.0472, 1.0472], device=device, dtype=dtype)
    thigh_min = torch.tensor([-1.5708, -1.5708, -0.5236, -0.5236], device=device, dtype=dtype)  # FL, FR, RL, RR
    thigh_max = torch.tensor([3.4907, 3.4907, 4.5379, 4.5379], device=device, dtype=dtype)
    calf_min = torch.tensor([-2.7227, -2.7227, -2.7227, -2.7227], device=device, dtype=dtype)
    calf_max = torch.tensor([-0.8378, -0.8378, -0.8378, -0.8378], device=device, dtype=dtype)
    return (hip_min, hip_max), (thigh_min, thigh_max), (calf_min, calf_max)


@dataclass(frozen=True)
class Go2VerifyMetrics:
    num_samples: int
    num_valid: int
    num_invalid: int
    num_nan: int
    pct_valid: float
    pct_invalid: float
    pct_nan: float
    viol_counts: dict[str, int]
    viol_mean: float | None
    viol_max: float | None


class Go2IKVerifier:
    def __init__(self, *, solver_dir: str = DEFAULT_IK_SOLVER_DIR, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        Go2Solver = import_go2_solver(solver_dir)
        self.solver = Go2Solver(device=str(device))

    @torch.no_grad()
    def evaluate(self, foot_pos: torch.Tensor, *, eps: float = 1e-3) -> Go2VerifyMetrics:
        """Evaluate IK validity & joint-limit violations for sampled foot positions.

        foot_pos: (N,12) or (N,4,3), assumed base-frame foot positions.
        """
        if foot_pos.dim() == 2 and foot_pos.size(1) == 12:
            foot_pos_ = foot_pos.view(-1, 4, 3)
        elif foot_pos.dim() == 3 and foot_pos.shape[1:] == (4, 3):
            foot_pos_ = foot_pos
        else:
            raise ValueError(f"Expected foot_pos shape (N,12) or (N,4,3), got {tuple(foot_pos.shape)}")

        foot_pos_ = foot_pos_.to(device=self.device, dtype=torch.float32)
        num_samples = int(foot_pos_.shape[0])

        ik_angles = self.solver.go2_ik_new(foot_pos_)
        nan_mask = torch.isnan(ik_angles).any(dim=1) | torch.isinf(ik_angles).any(dim=1)

        hip_angles = ik_angles[:, 0:4]
        thigh_angles = ik_angles[:, 4:8]
        calf_angles = ik_angles[:, 8:12]

        (hip_min, hip_max), (thigh_min, thigh_max), (calf_min, calf_max) = _go2_limits(
            device=ik_angles.device, dtype=ik_angles.dtype
        )

        valid_hip = (hip_angles >= hip_min - eps) & (hip_angles <= hip_max + eps)
        valid_thigh = (thigh_angles >= thigh_min - eps) & (thigh_angles <= thigh_max + eps)
        valid_calf = (calf_angles >= calf_min - eps) & (calf_angles <= calf_max + eps)

        all_valid_mask = valid_hip.all(dim=1) & valid_thigh.all(dim=1) & valid_calf.all(dim=1) & (~nan_mask)

        num_nan = int(nan_mask.sum().item())
        num_valid = int(all_valid_mask.sum().item())
        num_invalid = int(num_samples - num_valid - num_nan)

        hip_violation = torch.relu(hip_angles - (hip_max + eps)) + torch.relu((hip_min - eps) - hip_angles)
        thigh_violation = torch.relu(thigh_angles - (thigh_max + eps)) + torch.relu((thigh_min - eps) - thigh_angles)
        calf_violation = torch.relu(calf_angles - (calf_max + eps)) + torch.relu((calf_min - eps) - calf_angles)

        viol_counts = {
            "hip": int((hip_violation > 0).sum().item()),
            "thigh": int((thigh_violation > 0).sum().item()),
            "calf": int((calf_violation > 0).sum().item()),
        }

        total_violations = hip_violation + thigh_violation + calf_violation
        nonzero_viols = total_violations[total_violations > 0]
        if nonzero_viols.numel() > 0:
            viol_mean = float(nonzero_viols.mean().item())
            viol_max = float(nonzero_viols.max().item())
        else:
            viol_mean = None
            viol_max = None

        pct_valid = 100.0 * num_valid / max(1, num_samples)
        pct_nan = 100.0 * num_nan / max(1, num_samples)
        pct_invalid = 100.0 * num_invalid / max(1, num_samples)

        return Go2VerifyMetrics(
            num_samples=num_samples,
            num_valid=num_valid,
            num_invalid=num_invalid,
            num_nan=num_nan,
            pct_valid=pct_valid,
            pct_invalid=pct_invalid,
            pct_nan=pct_nan,
            viol_counts=viol_counts,
            viol_mean=viol_mean,
            viol_max=viol_max,
        )


def format_verify_line(epoch: int, metrics: Go2VerifyMetrics, *, prefix: str = "") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"{prefix}{ts} | Epoch {epoch:04d}",
        f"valid {metrics.num_valid}/{metrics.num_samples} ({metrics.pct_valid:.2f}%)",
        f"invalid {metrics.num_invalid}/{metrics.num_samples} ({metrics.pct_invalid:.2f}%)",
        f"nan {metrics.num_nan}/{metrics.num_samples} ({metrics.pct_nan:.2f}%)",
        f"viol_counts hip={metrics.viol_counts.get('hip',0)} thigh={metrics.viol_counts.get('thigh',0)} calf={metrics.viol_counts.get('calf',0)}",
    ]
    if metrics.viol_mean is not None and metrics.viol_max is not None:
        parts.append(f"viol(rad) mean={metrics.viol_mean:.6f} max={metrics.viol_max:.6f}")
    else:
        parts.append("viol(rad) none")
    return " | ".join(parts)


def default_verify_paths(save_path: str, *, suffix: str = "verify") -> dict[str, str]:
    """Derive default paths from a save_path."""
    if not save_path:
        base = Path.cwd() / f"{suffix}"
        return {
            "log": str(base.with_suffix(".txt")),
            "sample_pt": str(base.with_name(f"{base.name}_samples.pt")),
        }
    p = Path(save_path)
    stem = p.stem
    parent = p.parent if p.parent.as_posix() != "" else Path.cwd()
    return {
        "log": str(parent / f"{stem}_{suffix}.txt"),
        "sample_pt": str(parent / f"{stem}_{suffix}_samples.pt"),
    }


def append_line(path: str, line: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

