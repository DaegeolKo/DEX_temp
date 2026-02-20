import argparse
from pathlib import Path
import math
import sys
import re
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data

import normflows as nf
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from go2_verify_utils import DEFAULT_IK_SOLVER_DIR, Go2IKVerifier, append_line, default_verify_paths, format_verify_line
from coverage_utils import FootWorkspaceCoverage, format_coverage_line


class FixedPermutation1d(nn.Module):
    """Fixed (invertible) permutation over the feature dimension for (B, D) tensors."""

    def __init__(self, perm: torch.Tensor):
        super().__init__()
        if perm.dim() != 1:
            raise ValueError(f"perm must be 1D, got shape={tuple(perm.shape)}")
        perm = perm.to(dtype=torch.long)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x: torch.Tensor):
        x = x.index_select(1, self.perm)
        log_det = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        return x, log_det

    def inverse(self, x: torch.Tensor):
        x = x.index_select(1, self.inv_perm)
        log_det = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        return x, log_det


class VectorInvertible1x1Conv(nn.Module):
    """Adapter: apply nf.flows.Invertible1x1Conv to vector data (B, D) via (B, D, 1, 1)."""

    def __init__(self, dim: int, *, use_lu: bool = True):
        super().__init__()
        self.dim = int(dim)
        self._conv = nf.flows.Invertible1x1Conv(self.dim, use_lu=bool(use_lu))

    def forward(self, x: torch.Tensor):
        if x.dim() != 2 or x.size(1) != self.dim:
            raise ValueError(f"Expected (B,{self.dim}) input, got {tuple(x.shape)}")
        x4 = x[:, :, None, None]
        y4, log_det = self._conv(x4)
        y = y4[:, :, 0, 0]
        return y, log_det

    def inverse(self, y: torch.Tensor):
        if y.dim() != 2 or y.size(1) != self.dim:
            raise ValueError(f"Expected (B,{self.dim}) input, got {tuple(y.shape)}")
        y4 = y[:, :, None, None]
        x4, log_det = self._conv.inverse(y4)
        x = x4[:, :, 0, 0]
        return x, log_det


def build_model(
    dim: int,
    num_layers: int,
    hidden: int,
    num_blocks: int,
    num_bins: int,
    tail_bound: float,
    *,
    trainable_base: bool = True,
    actnorm: bool = False,
    mixing: str = "none",
    mixing_use_lu: bool = True,
    mixing_seed: int = 0,
):
    base = nf.distributions.base.DiagGaussian(dim, trainable=trainable_base)
    flows = []
    if bool(actnorm):
        flows.append(nf.flows.ActNorm((dim,)))
    mixing = (mixing or "").strip().lower()
    if mixing not in {"none", "permute", "inv_affine", "inv_1x1conv"}:
        raise ValueError("mixing must be one of: none|permute|inv_affine|inv_1x1conv")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(mixing_seed))
    for i in range(num_layers):
        flows.append(
            nf.flows.CoupledRationalQuadraticSpline(
                num_input_channels=dim,
                num_blocks=num_blocks,
                num_hidden_channels=hidden,
                num_bins=num_bins,
                tail_bound=tail_bound,
                activation=nn.ReLU,
                reverse_mask=bool(i % 2),
                init_identity=True,
            )
        )
        if mixing != "none" and i != num_layers - 1:
            if mixing == "permute":
                flows.append(FixedPermutation1d(torch.randperm(dim, generator=gen)))
            elif mixing == "inv_affine":
                flows.append(nf.flows.InvertibleAffine(dim, use_lu=bool(mixing_use_lu)))
            else:
                flows.append(VectorInvertible1x1Conv(dim, use_lu=bool(mixing_use_lu)))
    return nf.NormalizingFlow(base, flows)


def load_tensor(data_path: Path) -> torch.Tensor:
    ext = data_path.suffix.lower()
    if ext in {".pt", ".pth"}:
        x = torch.load(data_path)
    elif ext == ".npy":
        x = torch.from_numpy(np.load(data_path))
    elif ext == ".npz":
        npz = np.load(data_path)
        if "arr_0" in npz.files:
            key = "arr_0"
        elif len(npz.files) == 1:
            key = npz.files[0]
        else:
            raise ValueError(f"NPZ 파일에 하나의 배열만 있어야 합니다: {npz.files}")
        x = torch.from_numpy(npz[key])
    elif ext in {".csv", ".txt"}:
        delimiter = "," if ext == ".csv" else None
        x = torch.from_numpy(np.loadtxt(data_path, delimiter=delimiter))
    else:
        raise ValueError(f"지원하지 않는 데이터 확장자: {ext}")

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if x.dim() != 2:
        x = x.view(x.shape[0], -1)
    return x


def standardize(x: torch.Tensor, mean: torch.Tensor | None = None, std: torch.Tensor | None = None, eps: float = 1e-6):
    if mean is None or std is None:
        mean = x.mean(dim=0)
        std = x.std(dim=0)
    std = std.clamp_min(eps)
    x_std = (x - mean) / std
    return x_std, mean, std


def _import_go2_solver(ik_solver_dir: str):
    """Import Go2Solver from IsaacLab kinematics directory."""
    ik_solver_dir = (ik_solver_dir or "").strip()
    if not ik_solver_dir:
        raise ValueError("--ik-solver-dir 가 비어 있습니다.")
    if ik_solver_dir not in sys.path:
        sys.path.insert(0, ik_solver_dir)
    try:
        from FK_IK_Solver import Go2Solver  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Go2Solver import 실패: {e}\n"
            f"  -ik-solver-dir={ik_solver_dir}\n"
            "  예상 파일: FK_IK_Solver.py (Go2Solver 클래스 포함)"
        ) from e
    return Go2Solver


def _go2_limits(device: torch.device, dtype: torch.dtype):
    """Per-leg limits for Go2 from USD (verify.py와 동일)."""
    hip_min = torch.tensor([-1.0472, -1.0472, -1.0472, -1.0472], device=device, dtype=dtype)
    hip_max = torch.tensor([1.0472, 1.0472, 1.0472, 1.0472], device=device, dtype=dtype)
    thigh_min = torch.tensor([-1.5708, -1.5708, -0.5236, -0.5236], device=device, dtype=dtype)  # FL, FR, RL, RR
    thigh_max = torch.tensor([3.4907, 3.4907, 4.5379, 4.5379], device=device, dtype=dtype)
    calf_min = torch.tensor([-2.7227, -2.7227, -2.7227, -2.7227], device=device, dtype=dtype)
    calf_max = torch.tensor([-0.8378, -0.8378, -0.8378, -0.8378], device=device, dtype=dtype)
    return (hip_min, hip_max), (thigh_min, thigh_max), (calf_min, calf_max)


def _default_go2_joint_pose(device: torch.device, dtype: torch.dtype, *, layout: str = "grouped") -> torch.Tensor:
    layout = (layout or "grouped").strip().lower()
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
    elif layout == "interleaved":
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
    else:
        raise ValueError(f"Unsupported joint layout: {layout!r} (expected grouped|interleaved)")
    return torch.tensor(vals, device=device, dtype=dtype)


@torch.no_grad()
def _compute_stand_foot_offset_flat(solver, *, device: torch.device, dtype: torch.dtype, layout: str) -> torch.Tensor:
    q0 = _default_go2_joint_pose(device=device, dtype=dtype, layout=layout).view(1, 12)
    foot0 = solver.go2_fk_new(q0).view(1, 12)
    return foot0.to(device=device, dtype=dtype)


def _go2_joint_limit_violation(ik_angles: torch.Tensor, *, eps: float):
    """Return per-sample joint-limit violation magnitude (radians), sum over all 12 joints.

    ik_angles: (B, 12) joint angles in radians (NaN/Inf should be handled by caller)
    """
    (hip_min, hip_max), (thigh_min, thigh_max), (calf_min, calf_max) = _go2_limits(
        device=ik_angles.device, dtype=ik_angles.dtype
    )
    hip = ik_angles[:, 0:4]
    thigh = ik_angles[:, 4:8]
    calf = ik_angles[:, 8:12]

    hip_violation = torch.relu(hip - (hip_max + eps)) + torch.relu((hip_min - eps) - hip)
    thigh_violation = torch.relu(thigh - (thigh_max + eps)) + torch.relu((thigh_min - eps) - thigh)
    calf_violation = torch.relu(calf - (calf_max + eps)) + torch.relu((calf_min - eps) - calf)
    return (hip_violation + thigh_violation + calf_violation).sum(dim=1)


def _go2_unreachable_penalty(foot_pos: torch.Tensor, solver) -> torch.Tensor:
    """Differentiable proxy for 'IK returns NaN' (reachability violation).

    foot_pos: (B, 4, 3) in base frame.
    Returns: (B,) penalty >= 0
    """
    if foot_pos.dim() != 3 or foot_pos.shape[1:] != (4, 3):
        raise ValueError(f"Expected foot_pos shape (B,4,3), got {tuple(foot_pos.shape)}")
    # Copied from Go2Solver.go2_ik_new, but we keep it finite + differentiable.
    local = foot_pos - solver.HIP_OFFSETS.to(device=foot_pos.device, dtype=foot_pos.dtype)
    x = local[..., 0]
    y = local[..., 1]
    z = local[..., 2]

    side_sign = torch.tensor([1, -1, 1, -1], device=foot_pos.device, dtype=foot_pos.dtype).unsqueeze(0)
    y_signed = y * side_sign

    L_prime = x**2 + y_signed**2 + z**2 - solver.L_HIP**2
    cos_theta_2 = (L_prime - solver.L_THIGH**2 - solver.L_CALF**2) / (2 * solver.L_THIGH * solver.L_CALF)
    sqrt_arg_hip = z**2 + y_signed**2 - solver.L_HIP**2

    too_far = torch.relu(torch.abs(cos_theta_2) - 1.0)
    too_close = torch.relu(-sqrt_arg_hip)
    return (too_far + too_close).sum(dim=1)


def get_loader(x_std: torch.Tensor, batch_size: int, shuffle: bool = True):
    dataset = data.TensorDataset(x_std)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_loader_opt(
    x_std: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    dataset = data.TensorDataset(x_std)
    loader_kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return data.DataLoader(dataset, **loader_kwargs)


def _sample_softflow_sigma(x: torch.Tensor, args) -> torch.Tensor:
    mode = str(args.softflow_mode or "uniform").strip().lower()
    if mode == "fixed":
        sigma = float(args.softflow_sigma)
        if sigma < 0:
            raise ValueError(f"--softflow-sigma must be >= 0, got {sigma:g}")
        return torch.full((x.shape[0], 1), sigma, device=x.device, dtype=x.dtype)
    if mode == "uniform":
        sigma_min = float(args.softflow_sigma_min)
        sigma_max = float(args.softflow_sigma_max)
        if sigma_min < 0 or sigma_max < 0:
            raise ValueError(
                f"--softflow-sigma-min/max must be >= 0, got min={sigma_min:g}, max={sigma_max:g}"
            )
        if sigma_max < sigma_min:
            raise ValueError(
                f"--softflow-sigma-max must be >= --softflow-sigma-min, got min={sigma_min:g}, max={sigma_max:g}"
            )
        return torch.rand((x.shape[0], 1), device=x.device, dtype=x.dtype) * (sigma_max - sigma_min) + sigma_min
    raise ValueError(f"Unsupported --softflow-mode={mode!r}. Expected: 'fixed' or 'uniform'.")


def _apply_softflow_mollification(x: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = _sample_softflow_sigma(x, args)
    if bool(args.softflow_per_dim):
        sigma = sigma.expand(-1, x.shape[1])
    noise = torch.randn_like(x)
    return x + sigma * noise, sigma


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    resume_ckpt = None
    resume_path = (args.resume or "").strip()
    if resume_path:
        resume_ckpt = torch.load(resume_path, map_location="cpu")

    x_raw = load_tensor(Path(args.data_path))
    mean = None
    std = None
    if resume_ckpt is not None and bool(args.resume_use_stats):
        mean = resume_ckpt.get("mean", None)
        std = resume_ckpt.get("std", None)
        if mean is not None and std is not None:
            mean = torch.as_tensor(mean).view(-1).to(dtype=torch.float32)
            std = torch.as_tensor(std).view(-1).to(dtype=torch.float32)
    x_std, mean, std = standardize(x_raw, mean=mean, std=std, eps=args.std_eps)
    loader = get_loader_opt(
        x_std,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    if bool(args.fix_prior_standard_normal) and bool(args.trainable_base):
        print("[INFO] --fix-prior-standard-normal enabled: overriding --trainable-base -> False")
        args.trainable_base = False

    model = build_model(
        dim=args.dim,
        num_layers=args.num_layers,
        hidden=args.hidden,
        num_blocks=args.num_blocks,
        num_bins=args.num_bins,
        tail_bound=args.tail_bound,
        trainable_base=args.trainable_base,
        actnorm=args.actnorm,
        mixing=args.mixing,
        mixing_use_lu=args.mixing_use_lu,
        mixing_seed=args.mixing_seed,
    ).to(device)
    if bool(args.fix_prior_standard_normal):
        print("[INFO] prior fixed to standard normal: q0 = N(0, I) (non-trainable)")
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"[warn] torch.compile 실패: {e}", file=sys.stderr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    lr_scheduler_name = str(args.lr_scheduler or "cosine").strip().lower()
    if lr_scheduler_name == "none":
        scheduler = None
    elif lr_scheduler_name == "cosine":
        t_max = int(args.lr_t_max) if int(args.lr_t_max) > 0 else int(args.epochs)
        t_max = max(t_max, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=t_max,
            eta_min=float(args.lr_min),
        )
    elif lr_scheduler_name == "step":
        step_size = int(args.lr_step_size) if int(args.lr_step_size) > 0 else max(int(args.epochs) // 3, 1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=step_size,
            gamma=float(args.lr_gamma),
        )
    else:
        raise ValueError(f"Unsupported --lr-scheduler={args.lr_scheduler!r}. Expected: none|cosine|step")

    if scheduler is None:
        print(f"[INFO] lr scheduler: none (lr={float(args.lr):g})")
    elif lr_scheduler_name == "cosine":
        print(
            f"[INFO] lr scheduler: cosine (T_max={scheduler.T_max}, eta_min={float(args.lr_min):g}, "
            f"start_lr={float(args.lr):g})"
        )
    else:
        print(
            f"[INFO] lr scheduler: step (step_size={scheduler.step_size}, gamma={float(args.lr_gamma):g}, "
            f"start_lr={float(args.lr):g})"
        )

    start_epoch = 0
    global_step = 0
    if resume_ckpt is not None:
        state = resume_ckpt.get("model", None)
        if state is None and isinstance(resume_ckpt, dict):
            if all(isinstance(k, str) and k.startswith(("q0.", "flows.")) for k in resume_ckpt.keys()):
                state = resume_ckpt
        if state is None:
            raise ValueError(f"--resume 체크포인트에서 model state_dict를 찾지 못했습니다: {resume_path}")
        missing, unexpected = model.load_state_dict(state, strict=bool(args.resume_strict))
        if missing or unexpected:
            print(f"[warn] resume load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
        if bool(args.resume_optimizer) and "opt" in resume_ckpt:
            try:
                opt.load_state_dict(resume_ckpt["opt"])
            except Exception as e:
                print(f"[warn] resume optimizer state 로드 실패: {e}", file=sys.stderr)
        if scheduler is not None and bool(args.resume_optimizer) and "scheduler" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            except Exception as e:
                print(f"[warn] resume scheduler state 로드 실패: {e}", file=sys.stderr)
        if "epoch" in resume_ckpt:
            start_epoch = int(resume_ckpt.get("epoch", 0) or 0)
        else:
            m = re.findall(r"(?:^|_)ep(\d+)(?:$|_)", Path(resume_path).stem)
            start_epoch = int(m[-1]) if m else 0
        if "global_step" in resume_ckpt:
            global_step = int(resume_ckpt.get("global_step", 0) or 0)
        else:
            global_step = int(start_epoch * len(loader))
        print(f"[INFO] resumed from {resume_path} (start_epoch={start_epoch}, global_step={global_step})")

    use_amp = device.type == "cuda" and args.amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    log2 = math.log(2)
    log_disc = math.log(256)

    base_save = Path(args.save_path) if args.save_path else None
    def maybe_save(tag=None):
        if not base_save:
            return
        out_path = base_save if tag is None else base_save.with_name(f"{base_save.stem}_ep{tag}{base_save.suffix}")
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "mean": mean.cpu(),
                "std": std.cpu(),
                "epoch": int(tag) if tag is not None else int(start_epoch + args.epochs),
                "global_step": int(global_step),
                **({"opt": opt.state_dict()} if bool(args.save_optimizer) else {}),
                **({"scheduler": scheduler.state_dict()} if (bool(args.save_optimizer) and scheduler is not None) else {}),
            },
            out_path,
        )
        print(f"Saved checkpoint to {out_path}")

    verify_enabled = bool(args.verify)
    verifier = None
    verify_log = ""
    verify_sample_pt = ""
    if verify_enabled:
        defaults = default_verify_paths(args.save_path, suffix="verify")
        verify_log = str(args.verify_log or defaults["log"])
        verify_sample_pt = str(args.verify_sample_pt or defaults["sample_pt"])
        verifier = Go2IKVerifier(solver_dir=str(args.verify_solver_dir), device=device)
        print(f"[INFO] verify(log) -> {verify_log}")
        print(f"[INFO] verify(samples) -> {verify_sample_pt} (overwrite)")
        append_line(verify_log, f"# started: save_path={args.save_path} verify_num_samples={args.verify_num_samples}")
        if resume_path:
            append_line(verify_log, f"# resumed: {resume_path}")

    coverage_enabled = bool(args.coverage)
    coverage = None
    coverage_log = ""
    coverage_sample_pt = ""
    if coverage_enabled:
        defaults = default_verify_paths(args.save_path, suffix="coverage")
        coverage_log = str(args.coverage_log or defaults["log"])
        coverage_sample_pt = str(args.coverage_sample_pt or defaults["sample_pt"])

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(args.coverage_seed))
        ref_n = int(args.coverage_ref_size)
        if ref_n <= 0:
            raise ValueError("--coverage-ref-size must be > 0")
        ref_n = min(ref_n, int(x_raw.shape[0]))
        ref_idx = torch.randint(0, int(x_raw.shape[0]), (ref_n,), generator=gen)
        x_ref = x_raw.index_select(0, ref_idx).to(dtype=torch.float32, device="cpu")

        coverage = FootWorkspaceCoverage(
            x_ref,
            bins=int(args.coverage_bins),
            q_high=float(args.coverage_quantile),
        )
        print(f"[INFO] coverage(log) -> {coverage_log}")
        if bool(args.coverage_save_samples):
            print(f"[INFO] coverage(samples) -> {coverage_sample_pt} (overwrite)")
        append_line(
            coverage_log,
            f"# started: save_path={args.save_path} coverage_num_samples={args.coverage_num_samples} "
            f"coverage_ref_size={ref_n} bins={args.coverage_bins} q={args.coverage_quantile}",
        )
        if resume_path:
            append_line(coverage_log, f"# resumed: {resume_path}")

    ik_enabled = float(args.ik_penalty_weight) > 0.0
    ik_solver = None
    mean_d = None
    std_d = None
    if ik_enabled:
        Go2Solver = _import_go2_solver(args.ik_solver_dir)
        ik_solver = Go2Solver(device=str(device))
        mean_d = mean.to(device).view(1, -1)
        std_d = std.to(device).view(1, -1).clamp_min(args.std_eps)
        print(
            f"[INFO] IK penalty enabled: weight={args.ik_penalty_weight:g} "
            f"(samples={args.ik_penalty_samples or args.batch_size}, every={args.ik_penalty_every}, "
            f"warmup_epochs={args.ik_penalty_warmup_epochs})"
        )
    else:
        mean_d = mean.to(device).view(1, -1)
        std_d = std.to(device).view(1, -1).clamp_min(args.std_eps)

    stand_offset_flat = None
    if bool(args.ik_verify_add_stand_offset):
        if int(args.dim) != 12:
            raise ValueError("--ik-verify-add-stand-offset 는 --dim=12 에서만 지원합니다.")

        solver_for_offset = None
        if ik_solver is not None:
            solver_for_offset = ik_solver
        elif verifier is not None:
            solver_for_offset = verifier.solver
        elif ik_enabled or verify_enabled:
            # Fallback (normally not reached): instantiate from verify solver dir.
            Go2Solver = _import_go2_solver(args.verify_solver_dir)
            solver_for_offset = Go2Solver(device=str(device))

        if solver_for_offset is None:
            print(
                "[warn] --ik-verify-add-stand-offset enabled, but both IK penalty and verify are disabled. "
                "No effect."
            )
        else:
            stand_offset_flat = _compute_stand_foot_offset_flat(
                solver_for_offset,
                device=device,
                dtype=torch.float32,
                layout=str(args.stand_offset_joint_layout),
            )
            print(
                "[INFO] IK/verify stand offset enabled: "
                f"layout={args.stand_offset_joint_layout} "
                f"offset_norm={stand_offset_flat.norm().item():.4f}"
            )

    softflow_enabled = bool(args.softflow)
    if softflow_enabled:
        mode = str(args.softflow_mode or "uniform").strip().lower()
        if mode == "fixed":
            print(
                f"[INFO] softflow enabled: mode=fixed sigma={float(args.softflow_sigma):g} "
                f"per_dim={bool(args.softflow_per_dim)}"
            )
        else:
            print(
                "[INFO] softflow enabled: "
                f"mode=uniform sigma_range=[{float(args.softflow_sigma_min):g}, {float(args.softflow_sigma_max):g}] "
                f"per_dim={bool(args.softflow_per_dim)}"
            )

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        epoch_t0 = time.perf_counter()
        lr_this_epoch = float(opt.param_groups[0]["lr"])
        running_nll = 0.0
        running_nll_train = 0.0
        running_loss = 0.0
        running_ik = 0.0
        running_ik_nan = 0.0
        ik_updates = 0
        seen_samples = 0
        num_batches = len(loader)
        use_tqdm = bool(args.progress) and tqdm is not None
        epoch_loader = tqdm(
            loader,
            total=num_batches,
            desc=f"Epoch {epoch:03d}",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
        ) if use_tqdm else loader

        for batch_idx, (x,) in enumerate(epoch_loader, start=1):
            global_step += 1
            x = x.to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            seen_samples += int(x.shape[0])
            opt.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                x_train = x
                if softflow_enabled:
                    x_train, _ = _apply_softflow_mollification(x, args)
                nll_train = -model.log_prob(x_train).mean()
                loss = nll_train

            if softflow_enabled:
                with torch.no_grad():
                    nll_clean = -model.log_prob(x).mean()
            else:
                nll_clean = nll_train.detach()

            if ik_enabled and epoch > int(args.ik_penalty_warmup_epochs) and (global_step % int(args.ik_penalty_every) == 0):
                n_pen = int(args.ik_penalty_samples) if int(args.ik_penalty_samples) > 0 else int(x.shape[0])
                x_samp_std, _ = model.sample(n_pen)
                x_samp_raw = x_samp_std * std_d + mean_d
                x_samp_eval = x_samp_raw if stand_offset_flat is None else (x_samp_raw + stand_offset_flat)
                foot_pos = x_samp_eval.view(-1, 4, 3)

                ik_angles = ik_solver.go2_ik_new(foot_pos)
                nan_mask = torch.isnan(ik_angles).any(dim=1) | torch.isinf(ik_angles).any(dim=1)
                ik_safe = torch.nan_to_num(ik_angles, nan=0.0, posinf=0.0, neginf=0.0)

                limit_pen = _go2_joint_limit_violation(ik_safe, eps=float(args.ik_penalty_eps))
                unreach_pen = _go2_unreachable_penalty(foot_pos, ik_solver)
                nan_pen = float(args.ik_penalty_nan_weight) * nan_mask.to(dtype=limit_pen.dtype)

                ik_pen = (
                    float(args.ik_penalty_limit_weight) * limit_pen
                    + float(args.ik_penalty_unreach_weight) * unreach_pen
                    + nan_pen
                ).mean()
                loss = loss + float(args.ik_penalty_weight) * ik_pen

                running_ik += float(ik_pen.item())
                running_ik_nan += float(nan_mask.float().mean().item())
                ik_updates += 1
            if use_grad_scaler:
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()

            running_nll += float(nll_clean.item())
            running_nll_train += float(nll_train.item())
            running_loss += float(loss.item())

            if bool(args.progress):
                elapsed = max(time.perf_counter() - epoch_t0, 1e-9)
                it_s = batch_idx / elapsed
                samp_s = seen_samples / elapsed
                eta_s = (num_batches - batch_idx) / max(it_s, 1e-9)
                if use_tqdm:
                    epoch_loader.set_postfix_str(
                        f"it/s={it_s:.2f} samp/s={samp_s:.0f} eta={eta_s:.1f}s nll={running_nll / batch_idx:.4f}"
                    )
                elif args.progress_refresh > 0 and (batch_idx % int(args.progress_refresh) == 0 or batch_idx == num_batches):
                    msg = (
                        f"[Epoch {epoch:03d}] {batch_idx}/{num_batches} "
                        f"| it/s={it_s:.2f} | samp/s={samp_s:.0f} | eta={eta_s:.1f}s | nll={running_nll / batch_idx:.4f}"
                    )
                    print(msg)

        if use_tqdm:
            epoch_loader.close()

        avg_nll = running_nll / len(loader)
        avg_nll_train = running_nll_train / len(loader)
        avg_loss = running_loss / len(loader)
        epoch_elapsed = max(time.perf_counter() - epoch_t0, 1e-9)
        epoch_it_s = len(loader) / epoch_elapsed
        epoch_samp_s = seen_samples / epoch_elapsed
        bpd_cont = avg_nll / (log2 * args.dim)
        bpd_disc = (avg_nll + log_disc * args.dim) / (log2 * args.dim)
        if ik_enabled and ik_updates > 0:
            avg_ik = running_ik / ik_updates
            avg_ik_nan = running_ik_nan / ik_updates
            if softflow_enabled:
                print(
                    f"Epoch {epoch:03d} | NLL(clean) {avg_nll:.4f} | NLL(train) {avg_nll_train:.4f} | "
                    f"bits/dim(clean,cont) {bpd_cont:.4f} | bits/dim(clean,disc) {bpd_disc:.4f} | "
                    f"loss {avg_loss:.4f} | ik_pen {avg_ik:.4f} | ik_nan {avg_ik_nan:.4f} | "
                    f"t {epoch_elapsed:.1f}s | it/s {epoch_it_s:.2f} | samp/s {epoch_samp_s:.0f} | lr {lr_this_epoch:.3e}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | NLL {avg_nll:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f} | "
                    f"loss {avg_loss:.4f} | ik_pen {avg_ik:.4f} | ik_nan {avg_ik_nan:.4f} | "
                    f"t {epoch_elapsed:.1f}s | it/s {epoch_it_s:.2f} | samp/s {epoch_samp_s:.0f} | lr {lr_this_epoch:.3e}"
                )
        else:
            if softflow_enabled:
                print(
                    f"Epoch {epoch:03d} | NLL(clean) {avg_nll:.4f} | NLL(train) {avg_nll_train:.4f} | "
                    f"bits/dim(clean,cont) {bpd_cont:.4f} | bits/dim(clean,disc) {bpd_disc:.4f} | "
                    f"loss {avg_loss:.4f} | t {epoch_elapsed:.1f}s | it/s {epoch_it_s:.2f} | samp/s {epoch_samp_s:.0f} | "
                    f"lr {lr_this_epoch:.3e}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | NLL {avg_nll:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f} | "
                    f"t {epoch_elapsed:.1f}s | it/s {epoch_it_s:.2f} | samp/s {epoch_samp_s:.0f} | lr {lr_this_epoch:.3e}"
                )

        need_verify = verify_enabled and (epoch % int(args.verify_every) == 0)
        need_cov = coverage_enabled and (epoch % int(args.coverage_every) == 0)
        if need_verify or need_cov:
            model.eval()
            with torch.no_grad():
                n_samp = 0
                if need_verify:
                    n_samp = max(n_samp, int(args.verify_num_samples))
                if need_cov:
                    n_samp = max(n_samp, int(args.coverage_num_samples))

                x_samp_std, _ = model.sample(int(n_samp))
                x_samp_raw = (x_samp_std * std_d + mean_d).float()

                if need_verify:
                    x_v = x_samp_raw[: int(args.verify_num_samples)]
                    x_v_eval = x_v if stand_offset_flat is None else (x_v + stand_offset_flat)
                    metrics = verifier.evaluate(x_v_eval, eps=float(args.verify_eps))
                    if bool(args.verify_save_samples):
                        torch.save(x_v_eval.cpu(), verify_sample_pt)
                    line = format_verify_line(epoch, metrics, prefix="[VERIFY] ")
                    print(line)
                    append_line(verify_log, line)

                if need_cov:
                    x_c = x_samp_raw[: int(args.coverage_num_samples)]
                    cov = coverage.evaluate(x_c)
                    if bool(args.coverage_save_samples):
                        torch.save(x_c.cpu(), coverage_sample_pt)
                    line = format_coverage_line(epoch, cov, prefix="[COVERAGE] ")
                    print(line)
                    append_line(coverage_log, line)

            model.train(True)

        if args.save_every > 0 and epoch % args.save_every == 0:
            maybe_save(epoch)
        if scheduler is not None:
            scheduler.step()

    maybe_save(None)


def parse_args():
    p = argparse.ArgumentParser(description="Train NSF on standardized 12D vectors.")
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt", help="Path to .pt/.pth/.npy/.npz/.csv/.txt data of shape (N,12)")
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--tail-bound", type=float, default=5.0)
    p.add_argument(
        "--mixing",
        type=str,
        default="none",
        choices=["none", "permute", "inv_affine", "inv_1x1conv"],
        help="Coupling layer 사이에 feature mixing 추가",
    )
    p.add_argument(
        "--mixing-use-lu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="inv_affine/inv_1x1conv에서 LU 파라미터화 사용",
    )
    p.add_argument("--mixing-seed", type=int, default=0, help="mixing=permute 에서 permutation seed")
    p.add_argument(
        "--trainable-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="base DiagGaussian(q0)의 loc/log_scale 학습 여부 (끄면 고정 N(0,I))",
    )
    p.add_argument(
        "--fix-prior-standard-normal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="q0를 표준정규 N(0,I)로 고정(= --no-trainable-base 강제)",
    )
    p.add_argument(
        "--softflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SoftFlow-style data mollification 사용 (x -> x + sigma*noise 후 NLL 학습)",
    )
    p.add_argument(
        "--softflow-mode",
        type=str,
        default="uniform",
        choices=["fixed", "uniform"],
        help="softflow sigma 샘플링 방식",
    )
    p.add_argument(
        "--softflow-sigma",
        type=float,
        default=0.01,
        help="softflow-mode=fixed 일 때 sigma 값",
    )
    p.add_argument(
        "--softflow-sigma-min",
        type=float,
        default=0.0,
        help="softflow-mode=uniform 일 때 sigma 최소값",
    )
    p.add_argument(
        "--softflow-sigma-max",
        type=float,
        default=0.01,
        help="softflow-mode=uniform 일 때 sigma 최대값",
    )
    p.add_argument(
        "--softflow-per-dim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="True면 sigma를 (B,D)로 broadcast해서 적용 (False면 (B,1))",
    )
    p.add_argument(
        "--actnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="첫 layer로 ActNorm(learned affine, data-dependent init) 추가 (trainable_base 대체/보완)",
    )
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        help="learning rate scheduler 종류",
    )
    p.add_argument("--lr-min", type=float, default=1e-6, help="cosine scheduler의 eta_min")
    p.add_argument(
        "--lr-t-max",
        type=int,
        default=0,
        help="cosine scheduler T_max (<=0 이면 epochs 사용)",
    )
    p.add_argument(
        "--lr-step-size",
        type=int,
        default=0,
        help="step scheduler step_size (<=0 이면 epochs/3 사용)",
    )
    p.add_argument("--lr-gamma", type=float, default=0.5, help="step scheduler gamma")
    p.add_argument("--clip-grad", type=float, default=5.0, help="Clip global grad norm (<=0 to disable)")
    p.add_argument("--std-eps", type=float, default=1e-6, help="min std for standardization")
    p.add_argument("--resume", type=str, default="", help="체크포인트(.pt/.pth)에서 모델을 로드해서 이어서 학습")
    p.add_argument(
        "--resume-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume 시 state_dict strict 로드 여부",
    )
    p.add_argument(
        "--resume-use-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume 체크포인트에 mean/std가 있으면 그걸로 표준화 재사용 (권장)",
    )
    p.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume 체크포인트에 optimizer state가 있으면 로드",
    )
    p.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="체크포인트에 optimizer state 저장 (파일 크기 증가)",
    )
    p.add_argument(
        "--ik-penalty-weight",
        type=float,
        default=0.0,
        help="Go2 IK 기반 penalty 정규화 weight (0이면 비활성화). 샘플을 IK로 풀어 joint limit/unreachable을 벌점으로 줌",
    )
    p.add_argument("--ik-penalty-limit-weight", type=float, default=1.0, help="joint limit violation 항의 가중치")
    p.add_argument("--ik-penalty-unreach-weight", type=float, default=1.0, help="unreachable(=NaN proxy) 항의 가중치")
    p.add_argument("--ik-penalty-nan-weight", type=float, default=0.0, help="IK 결과가 NaN/Inf인 샘플에 추가 상수 벌점")
    p.add_argument("--ik-penalty-eps", type=float, default=1e-3, help="joint limit 허용 오차 (rad)")
    p.add_argument(
        "--ik-penalty-samples",
        type=int,
        default=1024,
        help="IK penalty 계산에 사용할 generated sample 개수 (0이면 batch_size만큼)",
    )
    p.add_argument("--ik-penalty-every", type=int, default=10, help="몇 step마다 IK penalty를 계산할지 (성능/속도 트레이드오프)")
    p.add_argument("--ik-penalty-warmup-epochs", type=int, default=0, help="이 에폭까지는 IK penalty 비활성화")
    p.add_argument(
        "--ik-solver-dir",
        type=str,
        default="/home/kdg/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/kinematics",
        help="IsaacLab Go2Solver(FK_IK_Solver.py)가 있는 디렉토리 경로",
    )
    p.add_argument(
        "--ik-verify-add-stand-offset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="IK penalty/verify 전에 샘플에 standing foot offset(절대좌표)을 더함 (rel 데이터셋용)",
    )
    p.add_argument(
        "--stand-offset-joint-layout",
        type=str,
        default="grouped",
        choices=["grouped", "interleaved"],
        help="standing foot offset 계산 시 사용할 default joint ordering",
    )
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 개수")
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="학습 중 epoch 진행바/속도 표시 (tqdm 미설치 시 텍스트 진행 출력)",
    )
    p.add_argument(
        "--progress-refresh",
        type=int,
        default=50,
        help="tqdm 미설치 시 진행 출력 주기(steps)",
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader pin_memory (CUDA에서 H2D 전송 최적화)",
    )
    p.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader persistent_workers (num_workers>0일 때만 의미 있음)",
    )
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch_factor (num_workers>0일 때)")
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TF32 matmul 허용 (속도↑, 정밀도↓; RTX 30/40에서 유효)",
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="AMP 사용 (속도/메모리↑)")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="torch.compile 사용(실험적)")
    p.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    p.add_argument("--save-every", type=int, default=1000, help="에폭마다 중간 체크포인트 저장 (0이면 비활성화)")
    p.add_argument("--save-path", type=str, default="")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify-every", type=int, default=1)
    p.add_argument("--verify-num-samples", type=int, default=10000)
    p.add_argument("--verify-eps", type=float, default=1e-3)
    p.add_argument("--verify-solver-dir", type=str, default=DEFAULT_IK_SOLVER_DIR)
    p.add_argument("--verify-log", type=str, default="")
    p.add_argument("--verify-sample-pt", type=str, default="")
    p.add_argument("--verify-save-samples", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coverage", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--coverage-every", type=int, default=1)
    p.add_argument("--coverage-num-samples", type=int, default=50000)
    p.add_argument("--coverage-ref-size", type=int, default=200000)
    p.add_argument("--coverage-bins", type=int, default=24)
    p.add_argument("--coverage-quantile", type=float, default=0.999)
    p.add_argument("--coverage-seed", type=int, default=0)
    p.add_argument("--coverage-log", type=str, default="")
    p.add_argument("--coverage-sample-pt", type=str, default="")
    p.add_argument("--coverage-save-samples", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
