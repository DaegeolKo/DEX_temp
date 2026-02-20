import argparse
from pathlib import Path
import math
import sys
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data

import normflows as nf

from go2_verify_utils import DEFAULT_IK_SOLVER_DIR, Go2IKVerifier, append_line, default_verify_paths, format_verify_line
from coverage_utils import FootWorkspaceCoverage, format_coverage_line


LEG_NAMES = ("FL", "FR", "RL", "RR")


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
    flows: list[nn.Module] = []
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


def split_legs(x12: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x12.dim() != 2 or x12.size(1) != 12:
        raise ValueError(f"Expected (B,12), got {tuple(x12.shape)}")
    return (x12[:, 0:3], x12[:, 3:6], x12[:, 6:9], x12[:, 9:12])


def concat_legs(xs: Iterable[torch.Tensor]) -> torch.Tensor:
    xs = list(xs)
    if len(xs) != 4:
        raise ValueError(f"Expected 4 leg tensors, got {len(xs)}")
    for x in xs:
        if x.dim() != 2 or x.size(1) != 3:
            raise ValueError(f"Each leg tensor must be (B,3), got {tuple(x.shape)}")
    return torch.cat(xs, dim=1)


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


@dataclass(frozen=True)
class ResumeState:
    ckpt: dict
    path: str
    start_epoch: int
    global_step: int


def _maybe_load_resume(args, loader_len: int) -> ResumeState | None:
    resume_path = (args.resume or "").strip()
    if not resume_path:
        return None
    ckpt = torch.load(resume_path, map_location="cpu")
    start_epoch = 0
    if "epoch" in ckpt:
        start_epoch = int(ckpt.get("epoch", 0) or 0)
    else:
        m = re.findall(r"(?:^|_)ep(\d+)(?:$|_)", Path(resume_path).stem)
        start_epoch = int(m[-1]) if m else 0
    if "global_step" in ckpt:
        global_step = int(ckpt.get("global_step", 0) or 0)
    else:
        global_step = int(start_epoch * loader_len)
    return ResumeState(ckpt=ckpt, path=resume_path, start_epoch=start_epoch, global_step=global_step)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    x_raw = load_tensor(Path(args.data_path))
    if x_raw.size(1) != 12:
        raise ValueError(f"Expected data with 12 dims, got shape={tuple(x_raw.shape)}")

    # Resume stats (optional).
    resume_for_stats = None
    if (args.resume or "").strip():
        resume_for_stats = torch.load(args.resume, map_location="cpu")
    mean = None
    std = None
    if resume_for_stats is not None and bool(args.resume_use_stats):
        mean = resume_for_stats.get("mean", None)
        std = resume_for_stats.get("std", None)
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

    models = nn.ModuleList(
        [
            build_model(
                dim=3,
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
            )
            for _ in range(4)
        ]
    ).to(device)

    if args.compile:
        for i in range(4):
            try:
                models[i] = torch.compile(models[i], mode=args.compile_mode)
            except Exception as e:
                print(f"[warn] torch.compile 실패 (leg={LEG_NAMES[i]}): {e}", file=sys.stderr)

    opt = torch.optim.Adam(models.parameters(), lr=args.lr)

    resume = _maybe_load_resume(args, loader_len=len(loader))
    start_epoch = 0
    global_step = 0
    if resume is not None:
        state = resume.ckpt.get("models", None)
        if state is None:
            state = resume.ckpt.get("model", None)
        if state is None:
            state = resume.ckpt
        missing, unexpected = models.load_state_dict(state, strict=bool(args.resume_strict))
        if missing or unexpected:
            print(f"[warn] resume load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
        if bool(args.resume_optimizer) and "opt" in resume.ckpt:
            try:
                opt.load_state_dict(resume.ckpt["opt"])
            except Exception as e:
                print(f"[warn] resume optimizer state 로드 실패: {e}", file=sys.stderr)
        start_epoch = int(resume.start_epoch)
        global_step = int(resume.global_step)
        print(f"[INFO] resumed from {resume.path} (start_epoch={start_epoch}, global_step={global_step})")

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
                "models": models.state_dict(),
                "args": vars(args),
                "mean": mean.cpu(),
                "std": std.cpu(),
                "epoch": int(tag) if tag is not None else int(start_epoch + args.epochs),
                "global_step": int(global_step),
                **({"opt": opt.state_dict()} if bool(args.save_optimizer) else {}),
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
        if resume is not None:
            append_line(verify_log, f"# resumed: {resume.path}")

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
        if resume is not None:
            append_line(coverage_log, f"# resumed: {resume.path}")

    ik_enabled = float(args.ik_penalty_weight) > 0.0
    ik_solver = None
    mean_d = mean.to(device).view(1, -1)
    std_d = std.to(device).view(1, -1).clamp_min(args.std_eps)
    if ik_enabled:
        Go2Solver = _import_go2_solver(args.ik_solver_dir)
        ik_solver = Go2Solver(device=str(device))
        print(
            f"[INFO] IK penalty enabled: weight={args.ik_penalty_weight:g} "
            f"(samples={args.ik_penalty_samples or args.batch_size}, every={args.ik_penalty_every}, "
            f"warmup_epochs={args.ik_penalty_warmup_epochs})"
        )

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        running_nll = 0.0
        running_loss = 0.0
        running_ik = 0.0
        running_ik_nan = 0.0
        ik_updates = 0

        for (x,) in loader:
            global_step += 1
            x = x.to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            x_fl, x_fr, x_rl, x_rr = split_legs(x)

            opt.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                nll = 0.0
                for m, x_leg in zip(models, (x_fl, x_fr, x_rl, x_rr)):
                    nll = nll + (-m.log_prob(x_leg).mean())
                loss = nll

            if (
                ik_enabled
                and epoch > int(args.ik_penalty_warmup_epochs)
                and (global_step % int(args.ik_penalty_every) == 0)
            ):
                assert ik_solver is not None
                n_pen = int(args.ik_penalty_samples) if int(args.ik_penalty_samples) > 0 else int(x.shape[0])
                x_samp_legs = []
                for m in models:
                    x_leg, _ = m.sample(int(n_pen))
                    x_samp_legs.append(x_leg)
                x_samp_std = concat_legs(x_samp_legs)
                x_samp_raw = x_samp_std * std_d + mean_d
                foot_pos = x_samp_raw.view(-1, 4, 3)

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
                    clip_grad_norm_(models.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    clip_grad_norm_(models.parameters(), args.clip_grad)
                opt.step()

            running_nll += float(nll.item())
            running_loss += float(loss.item())

        avg_nll = running_nll / len(loader)
        avg_loss = running_loss / len(loader)
        bpd_cont = avg_nll / (log2 * 12)
        bpd_disc = (avg_nll + log_disc * 12) / (log2 * 12)
        if ik_enabled and ik_updates > 0:
            avg_ik = running_ik / ik_updates
            avg_ik_nan = running_ik_nan / ik_updates
            print(
                f"Epoch {epoch:03d} | NLL {avg_nll:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f} | "
                f"loss {avg_loss:.4f} | ik_pen {avg_ik:.4f} | ik_nan {avg_ik_nan:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | NLL {avg_nll:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f}")

        need_verify = verify_enabled and (epoch % int(args.verify_every) == 0)
        need_cov = coverage_enabled and (epoch % int(args.coverage_every) == 0)
        if need_verify or need_cov:
            models.eval()
            with torch.no_grad():
                n_samp = 0
                if need_verify:
                    n_samp = max(n_samp, int(args.verify_num_samples))
                if need_cov:
                    n_samp = max(n_samp, int(args.coverage_num_samples))

                x_samp_legs = []
                for m in models:
                    x_leg, _ = m.sample(int(n_samp))
                    x_samp_legs.append(x_leg)
                x_samp_std = concat_legs(x_samp_legs)
                x_samp_raw = (x_samp_std * std_d + mean_d).float()

                if need_verify:
                    assert verifier is not None
                    x_v = x_samp_raw[: int(args.verify_num_samples)]
                    metrics = verifier.evaluate(x_v, eps=float(args.verify_eps))
                    if bool(args.verify_save_samples):
                        torch.save(x_v.cpu(), verify_sample_pt)
                    line = format_verify_line(epoch, metrics, prefix="[VERIFY] ")
                    print(line)
                    append_line(verify_log, line)

                if need_cov:
                    assert coverage is not None
                    x_c = x_samp_raw[: int(args.coverage_num_samples)]
                    cov = coverage.evaluate(x_c)
                    if bool(args.coverage_save_samples):
                        torch.save(x_c.cpu(), coverage_sample_pt)
                    line = format_coverage_line(epoch, cov, prefix="[COVERAGE] ")
                    print(line)
                    append_line(coverage_log, line)

            models.train(True)

        if args.save_every > 0 and epoch % args.save_every == 0:
            maybe_save(epoch)

    maybe_save(None)


def parse_args():
    p = argparse.ArgumentParser(
        description="Option A: train 4 independent 3D NSF models (FL/FR/RL/RR), concatenate to 12D foot positions."
    )
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt", help="Path to data of shape (N,12)")
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
    p.add_argument("--mixing-use-lu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mixing-seed", type=int, default=0)
    p.add_argument("--trainable-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--actnorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip-grad", type=float, default=5.0)
    p.add_argument("--std-eps", type=float, default=1e-6)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume-strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-use-stats", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-optimizer", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--ik-penalty-weight", type=float, default=0.0)
    p.add_argument("--ik-penalty-limit-weight", type=float, default=1.0)
    p.add_argument("--ik-penalty-unreach-weight", type=float, default=1.0)
    p.add_argument("--ik-penalty-nan-weight", type=float, default=0.0)
    p.add_argument("--ik-penalty-eps", type=float, default=1e-3)
    p.add_argument("--ik-penalty-samples", type=int, default=1024)
    p.add_argument("--ik-penalty-every", type=int, default=10)
    p.add_argument("--ik-penalty-warmup-epochs", type=int, default=0)
    p.add_argument("--ik-solver-dir", type=str, default=DEFAULT_IK_SOLVER_DIR)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])

    p.add_argument("--save-every", type=int, default=1000)
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
