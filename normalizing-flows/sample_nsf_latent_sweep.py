import argparse
import math
from pathlib import Path
from typing import Callable

import torch

# NOTE: We reuse the exact model builders used for training so that state_dict keys match.
from run_nsf_12_stand import build_model as build_gaussian_nsf
from nsf_12_stand_unform import build_model as build_uniform_nsf


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"체크포인트 포맷이 예상과 다릅니다: {path} (dict with key 'model' 기대)")
    args = ckpt.get("args", {})
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    data_rescale = float(ckpt.get("data_rescale", args.get("data_rescale", 1.0)))
    if mean is None or std is None:
        raise ValueError("체크포인트에 mean/std가 없습니다. 학습 스크립트가 저장한 ckpt를 사용하세요.")
    return ckpt["model"], args, mean, std, data_rescale


def train_to_raw(x_train: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, data_rescale: float):
    """Map training-space x_train -> raw foot positions (meters).

    - run_nsf_12_stand.py: x_train == standardized space -> data_rescale=1
    - nsf_12_stand_unform.py: x_train may be rescaled (e.g., into [-1,1]) so we undo it via data_rescale.
    """
    x_std = x_train * float(data_rescale)
    return x_std * std + mean


def _linf(z: torch.Tensor) -> torch.Tensor:
    return z.abs().amax(dim=1)


def _sample_truncated_standard_normal(
    n: int,
    dim: int,
    *,
    lo: float,
    hi: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample i.i.d. N(0,1) truncated per-dimension to [lo, hi]."""
    if not (lo < hi):
        raise ValueError(f"trunc normal requires lo < hi, got lo={lo}, hi={hi}")
    lo_t = torch.tensor(float(lo), device=device, dtype=dtype)
    hi_t = torch.tensor(float(hi), device=device, dtype=dtype)
    cdf_lo = torch.special.ndtr(lo_t)
    cdf_hi = torch.special.ndtr(hi_t)
    u = torch.rand((n, dim), device=device, dtype=dtype, generator=generator) * (cdf_hi - cdf_lo) + cdf_lo
    # ndtri expects values in (0,1)
    eps = torch.finfo(dtype).eps
    u = u.clamp(min=eps, max=1.0 - eps)
    return torch.special.ndtri(u)


def _sample_abs_band_standard_normal(
    n: int,
    dim: int,
    *,
    lo: float,
    hi: float | None,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample i.i.d. N(0,1) with per-dim |z_i| in [lo, hi] (or [lo, +inf) if hi is None).

    This is the "corner-heavy" interpretation used for 12D latent probes:
    every dimension is pushed into the same magnitude band.
    """
    lo = float(lo)
    if lo < 0:
        raise ValueError(f"abs-band sampling requires lo >= 0, got lo={lo}")

    lo_t = torch.tensor(lo, device=device, dtype=dtype)
    cdf_lo = torch.special.ndtr(lo_t)

    if hi is None:
        cdf_hi = torch.tensor(1.0, device=device, dtype=dtype)
    else:
        hi = float(hi)
        if not (lo < hi):
            raise ValueError(f"abs-band sampling requires lo < hi, got lo={lo}, hi={hi}")
        hi_t = torch.tensor(hi, device=device, dtype=dtype)
        cdf_hi = torch.special.ndtr(hi_t)

    u = torch.rand((n, dim), device=device, dtype=dtype, generator=generator) * (cdf_hi - cdf_lo) + cdf_lo
    eps = torch.finfo(dtype).eps
    u = u.clamp(min=eps, max=1.0 - eps)
    mag = torch.special.ndtri(u).abs()

    sign = torch.randint(0, 2, (n, dim), device=device, generator=generator, dtype=torch.int64)
    sign = sign * 2 - 1
    return mag * sign.to(dtype)


def _sample_rejection(
    n: int,
    *,
    propose: Callable[[int], torch.Tensor],
    accept: Callable[[torch.Tensor], torch.Tensor],
    proposal_batch: int,
) -> torch.Tensor:
    """Generic rejection sampler.

    propose(k) -> (k, D) tensor
    accept(z) -> boolean mask (k,)
    """
    out: list[torch.Tensor] = []
    remaining = int(n)
    while remaining > 0:
        k = max(int(proposal_batch), remaining)
        z = propose(k)
        mask = accept(z)
        if mask.any():
            take = z[mask]
            if take.shape[0] > remaining:
                take = take[:remaining]
            out.append(take)
            remaining -= int(take.shape[0])
    return torch.cat(out, dim=0)


def _save_pt(x_raw: torch.Tensor, out_path: Path, *, save_shape: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if save_shape == "4x3":
        if x_raw.dim() != 2 or x_raw.size(1) != 12:
            raise ValueError(f"Expected x_raw shape (N,12) before reshape, got {tuple(x_raw.shape)}")
        x_raw = x_raw.view(-1, 4, 3)
    torch.save(x_raw.cpu(), out_path)


def _infer_ckpt_kind(saved_args: dict) -> str:
    # nsf_12_stand_unform.py ckpt stores 'base' (uniform/modified_uniform)
    base = str(saved_args.get("base", "")).strip().lower()
    if base in {"uniform", "modified_uniform", "modified-uniform"}:
        return "moduniform"
    return "gaussian"


def _build_model_from_ckpt(kind: str, saved_args: dict, *, device: torch.device):
    dim = int(saved_args.get("dim", 12))
    num_layers = int(saved_args.get("num_layers", 32))
    hidden = int(saved_args.get("hidden", 512))
    num_blocks = int(saved_args.get("num_blocks", 2))
    num_bins = int(saved_args.get("num_bins", 16))
    tail_bound = float(saved_args.get("tail_bound", 5.0))
    actnorm = bool(saved_args.get("actnorm", False))
    mixing = str(saved_args.get("mixing", "none"))
    mixing_use_lu = bool(saved_args.get("mixing_use_lu", True))
    mixing_seed = int(saved_args.get("mixing_seed", 0))

    if kind == "gaussian":
        trainable_base = bool(saved_args.get("trainable_base", False))
        model = build_gaussian_nsf(
            dim=dim,
            num_layers=num_layers,
            hidden=hidden,
            num_blocks=num_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
            trainable_base=trainable_base,
            actnorm=actnorm,
            mixing=mixing,
            mixing_use_lu=mixing_use_lu,
            mixing_seed=mixing_seed,
        )
        return model.to(device), dim

    # moduniform / uniform
    base = str(saved_args.get("base", "modified_uniform"))
    uniform_bound = float(saved_args.get("uniform_bound", 1.0))
    modified_uniform_sigma = float(saved_args.get("modified_uniform_sigma", 1e-3))
    modified_uniform_normalize = bool(saved_args.get("modified_uniform_normalize", False))
    trainable_base = bool(saved_args.get("trainable_base", False))
    uniform_atanh = bool(saved_args.get("uniform_atanh", False))
    uniform_atanh_eps = float(saved_args.get("uniform_atanh_eps", 1e-6))

    model = build_uniform_nsf(
        dim=dim,
        num_layers=num_layers,
        hidden=hidden,
        num_blocks=num_blocks,
        num_bins=num_bins,
        tail_bound=tail_bound,
        base=base,
        uniform_bound=uniform_bound,
        modified_uniform_sigma=modified_uniform_sigma,
        modified_uniform_normalize=modified_uniform_normalize,
        trainable_base=trainable_base,
        uniform_atanh=uniform_atanh,
        uniform_atanh_eps=uniform_atanh_eps,
        actnorm=actnorm,
        mixing=mixing,
        mixing_use_lu=mixing_use_lu,
        mixing_seed=mixing_seed,
    )
    return model.to(device), dim


def _decode_and_save(
    *,
    model,
    z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    data_rescale: float,
    out_path: Path,
    save_shape: str,
):
    with torch.no_grad():
        x_train = model.forward(z)
        x_raw = train_to_raw(x_train, mean=mean, std=std, data_rescale=data_rescale)
    _save_pt(x_raw, out_path, save_shape=save_shape)


def _make_rng(device: torch.device, seed: int) -> torch.Generator | None:
    """
    Create RNG for sampling.

    NOTE:
    일부 torch 버전에서는 CUDA 텐서에 대해 `generator=`에 CPU generator를 넘기면
    `Expected a 'cuda' device type for generator but found 'cpu'` 에러가 발생합니다.
    그래서 CUDA에서는 가능하면 CUDA generator를 만들고, 실패하면 generator를 넘기지 않습니다.
    """
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        try:
            return torch.Generator(device="cuda").manual_seed(seed)
        except Exception:
            return None
    return torch.Generator(device="cpu").manual_seed(seed)


def _sanitize_generator(device: torch.device, gen: torch.Generator | None) -> torch.Generator | None:
    """
    Ensure generator is compatible with the target device.

    Some torch builds error if you pass a CPU generator when sampling CUDA tensors:
      RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
    In that case, the safest fallback is to not pass a generator at all.
    """
    if gen is None:
        return None
    try:
        gen_dev = getattr(gen, "device", None)
        # torch.Generator.device is a torch.device on modern PyTorch.
        if isinstance(gen_dev, torch.device) and gen_dev.type != device.type:
            return None
        # If it's a string-like (older variants), compare best-effort.
        if isinstance(gen_dev, str) and not str(gen_dev).startswith(device.type):
            return None
    except Exception:
        # If we can't introspect, avoid passing generator on CUDA.
        if device.type == "cuda":
            return None
    return gen


def run_full(args, *, model, dim: int, ckpt_kind: str, saved_args: dict, out_dir: Path, stem: str):
    device = next(model.parameters()).device
    dtype = torch.float32
    gen = _sanitize_generator(device, _make_rng(device, args.seed))

    if ckpt_kind == "gaussian":
        z = torch.randn((args.num_samples, dim), device=device, dtype=dtype, generator=gen)
        out_path = out_dir / f"{stem}__z_full.pt"
        return [(out_path, z)]

    # moduniform full: choose between hard box and mollified (as trained)
    uniform_bound = float(saved_args.get("uniform_bound", 1.0))
    sigma = float(saved_args.get("modified_uniform_sigma", 1e-3))
    if args.moduniform_full == "hard":
        u = torch.rand((args.num_samples, dim), device=device, dtype=dtype, generator=gen) * 2.0 - 1.0
        z = u * uniform_bound
        out_path = out_dir / f"{stem}__z_full_hard_box.pt"
    else:
        u = torch.rand((args.num_samples, dim), device=device, dtype=dtype, generator=gen) * 2.0 - 1.0
        eps = torch.randn((args.num_samples, dim), device=device, dtype=dtype, generator=gen)
        z = (u + sigma * eps) * uniform_bound
        out_path = out_dir / f"{stem}__z_full_mollified_sigma{sigma:g}.pt"
    return [(out_path, z)]


def run_gaussian_sweep(args, *, model, dim: int, out_dir: Path, stem: str):
    device = next(model.parameters()).device
    dtype = torch.float32
    gen = _sanitize_generator(device, _make_rng(device, args.seed))

    proposal_batch = int(args.proposal_batch)

    def propose_std(k: int) -> torch.Tensor:
        return torch.randn((k, dim), device=device, dtype=dtype, generator=gen)

    outs: list[tuple[Path, torch.Tensor]] = []

    # Full
    outs.append((out_dir / f"{stem}__z_full.pt", propose_std(int(args.num_samples))))

    # Truncated balls: ||z||_inf <= k
    # Include 0.5σ as a finer "near-mean" slice (requested for RL-centric analysis).
    for k in (0.5, 1.0, 2.0, 3.0):
        z = _sample_truncated_standard_normal(
            args.num_samples,
            dim,
            lo=-k,
            hi=k,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        outs.append((out_dir / f"{stem}__z_linf_le_{k:g}sigma.pt", z))

    # Shells: (0.5,1], (1,2], (2,3], (>3)
    shells = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]
    if args.interval_mode == "per_dim":
        # Corner-heavy per-dimension shells: for each i, |z_i| ∈ (lo, hi]
        for lo, hi in shells:
            z = _sample_abs_band_standard_normal(
                args.num_samples,
                dim,
                lo=lo,
                hi=hi,
                device=device,
                dtype=dtype,
                generator=gen,
            )
            outs.append((out_dir / f"{stem}__z_perdim_abs_{lo:g}_{hi:g}sigma.pt", z))

        z = _sample_abs_band_standard_normal(
            args.num_samples,
            dim,
            lo=3.0,
            hi=None,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        outs.append((out_dir / f"{stem}__z_perdim_abs_ge_3sigma.pt", z))
    else:
        # L∞ shells: lo < ||z||_inf <= hi
        for lo, hi in shells:
            def propose_trunc(k: int, *, _hi=hi):
                return _sample_truncated_standard_normal(k, dim, lo=-_hi, hi=_hi, device=device, dtype=dtype, generator=gen)

            z = _sample_rejection(
                args.num_samples,
                propose=lambda k, _p=propose_trunc: _p(k),
                accept=lambda zz, _lo=lo: _linf(zz) > _lo,
                proposal_batch=proposal_batch,
            )
            outs.append((out_dir / f"{stem}__z_linf_{lo:g}_{hi:g}sigma.pt", z))

        # >3sigma: reject from full gaussian
        z = _sample_rejection(
            args.num_samples,
            propose=propose_std,
            accept=lambda zz: _linf(zz) > 3.0,
            proposal_batch=proposal_batch,
        )
        outs.append((out_dir / f"{stem}__z_linf_gt_3sigma.pt", z))
    return outs


def run_moduniform_sweep(args, *, model, dim: int, saved_args: dict, out_dir: Path, stem: str):
    device = next(model.parameters()).device
    dtype = torch.float32
    gen = _sanitize_generator(device, _make_rng(device, args.seed))

    bound = float(saved_args.get("uniform_bound", 1.0))
    sigma = float(saved_args.get("modified_uniform_sigma", 1e-3))
    step = float(args.step)
    proposal_batch = int(args.proposal_batch)

    if step <= 0:
        raise ValueError("--step must be > 0")
    if bound <= 0:
        raise ValueError("uniform_bound must be > 0")

    # edges: 0.2, 0.4, ..., 1.0
    num_steps = int(round(bound / step))
    edges = [round(step * i, 10) for i in range(1, num_steps + 1)]
    edges[-1] = bound  # ensure exact

    def propose_box(k: int, *, box: float) -> torch.Tensor:
        u = torch.rand((k, dim), device=device, dtype=dtype, generator=gen) * 2.0 - 1.0
        z = u * float(box)
        if args.moduniform_noise == "mollified":
            eps = torch.randn((k, dim), device=device, dtype=dtype, generator=gen)
            z = z + (float(sigma) * eps)
        return z

    outs: list[tuple[Path, torch.Tensor]] = []

    # Full (for reference): hard box [-bound,bound]
    outs.append((out_dir / f"{stem}__z_full_hard_box.pt", propose_box(int(args.num_samples), box=bound)))

    # Cumulative boxes: ||z||_inf <= b
    for b in edges:
        z = _sample_rejection(
            args.num_samples,
            propose=lambda k, _b=b: propose_box(k, box=_b),
            accept=lambda zz, _b=b: _linf(zz) <= float(_b),
            proposal_batch=proposal_batch,
        )
        outs.append((out_dir / f"{stem}__z_linf_le_{b:g}.pt", z))

    # Shells (donut-like): (b_lo, b_hi]
    shells: list[tuple[float, float]] = []
    prev = 0.0
    for b in edges:
        shells.append((prev, b))
        prev = b

    for lo, hi in shells:
        if args.interval_mode == "per_dim" and lo > 0:
            # Each dimension independently in [-hi,-lo] ∪ [lo,hi] (corner-heavy).
            # This matches the literal "(-1,-0.8) or (0.8,1.0) per-dim" interpretation.
            def propose_per_dim(k: int, *, _lo=lo, _hi=hi) -> torch.Tensor:
                # magnitude ~ U(lo, hi), sign ~ Rademacher
                mag = torch.rand((k, dim), device=device, dtype=dtype, generator=gen) * (float(_hi) - float(_lo)) + float(_lo)
                sign = torch.randint(0, 2, (k, dim), device=device, generator=gen, dtype=torch.int64)
                sign = sign * 2 - 1
                z = mag * sign.to(dtype)
                if args.moduniform_noise == "mollified":
                    eps = torch.randn((k, dim), device=device, dtype=dtype, generator=gen)
                    z = z + (float(sigma) * eps)
                return z

            z = propose_per_dim(int(args.num_samples))
        else:
            # L∞ shell: lo < ||z||_inf <= hi
            z = _sample_rejection(
                args.num_samples,
                propose=lambda k, _hi=hi: propose_box(k, box=_hi),
                accept=lambda zz, _lo=lo, _hi=hi: (_linf(zz) > float(_lo)) & (_linf(zz) <= float(_hi)),
                proposal_batch=proposal_batch,
            )

        tag = "perdim" if (args.interval_mode == "per_dim" and lo > 0) else "linf"
        outs.append((out_dir / f"{stem}__z_{tag}_{lo:g}_{hi:g}.pt", z))

    # "outer / middle / inner" quick trio (defaults: outer 0.8-1.0, mid 0.4-0.6, inner 0-0.2)
    trio = [("inner", 0.0, min(0.2, bound)), ("middle", 0.4, min(0.6, bound)), ("outer", max(bound - 0.2, 0.0), bound)]
    for tag, lo, hi in trio:
        if hi <= 0 or hi <= lo:
            continue
        if args.interval_mode == "per_dim" and lo > 0:
            # "corner" trio: each dim in [-hi,-lo] ∪ [lo,hi]
            mag = torch.rand((args.num_samples, dim), device=device, dtype=dtype, generator=gen) * (float(hi) - float(lo)) + float(lo)
            sign = torch.randint(0, 2, (args.num_samples, dim), device=device, generator=gen, dtype=torch.int64)
            sign = sign * 2 - 1
            z = mag * sign.to(dtype)
            if args.moduniform_noise == "mollified":
                eps = torch.randn((args.num_samples, dim), device=device, dtype=dtype, generator=gen)
                z = z + (float(sigma) * eps)
        else:
            z = _sample_rejection(
                args.num_samples,
                propose=lambda k, _hi=hi: propose_box(k, box=_hi),
                accept=lambda zz, _lo=lo, _hi=hi: (_linf(zz) > float(_lo)) & (_linf(zz) <= float(_hi)),
                proposal_batch=proposal_batch,
            )
        mode = "perdim" if (args.interval_mode == "per_dim" and lo > 0) else "linf"
        outs.append((out_dir / f"{stem}__z_{tag}_{mode}_{lo:g}_{hi:g}.pt", z))

    # Outside-only bands (for "mollified uniform" boundary overflow probes):
    # sample |z_i| in [bound + k*delta, bound + (k+1)*delta] for k=0..steps-1.
    if args.outside_only:
        delta = float(args.outside_delta)
        if delta <= 0:
            delta = float(sigma)
        steps = int(args.outside_steps)
        if steps <= 0:
            raise ValueError("--outside-steps must be >= 1 when --outside-only is enabled.")
        if delta <= 0:
            raise ValueError("--outside-delta must be > 0 (or ckpt sigma must be > 0).")

        for k in range(steps):
            lo = float(bound) + float(k) * float(delta)
            hi = float(bound) + float(k + 1) * float(delta)
            if hi <= lo:
                continue

            if args.interval_mode == "per_dim":
                # Corner-heavy: all dims outside.
                mag = torch.rand((args.num_samples, dim), device=device, dtype=dtype, generator=gen) * (hi - lo) + lo
                sign = torch.randint(0, 2, (args.num_samples, dim), device=device, generator=gen, dtype=torch.int64)
                sign = sign * 2 - 1
                z = mag * sign.to(dtype)
                tag = "perdim"
            else:
                # L∞ outside band: lo < ||z||_inf <= hi
                z = _sample_rejection(
                    args.num_samples,
                    propose=lambda kk, _hi=hi: propose_box(kk, box=_hi),
                    accept=lambda zz, _lo=lo, _hi=hi: (_linf(zz) > float(_lo)) & (_linf(zz) <= float(_hi)),
                    proposal_batch=proposal_batch,
                )
                tag = "linf"

            outs.append((out_dir / f"{stem}__z_outside_{tag}_{lo:g}_{hi:g}.pt", z))

    return outs


def parse_args():
    p = argparse.ArgumentParser(description="Sampling-only: NSF checkpoints with latent range sweeps (for IsaacLab verify.py).")
    p.add_argument("--ckpt", type=str, required=True, help="NSF checkpoint (.pt)")
    p.add_argument("--out-dir", type=str, default="", help="Output directory (default: ckpt parent / latent_sweeps/<stem>)")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--save-shape", type=str, default="12", choices=["12", "4x3"], help="Save shape: (N,12) or (N,4,3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument(
        "--sweep",
        type=str,
        default="auto",
        choices=["auto", "full", "gaussian", "moduniform", "all"],
        help="Which sweep to run.",
    )

    # Rejection-sampling control
    p.add_argument("--proposal-batch", type=int, default=200000, help="Proposal batch size for rejection sampling.")

    # Moduniform-specific
    p.add_argument("--step", type=float, default=0.2, help="Shell step for moduniform sweep (default 0.2 for bound=1).")
    p.add_argument(
        "--moduniform-noise",
        type=str,
        default="none",
        choices=["none", "mollified"],
        help="If 'mollified', add N(0,sigma) noise (sigma from ckpt) when sampling moduniform sweeps.",
    )
    p.add_argument(
        "--moduniform-full",
        type=str,
        default="mollified",
        choices=["mollified", "hard"],
        help="How to sample FULL z for moduniform ckpt.",
    )
    p.add_argument(
        "--interval-mode",
        type=str,
        default="linf",
        choices=["linf", "per_dim"],
        help="How to interpret latent 'bands': L∞ shells (donut-like) or per-dimension (corner-heavy).",
    )
    p.add_argument(
        "--outside-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For moduniform sweeps: additionally sample ONLY the outside slab |z| in [bound, bound+delta] (and further bands).",
    )
    p.add_argument(
        "--outside-delta",
        type=float,
        default=0.0,
        help="Outside slab width beyond uniform_bound. If <= 0, uses ckpt mollifier sigma. Example: 1e-3.",
    )
    p.add_argument(
        "--outside-steps",
        type=int,
        default=3,
        help="How many outside bands to generate (default: 3). Each band width is outside-delta.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    state_dict, saved_args, mean_ckpt, std_ckpt, data_rescale = load_checkpoint(args.ckpt, device)

    ckpt_kind = _infer_ckpt_kind(saved_args)
    model, dim = _build_model_from_ckpt(ckpt_kind, saved_args, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    mean = mean_ckpt.to(device).view(1, -1)
    std = std_ckpt.to(device).view(1, -1).clamp_min(1e-6)
    save_shape = args.save_shape

    ckpt_path = Path(args.ckpt)
    stem = ckpt_path.stem
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ckpt_path.parent / "latent_sweeps" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    which = args.sweep
    if which == "auto":
        which = "moduniform" if ckpt_kind == "moduniform" else "gaussian"

    z_jobs: list[tuple[Path, torch.Tensor]] = []

    if which in {"full"}:
        z_jobs.extend(run_full(args, model=model, dim=dim, ckpt_kind=ckpt_kind, saved_args=saved_args, out_dir=out_dir, stem=stem))
    elif which in {"gaussian"}:
        if ckpt_kind != "gaussian":
            raise ValueError("이 ckpt는 gaussian NSF가 아닙니다. --sweep moduniform 또는 --sweep full 을 사용하세요.")
        z_jobs.extend(run_gaussian_sweep(args, model=model, dim=dim, out_dir=out_dir, stem=stem))
    elif which in {"moduniform"}:
        if ckpt_kind != "moduniform":
            raise ValueError("이 ckpt는 moduniform/uniform NSF가 아닙니다. --sweep gaussian 또는 --sweep full 을 사용하세요.")
        z_jobs.extend(run_moduniform_sweep(args, model=model, dim=dim, saved_args=saved_args, out_dir=out_dir, stem=stem))
    elif which == "all":
        z_jobs.extend(run_full(args, model=model, dim=dim, ckpt_kind=ckpt_kind, saved_args=saved_args, out_dir=out_dir, stem=stem))
        if ckpt_kind == "gaussian":
            z_jobs.extend(run_gaussian_sweep(args, model=model, dim=dim, out_dir=out_dir, stem=stem))
        else:
            z_jobs.extend(run_moduniform_sweep(args, model=model, dim=dim, saved_args=saved_args, out_dir=out_dir, stem=stem))
    else:
        raise ValueError(f"Unknown --sweep={which}")

    for out_path, z in z_jobs:
        _decode_and_save(
            model=model,
            z=z,
            mean=mean,
            std=std,
            data_rescale=float(data_rescale),
            out_path=out_path,
            save_shape=save_shape,
        )
        print(f"[OK] saved {out_path} (z shape={tuple(z.shape)}, save_shape={save_shape})")

    print(f"[DONE] wrote {len(z_jobs)} file(s) into {out_dir}")


if __name__ == "__main__":
    main()
