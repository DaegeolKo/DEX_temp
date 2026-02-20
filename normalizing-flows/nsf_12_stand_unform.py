import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils import clip_grad_norm_

import normflows as nf
from normflows.distributions.base import BaseDistribution
from normflows.flows.base import Flow

from go2_verify_utils import DEFAULT_IK_SOLVER_DIR, Go2IKVerifier, append_line, default_verify_paths, format_verify_line
from coverage_utils import FootWorkspaceCoverage, format_coverage_line


_LOG_HALF = -0.6931471805599453


class FixedPermutation1d(Flow):
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


class VectorInvertible1x1Conv(Flow):
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


def _log_ndtr(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.special, "log_ndtr"):
        return torch.special.log_ndtr(x)
    # Fallback: log(0.5 * (1 + erf(x/sqrt(2))))
    sqrt2 = math.sqrt(2.0)
    cdf = 0.5 * (1.0 + torch.erf(x / sqrt2))
    return torch.log(cdf.clamp_min(torch.finfo(cdf.dtype).tiny))


def _log1mexp(t: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(t)) for t <= 0 in a numerically stable way."""
    # NOTE: Do not use `torch.where` here. For very small |t| in float32,
    # `torch.exp(t)` may round to 1.0, making `log1p(-exp(t)) = -inf` even when
    # the other branch is selected. `torch.where` evaluates both branches, which
    # can then create NaN gradients (0 * inf) in backward.
    out = torch.empty_like(t)
    mask = t < _LOG_HALF  # exp(t) < 0.5
    if mask.any():
        out[mask] = torch.log1p(-torch.exp(t[mask]))
    if (~mask).any():
        out[~mask] = torch.log(-torch.expm1(t[~mask]))
    return out


def _log_ndtr_diff(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Compute log(Φ(b) - Φ(a)) stably for b >= a."""
    log_cdf_b = _log_ndtr(b)
    log_cdf_a = _log_ndtr(a)

    # Heuristic switch: Φ(b) close to 1  <=> logΦ(b) close to 0.
    use_tail = log_cdf_b > -1e-6

    # NOTE: Do not use `torch.where` here for the same reason as `_log1mexp`.
    # Computing both branches can create NaN gradients when the unused branch hits -inf
    # (e.g., log1p(-exp(t)) with exp(t) rounding to 1).
    out = torch.empty_like(log_cdf_b)

    use_reg = ~use_tail
    if use_reg.any():
        lb = log_cdf_b[use_reg]
        la = log_cdf_a[use_reg]
        out[use_reg] = lb + _log1mexp(la - lb)

    if use_tail.any():
        # Use the survival-function identity:
        #   Φ(b) - Φ(a) = Φ(-a) - Φ(-b)
        log_sf_a = _log_ndtr(-a[use_tail])
        log_sf_b = _log_ndtr(-b[use_tail])
        out[use_tail] = log_sf_a + _log1mexp(log_sf_b - log_sf_a)

    return out


class AtanhBoundedFlow(Flow):
    """Map a bounded variable z∈(-b,b) to an unbounded variable u∈R via atanh.

    Forward (sampling direction): u = atanh(z / b)
    Inverse (log_prob direction): z = b * tanh(u)
    """

    def __init__(self, *, bound: float, eps: float = 1e-6):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.bound = float(bound)
        self.eps = float(eps)

    def forward(self, z: torch.Tensor):
        if z.dim() != 2:
            raise ValueError(f"Expected z to have shape (B,D), got {tuple(z.shape)}")
        b = float(self.bound)
        y = z / b
        y = y.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)
        u = 0.5 * (torch.log1p(y) - torch.log1p(-y))
        logabsdet = (-math.log(b)) * z.shape[1] - torch.sum(torch.log1p(-y * y), dim=-1)
        return u, logabsdet

    def inverse(self, u: torch.Tensor):
        if u.dim() != 2:
            raise ValueError(f"Expected u to have shape (B,D), got {tuple(u.shape)}")
        b = float(self.bound)
        y = torch.tanh(u)
        z = y * b
        logabsdet = (math.log(b)) * u.shape[1] + torch.sum(torch.log1p(-y * y), dim=-1)
        return z, logabsdet


class DiagUniform(BaseDistribution):
    """Diagonal uniform distribution: z_i ~ Uniform(loc_i - scale_i, loc_i + scale_i)."""

    def __init__(self, dim: int, *, bound: float, trainable: bool = True):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        self.shape = (dim,)
        init_log_scale = float(math.log(bound))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, dim))
            self.log_scale = nn.Parameter(torch.full((1, dim), init_log_scale))
        else:
            self.register_buffer("loc", torch.zeros(1, dim))
            self.register_buffer("log_scale", torch.full((1, dim), init_log_scale))

    def forward(self, num_samples: int = 1, context=None):
        scale = torch.exp(self.log_scale)
        u = torch.rand((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        u = u * 2.0 - 1.0
        z = self.loc + scale * u
        log_p_const = -torch.sum(torch.log(2.0 * scale), dim=-1)
        # NOTE: avoid returning an expanded view because normflows' `sample()` does in-place ops.
        log_p = log_p_const.expand(num_samples).clone()
        return z, log_p

    def log_prob(self, z, context=None):
        scale = torch.exp(self.log_scale)
        low = self.loc - scale
        high = self.loc + scale
        out_range = torch.logical_or(z < low, z > high)
        ind_inf = torch.any(out_range.reshape(z.shape[0], -1), dim=-1)
        log_p_const = -torch.sum(torch.log(2.0 * scale), dim=-1).expand(z.shape[0])
        log_p = log_p_const.clone()
        log_p[ind_inf] = -float("inf")
        return log_p


class DiagModifiedUniform(BaseDistribution):
    """"Modified uniform" (soft-edged box) prior.

    Per dimension:
        x = u + δ,  u ~ Uniform(-1, 1),  δ ~ Normal(0, σ)
        z = loc + scale * x

    The (unnormalized) 1D PDF used in the paper:
        p(x) = Φ((1 - x)/σ) - Φ((-1 - x)/σ)
    where Φ is standard normal CDF.
    """

    def __init__(
        self,
        dim: int,
        *,
        bound: float,
        sigma: float = 0.01,
        normalize_pdf: bool = False,
        trainable: bool = True,
    ):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.shape = (dim,)
        self.sigma = float(sigma)
        self.normalize_pdf = bool(normalize_pdf)

        init_log_scale = float(math.log(bound))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, dim))
            self.log_scale = nn.Parameter(torch.full((1, dim), init_log_scale))
        else:
            self.register_buffer("loc", torch.zeros(1, dim))
            self.register_buffer("log_scale", torch.full((1, dim), init_log_scale))

    def forward(self, num_samples: int = 1, context=None):
        scale = torch.exp(self.log_scale)
        device = self.loc.device
        dtype = self.loc.dtype

        u = torch.rand((num_samples,) + self.shape, device=device, dtype=dtype) * 2.0 - 1.0
        eps = torch.randn((num_samples,) + self.shape, device=device, dtype=dtype)
        x = u + float(self.sigma) * eps
        z = self.loc + scale * x
        log_p = self.log_prob(z, context=context)
        return z, log_p

    def log_prob(self, z, context=None):
        scale = torch.exp(self.log_scale)
        x = (z - self.loc) / scale
        sigma = float(self.sigma)

        b = (1.0 - x) / sigma
        a = (-1.0 - x) / sigma
        log_diff = _log_ndtr_diff(b, a)
        if self.normalize_pdf:
            log_diff = log_diff - math.log(2.0)

        log_p_dim = log_diff - self.log_scale
        return torch.sum(log_p_dim, dim=-1)


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


def standardize(
    x: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float = 1e-6,
):
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


def compute_affine_stats(x: torch.Tensor, mode: str, eps: float):
    mode = (mode or "").strip().lower()
    if mode in {"raw", "none"}:
        mean = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        std = torch.ones(x.shape[1], dtype=x.dtype, device=x.device)
        return mean, std
    if mode in {"standardize", "zscore"}:
        return x.mean(dim=0), x.std(dim=0).clamp_min(eps)
    if mode in {"minmax", "min-max"}:
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        mean = 0.5 * (x_min + x_max)
        std = 0.5 * (x_max - x_min).clamp_min(eps)
        return mean, std
    raise ValueError(f"지원하지 않는 preprocess={mode!r} (raw|standardize|minmax)")


def affine_normalize(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float,
):
    std = std.clamp_min(eps)
    return (x - mean) / std, mean, std


def estimate_abs_quantile(
    x: torch.Tensor,
    *,
    q: float,
    max_samples: int,
    seed: int,
):
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")
    if max_samples <= 0:
        raise ValueError(f"max_samples must be > 0, got {max_samples}")
    if x.dim() != 2:
        raise ValueError(f"Expected x to have shape (N,D), got {tuple(x.shape)}")

    n, d = x.shape
    rows = int(math.ceil(max_samples / max(1, d)))
    rows = min(n, rows)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randint(0, n, (rows,), generator=g, device="cpu")
    sample = x[idx].abs().reshape(-1)
    return float(torch.quantile(sample, q).item())


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


def build_model(
    *,
    dim: int,
    num_layers: int,
    hidden: int,
    num_blocks: int,
    num_bins: int,
    tail_bound: float,
    base: str,
    uniform_bound: float,
    modified_uniform_sigma: float,
    modified_uniform_normalize: bool,
    trainable_base: bool,
    uniform_atanh: bool,
    uniform_atanh_eps: float,
    actnorm: bool = False,
    mixing: str = "none",
    mixing_use_lu: bool = True,
    mixing_seed: int = 0,
):
    base = (base or "").strip().lower()
    flows = []
    mixing = (mixing or "").strip().lower()
    if mixing not in {"none", "permute", "inv_affine", "inv_1x1conv"}:
        raise ValueError("mixing must be one of: none|permute|inv_affine|inv_1x1conv")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(mixing_seed))
    if base == "uniform":
        q0 = DiagUniform(dim, bound=uniform_bound, trainable=trainable_base)
        if uniform_atanh and trainable_base:
            print(
                "[warn] base=uniform 에서 trainable_base=True 이면 --uniform-atanh 를 무시합니다. "
                "고정 uniform[-b,b]를 원하면 --no-trainable-base 를 사용하세요.",
                file=sys.stderr,
            )
        elif uniform_atanh:
            flows.append(AtanhBoundedFlow(bound=uniform_bound, eps=uniform_atanh_eps))
    elif base in {"modified_uniform", "modified-uniform"}:
        q0 = DiagModifiedUniform(
            dim,
            bound=uniform_bound,
            sigma=modified_uniform_sigma,
            normalize_pdf=modified_uniform_normalize,
            trainable=trainable_base,
        )
    else:
        raise ValueError(f"지원하지 않는 base={base!r} (기대: 'uniform' or 'modified_uniform')")

    if bool(actnorm):
        flows.append(nf.flows.ActNorm((dim,)))

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
    return nf.NormalizingFlow(q0, flows)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    x_raw = load_tensor(Path(args.data_path))
    mean_aff, std_aff = compute_affine_stats(x_raw, args.preprocess, eps=args.std_eps)
    x_aff, mean, std = affine_normalize(x_raw, mean=mean_aff, std=std_aff, eps=args.std_eps)

    # Pick a default bound that safely contains the initial z distribution (identity init => z ~= x_std).
    if args.uniform_bound > 0:
        uniform_bound = float(args.uniform_bound)
    else:
        uniform_bound = float(x_aff.abs().max().item()) * 1.05 + 1e-6
        print(f"[INFO] uniform_bound auto = {uniform_bound:.6f}")

    # If the base support is bounded (e.g., Uniform[-b,b]) and you also standardize the data,
    # x_std can easily exceed the fixed bound (e.g., |x_std|>1). That yields -inf log_prob and
    # can create NaNs once combined with flow log-dets (inf - inf).
    data_rescale = 1.0
    x_train = x_aff
    if args.rescale_to_bound and uniform_bound > 0:
        base = (args.base or "").strip().lower()
        if args.rescale_target > 0:
            target = float(args.rescale_target)
        else:
            # FlowPG-style: modified_uniform은 초기에 box 경계/바깥쪽에 일부 데이터가 걸리게 해서
            # base log_prob의 기울기를 살립니다. (uniform은 hard support라서 불가)
            target = uniform_bound * (1.1 if base in {"modified_uniform", "modified-uniform"} else 1.0)

        if base == "uniform" and target > uniform_bound + 1e-12:
            raise ValueError(
                f"base=uniform 에서는 rescale_target({target:g})이 uniform_bound({uniform_bound:g})를 넘을 수 없습니다."
            )

        stat_mode = (args.rescale_stat or "").strip().lower()
        if stat_mode == "quantile":
            if base == "uniform":
                print("[warn] base=uniform 에서는 rescale_stat=quantile을 지원하지 않아 max로 대체합니다.", file=sys.stderr)
                stat = float(x_aff.abs().max().item())
            else:
                stat = estimate_abs_quantile(
                    x_aff,
                    q=float(args.rescale_quantile),
                    max_samples=int(args.rescale_quantile_samples),
                    seed=int(args.seed),
                )
        else:
            stat = float(x_aff.abs().max().item())

        if stat <= 0:
            raise ValueError("rescale_stat 결과가 0 이하입니다. 데이터가 모두 0인지 확인하세요.")
        data_rescale = stat / target
        x_train = x_aff / data_rescale
        print(
            f"[INFO] data_rescale = {data_rescale:.6f} (stat={stat:.6f} -> target={target:g}; x_aff -> x_train)"
        )
    args.data_rescale = float(data_rescale)

    loader = get_loader_opt(
        x_train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = build_model(
        dim=args.dim,
        num_layers=args.num_layers,
        hidden=args.hidden,
        num_blocks=args.num_blocks,
        num_bins=args.num_bins,
        tail_bound=args.tail_bound,
        mixing=args.mixing,
        mixing_use_lu=args.mixing_use_lu,
        mixing_seed=args.mixing_seed,
        base=args.base,
        uniform_bound=uniform_bound,
        modified_uniform_sigma=args.modified_uniform_sigma,
        modified_uniform_normalize=args.modified_uniform_normalize,
        trainable_base=args.trainable_base,
        uniform_atanh=args.uniform_atanh,
        uniform_atanh_eps=args.uniform_atanh_eps,
        actnorm=args.actnorm,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"[warn] torch.compile 실패: {e}", file=sys.stderr)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_amp = device.type == "cuda" and args.amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    logdet_penalty = float(args.logdet_penalty)
    if logdet_penalty > 0:
        print(f"[INFO] logdet_penalty = {logdet_penalty:g} (loss += λ * E[log_det^2])")

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
                "data_rescale": float(data_rescale),
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

    ik_enabled = float(args.ik_penalty_weight) > 0.0
    ik_solver = None
    mean_d = None
    std_d = None
    data_rescale_d = None
    if ik_enabled:
        Go2Solver = _import_go2_solver(args.ik_solver_dir)
        ik_solver = Go2Solver(device=str(device))
        mean_d = mean.to(device).view(1, -1)
        std_d = std.to(device).view(1, -1).clamp_min(args.std_eps)
        data_rescale_d = float(data_rescale)
        print(
            f"[INFO] IK penalty enabled: weight={args.ik_penalty_weight:g} "
            f"(samples={args.ik_penalty_samples or args.batch_size}, every={args.ik_penalty_every}, "
            f"warmup_epochs={args.ik_penalty_warmup_epochs})"
        )
    else:
        mean_d = mean.to(device).view(1, -1)
        std_d = std.to(device).view(1, -1).clamp_min(args.std_eps)
        data_rescale_d = float(data_rescale)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running_nll = 0.0
        running_loss = 0.0
        running_ik = 0.0
        running_ik_nan = 0.0
        ik_updates = 0
        for (x,) in loader:
            global_step += 1
            x = x.to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                if logdet_penalty > 0:
                    z, log_det = model.inverse_and_log_det(x)
                    log_q = model.q0.log_prob(z) + log_det
                    nll = -log_q.mean()
                    loss = nll + logdet_penalty * (log_det**2).mean()
                else:
                    nll = -model.log_prob(x).mean()
                    loss = nll

            if ik_enabled and epoch > int(args.ik_penalty_warmup_epochs) and (global_step % int(args.ik_penalty_every) == 0):
                n_pen = int(args.ik_penalty_samples) if int(args.ik_penalty_samples) > 0 else int(x.shape[0])
                x_samp_train, _ = model.sample(n_pen)
                x_samp_raw = (x_samp_train * float(data_rescale_d)) * std_d + mean_d
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
                    clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()
            running_nll += float(nll.item())
            running_loss += float(loss.item())

        avg_nll = running_nll / len(loader)
        avg_loss = running_loss / len(loader)
        bpd_cont = avg_nll / (log2 * args.dim)
        bpd_disc = (avg_nll + log_disc * args.dim) / (log2 * args.dim)
        if ik_enabled and ik_updates > 0:
            avg_ik = running_ik / ik_updates
            avg_ik_nan = running_ik_nan / ik_updates
            print(
                f"Epoch {epoch:03d} | NLL {avg_nll:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f} | "
                f"loss {avg_loss:.4f} | ik_pen {avg_ik:.4f} | ik_nan {avg_ik_nan:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | NLL {avg_nll:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f}"
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

                x_samp_train, _ = model.sample(int(n_samp))
                x_samp_raw = (x_samp_train * float(data_rescale_d)) * std_d + mean_d

                if need_verify:
                    x_v = x_samp_raw[: int(args.verify_num_samples)]
                    metrics = verifier.evaluate(x_v, eps=float(args.verify_eps))
                    if bool(args.verify_save_samples):
                        torch.save(x_v.cpu(), verify_sample_pt)
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

    maybe_save(None)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train NSF on standardized 12D vectors with uniform/modified-uniform latent."
    )
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt", help="Path to data of shape (N,12)")
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--num-bins", type=int, default=16)
    p.add_argument("--tail-bound", type=float, default=1.2)
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
        "--preprocess",
        type=str,
        default="standardize",
        choices=["raw", "standardize", "minmax"],
        help="학습 전 affine 정규화 방식. x_aff=(x-mean)/std. (기본: standardize)",
    )

    p.add_argument(
        "--base",
        type=str,
        default="uniform",
        choices=["uniform", "modified_uniform"],
        help="latent(base) distribution: uniform or modified_uniform(=uniform[-1,1] convolved with N(0,σ))",
    )
    p.add_argument(
        "--uniform-bound",
        type=float,
        default=0.0,
        help="uniform half-range bound. <=0이면 데이터에서 자동 추정 (초기 z=x_std가 범위 밖으로 나가지 않게)",
    )
    p.add_argument(
        "--modified-uniform-sigma",
        type=float,
        default=0.01,
        help="modified_uniform의 gaussian smoothing σ (논문 기본 0.01)",
    )
    p.add_argument(
        "--modified-uniform-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="modified_uniform PDF를 2로 나눠 정규화할지 여부 (기본 False: 논문식 그대로)",
    )
    p.add_argument(
        "--trainable-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="uniform/modified_uniform의 loc/log_scale(=center/half-range) 학습 여부",
    )
    p.add_argument(
        "--actnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="첫 layer로 ActNorm(learned affine, data-dependent init) 추가 (trainable_base 대체/보완)",
    )
    p.add_argument(
        "--uniform-atanh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="base=uniform일 때 atanh(z/bound) 변환을 앞에 추가해서 z가 항상 (-bound,bound)에 머물게 함(학습 안정화)",
    )
    p.add_argument(
        "--uniform-atanh-eps",
        type=float,
        default=1e-6,
        help="uniform-atanh clamp eps (|z/bound|<=1-eps)",
    )

    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-grad", type=float, default=5.0, help="Clip global grad norm (<=0 to disable)")
    p.add_argument("--std-eps", type=float, default=1e-6, help="min std for standardization")
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
        "--rescale-to-bound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preprocess 후 x_aff를 global rescale 해서 x_train을 만듦 (ckpt에 data_rescale 저장).",
    )
    p.add_argument(
        "--rescale-target",
        type=float,
        default=0.0,
        help="x_train의 타겟 max|.| 값. 0이면 자동( uniform=>bound, modified_uniform=>1.1*bound )",
    )
    p.add_argument(
        "--rescale-stat",
        type=str,
        default="max",
        choices=["max", "quantile"],
        help="rescale 통계값: max|.| 또는 |.| quantile (modified_uniform에서 outlier 영향 완화용)",
    )
    p.add_argument("--rescale-quantile", type=float, default=0.999, help="rescale_stat=quantile일 때 q (0<q<1)")
    p.add_argument(
        "--rescale-quantile-samples",
        type=int,
        default=2_000_000,
        help="rescale_stat=quantile일 때 랜덤 샘플 개수(근사). 큰 데이터에서 전체 quantile 계산 비용을 줄임",
    )
    p.add_argument(
        "--logdet-penalty",
        type=float,
        default=0.0,
        help="loss += λ * E[log_det^2]. (기본 0: 비활성화)",
    )
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 개수")
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
