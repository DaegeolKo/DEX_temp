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

from go2_verify_utils import DEFAULT_IK_SOLVER_DIR, Go2IKVerifier, append_line, default_verify_paths, format_verify_line


_LOG_HALF = -0.6931471805599453


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


def affine_normalize(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float,
):
    std = std.clamp_min(eps)
    return (x - mean) / std, mean, std


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


def preprocess_data(x_raw: torch.Tensor, args):
    mean, std = compute_affine_stats(x_raw, args.preprocess, eps=args.std_eps)
    x_std, mean, std = affine_normalize(x_raw, mean=mean, std=std, eps=args.std_eps)

    target_max_abs = float(args.target_max_abs)
    data_rescale = 1.0
    x_train = x_std
    if target_max_abs > 0:
        max_abs = float(x_std.abs().max().item())
        if max_abs <= 0:
            raise ValueError("데이터가 모두 0입니다. target_max_abs 스케일링을 할 수 없습니다.")
        data_rescale = max_abs / target_max_abs
        x_train = x_std / data_rescale
        print(f"[INFO] data_rescale = {data_rescale:.6f} (x_std -> x_train max|.|={target_max_abs:g})")
    return x_train, mean, std, float(data_rescale)


def train_to_raw(x_train: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, data_rescale: float):
    x_std = x_train * float(data_rescale)
    return x_std * std + mean


class FlowPGRealNVP(nn.Module):
    """FlowPG 구현 기반 RealNVP (조건부 y는 옵션이지만 여기서는 기본 0)."""

    def __init__(
        self,
        dim: int,
        transform_count: int,
        hidden_size: int,
        *,
        conditional_dim: int = 0,
        use_actnorm: bool = False,
        actnorm_eps: float = 1e-6,
        permute: str = "none",
        permute_seed: int = 0,
    ):
        super().__init__()
        if transform_count % 2 != 0:
            raise ValueError("transform_count는 짝수여야 합니다 (FlowPG와 동일).")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if conditional_dim < 0:
            raise ValueError(f"conditional_dim must be >= 0, got {conditional_dim}")

        self.dim = int(dim)
        self.conditional_dim = int(conditional_dim)
        self.input_dim = self.dim + self.conditional_dim
        self.transform_count = int(transform_count)
        self.hidden_size = int(hidden_size)
        self.use_actnorm = bool(use_actnorm)

        mask = self._get_masks(self.transform_count, self.dim)
        self.register_buffer("mask", mask)
        self.t = nn.ModuleList([self._get_nett() for _ in range(self.transform_count)])
        self.s = nn.ModuleList([self._get_nets() for _ in range(self.transform_count)])
        permute = (permute or "").strip().lower()
        if permute not in {"none", "flip", "random"}:
            raise ValueError("permute must be one of: none|flip|random")
        self.permute = permute
        if self.use_actnorm:
            self.actnorm = nn.ModuleList([ActNorm1d(self.dim, eps=actnorm_eps) for _ in range(self.transform_count)])
        else:
            self.actnorm = None
        if self.permute == "none":
            self.permutations = None
        else:
            perms = []
            if self.permute == "flip":
                base = torch.arange(self.dim - 1, -1, -1, dtype=torch.long)
                for _ in range(self.transform_count):
                    perms.append(base.clone())
            else:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(permute_seed))
                for _ in range(self.transform_count):
                    perms.append(torch.randperm(self.dim, generator=gen))
            self.permutations = nn.ModuleList([FixedPermutation1d(p) for p in perms])

    @staticmethod
    def _get_masks(transform_count: int, dim: int) -> torch.Tensor:
        left = dim // 2
        mask_one = torch.cat([torch.ones(left), torch.zeros(dim - left)], dim=0)
        mask_two = 1.0 - mask_one
        mask = torch.stack([mask_one, mask_two], dim=0).repeat(transform_count // 2, 1)
        return mask

    def _get_nets(self) -> nn.Sequential:
        # scale network: 마지막 Tanh로 s 범위를 제한 (FlowPG 핵심)
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.dim),
            nn.Tanh(),
        )

    def _get_nett(self) -> nn.Sequential:
        # shift network
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.dim),
        )

    def _concat_y(self, x_or_z: torch.Tensor, y: torch.Tensor | None):
        if y is None or self.conditional_dim == 0:
            return x_or_z
        return torch.cat([x_or_z, y], dim=1)

    def f(self, x: torch.Tensor, y: torch.Tensor | None = None):
        """x -> z (inverse direction)"""
        log_det_j = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for i in reversed(range(self.transform_count)):
            if self.permutations is not None:
                z = self.permutations[i].inverse(z)
            if self.use_actnorm:
                z, ldj = self.actnorm[i].inverse(z)
                log_det_j += ldj
            z_masked = self.mask[i] * z
            inp = self._concat_y(z_masked, y)
            s = self.s[i](inp) * (1.0 - self.mask[i])
            t = self.t[i](inp) * (1.0 - self.mask[i])
            z = (1.0 - self.mask[i]) * (z - t) * torch.exp(-s) + z_masked
            log_det_j -= torch.sum(s, dim=1)
        return z, log_det_j

    def g(self, z: torch.Tensor, y: torch.Tensor | None = None):
        """z -> x (forward direction)"""
        log_det_j = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        x = z
        for i in range(self.transform_count):
            x_masked = self.mask[i] * x
            inp = self._concat_y(x_masked, y)
            s = self.s[i](inp) * (1.0 - self.mask[i])
            t = self.t[i](inp) * (1.0 - self.mask[i])
            x = x_masked + (1.0 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_j += torch.sum(s, dim=1)
            if self.use_actnorm:
                x, ldj = self.actnorm[i].forward(x)
                log_det_j += ldj
            if self.permutations is not None:
                x = self.permutations[i](x)
        return x, log_det_j


class ActNorm1d(nn.Module):
    """Per-dimension affine transform with data-dependent init (Glow-style).

    forward:  y = (x + bias) * exp(log_scale)
    inverse:  x = y * exp(-log_scale) - bias

    Notes:
      - This implementation initializes on the first call in whichever direction is used first.
      - If initialized via inverse(), it normalizes the inverse output (mean=0, std=1) for that batch.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.bias = nn.Parameter(torch.zeros(1, self.dim))
        self.log_scale = nn.Parameter(torch.zeros(1, self.dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _maybe_init(self, x: torch.Tensor, *, inverse: bool):
        if bool(self.initialized.item()):
            return
        x_fp32 = x.detach().float()
        mean = x_fp32.mean(dim=0, keepdim=True)
        std = x_fp32.std(dim=0, keepdim=True).clamp_min(self.eps)
        with torch.no_grad():
            if inverse:
                # y = x*exp(-ls) - b ; want y ~ (x-mean)/std
                self.log_scale.copy_(std.log())
                self.bias.copy_(mean / std)
            else:
                # y = (x+b)*exp(ls) ; want y ~ (x-mean)/std
                self.log_scale.copy_(-std.log())
                self.bias.copy_(-mean)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor):
        self._maybe_init(x, inverse=False)
        y = (x + self.bias) * torch.exp(self.log_scale)
        ldj = self.log_scale.sum().expand(x.shape[0])
        return y, ldj

    def inverse(self, y: torch.Tensor):
        self._maybe_init(y, inverse=True)
        x = y * torch.exp(-self.log_scale) - self.bias
        ldj = (-self.log_scale.sum()).expand(y.shape[0])
        return x, ldj


class FixedPermutation1d(nn.Module):
    """Fixed (invertible) permutation over the last dimension."""

    def __init__(self, perm: torch.Tensor):
        super().__init__()
        if perm.dim() != 1:
            raise ValueError(f"perm must be 1D, got shape={tuple(perm.shape)}")
        perm = perm.to(dtype=torch.long)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(1, self.perm)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y.index_select(1, self.inv_perm)


class MollifiedUniformBoxPrior(nn.Module):
    """FlowPG 코드의 ConstrainedDistribution(BoxConstraint)와 동일한 형태의 'mollified uniform' prior.

    cv = sum(relu(|z|-bound))  (box constraint violation; inside box => 0)
    log p(z) = Normal(0,1).log_prob(cv / sigma)
    """

    def __init__(self, dim: int, *, bound: float = 1.0, sigma: float = 1e-4, aggregate: str = "sum"):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        aggregate = (aggregate or "").strip().lower()
        if aggregate not in {"sum", "max"}:
            raise ValueError("aggregate must be 'sum' or 'max'")
        self.dim = int(dim)
        self.bound = float(bound)
        self.sigma = float(sigma)
        self.aggregate = aggregate
        self.noise_prior = torch.distributions.Normal(0.0, 1.0)

    def _cv(self, z: torch.Tensor) -> torch.Tensor:
        # (B, D) -> (B,)
        violation = torch.relu(torch.abs(z) - float(self.bound))
        if self.aggregate == "max":
            return violation.max(dim=1).values
        return violation.sum(dim=1)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        cv = self._cv(z)
        return self.noise_prior.log_prob(cv / float(self.sigma))

    def sample(self, num_samples: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        z = (torch.rand(num_samples, self.dim, device=device, dtype=dtype) * 2.0 - 1.0) * float(self.bound)
        return z


class DiagModifiedUniformPrior(nn.Module):
    """NSF 쪽 `modified_uniform(Φdiff)`와 동일한 prior (Diag, independent dims).

    Per dim (x-space):
      x = u + δ,  u ~ Uniform(-1,1),  δ ~ Normal(0, σ)
      z = bound * x

    1D PDF (paper form, optional normalization):
      p(x) = Φ((1-x)/σ) - Φ((-1-x)/σ)          (unnormalized)
      p_norm(x) = 0.5 * p(x)                  (normalized)
    """

    def __init__(self, dim: int, *, bound: float = 1.0, sigma: float = 0.01, normalize_pdf: bool = False):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.dim = int(dim)
        self.bound = float(bound)
        self.sigma = float(sigma)
        self.normalize_pdf = bool(normalize_pdf)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.dim:
            raise ValueError(f"Expected z shape (B,{self.dim}), got {tuple(z.shape)}")
        b = float(self.bound)
        sigma = float(self.sigma)
        x = z / b
        b_ = (1.0 - x) / sigma
        a_ = (-1.0 - x) / sigma
        log_p = _log_ndtr_diff(b_, a_)
        if self.normalize_pdf:
            log_p = log_p - math.log(2.0)
        log_p = log_p - math.log(b)
        return log_p.sum(dim=1)

    def sample(self, num_samples: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = torch.rand(num_samples, self.dim, device=device, dtype=dtype) * 2.0 - 1.0
        delta = torch.randn_like(u) * float(self.sigma)
        x = u + delta
        z = x * float(self.bound)
        return z


def get_loader_opt(
    x_train: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    dataset = data.TensorDataset(x_train)
    loader_kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return data.DataLoader(dataset, **loader_kwargs)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    x_raw = load_tensor(Path(args.data_path))
    x_train, mean, std, data_rescale = preprocess_data(x_raw, args)

    loader = get_loader_opt(
        x_train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    flow = FlowPGRealNVP(
        dim=args.dim,
        transform_count=args.transform_count,
        hidden_size=args.hidden_size,
        conditional_dim=0,
        use_actnorm=bool(args.actnorm),
        actnorm_eps=float(args.actnorm_eps),
        permute=str(args.permute),
        permute_seed=int(args.permute_seed),
    ).to(device)
    prior_kind = (args.prior or "").strip().lower()
    if prior_kind == "box":
        prior = MollifiedUniformBoxPrior(
            dim=args.dim,
            bound=args.latent_bound,
            sigma=args.mollifier_sigma,
            aggregate=args.mollifier_aggregate,
        ).to(device)
    elif prior_kind in {"modified_uniform", "modified-uniform"}:
        if float(args.mollifier_sigma) < 1e-3:
            print(
                f"[warn] prior=modified_uniform 인데 mollifier_sigma={float(args.mollifier_sigma):g} 입니다. "
                "너무 작으면 hard-uniform처럼 동작해 학습 신호가 약해질 수 있어 보통 0.01 근처를 씁니다.",
                file=sys.stderr,
            )
        if args.mollifier_aggregate != "sum":
            print("[warn] prior=modified_uniform 에서는 --mollifier-aggregate 를 무시합니다.", file=sys.stderr)
        prior = DiagModifiedUniformPrior(
            dim=args.dim,
            bound=args.latent_bound,
            sigma=args.mollifier_sigma,
            normalize_pdf=bool(args.modified_uniform_normalize),
        ).to(device)
    else:
        raise ValueError("prior must be one of: modified_uniform|box")
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)

    use_amp = device.type == "cuda" and args.amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    base_save = Path(args.save_path) if args.save_path else None
    if base_save is None:
        print(
            "[WARN] --save-path 가 설정되지 않아 체크포인트를 저장하지 않습니다. "
            "저장을 원하면 예: --save-path sota/flowpg_realnvp_mollified.pt",
            file=sys.stderr,
        )

    def maybe_save(tag=None):
        if not base_save:
            return
        out_path = base_save if tag is None else base_save.with_name(f"{base_save.stem}_ep{tag}{base_save.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": flow.state_dict(),
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

    mean_d = mean.to(device).view(1, -1)
    std_d = std.to(device).view(1, -1).clamp_min(args.std_eps)

    log2 = math.log(2.0)
    for epoch in range(1, args.epochs + 1):
        running_nll = 0.0
        running_loss = 0.0
        for (x,) in loader:
            x = x.to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                z, log_det = flow.f(x, y=None)
                log_p = log_det + prior.log_prob(z)
                nll = (-log_p).mean()
                loss = nll
                if bool(args.take_log_again):
                    per_sample = -log_p
                    mask = per_sample > 1.0
                    per_sample = per_sample.clone()
                    per_sample[mask] = per_sample[mask].log()
                    per_sample[~mask] = per_sample[~mask] - 1.0
                    loss = per_sample.mean()

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if args.clip_grad > 0:
                    clip_grad_norm_(flow.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    clip_grad_norm_(flow.parameters(), args.clip_grad)
                opt.step()

            running_nll += float(nll.item())
            running_loss += float(loss.item())

        avg_nll = running_nll / len(loader)
        avg_loss = running_loss / len(loader)
        bpd = avg_nll / (log2 * args.dim)
        if bool(args.take_log_again):
            print(f"Epoch {epoch:03d} | NLL {avg_nll:.6f} | bits/dim {bpd:.6f} | loss {avg_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d} | NLL {avg_nll:.6f} | bits/dim {bpd:.6f}")

        if verify_enabled and (epoch % int(args.verify_every) == 0):
            flow.eval()
            with torch.no_grad():
                z = prior.sample(int(args.verify_num_samples), device=device, dtype=torch.float32)
                x_train_samp, _ = flow.g(z, y=None)
                x_raw_samp = train_to_raw(x_train_samp, mean=mean_d, std=std_d, data_rescale=float(data_rescale)).float()
                metrics = verifier.evaluate(x_raw_samp, eps=float(args.verify_eps))
                if bool(args.verify_save_samples):
                    torch.save(x_raw_samp.cpu(), verify_sample_pt)
            flow.train(True)
            line = format_verify_line(epoch, metrics, prefix="[VERIFY] ")
            print(line)
            append_line(verify_log, line)

        if args.save_every > 0 and epoch % args.save_every == 0:
            maybe_save(epoch)

    maybe_save(None)


def parse_args():
    p = argparse.ArgumentParser(description="FlowPG-style RealNVP + modified-uniform(Φdiff) prior training on 12D data.")
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt", help="Path to data of shape (N,12)")
    p.add_argument("--dim", type=int, default=12)

    # FlowPG defaults
    p.add_argument("--transform-count", type=int, default=6, help="RealNVP coupling layer count (must be even)")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--actnorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--actnorm-eps", type=float, default=1e-6)
    p.add_argument("--permute", type=str, default="flip", choices=["none", "flip", "random"])
    p.add_argument("--permute-seed", type=int, default=0, help="permute=random 일 때만 사용")

    p.add_argument("--prior", type=str, default="modified_uniform", choices=["modified_uniform", "box"])
    p.add_argument("--latent-bound", type=float, default=1.0, help="latent box half-range ([-b,b])")
    p.add_argument(
        "--mollifier-sigma",
        type=float,
        default=1e-4,
        help="prior=modified_uniform 일 때 smoothing σ, prior=box 일 때 constraint σ",
    )
    p.add_argument("--mollifier-aggregate", type=str, default="sum", choices=["sum", "max"], help="prior=box 에서만 사용")
    p.add_argument(
        "--modified-uniform-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="modified_uniform PDF에 0.5를 곱해 정규화할지 여부 (기본 False: paper form)",
    )

    # Preprocess: x_train = ((x_raw - mean)/std) / data_rescale
    p.add_argument("--preprocess", type=str, default="standardize", choices=["raw", "standardize", "minmax"])
    p.add_argument(
        "--target-max-abs",
        type=float,
        default=1.1,
        help="x_std의 global max|.|를 이 값으로 맞추도록 스케일링 (0이면 비활성화). 예: 1.1이면 일부 샘플이 box 경계에 걸려 gradient가 생김",
    )
    p.add_argument("--std-eps", type=float, default=1e-6)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip-grad", type=float, default=0.1)
    p.add_argument("--take-log-again", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-path", type=str, default="")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify-every", type=int, default=1)
    p.add_argument("--verify-num-samples", type=int, default=10000)
    p.add_argument("--verify-eps", type=float, default=1e-3)
    p.add_argument("--verify-solver-dir", type=str, default=DEFAULT_IK_SOLVER_DIR)
    p.add_argument("--verify-log", type=str, default="")
    p.add_argument("--verify-sample-pt", type=str, default="")
    p.add_argument("--verify-save-samples", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
