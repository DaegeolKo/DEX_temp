import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from go2_verify_utils import DEFAULT_IK_SOLVER_DIR, Go2IKVerifier, append_line, default_verify_paths, format_verify_line
from run_flowpg_realnvp_12_mollified import FlowPGRealNVP, get_loader_opt, load_tensor, preprocess_data


class StandardNormalPrior(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self._dist = torch.distributions.Normal(0.0, 1.0)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # (B, D) -> (B,)
        return self._dist.log_prob(z).sum(dim=1)


def train_to_raw(x_train: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, data_rescale: float) -> torch.Tensor:
    x_std = x_train * float(data_rescale)
    return x_std * std + mean


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
    prior = StandardNormalPrior(dim=args.dim).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)

    use_amp = device.type == "cuda" and args.amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    base_save = Path(args.save_path) if args.save_path else None
    if base_save is None:
        print(
            "[WARN] --save-path 가 설정되지 않아 체크포인트를 저장하지 않습니다. "
            "저장을 원하면 예: --save-path sota/flowpg_realnvp_gaussian.pt",
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
        running = 0.0
        for (x,) in loader:
            x = x.to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                z, log_det = flow.f(x, y=None)
                log_p = log_det + prior.log_prob(z)
                loss = -log_p.mean()

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

            running += float(loss.item())

        nll = running / len(loader)
        bpd = nll / (args.dim * log2)
        print(f"Epoch {epoch:03d} | NLL {nll:.6f} | bits/dim {bpd:.6f}")

        if verify_enabled and (epoch % int(args.verify_every) == 0):
            flow.eval()
            with torch.no_grad():
                z = torch.randn(int(args.verify_num_samples), int(args.dim), device=device)
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
    p = argparse.ArgumentParser(description="RealNVP (FlowPG architecture) + fixed N(0,I) prior training on 12D data.")
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt", help="Path to data of shape (N,12)")
    p.add_argument("--dim", type=int, default=12)

    # FlowPG-style RealNVP defaults
    p.add_argument("--transform-count", type=int, default=16, help="RealNVP coupling layer count (must be even)")
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--actnorm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--actnorm-eps", type=float, default=1e-6)
    p.add_argument("--permute", type=str, default="flip", choices=["none", "flip", "random"])
    p.add_argument("--permute-seed", type=int, default=0, help="permute=random 일 때만 사용")

    # Preprocess: x_train = ((x_raw - mean)/std) / data_rescale
    p.add_argument("--preprocess", type=str, default="standardize", choices=["raw", "standardize", "minmax"])
    p.add_argument(
        "--target-max-abs",
        type=float,
        default=0.0,
        help="x_std의 global max|.|를 이 값으로 맞추도록 스케일링 (0이면 비활성화).",
    )
    p.add_argument("--std-eps", type=float, default=1e-6)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-grad", type=float, default=1.0)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--save-path", type=str, default="sota/flowpg_realnvp_gaussian.pt")
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
