import argparse
from pathlib import Path
import math
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data

import normflows as nf


def build_model(dim: int, num_layers: int, hidden: int, num_blocks: int, num_bins: int, tail_bound: float):
    base = nf.distributions.base.DiagGaussian(dim)
    flows = []
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


def get_loader(data_path: Path, batch_size: int, shuffle: bool = True):
    x = load_tensor(data_path)
    dataset = data.TensorDataset(x)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_loader_opt(
    data_path: Path,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
):
    x = load_tensor(data_path)
    dataset = data.TensorDataset(x)
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

    loader = get_loader_opt(
        Path(args.data_path),
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

    log2 = math.log(2)
    log_disc = math.log(256)

    base_save = Path(args.save_path) if args.save_path else None
    def maybe_save(tag=None):
        if not base_save:
            return
        out_path = base_save if tag is None else base_save.with_name(f"{base_save.stem}_ep{tag}{base_save.suffix}")
        torch.save({"model": model.state_dict(), "args": vars(args)}, out_path)
        print(f"Saved checkpoint to {out_path}")

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for (x,) in loader:
            x = x.to(device, non_blocking=bool(args.pin_memory and device.type == "cuda"))
            opt.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()):
                loss = -model.log_prob(x).mean()
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
            running += loss.item()
        avg = running / len(loader)
        bpd_cont = avg / (log2 * args.dim)
        bpd_disc = (avg + log_disc * args.dim) / (log2 * args.dim)
        print(
            f"Epoch {epoch:03d} | NLL {avg:.4f} | bits/dim (cont) {bpd_cont:.4f} | bits/dim (disc) {bpd_disc:.4f}"
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            maybe_save(epoch)

    maybe_save(None)


def parse_args():
    p = argparse.ArgumentParser(description="Train NSF on 12D vectors (.pt/.npy/.npz/.csv/.txt).")
    p.add_argument(
        "--data-path",
        type=str,
        default="fk_positions_valid.pt",
        help="Path to .pt/.pth/.npy/.npz/.csv/.txt data of shape (N,12)",
    )
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clip-grad", type=float, default=5.0, help="Clip global grad norm (<=0 to disable)")
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
        default=True,
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
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
