import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from run_nsf_12_stand import build_model, load_tensor, standardize


def _parse_sigma_ranges(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return [("all", None)]

    out = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"all", "full", "*"}:
            out.append(("all", None))
            continue
        bound = float(token)
        if bound <= 0:
            raise ValueError(f"sigma 범위는 양수여야 합니다: {raw!r}")
        if float(int(bound)) == bound:
            label = f"pm{int(bound)}"
        else:
            label = f"pm{str(bound).replace('.', 'p')}"
        out.append((label, bound))

    if not out:
        out = [("all", None)]
    return out


def _add_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    return path.with_name(f"{path.name}_{suffix}")


def _sample_eps_truncated(
    num_samples: int,
    dim: int,
    device: torch.device,
    *,
    bound: float | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if bound is None:
        return torch.randn(num_samples, dim, device=device, dtype=dtype)

    compute_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    sqrt2 = math.sqrt(2.0)
    low = 0.5 * (1.0 + math.erf(-bound / sqrt2))
    high = 0.5 * (1.0 + math.erf(bound / sqrt2))
    u = torch.rand((num_samples, dim), device=device, dtype=compute_dtype)
    u = low + (high - low) * u
    eps = torch.finfo(compute_dtype).eps
    u = u.clamp(min=eps, max=1.0 - eps)
    z = sqrt2 * torch.erfinv(2.0 * u - 1.0)
    return z.to(dtype)


def _sample_from_base(
    model,
    num_samples: int,
    dim: int,
    device: torch.device,
    *,
    sigma_bound: float | None,
    sample_std: float,
) -> torch.Tensor:
    loc = model.q0.loc.to(device)
    scale = torch.exp(model.q0.log_scale.to(device)) * float(sample_std)
    eps = _sample_eps_truncated(
        num_samples,
        dim,
        device,
        bound=sigma_bound,
        dtype=loc.dtype,
    )
    z = loc + scale * eps
    return model(z)


def get_loader(x: torch.Tensor, batch_size: int):
    dataset = data.TensorDataset(x)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        saved_args = ckpt.get("args", {})
        mean = ckpt.get("mean")
        std = ckpt.get("std")
    else:
        state_dict = ckpt
        saved_args = {}
        mean = None
        std = None
    return state_dict, saved_args, mean, std


def destandardize(x_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return x_std * std + mean


def save_html(x_orig: torch.Tensor, x_rec: torch.Tensor, x_sample: torch.Tensor, out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit("plotly가 설치되어 있지 않습니다. `pip install plotly` 후 다시 실행하세요.") from e

    feet = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (9, 10, 11),
    ]

    fig = go.Figure()
    traces = [
        ("원본", x_orig, "blue", "circle"),
        ("재구성", x_rec, "orange", "square"),
        ("샘플", x_sample, "green", "diamond"),
    ]
    for name, data_tensor, color, symbol in traces:
        for idx, (a, b, c) in enumerate(feet):
            xyz = data_tensor[:, [a, b, c]].cpu().numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    name=f"{name} foot{idx}",
                    marker=dict(size=3, color=color, symbol=symbol, opacity=0.6),
                    legendgroup=name,
                )
            )

    fig.update_layout(
        title="12D NF (표준화 복원): 발끝 좌표 (원본/재구성/샘플)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"3D 발끝 시각화를 {out_path} 로 저장했습니다. 브라우저에서 열어 회전/확대 가능합니다.")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, saved_args, mean_ckpt, std_ckpt = load_checkpoint(args.ckpt, device)

    sigma_specs = _parse_sigma_ranges(args.sigma_ranges)

    dim = saved_args.get("dim", args.dim)
    model = build_model(
        dim=dim,
        num_layers=saved_args.get("num_layers", args.num_layers),
        hidden=saved_args.get("hidden", args.hidden),
        num_blocks=saved_args.get("num_blocks", args.num_blocks),
        num_bins=saved_args.get("num_bins", args.num_bins),
        tail_bound=saved_args.get("tail_bound", args.tail_bound),
        trainable_base=bool(saved_args.get("trainable_base", args.trainable_base)),
        actnorm=bool(saved_args.get("actnorm", args.actnorm)),
        mixing=str(saved_args.get("mixing", args.mixing)),
        mixing_use_lu=bool(saved_args.get("mixing_use_lu", args.mixing_use_lu)),
        mixing_seed=int(saved_args.get("mixing_seed", args.mixing_seed)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # NOTE: fk_positions_valid.pt 는 매우 커서(약 1천만 샘플) NLL 평가를 항상 돌리면 시간이 오래 걸립니다.
    # 기본 동작은 "샘플링만"이고, NLL 평가는 --eval-nll 로 켤 때만 수행합니다.
    x_all_raw = None
    if mean_ckpt is not None and std_ckpt is not None:
        mean = mean_ckpt.to(device)
        std = std_ckpt.to(device)
    else:
        # 오래된 ckpt 등 mean/std가 없으면 data에서 추정(느림).
        x_all_raw = load_tensor(Path(args.data_path))
        mean = x_all_raw.mean(dim=0).to(device)
        std = x_all_raw.std(dim=0).to(device)
    std = std.clamp_min(args.std_eps)

    if args.eval_nll:
        if x_all_raw is None:
            x_all_raw = load_tensor(Path(args.data_path))
        if args.eval_max > 0:
            x_all_raw = x_all_raw[: args.eval_max]

        x_all_std, _, _ = standardize(x_all_raw.to(device), mean=mean, std=std, eps=args.std_eps)
        loader = get_loader(x_all_std, args.batch_size)

        total_nll, total = 0.0, 0
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(device)
                nll = -model.log_prob(x)
                total_nll += nll.sum().item()
                total += x.size(0)

        avg_nll = total_nll / total
        log2 = math.log(2)
        bpd_cont = avg_nll / (log2 * dim)
        bpd_disc = (avg_nll + math.log(256) * dim) / (log2 * dim)
        print(
            f"Test NLL (std space): {avg_nll:.4f} | bits/dim (cont): {bpd_cont:.4f} | bits/dim (disc): {bpd_disc:.4f}"
        )

    need_samples = bool(args.sample_html or args.sample_pt)
    x_samps = {}
    if need_samples:
        with torch.no_grad():
            for label, bound in sigma_specs:
                x_samp_std = _sample_from_base(
                    model,
                    args.num_samples,
                    dim,
                    device,
                    sigma_bound=bound,
                    sample_std=args.sample_std,
                )
                x_samps[label] = destandardize(x_samp_std.cpu(), mean.cpu(), std.cpu())

        if args.sample_pt:
            base_pt = Path(args.sample_pt)
            multi = len(sigma_specs) > 1
            for label, _bound in sigma_specs:
                pt_path = _add_suffix(base_pt, label if multi else "")
                torch.save(x_samps[label], pt_path)
                print(f"샘플 {x_samps[label].shape[0]}개({label}, 원래 스케일)를 {pt_path} 로 저장했습니다.")

    if args.sample_html:
        if x_all_raw is None:
            x_all_raw = load_tensor(Path(args.data_path))
        with torch.no_grad():
            vis_n = min(args.max_points, x_all_raw.shape[0])
            x_vis_raw = x_all_raw[:vis_n].to(device)
            x_vis_std, _, _ = standardize(x_vis_raw, mean=mean, std=std, eps=args.std_eps)
            z_vis = model.inverse(x_vis_std)
            x_rec_std = model(z_vis)
        x_rec = destandardize(x_rec_std.cpu(), mean.cpu(), std.cpu())
        x_vis = x_vis_raw.cpu()

        base_html = Path(args.sample_html)
        multi = len(sigma_specs) > 1
        for label, _bound in sigma_specs:
            html_path = _add_suffix(base_html, label if multi else "")
            save_html(x_vis, x_rec, x_samps.get(label), html_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NSF (12D) with standardization and restore for viz.")
    p.add_argument("--ckpt", type=str, required=True, help="run_nsf_12_stand.py로 저장한 체크포인트")
    p.add_argument(
        "--data-path",
        type=str,
        default="fk_positions_valid.pt",
        help=".pt/.pth/.npy/.npz/.csv/.txt 데이터 경로 (N,12). --eval-nll 또는 --sample-html 또는 ckpt에 mean/std가 없을 때만 사용",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=3.0)
    p.add_argument(
        "--trainable-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ckpt에 값이 없을 때만 사용: base DiagGaussian(q0) 학습 여부",
    )
    p.add_argument(
        "--actnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ckpt에 값이 없을 때만 사용: ActNorm 적용 여부",
    )
    p.add_argument(
        "--mixing",
        type=str,
        default="none",
        choices=["none", "permute", "inv_affine", "inv_1x1conv"],
        help="ckpt에 값이 없을 때만 사용: Coupling layer 사이 mixing",
    )
    p.add_argument(
        "--mixing-use-lu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ckpt에 값이 없을 때만 사용: inv_affine/inv_1x1conv에서 LU 파라미터화 사용",
    )
    p.add_argument("--mixing-seed", type=int, default=0, help="ckpt에 값이 없을 때만 사용: permute seed")
    p.add_argument("--num-samples", type=int, default=2000, help="샘플링 개수 (sample_html 지정 시 사용)")
    p.add_argument("--sample-std", type=float, default=1.0, help="샘플링 시 base 가우시안 표준편차 스케일")
    p.add_argument(
        "--sigma-ranges",
        type=str,
        default="all",
        help="base 가우시안에서 샘플링 시 제한할 ±kσ (콤마구분: 1,2,3,all). 여러 개면 sample_pt/sample_html에 suffix를 붙여 각각 저장",
    )
    p.add_argument("--max-points", type=int, default=4000, help="원본/재구성에서 시각화에 사용할 최대 포인트 수")
    p.add_argument("--std-eps", type=float, default=1e-6, help="min std for standardization")
    p.add_argument(
        "--eval-nll",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="데이터셋 전체(또는 eval_max)로 NLL 평가까지 수행할지 여부 (기본 False: 샘플만 빠르게 뽑기)",
    )
    p.add_argument("--eval-max", type=int, default=0, help="eval_nll=True일 때 NLL 평가에 사용할 최대 샘플 수 (0이면 전체)")
    p.add_argument(
        "--sample-pt",
        type=str,
        default="",
        help="샘플로 생성한 좌표(원 스케일)를 저장할 .pt 파일 경로",
    )
    p.add_argument(
        "--sample-html",
        type=str,
        default="",
        help="인터랙티브 3D 샘플 플롯을 저장할 HTML 경로 (미지정 시 샘플 생성 안 함)",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
