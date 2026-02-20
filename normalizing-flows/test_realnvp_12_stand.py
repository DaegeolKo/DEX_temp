import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from run_realnvp_12_stand import build_model, load_tensor, standardize


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

    dim = saved_args.get("dim", args.dim)
    model = build_model(
        dim=dim,
        num_layers=saved_args.get("num_layers", args.num_layers),
        hidden=saved_args.get("hidden", args.hidden),
        scale_map=saved_args.get("scale_map", args.scale_map),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x_all_raw = load_tensor(Path(args.data_path))
    mean = (mean_ckpt if mean_ckpt is not None else x_all_raw.mean(dim=0)).to(device)
    std = (std_ckpt if std_ckpt is not None else x_all_raw.std(dim=0)).to(device)
    std = std.clamp_min(args.std_eps)

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
    x_samp_std = None
    x_samp = None
    if need_samples:
        with torch.no_grad():
            if args.sample_std != 1.0:
                z = torch.randn(args.num_samples, dim, device=device) * args.sample_std
                x_samp_std = model(z)
            else:
                x_samp_std, _ = model.sample(args.num_samples)
        x_samp = destandardize(x_samp_std.cpu(), mean.cpu(), std.cpu())
        if args.sample_pt:
            pt_path = Path(args.sample_pt)
            torch.save(x_samp, pt_path)
            print(f"샘플 {x_samp.shape[0]}개(원래 스케일)를 {pt_path} 로 저장했습니다.")

    if args.sample_html:
        with torch.no_grad():
            vis_n = min(args.max_points, x_all_raw.shape[0])
            x_vis_raw = x_all_raw[:vis_n].to(device)
            x_vis_std, _, _ = standardize(x_vis_raw, mean=mean, std=std, eps=args.std_eps)
            z_vis = model.inverse(x_vis_std)
            x_rec_std = model(z_vis)
        x_rec = destandardize(x_rec_std.cpu(), mean.cpu(), std.cpu())
        x_vis = x_vis_raw.cpu()
        save_html(x_vis, x_rec, x_samp, Path(args.sample_html))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RealNVP (12D) with standardization and restore for viz.")
    p.add_argument("--ckpt", type=str, required=True, help="run_realnvp_12_stand.py로 저장한 체크포인트")
    p.add_argument(
        "--data-path",
        type=str,
        default="fk_positions_valid.pt",
        help=".pt/.pth/.npy/.npz/.csv/.txt 데이터 경로 (N,12)",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--scale-map", type=str, default="sigmoid")
    p.add_argument("--num-samples", type=int, default=2000, help="샘플링 개수 (sample_html 지정 시 사용)")
    p.add_argument("--sample-std", type=float, default=1.0, help="샘플링 시 base 가우시안 표준편차 스케일")
    p.add_argument("--max-points", type=int, default=4000, help="원본/재구성에서 시각화에 사용할 최대 포인트 수")
    p.add_argument("--std-eps", type=float, default=1e-6, help="min std for standardization")
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
