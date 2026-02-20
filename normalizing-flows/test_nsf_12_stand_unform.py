import argparse
import math
from pathlib import Path

import torch
import torch.utils.data as data

from nsf_12_stand_unform import build_model, load_tensor, standardize


def get_loader(x: torch.Tensor, batch_size: int):
    dataset = data.TensorDataset(x)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"체크포인트 포맷이 예상과 다릅니다: {path} (dict with key 'model' 기대)")
    state_dict = ckpt["model"]
    saved_args = ckpt.get("args", {})
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    data_rescale = ckpt.get("data_rescale", saved_args.get("data_rescale", 1.0))
    return state_dict, saved_args, mean, std, float(data_rescale)


def destandardize(x_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return x_std * std + mean


def train_to_raw(
    x_train: torch.Tensor,
    *,
    data_rescale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
):
    x_std = x_train * float(data_rescale)
    return destandardize(x_std, mean, std)


def raw_to_train(
    x_raw: torch.Tensor,
    *,
    data_rescale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    std_eps: float,
):
    x_std, _, _ = standardize(x_raw, mean=mean, std=std, eps=std_eps)
    x_train = x_std / float(data_rescale)
    return x_train


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
        title="12D NF (uniform/modified_uniform, 표준화 복원): 발끝 좌표 (원본/재구성/샘플)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"3D 발끝 시각화를 {out_path} 로 저장했습니다. 브라우저에서 열어 회전/확대 가능합니다.")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, saved_args, mean_ckpt, std_ckpt, data_rescale = load_checkpoint(args.ckpt, device)

    dim = int(saved_args.get("dim", args.dim))
    base = str(saved_args.get("base", "unknown"))
    print(f"[INFO] ckpt base={base} dim={dim} data_rescale={data_rescale:g}")

    model = build_model(
        dim=dim,
        num_layers=int(saved_args.get("num_layers", args.num_layers)),
        hidden=int(saved_args.get("hidden", args.hidden)),
        num_blocks=int(saved_args.get("num_blocks", args.num_blocks)),
        num_bins=int(saved_args.get("num_bins", args.num_bins)),
        tail_bound=float(saved_args.get("tail_bound", args.tail_bound)),
        mixing=str(saved_args.get("mixing", args.mixing)),
        mixing_use_lu=bool(saved_args.get("mixing_use_lu", args.mixing_use_lu)),
        mixing_seed=int(saved_args.get("mixing_seed", args.mixing_seed)),
        base=str(saved_args.get("base", args.base)),
        uniform_bound=float(saved_args.get("uniform_bound", args.uniform_bound)),
        modified_uniform_sigma=float(saved_args.get("modified_uniform_sigma", args.modified_uniform_sigma)),
        modified_uniform_normalize=bool(saved_args.get("modified_uniform_normalize", args.modified_uniform_normalize)),
        trainable_base=bool(saved_args.get("trainable_base", args.trainable_base)),
        uniform_atanh=bool(saved_args.get("uniform_atanh", args.uniform_atanh)),
        uniform_atanh_eps=float(saved_args.get("uniform_atanh_eps", args.uniform_atanh_eps)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if mean_ckpt is None or std_ckpt is None:
        raise ValueError(
            "이 스크립트는 체크포인트에 mean/std가 있어야 합니다. "
            "nsf_12_stand_unform.py로 학습한 ckpt를 사용하세요."
        )
    mean = mean_ckpt.to(device).view(1, -1)
    std = std_ckpt.to(device).view(1, -1).clamp_min(args.std_eps)

    if args.eval_nll:
        x_all_raw = load_tensor(Path(args.data_path))
        if args.eval_max > 0:
            x_all_raw = x_all_raw[: args.eval_max]
        x_all_train = raw_to_train(
            x_all_raw.to(device),
            data_rescale=data_rescale,
            mean=mean,
            std=std,
            std_eps=args.std_eps,
        )
        loader = get_loader(x_all_train, args.batch_size)

        total_nll, total = 0.0, 0
        with torch.no_grad():
            for (x_train,) in loader:
                x_train = x_train.to(device)
                nll = -model.log_prob(x_train)
                total_nll += nll.sum().item()
                total += x_train.size(0)

        avg_nll = total_nll / total
        log2 = math.log(2)
        bpd_cont = avg_nll / (log2 * dim)
        bpd_disc = (avg_nll + math.log(256) * dim) / (log2 * dim)
        print(
            f"Test NLL (train space): {avg_nll:.4f} | bits/dim (cont): {bpd_cont:.4f} | bits/dim (disc): {bpd_disc:.4f}"
        )

    need_samples = bool(args.sample_pt or args.sample_html)
    x_samp_raw = None
    if need_samples:
        with torch.no_grad():
            x_samp_train, _ = model.sample(args.num_samples)
            x_samp_raw = train_to_raw(x_samp_train, data_rescale=data_rescale, mean=mean, std=std).cpu()

        if args.sample_pt:
            pt_path = Path(args.sample_pt)
            torch.save(x_samp_raw, pt_path)
            print(f"샘플 {x_samp_raw.shape[0]}개(원 스케일)를 {pt_path} 로 저장했습니다.")

    if args.sample_html:
        x_all_raw = load_tensor(Path(args.data_path))
        with torch.no_grad():
            vis_n = min(args.max_points, x_all_raw.shape[0])
            x_vis_raw = x_all_raw[:vis_n].to(device)
            x_vis_train = raw_to_train(
                x_vis_raw,
                data_rescale=data_rescale,
                mean=mean,
                std=std,
                std_eps=args.std_eps,
            )
            z_vis = model.inverse(x_vis_train)
            x_rec_train = model(z_vis)
            x_rec_raw = train_to_raw(x_rec_train, data_rescale=data_rescale, mean=mean, std=std).cpu()

        save_html(x_vis_raw.cpu(), x_rec_raw, x_samp_raw, Path(args.sample_html))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate NSF checkpoints trained with nsf_12_stand_unform.py (uniform/modified_uniform)."
    )
    p.add_argument("--ckpt", type=str, required=True, help="nsf_12_stand_unform.py로 저장한 체크포인트")
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt", help="(N,12) 데이터 경로")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-samples", type=int, default=2000, help="샘플링 개수")
    p.add_argument("--max-points", type=int, default=4000, help="원본/재구성 시각화 포인트 수")
    p.add_argument("--std-eps", type=float, default=1e-6, help="min std for standardization")
    p.add_argument(
        "--eval-nll",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="데이터셋 전체(또는 eval_max)로 NLL 평가까지 수행할지 여부 (기본 False: 샘플만 빠르게 뽑기)",
    )
    p.add_argument("--eval-max", type=int, default=0, help="eval_nll=True일 때 NLL 평가에 사용할 최대 샘플 수 (0이면 전체)")
    p.add_argument("--sample-pt", type=str, default="", help="샘플을 저장할 .pt 파일 경로")
    p.add_argument("--sample-html", type=str, default="", help="원본/재구성/샘플 3D plot(html) 저장 경로")

    # Fallback defaults (ckpt args가 없을 때만 사용)
    p.add_argument("--dim", type=int, default=12)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--tail-bound", type=float, default=3.0)
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
    p.add_argument("--base", type=str, default="uniform")
    p.add_argument("--uniform-bound", type=float, default=1.0)
    p.add_argument("--modified-uniform-sigma", type=float, default=0.01)
    p.add_argument("--modified-uniform-normalize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--trainable-base", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--uniform-atanh", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uniform-atanh-eps", type=float, default=1e-6)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
