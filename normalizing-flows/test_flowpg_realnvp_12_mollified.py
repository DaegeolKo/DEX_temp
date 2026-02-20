import argparse
import math
from pathlib import Path

import torch
import torch.utils.data as data

from run_flowpg_realnvp_12_mollified import (
    DiagModifiedUniformPrior,
    FlowPGRealNVP,
    MollifiedUniformBoxPrior,
    affine_normalize,
    load_tensor,
)


def get_loader(x: torch.Tensor, batch_size: int):
    dataset = data.TensorDataset(x)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"체크포인트 포맷이 예상과 다릅니다: {path} (dict with key 'model' 기대)")
    args = ckpt.get("args", {})
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    data_rescale = float(ckpt.get("data_rescale", args.get("data_rescale", 1.0)))
    if mean is None or std is None:
        raise ValueError("체크포인트에 mean/std가 없습니다. run_flowpg_realnvp_12_mollified.py로 학습한 ckpt를 사용하세요.")
    return ckpt["model"], args, mean, std, data_rescale


def train_to_raw(x_train: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, data_rescale: float):
    x_std = x_train * float(data_rescale)
    return x_std * std + mean


def raw_to_train(x_raw: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, data_rescale: float, std_eps: float):
    x_std, _, _ = affine_normalize(x_raw, mean=mean, std=std, eps=std_eps)
    return x_std / float(data_rescale)


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
        title="FlowPG RealNVP: 발끝 좌표 (원본/재구성/샘플)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"3D 발끝 시각화를 {out_path} 로 저장했습니다. 브라우저에서 열어 회전/확대 가능합니다.")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, saved_args, mean_ckpt, std_ckpt, data_rescale = load_checkpoint(args.ckpt, device)

    dim = int(saved_args.get("dim", 12))
    transform_count = int(saved_args.get("transform_count", 6))
    hidden_size = int(saved_args.get("hidden_size", 256))
    latent_bound = float(saved_args.get("latent_bound", 1.0))
    mollifier_sigma = float(saved_args.get("mollifier_sigma", 1e-4))
    mollifier_aggregate = str(saved_args.get("mollifier_aggregate", "sum"))
    prior_kind = str(saved_args.get("prior", "box")).strip().lower()
    modified_uniform_normalize = bool(saved_args.get("modified_uniform_normalize", False))
    use_actnorm = bool(saved_args.get("actnorm", False))
    actnorm_eps = float(saved_args.get("actnorm_eps", 1e-6))
    permute = str(saved_args.get("permute", "none"))
    permute_seed = int(saved_args.get("permute_seed", 0))

    flow = FlowPGRealNVP(
        dim=dim,
        transform_count=transform_count,
        hidden_size=hidden_size,
        conditional_dim=0,
        use_actnorm=use_actnorm,
        actnorm_eps=actnorm_eps,
        permute=permute,
        permute_seed=permute_seed,
    ).to(device)
    flow.load_state_dict(state_dict)
    flow.eval()

    mean = mean_ckpt.to(device).view(1, -1)
    std = std_ckpt.to(device).view(1, -1).clamp_min(args.std_eps)

    if args.eval_nll or args.sample_html:
        x_all_raw = load_tensor(Path(args.data_path))
        if args.eval_max > 0:
            x_all_raw = x_all_raw[: args.eval_max]

    if args.eval_nll:
        if prior_kind == "box":
            prior = MollifiedUniformBoxPrior(
                dim=dim,
                bound=latent_bound,
                sigma=mollifier_sigma,
                aggregate=mollifier_aggregate,
            ).to(device)
        else:
            prior = DiagModifiedUniformPrior(
                dim=dim,
                bound=latent_bound,
                sigma=mollifier_sigma,
                normalize_pdf=modified_uniform_normalize,
            ).to(device)
        x_all_train = raw_to_train(
            x_all_raw.to(device),
            mean=mean,
            std=std,
            data_rescale=data_rescale,
            std_eps=args.std_eps,
        )
        loader = get_loader(x_all_train, args.batch_size)
        total_nll, total = 0.0, 0
        with torch.no_grad():
            for (x_train,) in loader:
                x_train = x_train.to(device)
                z, log_det = flow.f(x_train, y=None)
                nll = -(log_det + prior.log_prob(z))
                total_nll += nll.sum().item()
                total += x_train.size(0)
        avg_nll = total_nll / total
        log2 = math.log(2)
        bpd_cont = avg_nll / (log2 * dim)
        bpd_disc = (avg_nll + math.log(256) * dim) / (log2 * dim)
        print(
            f"Test NLL (train space): {avg_nll:.4f} | bits/dim (cont): {bpd_cont:.4f} | bits/dim (disc): {bpd_disc:.4f}"
        )

    x_samp_raw = None
    if args.sample_pt or args.sample_html:
        with torch.no_grad():
            if prior_kind == "box":
                z = (torch.rand(args.num_samples, dim, device=device) * 2.0 - 1.0) * latent_bound
            else:
                z = DiagModifiedUniformPrior(
                    dim=dim,
                    bound=latent_bound,
                    sigma=mollifier_sigma,
                    normalize_pdf=modified_uniform_normalize,
                ).to(device).sample(args.num_samples, device=device, dtype=torch.float32)
            x_samp_train, _ = flow.g(z, y=None)
            x_samp_raw = train_to_raw(x_samp_train, mean=mean, std=std, data_rescale=data_rescale).cpu()
        if args.sample_pt:
            out_path = Path(args.sample_pt)
            torch.save(x_samp_raw, out_path)
            print(f"샘플 {x_samp_raw.shape[0]}개(원 스케일)를 {out_path} 로 저장했습니다.")

    if args.sample_html:
        with torch.no_grad():
            vis_n = min(args.max_points, x_all_raw.shape[0])
            x_vis_raw = x_all_raw[:vis_n].to(device)
            x_vis_train = raw_to_train(
                x_vis_raw,
                mean=mean,
                std=std,
                data_rescale=data_rescale,
                std_eps=args.std_eps,
            )
            z_vis, _ = flow.f(x_vis_train, y=None)
            x_rec_train, _ = flow.g(z_vis, y=None)
            x_rec_raw = train_to_raw(x_rec_train, mean=mean, std=std, data_rescale=data_rescale).cpu()
        save_html(x_vis_raw.cpu(), x_rec_raw, x_samp_raw, Path(args.sample_html))


def parse_args():
    p = argparse.ArgumentParser(description="Test FlowPG-style RealNVP + MollifiedUniform(Box).")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-pt", type=str, default="")
    p.add_argument("--sample-html", type=str, default="")

    p.add_argument(
        "--data-path",
        type=str,
        default="fk_positions_valid.pt",
        help="--eval-nll 또는 --sample-html 일 때만 필요",
    )
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--max-points", type=int, default=4000)
    p.add_argument("--std-eps", type=float, default=1e-6)

    p.add_argument("--eval-nll", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval-max", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
