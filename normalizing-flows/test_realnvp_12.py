import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from run_realnvp_12 import build_model


def load_tensor(pt_path: Path) -> torch.Tensor:
    ext = pt_path.suffix.lower()
    if ext in {".pt", ".pth"}:
        x = torch.load(pt_path)
    elif ext == ".npy":
        x = torch.from_numpy(np.load(pt_path))
    elif ext == ".npz":
        npz = np.load(pt_path)
        if "arr_0" in npz.files:
            key = "arr_0"
        elif len(npz.files) == 1:
            key = npz.files[0]
        else:
            raise ValueError(f"NPZ 파일에 하나의 배열만 있어야 합니다: {npz.files}")
        x = torch.from_numpy(npz[key])
    elif ext in {".csv", ".txt"}:
        delimiter = "," if ext == ".csv" else None
        x = torch.from_numpy(np.loadtxt(pt_path, delimiter=delimiter))
    else:
        raise ValueError(f"지원하지 않는 데이터 확장자: {ext}")

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if x.dim() != 2:
        x = x.view(x.shape[0], -1)
    return x


def get_loader(x: torch.Tensor, batch_size: int):
    dataset = data.TensorDataset(x)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        saved_args = ckpt.get("args", {})
    else:
        state_dict = ckpt
        saved_args = {}
    return state_dict, saved_args


def save_html(
    x_orig: torch.Tensor,
    x_rec: torch.Tensor,
    x_sample: torch.Tensor,
    out_path: Path,
    axis_ranges=None,
    outliers=None,
    enabled_sets=None,
):
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise SystemExit("plotly가 설치되어 있지 않습니다. `pip install plotly` 후 다시 실행하세요.") from e

    # 네 개의 발끝 좌표 (012, 345, 678, 9 10 11)
    feet = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (9, 10, 11),
    ]

    fig = go.Figure()
    enabled_sets = enabled_sets or {"orig", "rec", "sample"}
    traces = []
    if "orig" in enabled_sets:
        traces.append(("원본", x_orig, "blue", "circle", "orig"))
    if "rec" in enabled_sets:
        traces.append(("재구성", x_rec, "orange", "square", "rec"))
    if "sample" in enabled_sets:
        traces.append(("샘플", x_sample, "green", "diamond", "sample"))
    for name, data_tensor, color, symbol, _key in traces:
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
            # 축 범위 계산에 잡히지 않을 수 있는 극단값을 별도 trace로 표시
            ext_idx = np.unique(np.concatenate([xyz.argmin(axis=0), xyz.argmax(axis=0)]))
            ext_xyz = xyz[ext_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=ext_xyz[:, 0],
                    y=ext_xyz[:, 1],
                    z=ext_xyz[:, 2],
                    mode="markers",
                    name=f"{name} foot{idx} extremes",
                    marker=dict(size=4, color=color, symbol="x", opacity=1.0),
                    legendgroup=name,
                    showlegend=False,
                )
            )

    # outlier들을 별도 표시해서 decimation에서 빠지지 않게 함
    if outliers:
        for name, data_tensor, color, key in outliers:
            if key not in enabled_sets:
                continue
            if data_tensor is None or data_tensor.numel() == 0:
                continue
            for idx, (a, b, c) in enumerate(feet):
                xyz = data_tensor[:, [a, b, c]].cpu().numpy()
                fig.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode="markers",
                        name=f"{name} outliers foot{idx}",
                        marker=dict(size=6, color=color, symbol="cross", opacity=1.0),
                        legendgroup=name,
                        showlegend=True,
                    )
                )

    scene_cfg = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
    if axis_ranges:
        scene_cfg["xaxis"] = dict(title="x", range=axis_ranges[0])
        scene_cfg["yaxis"] = dict(title="y", range=axis_ranges[1])
        scene_cfg["zaxis"] = dict(title="z", range=axis_ranges[2])

    fig.update_layout(
        title="12D NF: 발끝 좌표 (원본/재구성/샘플)",
        scene=scene_cfg,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"3D 발끝 시각화를 {out_path} 로 저장했습니다. 브라우저에서 열어 회전/확대 가능합니다.")


def compute_axis_ranges(*tensors):
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    flat = torch.cat(tensors, dim=0).view(-1, 3)
    xyz_min = flat.min(dim=0).values
    xyz_max = flat.max(dim=0).values
    padding = float((xyz_max - xyz_min).max()) * 0.1 + 1e-6
    return [
        (float(xyz_min[i] - padding), float(xyz_max[i] + padding)) for i in range(3)
    ]


def find_outliers(tensor: torch.Tensor, thresh: float):
    if tensor is None:
        return None
    mask = tensor.abs().max(dim=1).values > thresh
    return tensor[mask] if mask.any() else None


def describe(name: str, tensor: torch.Tensor):
    if tensor is None:
        print(f"{name}: 없음")
        return
    flat = tensor.view(-1)
    print(f"{name}: shape={tuple(tensor.shape)} min={flat.min().item():.4f} max={flat.max().item():.4f}")


def evaluate(args):
    plot_sets = {s.strip() for s in args.plot_sets.split(",") if s.strip()}
    if not plot_sets:
        plot_sets = {"orig", "rec", "sample"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, saved_args = load_checkpoint(args.ckpt, device)

    dim = saved_args.get("dim", args.dim)
    model = build_model(
        dim=dim,
        num_layers=saved_args.get("num_layers", args.num_layers),
        hidden=saved_args.get("hidden", args.hidden),
        scale_map=saved_args.get("scale_map", args.scale_map),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    x_all = load_tensor(Path(args.data_path))
    loader = get_loader(x_all, args.batch_size)

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
        f"Test NLL: {avg_nll:.4f} | bits/dim (cont): {bpd_cont:.4f} | bits/dim (disc): {bpd_disc:.4f}"
    )

    need_samples = bool(args.sample_html or args.sample_pt or "sample" in plot_sets)
    x_samp = None
    if need_samples:
        with torch.no_grad():
            if args.sample_std != 1.0:
                z = torch.randn(args.num_samples, dim, device=device) * args.sample_std
                x_samp = model(z)
            else:
                x_samp, _ = model.sample(args.num_samples)
        if args.sample_pt and x_samp is not None:
            pt_path = Path(args.sample_pt)
            torch.save(x_samp.cpu(), pt_path)
            print(f"샘플 {x_samp.shape[0]}개를 {pt_path} 로 저장했습니다.")

    if args.sample_html:
        model.eval()
        with torch.no_grad():
            # 시각화에 사용할 부분 샘플
            vis_n = min(args.max_points, x_all.shape[0])
            x_vis = x_all[:vis_n].to(device)
            z_vis = model.inverse(x_vis)
            x_rec = model(z_vis)
        x_vis_cpu = x_vis.cpu()
        x_rec_cpu = x_rec.cpu()
        x_samp_cpu = x_samp.cpu() if x_samp is not None else None
        describe("원본", x_vis_cpu)
        describe("재구성", x_rec_cpu)
        describe("샘플", x_samp_cpu)
        tensors_for_range = []
        if "orig" in plot_sets:
            tensors_for_range.append(x_vis_cpu)
        if "rec" in plot_sets:
            tensors_for_range.append(x_rec_cpu)
        if "sample" in plot_sets and x_samp_cpu is not None:
            tensors_for_range.append(x_samp_cpu)
        ranges = compute_axis_ranges(*tensors_for_range)
        outliers = [
            ("원본", find_outliers(x_vis_cpu, args.outlier_thresh), "red", "orig"),
            ("재구성", find_outliers(x_rec_cpu, args.outlier_thresh), "purple", "rec"),
            ("샘플", find_outliers(x_samp_cpu, args.outlier_thresh) if x_samp_cpu is not None else None, "black", "sample"),
        ]
        for n, t, _c, _key in outliers:
            if t is not None:
                print(f"{n} outliers (>|{args.outlier_thresh}|): {t.shape[0]}")
        save_html(
            x_vis,
            x_rec,
            x_samp,
            Path(args.sample_html),
            axis_ranges=ranges,
            outliers=outliers,
            enabled_sets=plot_sets,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RealNVP (12D) and optionally export interactive samples.")
    p.add_argument("--ckpt", type=str, required=True, help="run_realnvp_12.py로 저장한 체크포인트")
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
    p.add_argument(
        "--plot-sets",
        type=str,
        default="orig,rec,sample",
        help="시각화에 포함할 집합 콤마구분 (orig, rec, sample)",
    )
    p.add_argument("--outlier-thresh", type=float, default=5.0, help="|값|이 이보다 크면 outlier로 표시")
    p.add_argument(
        "--sample-pt",
        type=str,
        default="",
        help="샘플로 생성한 좌표를 저장할 .pt 파일 경로 (지정 시 num_samples 사용)",
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
