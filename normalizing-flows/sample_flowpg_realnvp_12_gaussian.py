import argparse
from pathlib import Path

import torch

from run_flowpg_realnvp_12_gaussian import FlowPGRealNVP


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"체크포인트 포맷이 예상과 다릅니다: {path} (dict with key 'model' 기대)")
    args = ckpt.get("args", {})
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    data_rescale = float(ckpt.get("data_rescale", args.get("data_rescale", 1.0)))
    if mean is None or std is None:
        raise ValueError("체크포인트에 mean/std가 없습니다. run_flowpg_realnvp_12_gaussian.py로 학습한 ckpt를 사용하세요.")
    return ckpt["model"], args, mean, std, data_rescale


def train_to_raw(x_train: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, data_rescale: float):
    x_std = x_train * float(data_rescale)
    return x_std * std + mean


def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state_dict, saved_args, mean_ckpt, std_ckpt, data_rescale = load_checkpoint(args.ckpt, device)

    dim = int(saved_args.get("dim", 12))
    transform_count = int(saved_args.get("transform_count", 16))
    hidden_size = int(saved_args.get("hidden_size", 512))
    use_actnorm = bool(saved_args.get("actnorm", False))
    actnorm_eps = float(saved_args.get("actnorm_eps", 1e-6))
    permute = str(saved_args.get("permute", "none"))
    permute_seed = int(saved_args.get("permute_seed", 0))

    torch.manual_seed(int(args.seed))

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

    with torch.no_grad():
        z = torch.randn(args.num_samples, dim, device=device) * float(args.z_scale)
        x_train, _ = flow.g(z, y=None)
        x_raw = train_to_raw(x_train, mean=mean, std=std, data_rescale=data_rescale).cpu()

    out_path = Path(args.sample_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x_raw, out_path)
    print(f"샘플 {x_raw.shape[0]}개(원 스케일)를 {out_path} 로 저장했습니다.")


def parse_args():
    p = argparse.ArgumentParser(description="Sampling-only: RealNVP (FlowPG arch) + fixed N(0,I) prior ckpt.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-pt", type=str, required=True)
    p.add_argument("--z-scale", type=float, default=1.0, help="z ~ N(0, z_scale^2 I)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--std-eps", type=float, default=1e-6)
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


if __name__ == "__main__":
    sample(parse_args())
