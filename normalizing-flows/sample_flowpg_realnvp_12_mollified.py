import argparse
from pathlib import Path

import torch

from run_flowpg_realnvp_12_mollified import DiagModifiedUniformPrior, FlowPGRealNVP, MollifiedUniformBoxPrior


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


def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state_dict, saved_args, mean_ckpt, std_ckpt, data_rescale = load_checkpoint(args.ckpt, device)

    dim = int(saved_args.get("dim", 12))
    transform_count = int(saved_args.get("transform_count", 6))
    hidden_size = int(saved_args.get("hidden_size", 256))
    use_actnorm = bool(saved_args.get("actnorm", False))
    actnorm_eps = float(saved_args.get("actnorm_eps", 1e-6))
    permute = str(saved_args.get("permute", "none"))
    permute_seed = int(saved_args.get("permute_seed", 0))
    latent_bound = float(saved_args.get("latent_bound", 1.0))
    if args.latent_bound > 0:
        latent_bound = float(args.latent_bound)
    mollifier_sigma = float(saved_args.get("mollifier_sigma", 1e-4))
    if args.mollifier_sigma > 0:
        mollifier_sigma = float(args.mollifier_sigma)
    prior_kind = str(saved_args.get("prior", "modified_uniform")).strip().lower()
    modified_uniform_normalize = bool(saved_args.get("modified_uniform_normalize", False))

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
        if prior_kind == "box":
            prior = MollifiedUniformBoxPrior(dim=dim, bound=latent_bound, sigma=mollifier_sigma, aggregate="sum").to(device)
        else:
            prior = DiagModifiedUniformPrior(
                dim=dim,
                bound=latent_bound,
                sigma=mollifier_sigma,
                normalize_pdf=modified_uniform_normalize,
            ).to(device)
        z = prior.sample(args.num_samples, device=device, dtype=torch.float32)
        x_train, _ = flow.g(z, y=None)
        x_raw = train_to_raw(x_train, mean=mean, std=std, data_rescale=data_rescale).cpu()

    out_path = Path(args.sample_pt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x_raw, out_path)
    print(f"샘플 {x_raw.shape[0]}개(원 스케일)를 {out_path} 로 저장했습니다.")


def parse_args():
    p = argparse.ArgumentParser(description="Sampling-only: FlowPG-style RealNVP + (modified_uniform|box) prior ckpt.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-pt", type=str, required=True)
    p.add_argument("--latent-bound", type=float, default=0.0, help="0이면 ckpt에 저장된 값을 사용")
    p.add_argument("--mollifier-sigma", type=float, default=0.0, help="0이면 ckpt에 저장된 값을 사용")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--std-eps", type=float, default=1e-6)
    p.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


if __name__ == "__main__":
    sample(parse_args())
