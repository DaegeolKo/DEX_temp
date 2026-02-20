import argparse
from pathlib import Path

import torch


def split_legs(x12: torch.Tensor) -> torch.Tensor:
    if x12.dim() != 2 or x12.size(1) != 12:
        raise ValueError(f"Expected (N,12), got {tuple(x12.shape)}")
    return x12.view(-1, 4, 3)


def corrcoef(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True).clamp_min(1e-12))
    return (x.T @ x) / (x.shape[0] - 1)


def main():
    p = argparse.ArgumentParser(description="Quick sanity check: are Go2 legs approximately independent in fk_positions_valid.pt?")
    p.add_argument("--data-path", type=str, default="fk_positions_valid.pt")
    p.add_argument("--sample-size", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    x = torch.load(Path(args.data_path))
    if x.dim() != 2 or x.size(1) != 12:
        raise ValueError(f"Expected (N,12), got {tuple(x.shape)}")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.seed))
    n = min(int(args.sample_size), int(x.shape[0]))
    idx = torch.randint(0, int(x.shape[0]), (n,), generator=g)
    xs = x.index_select(0, idx).float()

    legs = split_legs(xs)  # (n,4,3)
    flat = legs.reshape(n, 12)
    c = corrcoef(flat).abs()

    # Zero out within-leg blocks to measure cross-leg dependence.
    mask = torch.ones((12, 12), dtype=torch.bool)
    for leg in range(4):
        s = leg * 3
        mask[s : s + 3, s : s + 3] = False
    cross = c[mask]
    print(f"sample_size={n}")
    print(f"max_abs_corr_cross_leg={float(cross.max().item()):.6f}")

    # Also show within-leg max abs corr (x/y/z coupling).
    within = []
    for leg in range(4):
        s = leg * 3
        block = c[s : s + 3, s : s + 3].clone()
        block.fill_diagonal_(0.0)
        within.append(float(block.max().item()))
    print(f"max_abs_corr_within_leg={max(within):.6f} (per-leg: {within})")


if __name__ == "__main__":
    main()

