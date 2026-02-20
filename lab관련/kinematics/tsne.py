"""
Compute FK/IK datasets and visualize their relation with t-SNE.

Steps:
1) Load joint angles (default: joint_angles_100k_ordered.pt).
2) Run FK -> foot positions, save.
3) Run IK on those positions, save.
4) Run t-SNE on a subset of samples (default 5k) for both foot positions and IK joints.
   Color by ||original_joint - IK_joint|| to see local distortion.

Output:
- foot_positions_from_joints.pt
- ik_joints_from_fk.pt
- tsne_fk_ik.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
try:
    import cupy as cp
    from cuml.manifold import TSNE as cuTSNE

    _CUM_TSNE_AVAILABLE = True
except ImportError:
    _CUM_TSNE_AVAILABLE = False

from FK_IK_Solver import Go2Solver


def chunked_fk_ik(joints: torch.Tensor, solver: Go2Solver, chunk: int, device: str):
    """Run FK then IK in chunks to avoid OOM."""
    fk_list = []
    ik_list = []
    with torch.no_grad():
        for i in range(0, len(joints), chunk):
            batch = joints[i : i + chunk].to(device)
            fk_flat = solver.go2_fk_new(batch)
            fk_list.append(fk_flat.cpu())
            fk_pos = fk_flat.view(batch.size(0), 4, 3)
            ik_batch = solver.go2_ik_new(fk_pos)
            ik_list.append(ik_batch.cpu())
    fk = torch.cat(fk_list, dim=0)
    ik = torch.cat(ik_list, dim=0)
    return fk, ik


def run_tsne(data: np.ndarray, perplexity: float, seed: int, backend: str = "auto"):
    """Run t-SNE; prefer GPU (cuml) if available and backend allows."""
    backend = backend.lower()
    if backend not in ("auto", "gpu", "cpu"):
        backend = "auto"

    if backend != "cpu" and _CUM_TSNE_AVAILABLE:
        try:
            data_gpu = cp.asarray(data, dtype=cp.float32)
            tsne = cuTSNE(
                n_components=2,
                perplexity=perplexity,
                init="random",
                random_state=seed,
                verbose=1,
            )
            embed_gpu = tsne.fit_transform(data_gpu)
            return cp.asnumpy(embed_gpu)
        except Exception as e:
            print(f"[warn] cuML TSNE failed, falling back to sklearn TSNE. Error: {e}")

    # CPU fallback (sklearn)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
        verbose=1,
    )
    return tsne.fit_transform(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-path", default="joint_angles_100k_ordered.pt", help="Input joint angles (.pt, shape [N,12])")
    parser.add_argument("--chunk", type=int, default=10000, help="Batch size for FK/IK")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tsne-samples", type=int, default=5000, help="Subset size for t-SNE (per space)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lip-k", type=int, default=5, help="k-NN for Lipschitz ratio")
    parser.add_argument("--lip-eps", type=float, default=1e-6, help="epsilon to avoid div by zero in ratio")
    parser.add_argument("--lip-metric", choices=["euclidean", "weighted"], default="euclidean",
                        help="Distance metric for Δp (default euclidean; weighted scales xyz by scales)")
    parser.add_argument("--scale-x", type=float, default=1.0, help="Scale for x when metric=weighted")
    parser.add_argument("--scale-y", type=float, default=1.0, help="Scale for y when metric=weighted")
    parser.add_argument("--scale-z", type=float, default=1.0, help="Scale for z when metric=weighted")
    parser.add_argument("--tsne-backend", choices=["auto", "gpu", "cpu"], default="auto",
                        help="t-SNE backend: try GPU (cuml) or force CPU")
    args = parser.parse_args()

    if not os.path.exists(args.joint_path):
        raise FileNotFoundError(f"Input joint file not found: {args.joint_path}")

    joints = torch.load(args.joint_path, map_location="cpu")
    if joints.dim() != 2 or joints.size(1) != 12:
        raise ValueError(f"Expected joint tensor shape (N,12), got {tuple(joints.shape)}")

    solver = Go2Solver(device=args.device)
    fk_path = "foot_positions_from_joints.pt"
    ik_path = "ik_joints_from_fk.pt"

    print(f"[1/4] Running FK/IK on {len(joints)} samples (chunk={args.chunk}, device={args.device})...")
    fk, ik = chunked_fk_ik(joints, solver, args.chunk, args.device)
    torch.save(fk, fk_path)
    torch.save(ik, ik_path)
    print(f"Saved FK to {fk_path}, IK to {ik_path}")

    # Prepare subset for t-SNE (drop any NaN rows from IK/FK first)
    valid_mask = (~torch.isnan(fk).any(dim=1)) & (~torch.isnan(ik).any(dim=1))
    num_valid = valid_mask.sum().item()
    num_invalid = len(fk) - num_valid
    if num_invalid > 0:
        print(f"[info] Dropping {num_invalid} samples with NaNs before t-SNE.")
    fk_valid = fk[valid_mask]
    ik_valid = ik[valid_mask]
    joints_valid = joints[valid_mask]

    N = len(fk_valid)
    tsne_n = min(args.tsne_samples, N)
    rng = np.random.default_rng(args.seed)
    subset_idx = rng.choice(N, size=tsne_n, replace=False)
    fk_sub = fk_valid[subset_idx].numpy()
    ik_sub = ik_valid[subset_idx].numpy()
    joints_sub = joints_valid[subset_idx].numpy()

    # Color: L2 norm difference between original joints and IK output
    diff = np.linalg.norm(joints_sub - ik_sub, axis=1)

    print(f"[2/4] Running t-SNE on FK positions (subset {tsne_n}, perplexity {args.perplexity}, backend={args.tsne_backend})...")
    tsne_fk = run_tsne(fk_sub, args.perplexity, args.seed, backend=args.tsne_backend)

    print(f"[3/4] Running t-SNE on IK joints (subset {tsne_n}, perplexity {args.perplexity}, backend={args.tsne_backend})...")
    tsne_ik = run_tsne(ik_sub, args.perplexity, args.seed, backend=args.tsne_backend)

    print("[4/4] Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sc1 = axes[0].scatter(tsne_fk[:, 0], tsne_fk[:, 1], c=diff, cmap="viridis", s=6, alpha=0.7)
    axes[0].set_title("t-SNE of FK foot positions")
    axes[0].set_xlabel("dim 1")
    axes[0].set_ylabel("dim 2")
    cb1 = fig.colorbar(sc1, ax=axes[0])
    cb1.set_label("||q_orig - q_IK|| (rad)")

    sc2 = axes[1].scatter(tsne_ik[:, 0], tsne_ik[:, 1], c=diff, cmap="viridis", s=6, alpha=0.7)
    axes[1].set_title("t-SNE of IK joint angles")
    axes[1].set_xlabel("dim 1")
    axes[1].set_ylabel("dim 2")
    cb2 = fig.colorbar(sc2, ax=axes[1])
    cb2.set_label("||q_orig - q_IK|| (rad)")

    plt.tight_layout()
    out_png = "tsne_fk_ik.png"
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    # Lipschitz proxy per-leg: k-NN in each leg FK space (3D), ratio ||Δq_leg||/||Δp_leg||
    # Δp_leg distance can be weighted to reflect x/y/z scaling if needed.
    print("[5/5] Computing per-leg Lipschitz ratios...")
    k = min(args.lip_k + 1, tsne_n)  # include self, drop later
    leg_names = ["FL", "FR", "RL", "RR"]
    ratios_all = []
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    axes2 = axes2.flatten()
    scales = np.array([args.scale_x, args.scale_y, args.scale_z], dtype=np.float32)

    for leg in range(4):
        fk_leg = fk_sub.reshape(tsne_n, 4, 3)[:, leg, :]  # (N, 3)
        ik_leg = ik_sub.reshape(tsne_n, 4, 3)[:, leg, :]  # (N, 3)
        if args.lip_metric == "weighted":
            fk_leg_scaled = fk_leg * scales
        else:
            fk_leg_scaled = fk_leg
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(fk_leg_scaled)
        dist_fk, idxs = nn.kneighbors(fk_leg_scaled)
        dist_fk = dist_fk[:, 1:]
        idxs = idxs[:, 1:]
        dist_q = np.linalg.norm(ik_leg[:, None, :] - ik_leg[idxs], axis=2)
        ratio = dist_q / (dist_fk + args.lip_eps)
        ratio = ratio.flatten()
        dist_fk_flat = dist_fk.flatten()
        dist_q_flat = dist_q.flatten()
        ratio = ratio[np.isfinite(ratio)]
        ratios_all.append(ratio)

        ax = axes2[leg]
        ax.scatter(dist_fk_flat, dist_q_flat, s=3, alpha=0.25)
        ax.set_xlabel("||Δp_leg|| (m)")
        ax.set_ylabel("||Δq_leg|| (rad)")
        ax.set_title(f"{leg_names[leg]}: Δp vs Δq (k={args.lip_k}, metric={args.lip_metric})")

    plt.tight_layout()
    lip_scatter_png = "lip_leg_scatter.png"
    plt.savefig(lip_scatter_png, dpi=200)
    plt.close(fig2)

    # Histogram of ratios for all legs combined
    ratios_concat = np.concatenate(ratios_all) if ratios_all else np.array([])
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    ax3.hist(ratios_concat, bins=80, color="#55a868", alpha=0.8)
    ax3.set_xlabel("||Δq_leg|| / ||Δp_leg||")
    ax3.set_ylabel("count")
    ax3.set_title(
        f"Lipschitz ratio per-leg (all legs, k={args.lip_k}, eps={args.lip_eps}, metric={args.lip_metric})"
    )
    plt.tight_layout()
    lip_hist_png = "lip_leg_hist.png"
    plt.savefig(lip_hist_png, dpi=200)
    plt.close(fig3)

    print(
        f"Done.\n"
        f"- t-SNE plot: {out_png}\n"
        f"- Lipschitz scatter per-leg: {lip_scatter_png}\n"
        f"- Lipschitz ratio hist (all legs): {lip_hist_png}\n"
        f"- Ratio stats (all legs): min={ratios_concat.min():.4f}, max={ratios_concat.max():.4f}, mean={ratios_concat.mean():.4f}"
    )


if __name__ == "__main__":
    main()
