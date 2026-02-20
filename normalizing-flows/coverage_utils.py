from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch


_FOOT_NAMES = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class FootCoverage:
    ref_bins: int
    gen_bins: int
    inter_bins: int
    recall: float
    precision: float
    jaccard: float
    in_range_pct: float
    out_range_pct: float


@dataclass(frozen=True)
class CoverageMetrics:
    num_samples: int
    num_ref: int
    bins: int
    q_low: float
    q_high: float
    feet: tuple[FootCoverage, FootCoverage, FootCoverage, FootCoverage]
    recall_mean: float
    precision_mean: float
    jaccard_mean: float
    in_range_mean_pct: float
    out_range_mean_pct: float


class FootWorkspaceCoverage:
    """Simple coverage proxy for 12D (4 feet x 3D) samples.

    We build a per-foot 3D voxel occupancy map from a reference subset, then compare
    generated samples against it:
      - recall:   |bins(gen) ∩ bins(ref)| / |bins(ref)|
      - precision:|bins(gen) ∩ bins(ref)| / |bins(gen)|
      - jaccard:  |bins(gen) ∩ bins(ref)| / |bins(gen) ∪ bins(ref)|

    Bounds are derived from reference quantiles (q_low, q_high) per-foot per-axis.
    Samples outside bounds are counted as out-of-range and excluded from binning.
    """

    def __init__(
        self,
        ref: torch.Tensor,
        *,
        bins: int = 24,
        q_high: float = 0.999,
        q_low: float | None = None,
    ):
        if ref.dim() != 2 or ref.size(1) != 12:
            raise ValueError(f"ref must have shape (N,12), got {tuple(ref.shape)}")
        if bins <= 1:
            raise ValueError(f"bins must be > 1, got {bins}")
        q_high = float(q_high)
        if not (0.0 < q_high <= 1.0):
            raise ValueError(f"q_high must be in (0,1], got {q_high}")
        if q_low is None:
            q_low = 1.0 - q_high
        q_low = float(q_low)
        if not (0.0 <= q_low < q_high):
            raise ValueError(f"q_low must satisfy 0<=q_low<q_high, got q_low={q_low}, q_high={q_high}")

        self.bins = int(bins)
        self.q_low = q_low
        self.q_high = q_high

        ref = ref.detach().to(dtype=torch.float32, device="cpu")
        self.num_ref = int(ref.shape[0])

        lows = []
        highs = []
        ref_bins = []
        for foot in range(4):
            coords = ref[:, foot * 3 : foot * 3 + 3]
            lo = torch.quantile(coords, q_low, dim=0)
            hi = torch.quantile(coords, q_high, dim=0)
            # Avoid degenerate bounds.
            span = (hi - lo).clamp_min(1e-9)
            hi = lo + span
            lows.append(lo)
            highs.append(hi)
            ref_bins.append(self._unique_bins(coords, lo=lo, hi=hi))

        self._lo = torch.stack(lows, dim=0)  # (4,3)
        self._hi = torch.stack(highs, dim=0)  # (4,3)
        self._ref_bins = tuple(ref_bins)

    def _unique_bins(self, coords: torch.Tensor, *, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        if coords.dim() != 2 or coords.size(1) != 3:
            raise ValueError(f"coords must be (N,3), got {tuple(coords.shape)}")
        in_range = (coords >= lo) & (coords <= hi)
        in_range = in_range.all(dim=1)
        coords = coords[in_range]
        if coords.numel() == 0:
            return torch.empty((0,), dtype=torch.int64)

        span = (hi - lo).clamp_min(1e-9)
        t = (coords - lo) / span
        # Ensure t ∈ [0, 1) so that bin index is always in [0, bins-1].
        t = t.clamp(min=0.0, max=1.0 - 1e-9)
        idx = (t * float(self.bins)).to(torch.int64)
        idx = idx.clamp(min=0, max=self.bins - 1)
        flat = idx[:, 0] + self.bins * idx[:, 1] + (self.bins * self.bins) * idx[:, 2]
        return torch.unique(flat)

    @staticmethod
    def _intersection_count(sorted_ref: torch.Tensor, sorted_gen: torch.Tensor) -> int:
        if sorted_ref.numel() == 0 or sorted_gen.numel() == 0:
            return 0
        pos = torch.searchsorted(sorted_ref, sorted_gen)
        mask = pos < sorted_ref.numel()
        if not bool(mask.any()):
            return 0
        pos2 = pos[mask]
        gen2 = sorted_gen[mask]
        return int((sorted_ref[pos2] == gen2).sum().item())

    @torch.no_grad()
    def evaluate(self, samples: torch.Tensor) -> CoverageMetrics:
        if samples.dim() != 2 or samples.size(1) != 12:
            raise ValueError(f"samples must have shape (N,12), got {tuple(samples.shape)}")
        samples = samples.detach().to(dtype=torch.float32, device="cpu")
        num_samples = int(samples.shape[0])

        feet: list[FootCoverage] = []
        recalls = []
        precisions = []
        jaccards = []
        in_ranges = []
        out_ranges = []

        for foot in range(4):
            lo = self._lo[foot]
            hi = self._hi[foot]
            coords = samples[:, foot * 3 : foot * 3 + 3]
            in_range = ((coords >= lo) & (coords <= hi)).all(dim=1)
            in_range_pct = 100.0 * float(in_range.float().mean().item())
            out_range_pct = 100.0 - in_range_pct

            gen_bins = self._unique_bins(coords, lo=lo, hi=hi)
            ref_bins = self._ref_bins[foot]
            inter = self._intersection_count(ref_bins, gen_bins)

            ref_n = int(ref_bins.numel())
            gen_n = int(gen_bins.numel())
            denom_union = ref_n + gen_n - inter

            recall = float(inter / ref_n) if ref_n > 0 else 0.0
            precision = float(inter / gen_n) if gen_n > 0 else 0.0
            jaccard = float(inter / denom_union) if denom_union > 0 else 0.0

            feet.append(
                FootCoverage(
                    ref_bins=ref_n,
                    gen_bins=gen_n,
                    inter_bins=int(inter),
                    recall=recall,
                    precision=precision,
                    jaccard=jaccard,
                    in_range_pct=in_range_pct,
                    out_range_pct=out_range_pct,
                )
            )
            recalls.append(recall)
            precisions.append(precision)
            jaccards.append(jaccard)
            in_ranges.append(in_range_pct)
            out_ranges.append(out_range_pct)

        recall_mean = float(sum(recalls) / 4.0)
        precision_mean = float(sum(precisions) / 4.0)
        jaccard_mean = float(sum(jaccards) / 4.0)
        in_range_mean_pct = float(sum(in_ranges) / 4.0)
        out_range_mean_pct = float(sum(out_ranges) / 4.0)

        return CoverageMetrics(
            num_samples=num_samples,
            num_ref=self.num_ref,
            bins=self.bins,
            q_low=self.q_low,
            q_high=self.q_high,
            feet=(feet[0], feet[1], feet[2], feet[3]),
            recall_mean=recall_mean,
            precision_mean=precision_mean,
            jaccard_mean=jaccard_mean,
            in_range_mean_pct=in_range_mean_pct,
            out_range_mean_pct=out_range_mean_pct,
        )


def format_coverage_line(epoch: int, metrics: CoverageMetrics, *, prefix: str = "") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"{prefix}{ts} | Epoch {epoch:04d}",
        f"samples {metrics.num_samples} | ref {metrics.num_ref} | bins {metrics.bins} | q {metrics.q_low:.3g}-{metrics.q_high:.3g}",
        f"mean recall={100.0*metrics.recall_mean:.2f}% prec={100.0*metrics.precision_mean:.2f}% jac={100.0*metrics.jaccard_mean:.2f}% | "
        f"oor(mean)={metrics.out_range_mean_pct:.2f}%",
    ]
    for name, fc in zip(_FOOT_NAMES, metrics.feet):
        parts.append(
            f"{name} r={100.0*fc.recall:.1f}% p={100.0*fc.precision:.1f}% jac={100.0*fc.jaccard:.1f}% "
            f"oor={fc.out_range_pct:.1f}% bins={fc.inter_bins}/{fc.ref_bins}"
        )
    return " | ".join(parts)

