"""Plot observed development weights without fitting a model or trimming tails.

Run: .venv/bin/python scripts/plot_weight_quality.py
Missing measurements are omitted from plots only and counted in the caption.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

# Keep an explicitly configured cache; otherwise use a writable temporary path.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "freight-matplotlib")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITIVE_COLOR = "#16697A"
NEGATIVE_COLOR = "#D67732"


def save_weight_quality_plot(train: pd.DataFrame, output: Path) -> None:
    """Save full-range distributions and rates for supplied development rows.

    No input rows or values are modified. Numeric conversion rejects malformed
    nonmissing values. Plot-only finite-value exclusions are counted explicitly.
    """
    if output.suffix.lower() != ".png":
        raise ValueError("Plot output must have a .png extension")
    weight = pd.to_numeric(train["weight"], errors="raise").astype(float)
    distance = pd.to_numeric(train["distance"], errors="raise").astype(float)
    rate = pd.to_numeric(train["posted_rate"], errors="raise").astype(float)
    finite_weight = np.isfinite(weight)
    positive = finite_weight & weight.gt(0)
    negative = finite_weight & weight.lt(0)
    zero_count = int((finite_weight & weight.eq(0)).sum())
    missing_count = int(weight.isna().sum())
    infinite_count = int(np.isinf(weight).sum())
    if not (positive | negative).any():
        raise ValueError("No finite nonzero weights available for plotting")

    magnitude = weight.abs() / 1_000
    rate_per_mile = rate / distance
    eligible = (
        finite_weight
        & np.isfinite(distance)
        & distance.gt(0)
        & np.isfinite(rate)
        & rate.gt(0)
        & np.isfinite(rate_per_mile)
        & rate_per_mile.gt(0)
    )
    invalid_rate_distance_count = int((finite_weight & ~eligible).sum())
    bins = np.linspace(
        magnitude[positive | negative].min(),
        magnitude[positive | negative].max(),
        31,
    )
    if bins[0] == bins[-1]:
        bins = np.linspace(bins[0] - 0.5, bins[-1] + 0.5, 31)

    with plt.rc_context({"font.size": 10, "axes.titlesize": 12}):
        figure, axes = plt.subplots(ncols=2, figsize=(12.5, 5.1), dpi=160)
        groups = [
            (positive, "Positive", POSITIVE_COLOR),
            (negative, "Negative → absolute", NEGATIVE_COLOR),
        ]
        for mask, label, color in groups:
            if mask.any():
                axes[0].hist(
                    magnitude[mask], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color,
                    label=f"{label} (n={int(mask.sum()):,})",
                )
            plot_mask = mask & eligible
            if plot_mask.any():
                axes[1].scatter(
                    magnitude[plot_mask], rate_per_mile[plot_mask],
                    s=7 if label == "Positive" else 20,
                    alpha=0.16 if label == "Positive" else 0.8,
                    color=color, edgecolors="none", rasterized=True,
                    label=f"{label} (n={int(plot_mask.sum()):,})",
                )
        if zero_count:
            plot_mask = eligible & weight.eq(0)
            axes[1].scatter(
                magnitude[plot_mask], rate_per_mile[plot_mask], s=20,
                color="#6E7080", label=f"Zero (n={int(plot_mask.sum()):,})",
            )
        axes[0].set_title("Do negative magnitudes resemble positive weights?")
        axes[0].set_ylabel("Density per 1,000 lb (each group normalized)")
        axes[1].set_title("Weight and observed rate per mile — complete tail")
        axes[1].set_ylabel("Observed posted rate / distance ($ per mile, log scale)")
        axes[1].set_yscale("log")
        for axis in axes:
            axis.set_xlabel("Weight magnitude (1,000 lb)")
            axis.grid(axis="y", alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=8.5, loc="upper left")
        figure.suptitle(
            f"Development data: weight quality ({len(train):,} rows)",
            fontsize=15, fontweight="bold", x=0.075, ha="left",
        )
        figure.text(
            0.075, 0.025,
            f"Plot-only exclusions: {missing_count:,} missing weights; "
            f"{infinite_count:,} infinite weights; "
            f"{invalid_rate_distance_count:,} invalid rate/distance rows. "
            f"Zero weights: {zero_count:,}.\n"
            "All finite eligible observations are shown; no input rows are deleted "
            "and no statistical outliers are removed.",
            fontsize=9, color="#455A60",
        )
        figure.tight_layout(rect=(0, 0.115, 1, 0.93))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, default=PROJECT_ROOT / "data/train_test.csv"
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports/figures/weight_quality.png",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.train.resolve():
        raise ValueError("Plot output must not overwrite its input dataset")
    save_weight_quality_plot(pd.read_csv(args.train), args.output)
    print(f"Saved development-only weight quality plot: {args.output}")


if __name__ == "__main__":
    main()
