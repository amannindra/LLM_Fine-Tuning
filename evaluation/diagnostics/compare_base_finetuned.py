"""Compare paired base and fine-tuned PubMedQA generations and save diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
BASE_CSV = EVAL / "bertscore" / "test_1" / "bertscore_results.csv"
FINE_CSV = EVAL / "evaluation" / "bertscore" / "test_1" / "bertscore_results.csv"
OUT_PNG = HERE / "base_vs_finetuned.png"


def main() -> None:
    base = pd.read_csv(BASE_CSV, index_col=0)
    fine = pd.read_csv(FINE_CSV, index_col=0)
    if len(base) != len(fine):
        raise ValueError(f"Row count mismatch: base={len(base)}, fine={len(fine)}")
    if not base["reference"].equals(fine["reference"]):
        raise ValueError("References are not aligned row by row")

    paired = pd.DataFrame(
        {
            "base_f1": base["f1"],
            "fine_f1": fine["f1"],
            "fine_chars": fine["student_output"].fillna("").str.len(),
            "base_chars": base["student_output"].fillna("").str.len(),
        }
    )
    paired["delta_f1"] = paired["fine_f1"] - paired["base_f1"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    bins = np.linspace(0, 1, 45)
    axes[0].hist(paired["base_f1"], bins=bins, alpha=0.65, label="Base")
    axes[0].hist(paired["fine_f1"], bins=bins, alpha=0.65, label="Fine-tuned")
    axes[0].set(title="BERTScore F1 distributions", xlabel="F1", ylabel="Examples")
    axes[0].legend()

    axes[1].scatter(paired["base_f1"], paired["fine_f1"], s=5, alpha=0.18)
    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Paired scores (same 10,000 rows)",
        xlabel="Base F1",
        ylabel="Fine-tuned F1",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    points = axes[2].scatter(
        paired["fine_chars"], paired["fine_f1"], c=paired["base_f1"],
        s=7, alpha=0.35, cmap="viridis", vmin=0, vmax=1,
    )
    axes[2].set(
        title="Fine-tuned output length vs F1",
        xlabel="Generated output length (characters)",
        ylabel="Fine-tuned F1",
    )
    fig.colorbar(points, ax=axes[2], label="Base F1")
    fig.suptitle(
        f"Mean F1: base {paired.base_f1.mean():.3f} → fine-tuned {paired.fine_f1.mean():.3f}; "
        f"fine-tuned lower on {(paired.delta_f1 < 0).mean():.1%} of rows"
    )
    fig.savefig(OUT_PNG, dpi=180)
    print(f"Saved {OUT_PNG}")
    print(f"Mean paired delta: {paired.delta_f1.mean():.4f}")
    print(f"Fine-tuned lower on {(paired.delta_f1 < 0).mean():.2%} of rows")
    print(f"Output length means: base {paired.base_chars.mean():.1f}, fine {paired.fine_chars.mean():.1f} chars")


if __name__ == "__main__":
    main()
