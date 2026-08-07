import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core_algorithms import (
    generate_implication_matrix,
    gras_algorithm,
    greedy_algorithm,
    directional_distances,
)

OUT = Path("results_inclusion")
OUT.mkdir(exist_ok=True)

SEED = 20260804
N_VALUES = [5, 8, 10, 12, 15]
ALPHAS = [
    0.51,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]
B = 100

rng = np.random.default_rng(SEED)
rows = []

for n in N_VALUES:

    for rep in range(B):

        M = generate_implication_matrix(
            n,
            rng,
        )

        SG, _, _ = gras_algorithm(M)

        for alpha in ALPHAS:

            (
                SN,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = greedy_algorithm(
                M,
                alpha=alpha,
            )

            d = directional_distances(
                SG,
                SN,
            )

            inter = SG & SN

            rows.append(
                {
                    "n": n,
                    "rep": rep + 1,
                    "alpha": alpha,
                    "SG_size": len(SG),
                    "SN_size": len(SN),
                    "intersection_size": len(inter),
                    "missing_size": len(SG - SN),
                    "surplus_size": len(SN - SG),
                    "D_G": d["D_G"],
                    "D_N": d["D_N"],
                    "recall_G": d["recall_G"],
                    "precision_N": d["precision_N"],
                    "SG_subset_SN": d["SG_subset_SN"],
                }
            )

results = pd.DataFrame(rows)

summary = (
    results.groupby("alpha")
    .agg(
        recall_mean=("recall_G", "mean"),
        recall_median=("recall_G", "median"),
        missing_mean=("D_G", "mean"),
        precision_mean=("precision_N", "mean"),
        surplus_mean=("D_N", "mean"),
        prop_exact_inclusion=(
            "SG_subset_SN",
            "mean",
        ),
        SG_size_mean=("SG_size", "mean"),
        SN_size_mean=("SN_size", "mean"),
    )
    .reset_index()
)

print(summary)

results.to_csv(
    OUT / "SG_inclusion_raw.csv",
    index=False,
)

summary.to_csv(
    OUT / "SG_inclusion_summary.csv",
    index=False,
)

fig = plt.figure(figsize=(7, 4.5))

plt.plot(
    summary["alpha"],
    summary["recall_mean"],
    marker="o",
    label="|SG ∩ SN| / |SG|",
)

plt.plot(
    summary["alpha"],
    summary["precision_mean"],
    marker="s",
    label="|SG ∩ SN| / |SN|",
)

plt.xlabel("alpha")
plt.ylabel("Proportion moyenne")
plt.title("Couverture de Gras et précision du glouton")
plt.ylim(0, 1.02)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "recall_precision_vs_alpha.png",
    dpi=180,
)

plt.close(fig)

fig = plt.figure(figsize=(7, 4.5))

plt.plot(
    summary["alpha"],
    summary["missing_mean"],
    marker="o",
    label="Part de SG manquante",
)

plt.plot(
    summary["alpha"],
    summary["surplus_mean"],
    marker="s",
    label="Part de SN en surplus",
)

plt.xlabel("alpha")
plt.ylabel("Proportion moyenne")
plt.title("Omission versus surplus")
plt.ylim(0, 1.02)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "missing_vs_surplus.png",
    dpi=180,
)

plt.close(fig)
