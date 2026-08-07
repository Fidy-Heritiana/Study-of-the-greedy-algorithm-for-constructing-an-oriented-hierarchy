import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core_algorithms import (
    generate_implication_matrix,
    gras_algorithm,
    greedy_algorithm,
    jaccard_dissimilarity,
)

OUT = Path("results_alpha")
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

# Même matrice pour toutes les valeurs de alpha
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
                paths,
                R_top2,
                lbar_top2,
                R_strong,
                lbar_strong,
                _,
            ) = greedy_algorithm(
                M,
                alpha=alpha,
            )

            DJ = jaccard_dissimilarity(
                SG,
                SN,
            )

            C_size = len(SN)

            c_hat_strong = np.nan

            if (
                R_strong > 0
                and lbar_strong > 0
            ):
                c_hat_strong = (
                    DJ * C_size
                    / (
                        R_strong
                        * lbar_strong
                    )
                )

            rows.append(
                {
                    "n": n,
                    "rep": rep + 1,
                    "alpha": alpha,
                    "DJ": DJ,
                    "SG_size": len(SG),
                    "SN_size": len(SN),
                    "n_final_paths": len(paths),
                    "R_top2": R_top2,
                    "R_strong": R_strong,
                    "lbar_strong": lbar_strong,
                    "c_hat_strong": c_hat_strong,
                }
            )

results = pd.DataFrame(rows)

summary = (
    results.groupby("alpha")
    .agg(
        DJ_mean=("DJ", "mean"),
        DJ_sd=("DJ", "std"),
        DJ_median=("DJ", "median"),
        SN_size_mean=("SN_size", "mean"),
        R_strong_mean=("R_strong", "mean"),
        c_hat_mean=("c_hat_strong", "mean"),
        c_hat_q95=(
            "c_hat_strong",
            lambda s: s.quantile(0.95),
        ),
        c_hat_max=("c_hat_strong", "max"),
    )
    .reset_index()
)

summary_by_n = (
    results.groupby(["n", "alpha"])
    .agg(
        DJ_mean=("DJ", "mean"),
        DJ_sd=("DJ", "std"),
        SN_size_mean=("SN_size", "mean"),
        R_strong_mean=("R_strong", "mean"),
        c_hat_mean=("c_hat_strong", "mean"),
    )
    .reset_index()
)

print(summary)

results.to_csv(
    OUT / "alpha_sensitivity_raw.csv",
    index=False,
)

summary.to_csv(
    OUT / "alpha_sensitivity_summary.csv",
    index=False,
)

summary_by_n.to_csv(
    OUT / "alpha_sensitivity_by_n.csv",
    index=False,
)

fig = plt.figure(figsize=(7, 4.5))

plt.errorbar(
    summary["alpha"],
    summary["DJ_mean"],
    yerr=summary["DJ_sd"],
    marker="o",
    capsize=4,
)

plt.xlabel("alpha")
plt.ylabel("D_J moyen")
plt.title("Effet du seuil alpha sur D_J")
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "DJ_vs_alpha.png",
    dpi=180,
)

plt.close(fig)

fig = plt.figure(figsize=(7.5, 4.8))

for n in N_VALUES:

    sub = summary_by_n[
        summary_by_n["n"] == n
    ]

    plt.plot(
        sub["alpha"],
        sub["DJ_mean"],
        marker="o",
        label=f"n={n}",
    )

plt.xlabel("alpha")
plt.ylabel("D_J moyen")
plt.title("D_J selon alpha et n")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "DJ_vs_alpha_by_n.png",
    dpi=180,
)

plt.close(fig)
