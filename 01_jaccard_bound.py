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

OUT = Path("results_jaccard")
OUT.mkdir(exist_ok=True)

SEED = 20260804
ALPHA = 0.70
N_VALUES = [5, 8, 10, 12, 15]
B = 100

rng = np.random.default_rng(SEED)
rows = []

for n in N_VALUES:

    for rep in range(B):

        M = generate_implication_matrix(n, rng)

        SG, _, _ = gras_algorithm(M)

        (
            SN,
            _,
            final_paths,
            R_top2,
            lbar_top2,
            R_strong,
            lbar_strong,
            _,
        ) = greedy_algorithm(M, alpha=ALPHA)

        DJ = jaccard_dissimilarity(SG, SN)

        C_size = len(SN)

        c_hat_top2 = np.nan

        if R_top2 > 0 and lbar_top2 > 0:
            c_hat_top2 = (
                DJ * C_size
                / (R_top2 * lbar_top2)
            )

        c_hat_strong = np.nan

        if R_strong > 0 and lbar_strong > 0:
            c_hat_strong = (
                DJ * C_size
                / (R_strong * lbar_strong)
            )

        rows.append(
            {
                "n": n,
                "rep": rep + 1,
                "alpha": ALPHA,
                "DJ": DJ,
                "R_top2": R_top2,
                "lbar_top2": lbar_top2,
                "R_strong": R_strong,
                "lbar_strong": lbar_strong,
                "C_size": C_size,
                "SG_size": len(SG),
                "SN_size": len(SN),
                "n_final_paths": len(final_paths),
                "c_hat_top2": c_hat_top2,
                "c_hat_strong": c_hat_strong,
            }
        )

results = pd.DataFrame(rows)

summary = (
    results.groupby("n")
    .agg(
        DJ_mean=("DJ", "mean"),
        DJ_sd=("DJ", "std"),
        R_top2_mean=("R_top2", "mean"),
        R_strong_mean=("R_strong", "mean"),
        c_top2_mean=("c_hat_top2", "mean"),
        c_top2_q95=("c_hat_top2", lambda s: s.quantile(0.95)),
        c_top2_max=("c_hat_top2", "max"),
        c_strong_mean=("c_hat_strong", "mean"),
        c_strong_q95=("c_hat_strong", lambda s: s.quantile(0.95)),
        c_strong_max=("c_hat_strong", "max"),
    )
    .reset_index()
)

bad_top2 = results[
    (results["R_top2"] == 0)
    & (results["DJ"] > 0)
]

bad_strong = results[
    (results["R_strong"] == 0)
    & (results["DJ"] > 0)
]

print(summary)
print()
print("R_top2 = 0 mais DJ > 0 :", len(bad_top2))
print("R_strong = 0 mais DJ > 0 :", len(bad_strong))

results.to_csv(
    OUT / "jaccard_bound_raw.csv",
    index=False,
)

summary.to_csv(
    OUT / "jaccard_bound_summary.csv",
    index=False,
)

fig = plt.figure(figsize=(7, 4.5))

plt.errorbar(
    summary["n"],
    summary["DJ_mean"],
    yerr=summary["DJ_sd"],
    marker="o",
    capsize=4,
)

plt.xlabel("n")
plt.ylabel("D_J moyen")
plt.title(
    rf"Distance de Jaccard, alpha={ALPHA}"
)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "DJ_mean_vs_n.png",
    dpi=180,
)

plt.close(fig)

fig = plt.figure(figsize=(7, 4.5))

plt.plot(
    summary["n"],
    summary["c_strong_mean"],
    marker="o",
    label="moyenne",
)

plt.plot(
    summary["n"],
    summary["c_strong_q95"],
    marker="s",
    label="quantile 95%",
)

plt.plot(
    summary["n"],
    summary["c_strong_max"],
    marker="^",
    label="maximum",
)

plt.xlabel("n")
plt.ylabel("c-hat")
plt.title("Constante empirique de la borne")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "c_hat_strong_vs_n.png",
    dpi=180,
)

plt.close(fig)
