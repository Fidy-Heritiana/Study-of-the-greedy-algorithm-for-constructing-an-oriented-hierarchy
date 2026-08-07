import numpy as np
import pandas as pd

from core_algorithms import (
    generate_implication_matrix,
    gras_algorithm,
    greedy_algorithm,
    jaccard_dissimilarity,
)

# ------------------------------------------------------------
# PREMIER ESSAI HISTORIQUE
# alpha = 0.5
# Cette valeur est au bord de l'intervalle et ne doit pas être
# retenue comme expérience finale si l'article impose alpha > 0.5.
# ------------------------------------------------------------

SEED = 20260804
ALPHA = 0.5
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
            paths,
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
            c_hat_top2 = DJ * C_size / (R_top2 * lbar_top2)

        c_hat_strong = np.nan
        if R_strong > 0 and lbar_strong > 0:
            c_hat_strong = DJ * C_size / (R_strong * lbar_strong)

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
                "n_paths": len(paths),
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

print(summary)
