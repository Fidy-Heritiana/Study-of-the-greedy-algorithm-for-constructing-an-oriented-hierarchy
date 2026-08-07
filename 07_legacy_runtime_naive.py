import time
import numpy as np
import pandas as pd

from core_algorithms import (
    generate_implication_matrix,
    gras_algorithm,
    greedy_algorithm,
)

# ------------------------------------------------------------
# PREMIER BENCHMARK HISTORIQUE
# Ici le glouton rescannait les candidats à chaque fusion.
# Ce benchmark a été remplacé par la version optimisée par tas.
# ------------------------------------------------------------

SEED = 20260804
ALPHA = 0.70
N_VALUES = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40]
B = 20

rng = np.random.default_rng(SEED)
rows = []

for n in N_VALUES:

    for rep in range(B):

        M = generate_implication_matrix(n, rng)

        t0 = time.perf_counter()
        SG, _, _ = gras_algorithm(M)
        t_gras = time.perf_counter() - t0

        t0 = time.perf_counter()
        SN, _, _, _, _, _, _, _ = greedy_algorithm(
            M,
            alpha=ALPHA,
        )
        t_glouton = time.perf_counter() - t0

        rows.append(
            {
                "n": n,
                "rep": rep + 1,
                "alpha": ALPHA,
                "time_gras_s": t_gras,
                "time_greedy_s": t_glouton,
                "speedup": (
                    t_gras / t_glouton
                    if t_glouton > 0
                    else np.nan
                ),
                "SG_size": len(SG),
                "SN_size": len(SN),
            }
        )

results = pd.DataFrame(rows)

summary = (
    results.groupby("n")
    .agg(
        gras_mean_s=("time_gras_s", "mean"),
        gras_sd_s=("time_gras_s", "std"),
        greedy_mean_s=("time_greedy_s", "mean"),
        greedy_sd_s=("time_greedy_s", "std"),
        speedup_mean=("speedup", "mean"),
        speedup_median=("speedup", "median"),
    )
    .reset_index()
)

print(summary)
