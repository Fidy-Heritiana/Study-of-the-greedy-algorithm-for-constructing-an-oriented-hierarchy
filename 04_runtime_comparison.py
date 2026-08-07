import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core_algorithms import (
    generate_implication_matrix,
    gras_algorithm,
    greedy_heap_full,
)

OUT = Path("results_runtime")
OUT.mkdir(exist_ok=True)

SEED = 20260804
ALPHA = 0.70

N_REPS = {
    10: 30,
    20: 30,
    30: 25,
    40: 20,
    50: 15,
    60: 12,
    80: 8,
    100: 5,
}

rng = np.random.default_rng(SEED)
rows = []

for n, reps in N_REPS.items():

    for rep in range(reps):

        M = generate_implication_matrix(
            n,
            rng,
        )

        t0 = time.perf_counter()

        SG, _, _ = gras_algorithm(M)

        t_gras = (
            time.perf_counter()
            - t0
        )

        t0 = time.perf_counter()

        SN, paths, edges = greedy_heap_full(
            M,
            alpha=ALPHA,
        )

        t_glouton = (
            time.perf_counter()
            - t0
        )

        rows.append(
            {
                "n": n,
                "rep": rep + 1,
                "gras_s": t_gras,
                "greedy_s": t_glouton,
                "speedup": (
                    t_gras / t_glouton
                    if t_glouton > 0
                    else np.nan
                ),
                "SG_size": len(SG),
                "SN_size": len(SN),
                "n_edges": len(edges),
                "n_paths": len(paths),
            }
        )

results = pd.DataFrame(rows)

summary = (
    results.groupby("n")
    .agg(
        reps=("rep", "count"),
        gras_mean_s=("gras_s", "mean"),
        gras_median_s=("gras_s", "median"),
        greedy_mean_s=("greedy_s", "mean"),
        greedy_median_s=("greedy_s", "median"),
        speedup_mean=("speedup", "mean"),
        speedup_median=("speedup", "median"),
    )
    .reset_index()
)


def fit_power_law(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p, loga = np.polyfit(
        np.log(x),
        np.log(y),
        1,
    )

    return np.exp(loga), p


aG, pG = fit_power_law(
    summary["n"],
    summary["gras_mean_s"],
)

aN, pN = fit_power_law(
    summary["n"],
    summary["greedy_mean_s"],
)

print(summary)
print()
print(
    f"Gras : t ≈ {aG:.3e} n^{pG:.3f}"
)
print(
    f"Glouton : t ≈ {aN:.3e} n^{pN:.3f}"
)

results.to_csv(
    OUT / "runtime_raw.csv",
    index=False,
)

summary.to_csv(
    OUT / "runtime_summary.csv",
    index=False,
)

fig = plt.figure(figsize=(7, 4.5))

plt.plot(
    summary["n"],
    summary["gras_mean_s"],
    marker="o",
    label="Gras",
)

plt.plot(
    summary["n"],
    summary["greedy_mean_s"],
    marker="s",
    label="Glouton (tas)",
)

plt.xlabel("n")
plt.ylabel("Temps moyen (s)")
plt.title(
    rf"Temps d'exécution, alpha={ALPHA}"
)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "runtime_comparison.png",
    dpi=180,
)

plt.close(fig)

fig = plt.figure(figsize=(7, 4.5))

plt.loglog(
    summary["n"],
    summary["gras_mean_s"],
    marker="o",
    label=f"Gras, pente≈{pG:.2f}",
)

plt.loglog(
    summary["n"],
    summary["greedy_mean_s"],
    marker="s",
    label=f"Glouton, pente≈{pN:.2f}",
)

plt.xlabel("n")
plt.ylabel("Temps moyen (s)")
plt.title("Comparaison log-log")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "runtime_loglog.png",
    dpi=180,
)

plt.close(fig)

fig = plt.figure(figsize=(7, 4.5))

plt.plot(
    summary["n"],
    summary["speedup_median"],
    marker="o",
)

plt.xlabel("n")
plt.ylabel("T_Gras / T_glouton")
plt.title("Accélération médiane")
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "runtime_speedup.png",
    dpi=180,
)

plt.close(fig)
