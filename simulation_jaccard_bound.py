
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(".")
SEED = 20260804
ALPHA = 0.5
N_VALUES = [5, 8, 10, 12, 15]
B = 100

def g_pair(p):
    p = float(p)
    if p < 0.5:
        return 0.0
    if p >= 1.0:
        H = 0.0
    else:
        H = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return math.sqrt(max(0.0, 1.0 - H / 2.0))

def class_stats(C, M, cache):
    C = tuple(C)
    if C in cache:
        return cache[C]
    if len(C) <= 1:
        ans = (1.0, 1.0, 1.0)
        cache[C] = ans
        return ans
    pvals, gvals = [], []
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            p = float(M[C[i], C[j]])
            pvals.append(p)
            gvals.append(g_pair(p))
    coh = 0.0 if any(v <= 0 for v in gvals) else math.exp(
        sum(math.log(v) for v in gvals) / len(gvals)
    )
    ans = (coh, min(pvals), max(pvals))
    cache[C] = ans
    return ans

def cohesion(C, M, cache):
    return class_stats(C, M, cache)[0]

def delta(C, M, cache):
    _, pmin, pmax = class_stats(C, M, cache)
    return g_pair(pmax) - g_pair(pmin)

def generate_implication_matrix(n, rng):
    M = rng.random((n, n))
    np.fill_diagonal(M, 0.0)
    return M

def gras_algorithm(M):
    n = M.shape[0]
    active = [(i,) for i in range(n)]
    selected = []
    cache = {}
    while len(active) >= 2:
        best = None
        best_coh = 0.0
        for ia, A in enumerate(active):
            for ib, B in enumerate(active):
                if ia == ib:
                    continue
                C = A + B
                coh = cohesion(C, M, cache)
                if (
                    coh > best_coh + 1e-15
                    or (
                        abs(coh - best_coh) <= 1e-15
                        and best is not None
                        and C < best[2]
                    )
                ):
                    best = (ia, ib, C)
                    best_coh = coh
        if best is None or best_coh <= 0:
            break
        ia, ib, C = best
        selected.append(C)
        active = [
            chain for k, chain in enumerate(active)
            if k not in (ia, ib)
        ] + [C]
    return set(selected), selected, active

def greedy_algorithm(M, alpha=0.5):
    n = M.shape[0]
    active = [(i,) for i in range(n)]
    selected = []
    cache = {}

    R_top2 = 0
    R_strong = 0
    vertices_top2 = set()
    vertices_strong = set()

    while len(active) >= 2:
        candidates = []
        for ia, A in enumerate(active):
            for ib, B in enumerate(active):
                if ia == ib:
                    continue
                w = float(M[A[-1], B[0]])
                if w >= alpha:
                    candidates.append((w, ia, ib, A + B))

        if not candidates:
            break

        candidates.sort(key=lambda x: (-x[0], x[3]))
        w1, ia1, ib1, C1 = candidates[0]
        g1 = g_pair(w1)
        d1 = delta(C1, M, cache)
        coh1 = cohesion(C1, M, cache)

        if len(candidates) >= 2:
            w2, _, _, C2 = candidates[1]
            d2 = delta(C2, M, cache)
            if abs(g1 - g_pair(w2)) <= d1 + d2:
                R_top2 += 1
                vertices_top2.update(C1)
                vertices_top2.update(C2)

        violating_competitors = []
        for wj, _, _, Cj in candidates[1:]:
            dj = delta(Cj, M, cache)
            if g1 - g_pair(wj) <= d1 + dj:
                violating_competitors.append(Cj)

        stop_mismatch_risk = coh1 <= 0.0
        if stop_mismatch_risk or violating_competitors:
            R_strong += 1
            vertices_strong.update(C1)
            for Cj in violating_competitors:
                vertices_strong.update(Cj)

        selected.append(C1)
        active = [
            chain for k, chain in enumerate(active)
            if k not in (ia1, ib1)
        ] + [C1]

    lengths_top2 = [
        len(chain) for chain in active
        if set(chain) & vertices_top2
    ]
    lengths_strong = [
        len(chain) for chain in active
        if set(chain) & vertices_strong
    ]

    lbar_top2 = float(np.mean(lengths_top2)) if lengths_top2 else 0.0
    lbar_strong = float(np.mean(lengths_strong)) if lengths_strong else 0.0

    return (
        set(selected), selected, active,
        R_top2, lbar_top2,
        R_strong, lbar_strong
    )

def jaccard_dissimilarity(SG, SN):
    union = SG | SN
    return 0.0 if not union else 1.0 - len(SG & SN) / len(union)

def one_simulation(n, rng, alpha=0.5):
    M = generate_implication_matrix(n, rng)
    SG, _, _ = gras_algorithm(M)
    SN, _, paths, Rt, lt, Rs, ls = greedy_algorithm(M, alpha)

    DJ = jaccard_dissimilarity(SG, SN)
    C_size = len(SN)

    c_top2 = np.nan if Rt == 0 or lt == 0 else DJ * C_size / (Rt * lt)
    c_strong = np.nan if Rs == 0 or ls == 0 else DJ * C_size / (Rs * ls)

    return {
        "n": n,
        "DJ": DJ,
        "R_top2": Rt,
        "lbar_top2": lt,
        "R_strong": Rs,
        "lbar_strong": ls,
        "C_size": C_size,
        "SG_size": len(SG),
        "SN_size": len(SN),
        "n_final_paths": len(paths),
        "c_hat_top2": c_top2,
        "c_hat_strong": c_strong,
    }

rng = np.random.default_rng(SEED)
rows = []

for n in N_VALUES:
    for rep in range(B):
        row = one_simulation(n, rng, ALPHA)
        row["rep"] = rep + 1
        row["alpha"] = ALPHA
        rows.append(row)

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

results.to_csv(OUT / "simulation_jaccard_bound_results.csv", index=False)
summary.to_csv(OUT / "simulation_jaccard_bound_summary.csv", index=False)

bad_top2 = results[(results.R_top2 == 0) & (results.DJ > 0)]
bad_strong = results[(results.R_strong == 0) & (results.DJ > 0)]

print(summary.to_string(index=False))
print("\nR_top2=0 mais DJ>0 :", len(bad_top2))
print("R_strong=0 mais DJ>0 :", len(bad_strong))
print("max c_hat_top2 :", np.nanmax(results.c_hat_top2))
print("max c_hat_strong :", np.nanmax(results.c_hat_strong))

fig = plt.figure(figsize=(7, 4.5))
plt.errorbar(summary.n, summary.DJ_mean, yerr=summary.DJ_sd, marker="o", capsize=4)
plt.xlabel("n")
plt.ylabel("D_J moyen")
plt.title("Dissimilarité de Jaccard")
plt.tight_layout()
plt.savefig(OUT / "jaccard_mean_vs_n.png", dpi=180)
plt.close(fig)

fig = plt.figure(figsize=(7, 4.5))
plt.plot(summary.n, summary.c_strong_mean, marker="o", label="moyenne")
plt.plot(summary.n, summary.c_strong_q95, marker="s", label="quantile 95%")
plt.plot(summary.n, summary.c_strong_max, marker="^", label="maximum")
plt.xlabel("n")
plt.ylabel("c-hat")
plt.title("Constante empirique pour R_strong")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "c_hat_strong_vs_n.png", dpi=180)
plt.close(fig)
