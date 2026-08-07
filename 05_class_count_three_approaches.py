import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results_class_count")
OUT.mkdir(exist_ok=True)

SEED = 20260804

N_REPS = {
    20: 300,
    50: 250,
    100: 200,
    200: 120,
    400: 60,
}

ALPHAS = [
    0.0,
    0.7,
]

rng = np.random.default_rng(SEED)


def simulate_greedy_paths(
    n,
    alpha,
    rng,
):
    """
    Simulation efficace du glouton sous M_ij iid U(0,1).

    Pour chaque paire non ordonnée {i,j} :
        max(Mij,Mji) a pour CDF z^2,
        donc max = sqrt(U).

    L'orientation gagnante est équiprobable.
    """

    ii, jj = np.triu_indices(
        n,
        k=1,
    )

    m = len(ii)

    weights = np.sqrt(
        rng.random(m)
    )

    orient = rng.integers(
        0,
        2,
        size=m,
    )

    mask = (
        weights >= alpha
    )

    ii = ii[mask]
    jj = jj[mask]
    weights = weights[mask]
    orient = orient[mask]

    src = np.where(
        orient == 0,
        ii,
        jj,
    )

    dst = np.where(
        orient == 0,
        jj,
        ii,
    )

    order = np.argsort(
        weights
    )[::-1]

    out_used = np.zeros(
        n,
        dtype=bool,
    )

    in_used = np.zeros(
        n,
        dtype=bool,
    )

    succ = np.full(
        n,
        -1,
        dtype=int,
    )

    pred = np.full(
        n,
        -1,
        dtype=int,
    )

    for idx in order:

        x = int(src[idx])
        y = int(dst[idx])

        if (
            out_used[x]
            or in_used[y]
        ):
            continue

        z = y
        cycle = False

        while z != -1:

            if z == x:
                cycle = True
                break

            z = succ[z]

        if cycle:
            continue

        succ[x] = y
        pred[y] = x

        out_used[x] = True
        in_used[y] = True

    starts = np.where(
        pred == -1
    )[0]

    visited = np.zeros(
        n,
        dtype=bool,
    )

    lengths = []

    for s in starts:

        z = int(s)
        L = 0

        while (
            z != -1
            and not visited[z]
        ):
            visited[z] = True
            L += 1
            z = succ[z]

        if L > 0:
            lengths.append(L)

    for i in range(n):

        if not visited[i]:
            lengths.append(1)

    n0 = len(lengths)

    # Nombre réel de classes :
    # singletons inclus, conformément au lemme.
    C_actual = sum(
        L * (L + 1) / 2
        for L in lengths
    )

    # ------------------------------------
    # Approche 1 : indépendance locale
    # ------------------------------------

    q_hat = (
        1.0
        - n0 / n
    )

    C_local = (
        n / (1.0 - q_hat)
    )

    # ------------------------------------
    # Approche 2 : combinatoire
    # n0 ~ sqrt(n), donc |C| ~ n^(3/2)
    # ------------------------------------

    C_comb = n ** 1.5

    # ------------------------------------
    # Approche 3 : coagulation
    # formule du manuscrit
    # ------------------------------------

    C_coag = (
        n**2 / (n0 + 1.0)
        + n / 2.0
    )

    return {
        "n0": n0,
        "q_hat": q_hat,
        "C_actual": C_actual,
        "C_local": C_local,
        "C_comb": C_comb,
        "C_coag": C_coag,
    }


rows = []

for alpha in ALPHAS:

    for n, B in N_REPS.items():

        for rep in range(B):

            r = simulate_greedy_paths(
                n,
                alpha,
                rng,
            )

            r.update(
                {
                    "alpha": alpha,
                    "n": n,
                    "rep": rep + 1,
                }
            )

            rows.append(r)

results = pd.DataFrame(rows)

for model in [
    "local",
    "comb",
    "coag",
]:

    pred = results[
        f"C_{model}"
    ]

    actual = results[
        "C_actual"
    ]

    results[
        f"relerr_{model}"
    ] = (
        pred - actual
    ) / actual

    results[
        f"absrel_{model}"
    ] = np.abs(
        results[
            f"relerr_{model}"
        ]
    )

    results[
        f"sqrel_{model}"
    ] = (
        results[
            f"relerr_{model}"
        ] ** 2
    )

summary_rows = []

for (alpha, n), g in results.groupby(
    ["alpha", "n"]
):

    row = {
        "alpha": alpha,
        "n": n,
        "B": len(g),
        "C_actual_mean": g[
            "C_actual"
        ].mean(),
        "C_actual_sd": g[
            "C_actual"
        ].std(),
        "n0_mean": g[
            "n0"
        ].mean(),
        "n0_sd": g[
            "n0"
        ].std(),
        "sqrt_n": np.sqrt(n),
        "q_mean": g[
            "q_hat"
        ].mean(),
    }

    for model in [
        "local",
        "comb",
        "coag",
    ]:

        row[
            f"{model}_pred_mean"
        ] = g[
            f"C_{model}"
        ].mean()

        row[
            f"{model}_MAPE"
        ] = 100 * g[
            f"absrel_{model}"
        ].mean()

        row[
            f"{model}_bias_pct"
        ] = 100 * g[
            f"relerr_{model}"
        ].mean()

        row[
            f"{model}_RMSRE"
        ] = 100 * np.sqrt(
            g[
                f"sqrel_{model}"
            ].mean()
        )

    summary_rows.append(row)

summary = (
    pd.DataFrame(summary_rows)
    .sort_values(
        ["alpha", "n"]
    )
)

print(summary)

results.to_csv(
    OUT / "class_count_raw.csv",
    index=False,
)

summary.to_csv(
    OUT / "class_count_summary.csv",
    index=False,
)

for alpha in ALPHAS:

    s = summary[
        summary["alpha"]
        == alpha
    ]

    fig = plt.figure(
        figsize=(7.5, 4.8)
    )

    plt.plot(
        s["n"],
        s["C_actual_mean"],
        marker="o",
        label="Simulation glouton",
    )

    plt.plot(
        s["n"],
        s["local_pred_mean"],
        marker="s",
        label="Indépendance locale",
    )

    plt.plot(
        s["n"],
        s["comb_pred_mean"],
        marker="^",
        label="Combinatoire",
    )

    plt.plot(
        s["n"],
        s["coag_pred_mean"],
        marker="d",
        label="Coagulation",
    )

    plt.xlabel("n")
    plt.ylabel("|C| moyen")
    plt.title(
        f"Trois modèles, alpha={alpha}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        OUT
        / f"class_count_alpha_{alpha}.png",
        dpi=180,
    )

    plt.close(fig)

fig = plt.figure(
    figsize=(7.5, 4.8)
)

for alpha in ALPHAS:

    s = summary[
        summary["alpha"]
        == alpha
    ]

    plt.plot(
        s["n"],
        s["n0_mean"],
        marker="o",
        label=f"E[n0], alpha={alpha}",
    )

n_grid = np.array(
    sorted(N_REPS.keys())
)

plt.plot(
    n_grid,
    np.sqrt(n_grid),
    marker="s",
    label="sqrt(n)",
)

plt.xlabel("n")
plt.ylabel("n0 moyen")
plt.title("Comportement de n0")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "n0_vs_sqrt_n.png",
    dpi=180,
)

plt.close(fig)
