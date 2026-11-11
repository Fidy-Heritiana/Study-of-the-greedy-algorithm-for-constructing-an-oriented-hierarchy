import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core_algorithms import (
    directional_distances,
    generate_implication_matrix,
    gras_algorithm,
    greedy_algorithm,
    greedy_heap_full,
    jaccard_dissimilarity,
)

REPO_URL = (
    "https://github.com/Fidy-Heritiana/"
    "Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy"
)
ALPHAS = [0.51, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95]

st.set_page_config(
    page_title="Greedy oriented hierarchy",
    page_icon="🌳",
    layout="wide",
)

st.title("Greedy algorithm for constructing an oriented hierarchy")
st.markdown(
    "Interactive companion to the "
    f"[GitHub repository]({REPO_URL}). "
    "A random implication matrix is drawn, then **Gras's algorithm** (set SG) "
    "and the **greedy algorithm** with threshold alpha (set SN) are run on it."
)

# ----------------------------------------------------------------------
# Computations (cached so that moving between tabs stays instant)
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def run_once(n, alpha, seed):
    rng = np.random.default_rng(seed)
    M = generate_implication_matrix(n, rng)

    SG, _, active_g = gras_algorithm(M)
    SN, _, active_n, R_top2, _, R_strong, _, _ = greedy_algorithm(M, alpha=alpha)

    return {
        "M": M,
        "SG": SG,
        "SN": SN,
        "active_g": active_g,
        "active_n": active_n,
        "R_top2": R_top2,
        "R_strong": R_strong,
    }


@st.cache_data(show_spinner=False)
def alpha_sweep(n, reps, seed):
    """Same matrix for every alpha, as in 02_alpha_sensitivity.py."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        M = generate_implication_matrix(n, rng)
        SG, _, _ = gras_algorithm(M)
        for alpha in ALPHAS:
            SN = greedy_algorithm(M, alpha=alpha)[0]
            d = directional_distances(SG, SN)
            rows.append(
                {
                    "rep": rep + 1,
                    "alpha": alpha,
                    "DJ": jaccard_dissimilarity(SG, SN),
                    "recall_G": d["recall_G"],
                    "precision_N": d["precision_N"],
                    "SG_size": len(SG),
                    "SN_size": len(SN),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def runtime_benchmark(n_values, reps, alpha, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        for _ in range(reps):
            M = generate_implication_matrix(n, rng)

            t0 = time.perf_counter()
            gras_algorithm(M)
            t_gras = time.perf_counter() - t0

            t0 = time.perf_counter()
            greedy_heap_full(M, alpha=alpha)
            t_greedy = time.perf_counter() - t0

            rows.append({"n": n, "gras_s": t_gras, "greedy_s": t_greedy})
    df = pd.DataFrame(rows)
    return df.groupby("n").mean().reset_index()


def chain_text(chains):
    return "\n".join(" → ".join(str(v) for v in chain) for chain in chains)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.header("Parameters")
n = st.sidebar.slider("n (number of variables)", 5, 60, 12)
alpha = st.sidebar.slider("alpha (greedy threshold)", 0.51, 0.95, 0.70, 0.01)
seed = st.sidebar.number_input("Random seed", min_value=0, value=1, step=1)
st.sidebar.caption("Change the seed to draw another random matrix.")

tab_run, tab_alpha, tab_time = st.tabs(
    ["Single run", "Sensitivity to alpha", "Runtime comparison"]
)

# ----------------------------------------------------------------------
# Tab 1: one run
# ----------------------------------------------------------------------

with tab_run:
    res = run_once(n, alpha, int(seed))
    SG, SN = res["SG"], res["SN"]
    dd = directional_distances(SG, SN)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("|SG| (Gras)", len(SG))
    c2.metric("|SN| (greedy)", len(SN))
    c3.metric("Jaccard dissimilarity", f"{jaccard_dissimilarity(SG, SN):.3f}")
    c4.metric("R_top2", res["R_top2"])
    c5.metric("R_strong", res["R_strong"])

    st.write(
        f"Share of SG recovered by the greedy algorithm: **{dd['recall_G']:.2f}**. "
        f"Share of SN that belongs to SG: **{dd['precision_N']:.2f}**. "
        f"SG included in SN: **{'yes' if dd['SG_subset_SN'] else 'no'}**."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Implication matrix")
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(res["M"], cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, label="M[i, j]")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Final chains")
        len_g = [len(c) for c in res["active_g"]]
        len_n = [len(c) for c in res["active_n"]]
        top = max(len_g + len_n)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.hist(
            [len_g, len_n],
            bins=np.arange(1, top + 2) - 0.5,
            label=["Gras", "greedy"],
        )
        ax.set_xlabel("length of final chains")
        ax.set_ylabel("number of chains")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Show the final chains"):
        a, b = st.columns(2)
        a.markdown("**Gras**")
        a.code(chain_text(res["active_g"]))
        b.markdown("**Greedy**")
        b.code(chain_text(res["active_n"]))

# ----------------------------------------------------------------------
# Tab 2: sensitivity to alpha
# ----------------------------------------------------------------------

with tab_alpha:
    st.write(
        "Mean Jaccard dissimilarity between SG and SN for several thresholds, "
        "using the same random matrices for all values of alpha."
    )
    n_sw = st.slider("n", 5, 30, 10, key="n_sw")
    reps = st.slider("Number of random matrices", 10, 100, 50, step=10, key="reps_sw")

    if st.button("Run the sweep"):
        st.session_state["sweep"] = (n_sw, reps, int(seed))

    if "sweep" in st.session_state:
        n_s, reps_s, seed_s = st.session_state["sweep"]
        with st.spinner("Running..."):
            df = alpha_sweep(n_s, reps_s, seed_s)

        summ = (
            df.groupby("alpha")
            .agg(
                DJ_mean=("DJ", "mean"),
                DJ_sd=("DJ", "std"),
                recall_mean=("recall_G", "mean"),
                precision_mean=("precision_N", "mean"),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(
            summ["alpha"], summ["DJ_mean"], yerr=summ["DJ_sd"], marker="o", capsize=4
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("mean D_J")
        ax.set_title(f"n = {n_s}, {reps_s} random matrices")
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(summ)
        st.download_button(
            "Download the results (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="alpha_sweep.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------
# Tab 3: runtime
# ----------------------------------------------------------------------

with tab_time:
    st.write(
        "Average running time of Gras's algorithm and of the heap-based greedy "
        "algorithm (`greedy_heap_full`) as n grows."
    )
    n_max = st.slider("Largest n", 20, 120, 80, step=10)
    reps_t = st.slider("Repetitions per n", 1, 5, 3, key="reps_t")
    loglog = st.checkbox("Log-log scale", value=True)

    if st.button("Run the benchmark"):
        st.session_state["bench"] = (n_max, reps_t, alpha, int(seed))

    if "bench" in st.session_state:
        n_m, reps_m, alpha_m, seed_m = st.session_state["bench"]
        n_values = tuple(range(10, n_m + 1, 10))
        with st.spinner("Running..."):
            bench = runtime_benchmark(n_values, reps_m, alpha_m, seed_m)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bench["n"], bench["gras_s"], marker="o", label="Gras")
        ax.plot(bench["n"], bench["greedy_s"], marker="s", label="Greedy (heap)")
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("mean time (s)")
        ax.set_title(f"alpha = {alpha_m}, {reps_m} repetitions per n")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(bench)

st.divider()
st.caption(f"Source code and experiments: {REPO_URL}")
