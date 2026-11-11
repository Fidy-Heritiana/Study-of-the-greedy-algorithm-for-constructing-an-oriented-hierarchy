# Study of the greedy algorithm for constructing an oriented hierarchy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy/blob/main/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy/main?labpath=demo.ipynb)
[![Streamlit app](https://img.shields.io/badge/Streamlit-Live%20app-FF4B4B?logo=streamlit&logoColor=white)](https://VOTRE-APP.streamlit.app)
[![CI](https://github.com/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy/actions/workflows/ci.yml/badge.svg)](https://github.com/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24129903.svg)](https://doi.org/10.5281/zenodo.24129903)

We discuss the typical time complexity of the greedy algorithm for constructing an oriented hierarchy in statistical implicative analysis, and compare this algorithm with Gras's original algorithm. We distinguished two phases to calculate the time complexity, including the greedy selection phase, with a typical complexity of $O(n^2p_{\alpha}\log(n^2p_{\alpha}))$, and the implicative class construction phase. For the second phase, we tested three different hypotheses, namely: the local independence hypothesis on vertices, the equiprobability of the different partitions of $[n]$, and constant kernel coagulation. A numerical simulation confirms that the greedy algorithm behaves rather like constant kernel coagulation, which induces a $\Theta(n^2)$ complexity for the class construction phase. The study of the gap between the greedy algorithm and Gras's algorithm formalizes the rank-preservation condition, under which, at a given step, both algorithms select the same candidate class. We have also established a conjecture to bound this gap, measured by the Jaccard distance on the sets of classes produced by both algorithms.

Python code accompanying a study of a greedy algorithm that builds an oriented hierarchy. The scripts reproduce the experiments one by one.

## Repository content

| File | Purpose |
|------|---------|
| `core_algorithms.py` | Core implementations of the algorithms |
| `01_jaccard_bound.py` | Jaccard bound experiment |
| `02_alpha_sensitivity.py` | Sensitivity to the parameter alpha |
| `03_inclusion_SG_SN.py` | Inclusion between SG and SN |
| `04_runtime_comparison.py` | Runtime comparison |
| `05_class_count_three_approaches.py` | Class count for three approaches |
| `06_legacy_jaccard_alpha_0_5.py`, `07_legacy_runtime_naive.py` | Earlier versions kept for reference |
| `simulation_jaccard_bound.py` | Standalone version of the Jaccard bound experiment (alpha = 0.5, does not use `core_algorithms.py`) |
| `demo.ipynb` | Interactive demo notebook |
| `requirements.txt` | Python dependencies |
| `run_all.py` | Runs all experiments |

## How to run

```bash
git clone https://github.com/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy.git
cd Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy
pip install -r requirements.txt
python run_all.py
```

## Online demo

   - **Colab / Binder**: run `demo.ipynb` in your browser, without installing anything.
   - **Streamlit app**: explore the algorithms interactively (parameters, alpha sweep, runtime comparison) at https://greedy-oriented-hierarchy.streamlit.app/
