# Study of the greedy algorithm for constructing an oriented hierarchy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy/blob/main/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy/main?labpath=demo.ipynb)
[![Streamlit app](https://img.shields.io/badge/Streamlit-Live%20app-FF4B4B?logo=streamlit&logoColor=white)](https://VOTRE-APP.streamlit.app)

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
| `run_all.py` | Runs all experiments |

## How to run

```bash
git clone https://github.com/Fidy-Heritiana/Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy.git
cd Study-of-the-greedy-algorithm-for-constructing-an-oriented-hierarchy
pip install -r requirements.txt
python run_all.py
```

## Online demo

   - **Streamlit app**: explore the algorithms interactively (parameters, alpha sweep, runtime comparison) at https://VOTRE-APP.streamlit.app
   - **Colab / Binder**: run `demo.ipynb` in your browser, without installing anything.
