import subprocess
import sys

scripts = [
    "01_jaccard_bound.py",
    "02_alpha_sensitivity.py",
    "03_inclusion_SG_SN.py",
    "04_runtime_comparison.py",
    "05_class_count_three_approaches.py",
]

for script in scripts:
    print("=" * 70)
    print("Exécution :", script)
    print("=" * 70)

    subprocess.run(
        [sys.executable, script],
        check=True,
    )

print("\nToutes les expériences principales sont terminées.")
