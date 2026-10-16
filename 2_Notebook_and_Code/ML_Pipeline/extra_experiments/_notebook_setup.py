"""Shared setup for extra experiments 06-12.

Runs the notebook's own setup cells (imports, data load, quality checks, leakage audit, target, chronological
split, feature sets and the preprocessing / model factories) so every script uses exactly the same data,
split, features and pipelines as Machine_Learning_Pipeline.ipynb. Nothing is trained or evaluated here.
Run the scripts from the ML_Pipeline folder (where the notebook and the dataset are).
"""
import json

import matplotlib

matplotlib.use("Agg")

SETUP_CELLS = (3, 5, 7, 9, 11, 13, 15, 21, 23)


def load(notebook="Machine_Learning_Pipeline.ipynb"):
    cells = json.load(open(notebook, encoding="utf-8"))["cells"]
    ns = {"__name__": "notebook_setup", "display": lambda *a, **k: None, "print": lambda *a, **k: None}
    for i in SETUP_CELLS:
        src = "".join(cells[i]["source"])
        src = src.replace("from IPython.display import display", "").replace("plt.show()", "plt.close('all')")
        exec(compile(src, f"notebook cell {i}", "exec"), ns)
    assert (len(ns["train_df"]), len(ns["valid_df"]), len(ns["test_df"])) == (42795, 2802, 5604), "unexpected split sizes"
    return ns
