"""Re-run the notebook's own code cells up to the helper definitions, then test SGD+momentum at lr=0.01."""
import json, os, matplotlib
matplotlib.use("Agg")
nb = json.load(open("build/pass3_exec.ipynb"))
g = {}
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    if any(k in src for k in ("RandomForestRegressor", "rf.", "rf_", "regression_report")): continue
    src = "\n".join(l for l in src.split("\n") if not l.startswith("%"))
    exec(compile(src, "cell", "exec"), g)
    if src.startswith("results = {}"): break
import torch
for lr in (0.01, 0.001):
    g["run"](f"sgdm {lr}", "t", optimizer="sgd_momentum", lr=lr, epochs=5)
    h = g["histories"][f"sgdm {lr}"]
    print("SGD+momentum lr", lr, "-> val RMSE per epoch:", [round(v, 3) for v in h["val"]])
