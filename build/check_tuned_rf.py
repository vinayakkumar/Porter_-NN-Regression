"""Fairness check: tune the Random Forest on the VALIDATION rows only, then score the test rows once."""
import json, matplotlib, time
matplotlib.use("Agg")
nb = json.load(open("build/run2_exec.ipynb"))
g = {}
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    if any(k in src for k in ("RandomForestRegressor", "rf.", "rf_", "regression_report", "rng = np.random", "final_net_rmse", "customer_view")): continue
    src = "\n".join(l for l in src.split("\n") if not l.startswith("%"))
    exec(compile(src, "cell", "exec"), g)
    if src.startswith("results = {}"): break
from sklearn.ensemble import RandomForestRegressor
import numpy as np
Xtr, ytr, Xva, yva, Xte, yte = g["X_train_nn"], g["y_train_nn"], g["X_val"], g["y_val"], g["X_test"], g["y_test"]
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - b) ** 2)))
rows = []
for leaf in [1, 5, 10, 20, 40]:
    for feats in [1.0, 0.5]:
        t0 = time.time()
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf, max_features=feats, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        rows.append((leaf, feats, rmse(yva, m.predict(Xva)), m, time.time() - t0))
        print(f"min_samples_leaf={leaf:>2} max_features={feats}: validation RMSE {rows[-1][2]:.3f}  ({rows[-1][4]:.0f}s)", flush=True)
best = min(rows, key=lambda r: r[2])
print("best on validation:", best[0], best[1], round(best[2], 3))
print("TEST RMSE of that forest (scored once):", round(rmse(yte, best[3].predict(Xte)), 3))
