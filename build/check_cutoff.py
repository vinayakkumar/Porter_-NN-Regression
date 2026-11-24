"""Which cutoff on total_items removes the 411-item order without throwing away real orders?"""
import pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
df = pd.read_csv("data/porter_delivery.csv")
df["created_at"] = pd.to_datetime(df["created_at"]); df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
df["time_taken_mins"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 60
df = df.drop(columns=["created_at", "actual_delivery_time", "estimated_store_to_consumer_driving_duration"])
bad = ((df.min_item_price < 0) | (df.total_onshift_dashers < 0) | (df.total_busy_dashers < 0) | (df.total_outstanding_orders < 0) | (df.subtotal == 0) | (df.min_item_price > df.max_item_price))
df = df[~bad].reset_index(drop=True)
num_cols = ["total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price", "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]
iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1).fit(df[num_cols])
df = df[iso.predict(df[num_cols]) != -1].reset_index(drop=True)
x = df["total_items"]
print("rows after Isolation Forest:", len(df))
q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
print("Q1, Q3, IQR:", q1, q3, iqr, "| Tukey fence Q3+1.5*IQR =", q3 + 1.5 * iqr, "| Q3+3*IQR =", q3 + 3 * iqr)
for t in [10, 15, 20, 25, 30, 40, 60]:
    print(f"rows with total_items > {t:>2}: {(x > t).sum():>5}  ({(x > t).mean() * 100:.3f}%)  mean time {df.loc[x > t, 'time_taken_mins'].mean():.1f}")
print("99th / 99.9th / 99.99th percentile:", x.quantile([.99, .999, .9999]).tolist())
print("largest 8 values:", sorted(x)[-8:])
