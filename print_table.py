import pandas as pd

d = pd.read_csv("model/metrics_summary.csv")
print("| ML Model Name | " + " | ".join(d.columns[1:]) + " |")
print("|" + "---|" * len(d.columns))
for _, r in d.iterrows():
    print("| " + r["Model"] + " | " + " | ".join(f"{v:.4f}" for v in r.iloc[1:]) + " |")
