"""Write a per-segment, per-label profile of CIC-IDS2017 to results/."""
import pandas as pd

from ids.data import DAY_FILES, REPO_ROOT, load_day

rows = []
for name, _ in DAY_FILES:
    df = load_day(name)
    g = df.groupby("Label")["Timestamp"].agg(["size", "min", "max"]).reset_index()
    g.insert(0, "segment", name)
    rows.append(g.rename(columns={"size": "flows", "min": "first_time", "max": "last_time"}))
    del df

out = pd.concat(rows, ignore_index=True)
path = REPO_ROOT / "results" / "dataset_profile.csv"
out.to_csv(path, index=False)
print("wrote", path)
print(out.to_string(index=False))
