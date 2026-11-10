"""Fit the preprocessing spec on Monday's training split and save it."""
from ids.data import REPO_ROOT, load_day
from ids.preprocess import clean_features, fit_spec, save_spec, time_split

train, _, _ = time_split(load_day("mon"))
X, medians = clean_features(train)
spec = fit_spec(X, medians)

path = REPO_ROOT / "models" / "preprocess_spec.json"
save_spec(spec, path)
print("wrote", path)
print("features:", len(spec["columns"]))
print("first 5:", spec["columns"][:5])
