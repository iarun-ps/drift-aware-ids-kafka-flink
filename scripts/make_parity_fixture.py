"""Generate 100 Monday-holdout records for streaming inference parity tests."""

import json

import numpy as np
import pandas as pd
import torch

from ids.data import REPO_ROOT, load_day
from ids.models import AutoEncoder, score
from ids.preprocess import load_spec, time_split, transform


def main():
    spec = load_spec(REPO_ROOT / "models" / "preprocess_spec.json")
    meta = json.loads(
        (REPO_ROOT / "models" / "static_ae_meta.json").read_text()
    )

    model = AutoEncoder(meta["n_features"])
    model.load_state_dict(
        torch.load(
            REPO_ROOT / "models" / "static_ae.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    _, _, holdout = time_split(load_day("mon"))
    sample = holdout.head(100).copy()

    prepared = transform(sample, spec)
    assert len(sample) == 100
    assert prepared.shape == (100, 67)
    assert list(prepared.columns) == spec["columns"]

    scores = score(model, prepared.to_numpy(dtype=np.float32))
    assert np.isfinite(scores).all()

    fixture = prepared.reset_index(drop=True).copy()
    fixture.insert(
        0,
        "flow_id",
        sample["Flow ID"].astype(str).to_numpy(),
    )
    fixture["expected_score"] = scores

    output = REPO_ROOT / "tests" / "fixtures" / "parity_100.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(output, index=False, float_format="%.10g")

    print(f"PASS: Generated {len(fixture)} parity records")
    print(f"Features: {len(spec['columns'])}")
    print(f"Columns: {len(fixture.columns)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
