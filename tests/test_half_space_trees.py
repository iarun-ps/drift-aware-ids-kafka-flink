import numpy as np
import pytest

from ids.half_space_trees import HalfSpaceTreesBaseline


def create_model():
    return HalfSpaceTreesBaseline(
        n_features=3,
        n_trees=3,
        height=3,
        window_size=5,
        seed=42,
    )


def test_anomaly_scoring_and_updates():
    hst = create_model()
    hst.initialize(np.zeros((20, 3), dtype=np.float32))

    X = np.array([
        [0, 0, 0],
        [5, 5, 5],
    ], dtype=np.float32)

    scores = hst.score_batch(X)

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]
    assert hst.updates == 20

    hst.update_batch(X, scores)
    assert hst.updates == 22


def test_invalid_feature_dimensions():
    hst = create_model()

    with pytest.raises(ValueError):
        hst.score_batch(np.zeros((2, 4)))


def test_nonfinite_features():
    hst = create_model()

    X = np.array([[1, np.nan, 3]])

    with pytest.raises(ValueError):
        hst.score_batch(X)


def test_mismatched_update_scores():
    hst = create_model()

    with pytest.raises(ValueError):
        hst.update_batch(
            np.zeros((2, 3)),
            np.zeros(1),
        )
