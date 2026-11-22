"""Regression tests for drift-aware autoencoder adaptation."""

import numpy as np
import torch

from ids.adaptive_engine import AdaptiveAE
from ids.adaptive_controller import AdaptiveController


def test_deferred_adaptation():
    controller = AdaptiveController(
        threshold=0.2,
        cutoff=0.001,
        min_samples=20,
    )

    X = np.zeros((20, 67), dtype=np.float32)
    events = iter([True, False, False])

    controller.monitor.update_minute = lambda scores: {
        "median_ratio": 0.0,
        "drift_detected": next(events),
    }

    first = controller.observe_minute(X, np.full(20, 0.0005))
    assert first["pending_drift"]
    assert first["accepted"] == 0

    blocked = controller.observe_minute(X, np.full(20, 0.5))
    assert blocked["suspicious_minute"]
    assert blocked["buffer_size"] == 0

    safe = controller.observe_minute(X, np.full(20, 0.0005))
    assert safe["candidate_ready"]

    controller.mark_adapted()
    assert not controller.pending_drift


def test_candidate_does_not_replace_active_model():
    torch.manual_seed(42)

    model = torch_model = None
    from ids.models import AutoEncoder

    model = AutoEncoder(67)
    engine = AdaptiveAE(
        model,
        threshold=100.0,
        cutoff=100.0,
        min_samples=20,
    )

    X = np.zeros((20, 67), dtype=np.float32)
    original = [
        p.detach().clone()
        for p in engine.active_model.parameters()
    ]

    events = iter([True, False])
    engine.controller.monitor.update_minute = lambda scores: {
        "median_ratio": 0.0,
        "drift_detected": next(events),
    }

    engine.process_minute(X)
    _, event = engine.process_minute(X)

    assert event["candidate_proposed"]
    assert engine.candidate_model is not None

    assert all(
        torch.equal(p, old)
        for p, old in zip(engine.active_model.parameters(), original)
    )

    assert engine.score_candidate(X).shape == (20,)


def test_candidate_promotion_and_rejection():
    import copy

    from ids.models import AutoEncoder

    torch.manual_seed(42)

    engine = AdaptiveAE(
        AutoEncoder(67),
        threshold=100.0,
        cutoff=100.0,
        min_samples=20,
    )

    X = np.zeros((20, 67), dtype=np.float32)

    def prepare(candidate):
        engine.controller.pending_drift = True
        engine.controller.buffer.add_batch(X, np.zeros(20))
        engine.candidate_model = candidate

    # An unchanged candidate should pass validation.
    prepare(copy.deepcopy(engine.active_model))
    approved = engine.validate_and_promote(X)

    assert approved["approved"]
    assert engine.model_version == 1
    assert not engine.controller.pending_drift

    # A severely degraded candidate must be rejected.
    degraded = copy.deepcopy(engine.active_model)

    with torch.no_grad():
        degraded.decoder[-1].bias.add_(100.0)

    prepare(degraded)
    rejected = engine.validate_and_promote(X)

    assert not rejected["approved"]
    assert engine.model_version == 1
    assert engine.candidate_model is None
    assert not engine.controller.pending_drift
