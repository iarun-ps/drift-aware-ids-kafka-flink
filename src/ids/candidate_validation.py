"""Label-free safety checks before promoting an adapted autoencoder."""

import numpy as np

from ids.models import score


def validate_candidate(
    active_model,
    candidate_model,
    reference_X,
    adaptation_X,
    threshold,
    max_fpr_increase=0.005,
    max_loss_increase=0.05,
):
    """Compare candidate against active model using historical reference data."""
    reference_X = np.asarray(reference_X, dtype=np.float32)
    adaptation_X = np.asarray(adaptation_X, dtype=np.float32)

    if (
        reference_X.ndim != 2
        or adaptation_X.ndim != 2
        or len(reference_X) == 0
        or len(adaptation_X) == 0
        or reference_X.shape[1] != adaptation_X.shape[1]
    ):
        raise ValueError("Invalid validation data")

    if not np.isfinite(reference_X).all() or not np.isfinite(adaptation_X).all():
        raise ValueError("Non-finite validation features")

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid threshold")

    old_ref = score(active_model, reference_X)
    new_ref = score(candidate_model, reference_X)

    old_adapt = score(active_model, adaptation_X)
    new_adapt = score(candidate_model, adaptation_X)

    old_fpr = float(np.mean(old_ref > threshold))
    new_fpr = float(np.mean(new_ref > threshold))

    old_loss = float(np.mean(old_ref))
    new_loss = float(np.mean(new_ref))

    adaptation_improved = float(np.mean(new_adapt)) <= float(np.mean(old_adapt))
    fpr_ok = new_fpr <= old_fpr + max_fpr_increase
    loss_ok = new_loss <= old_loss * (1 + max_loss_increase)

    approved = bool(adaptation_improved and fpr_ok and loss_ok)

    return {
        "approved": approved,
        "reference_fpr_before": old_fpr,
        "reference_fpr_after": new_fpr,
        "reference_loss_before": old_loss,
        "reference_loss_after": new_loss,
        "adaptation_improved": adaptation_improved,
        "fpr_ok": fpr_ok,
        "loss_ok": loss_ok,
    }
