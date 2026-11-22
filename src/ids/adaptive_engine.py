"""Combine ADWIN detection, candidate collection, and safe model proposals."""

import copy

import numpy as np

from ids.models import score
from ids.adaptive_controller import AdaptiveController
from ids.adaptive_finetune import fine_tune_candidate
from ids.candidate_validation import validate_candidate


class AdaptiveAE:
    def __init__(self, model, threshold, cutoff, **controller_kwargs):
        self.active_model = copy.deepcopy(model).cpu().eval()

        self.controller = AdaptiveController(
            threshold=threshold,
            cutoff=cutoff,
            **controller_kwargs,
        )

        self.candidate_model = None
        self.candidate_info = None
        self.model_version = 0

    def process_minute(self, X):
        """Score traffic first, then consider a model update."""
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or len(X) == 0:
            raise ValueError("Expected a nonempty feature matrix")

        # Predictions always use the current active model.
        scores = score(self.active_model, X)

        # Never automatically deploy an unvalidated candidate.
        if self.candidate_model is not None:
            return scores, {"candidate_awaiting_validation": True}

        event = self.controller.observe_minute(X, scores)

        if event is None:
            return scores, None

        event["candidate_proposed"] = False

        if event["candidate_ready"]:
            training_data = self.controller.buffer.snapshot()

            candidate, info = fine_tune_candidate(
                self.active_model,
                training_data,
            )

            self.candidate_model = candidate
            self.candidate_info = info

            event["candidate_proposed"] = True
            event["training_info"] = info

        return scores, event

    def score_candidate(self, X):
        """Evaluate the proposed model without deploying it."""
        if self.candidate_model is None:
            raise RuntimeError("No candidate available")

        return score(
            self.candidate_model,
            np.asarray(X, dtype=np.float32),
        )

    def validate_and_promote(self, reference_X):
        """Promote only if historical, label-free checks pass."""
        if self.candidate_model is None:
            raise RuntimeError("No candidate awaiting validation")

        training_data = self.controller.buffer.snapshot()
        if training_data is None:
            raise RuntimeError("Adaptation buffer is not ready")

        report = validate_candidate(
            self.active_model,
            self.candidate_model,
            reference_X,
            training_data,
            self.controller.threshold,
        )

        if report["approved"]:
            self.controller.mark_adapted()
            self.active_model = self.candidate_model.cpu().eval()
            self.model_version += 1
        else:
            self.controller.buffer.clear()
            self.controller.pending_drift = False

        self.candidate_model = None
        self.candidate_info = None

        return {**report, "model_version": self.model_version}
