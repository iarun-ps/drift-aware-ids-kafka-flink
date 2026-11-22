"""Safe model loading and reloading for the Flink scoring job.

A small pointer file, models/active_model.json, names the model version to use:

    {"model_version": "adaptive-v1"}

The holder looks at that file at most every few seconds. A new version becomes active
only after it has been fully validated (files, metadata, ONNX interface, a test run).
If anything fails, the model that is already active keeps scoring.
"""
import json
import logging
import math
import os
import re
import time

import numpy as np
import onnxruntime as ort

log = logging.getLogger("model_holder")

VERSION_RE = re.compile(r"^(static-v1|adaptive-v[0-9]{1,3})$")
POINTER_NAME = "active_model.json"


class ModelLoadError(Exception):
    """A model version could not be validated or loaded."""


def _paths_for(models_dir, version):
    if version == "static-v1":
        return (os.path.join(models_dir, "static_ae.onnx"),
                os.path.join(models_dir, "static_ae_meta.json"))
    return (os.path.join(models_dir, "adaptive", f"{version}.onnx"),
            os.path.join(models_dir, "adaptive", f"{version}.json"))


def load_model(models_dir, version, n_features):
    """Return (session, threshold, version) or raise ModelLoadError. Never half-loads."""
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise ModelLoadError(f"unsupported model version name: {version!r}")
    onnx_path, meta_path = _paths_for(models_dir, version)
    for path in (onnx_path, meta_path):
        if not os.path.isfile(path):
            raise ModelLoadError(f"missing file: {path}")

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (ValueError, OSError) as exc:
        raise ModelLoadError(f"unreadable metadata: {exc}")
    if not isinstance(meta, dict):
        raise ModelLoadError("metadata is not a JSON object")
    threshold = meta.get("threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold <= 0):
        raise ModelLoadError("metadata has no valid threshold")
    if "model_version" in meta and meta["model_version"] != version:
        raise ModelLoadError("metadata model_version does not match the requested version")
    if meta.get("n_features", n_features) != n_features:
        raise ModelLoadError("metadata n_features does not match the feature spec")
    if meta.get("input_name", "x") != "x" or meta.get("output_name", "score") != "score":
        raise ModelLoadError("metadata input/output names differ from x/score")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        session = ort.InferenceSession(onnx_path, sess_options=options,
                                       providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ModelLoadError(f"ONNX file could not be loaded: {exc}")
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    if inp.name != "x" or out.name != "score":
        raise ModelLoadError("ONNX input/output names are not x/score")
    if inp.type != "tensor(float)":
        raise ModelLoadError(f"ONNX input type is {inp.type}, expected float32")
    if len(inp.shape) != 2 or inp.shape[1] != n_features:
        raise ModelLoadError(f"ONNX input shape {inp.shape} does not match {n_features} features")

    # self-test: two fixed rows must give two finite, non-negative scores
    rng = np.random.default_rng(0)
    test = np.vstack([np.zeros((1, n_features)), rng.random((1, n_features))]).astype(np.float32)
    try:
        scores = session.run(["score"], {"x": test})[0]
    except Exception as exc:
        raise ModelLoadError(f"self-test run failed: {exc}")
    if scores.shape != (2,) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ModelLoadError("self-test scores are not finite non-negative numbers")
    return session, float(threshold), version


class ModelHolder:
    """Holds the active model and swaps it only after a replacement is fully validated."""

    def __init__(self, models_dir, n_features, check_every_s=2.0,
                 default_version="static-v1", clock=time.monotonic):
        self.models_dir = str(models_dir)
        self.n_features = n_features
        self.check_every_s = check_every_s
        self.default_version = default_version
        self._clock = clock
        self.pointer_path = os.path.join(self.models_dir, POINTER_NAME)
        self.active = None          # (session, threshold, version), replaced in one step
        self.reload_failures = 0
        self.last_error = None
        self._last_check = float("-inf")
        self._last_stamp = None

    @property
    def version(self):
        return self.active[2]

    def _pointer_stamp(self):
        try:
            st = os.stat(self.pointer_path)
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _read_pointer(self):
        try:
            with open(self.pointer_path) as f:
                data = json.load(f)
        except (ValueError, OSError) as exc:
            raise ModelLoadError(f"pointer file unreadable: {exc}")
        version = data.get("model_version") if isinstance(data, dict) else None
        if not isinstance(version, str):
            raise ModelLoadError("pointer file has no model_version")
        return version

    def _fail(self, message):
        self.reload_failures += 1
        self.last_error = message
        log.warning("model reload rejected, keeping %s: %s",
                    self.active[2] if self.active else "nothing", message)

    def load_initial(self):
        """Start with the pointed model; fall back to the default if it is not usable."""
        self._last_stamp = self._pointer_stamp()
        if self._last_stamp is not None:
            try:
                wanted = self._read_pointer()
                if wanted != self.default_version:
                    self.active = load_model(self.models_dir, wanted, self.n_features)
                    log.info("model %s active at start", wanted)
                    return
            except ModelLoadError as exc:
                self._fail(f"start-up pointer not usable, using {self.default_version}: {exc}")
        self.active = load_model(self.models_dir, self.default_version, self.n_features)
        log.info("model %s active at start", self.default_version)

    def maybe_reload(self):
        """Cheap to call for every record. Returns True if a new model became active."""
        now = self._clock()
        if now - self._last_check < self.check_every_s:
            return False
        self._last_check = now
        try:
            stamp = self._pointer_stamp()
            if stamp is None or stamp == self._last_stamp:
                return False
            self._last_stamp = stamp
            version = self._read_pointer()
            if version == self.version:
                return False
            self.active = load_model(self.models_dir, version, self.n_features)
            log.info("model switched to %s", version)
            return True
        except Exception as exc:  # reload problems must never stop scoring
            self._fail(str(exc))
            return False