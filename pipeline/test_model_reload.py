"""Offline test of safe model reload (uses a temporary copy of models/, nothing real is changed)."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from model_holder import ModelHolder, load_model

REAL = Path(__file__).resolve().parent.parent / "models"
N = len(json.load(open(REAL / "preprocess_spec.json"))["columns"])
passed = 0


def check(name, ok):
    global passed
    print(("PASS  " if ok else "FAIL  ") + name)
    passed += ok
    if not ok:
        raise SystemExit(1)


def point(tmp, version):
    (tmp / "active_model.json").write_text(json.dumps({"model_version": version}))


tmp = Path(tempfile.mkdtemp())
shutil.copytree(REAL, tmp, dirs_exist_ok=True)
x = np.random.default_rng(1).random((5, N)).astype(np.float32)

# 1. every version loads and scores
for v in ["static-v1", "adaptive-v1", "adaptive-v2", "adaptive-v3"]:
    session, thr, ver = load_model(str(tmp), v, N)
    s = session.run(["score"], {"x": x})[0]
    check(f"{v} loads, threshold {thr:.4f}, scores finite", ver == v and np.isfinite(s).all())

# 2. start on static, switch by pointer
h = ModelHolder(str(tmp), N, check_every_s=0)
h.load_initial()
check("starts on static-v1", h.version == "static-v1")
point(tmp, "adaptive-v1")
check("switches to adaptive-v1", h.maybe_reload() and h.version == "adaptive-v1")
point(tmp, "adaptive-v2")
check("switches to adaptive-v2", h.maybe_reload() and h.version == "adaptive-v2")

# 3. missing version: rejected, old model kept
point(tmp, "adaptive-v9")
h.maybe_reload()
check("missing version rejected, still adaptive-v2", h.version == "adaptive-v2" and h.reload_failures == 1)

# 4. bad pointer text and bad version name
(tmp / "active_model.json").write_text("{ not json")
h.maybe_reload()
check("corrupt pointer rejected, still adaptive-v2", h.version == "adaptive-v2" and h.reload_failures == 2)
point(tmp, "../../etc/passwd")
h.maybe_reload()
check("unsafe version name rejected", h.version == "adaptive-v2" and h.reload_failures == 3)

# 5. corrupt model file
(tmp / "adaptive" / "adaptive-v3.onnx").write_bytes(b"garbage")
point(tmp, "adaptive-v3")
h.maybe_reload()
check("corrupt ONNX rejected, still adaptive-v2", h.version == "adaptive-v2" and h.reload_failures == 4)

# 6. recovery: a good pointer works again
point(tmp, "adaptive-v1")
check("recovers to adaptive-v1", h.maybe_reload() and h.version == "adaptive-v1")

print(f"\nALL {passed} CHECKS PASSED. Last error seen: {h.last_error}")
shutil.rmtree(tmp)