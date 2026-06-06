"""Preflight validator for talking-head reference images.

Root-cause hardening for the failure documented in
``docs/inspect/04_output_quality.md``: a bad reference image (occluded mouth,
heavy blur, or no detectable face) silently produced a degraded EchoMimic
generation (ghostly lower face, lip-sync 2/10). This validator runs BEFORE
generation and FAILS LOUDLY on an unusable reference instead of letting the
renderer produce garbage.

Decision signals (grounded in measured data over the 6 real Santa references,
see ``tests/phoenix/test_reference_validator.py``):

  * face presence/quality  — exactly one face, RetinaFace score >= min_score.
    (santa_ref_face.jpg -> NO_FACE -> reject)
  * face size              — face box min side >= min_face_px and area fraction
    >= min_face_area_frac (rejects tiny faces in full-frame shots).
  * global sharpness       — variance-of-Laplacian >= min_sharpness.
    (santa_fullframe_600s lap=5, santa_portrait_768 lap=9 -> reject;
     v5=25, portrait=55, portrait_320=77 -> pass)
  * mouth-region sharpness — variance-of-Laplacian inside the mouth box.
    A specifically blurry/smeared mouth region is the visible signature of the
    occluded-tissue reference even when global sharpness is borderline.

KNOWN LIMITATION (documented honestly, not hidden): semantic occlusion of the
mouth by a *sharp* object (a hand or patterned cloth that is itself in focus)
is NOT reliably caught by sharpness alone. A near-white HSV heuristic was tried
and rejected because the white beard dominates the mouth region. Full semantic
mouth-occlusion detection needs a face-parsing segmentation model and is tracked
as a follow-up. This validator catches the blur + no-face + tiny-face failure
modes that actually occurred, and a blurred mouth region.

This module does NOT use mocks or fallbacks. If RetinaFace cannot load, it
raises — a broken detector must not silently pass references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# tensorflow 2.15 lazy-loads `tensorflow.keras`; retina-face 0.0.17 imports it
# eagerly and fails unless the lazy loader is triggered first. Import order fix.
import tensorflow.keras  # noqa: F401,E402  (must precede retinaface import)
import cv2  # noqa: E402
from retinaface import RetinaFace  # noqa: E402


# Defaults grounded in the 6-reference measurement (see module docstring).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_score": 0.90,          # RetinaFace detection confidence
    "min_face_px": 96,          # face box shortest side, pixels
    "min_face_area_frac": 0.04, # face box area / image area
    "min_sharpness": 15.0,      # global variance-of-Laplacian
    "min_mouth_sharpness": 8.0, # mouth-region variance-of-Laplacian
}


@dataclass
class ReferenceCheck:
    """Outcome of validating one reference image."""

    valid: bool
    image_path: str
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def _variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mouth_box(landmarks: dict, w: int, h: int) -> tuple[int, int, int, int]:
    ml = np.asarray(landmarks["mouth_left"], dtype=np.float32)
    mr = np.asarray(landmarks["mouth_right"], dtype=np.float32)
    cx, cy = (ml + mr) / 2.0
    span = float(np.linalg.norm(mr - ml))
    half_x = max(int(span * 0.9), 12)
    half_y = max(int(span * 0.7), 10)
    x0 = max(int(cx - half_x), 0)
    y0 = max(int(cy - half_y), 0)
    x1 = min(int(cx + half_x), w)
    y1 = min(int(cy + half_y), h)
    return x0, y0, x1, y1


def validate_reference(
    image_path: str,
    thresholds: dict[str, float] | None = None,
) -> ReferenceCheck:
    """Validate a reference image for talking-head generation.

    Returns a ReferenceCheck. ``valid`` is True only if every gate passes.
    Always populates ``metrics`` so the decision is transparent and auditable.
    Raises FileNotFoundError if the image cannot be read (fail loud).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Reference image unreadable: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    metrics: dict[str, Any] = {
        "width": w,
        "height": h,
        "sharpness": _variance_of_laplacian(gray),
        "n_faces": 0,
        "face_score": None,
        "face_min_px": None,
        "face_area_frac": None,
        "mouth_sharpness": None,
    }
    failures: list[str] = []

    resp = RetinaFace.detect_faces(image_path)
    if not isinstance(resp, dict) or not resp:
        failures.append("no_face_detected")
        logger.warning("REJECT %s: no face detected", image_path)
        return ReferenceCheck(False, image_path, metrics, failures)

    metrics["n_faces"] = len(resp)
    if len(resp) != 1:
        failures.append(f"expected_1_face_got_{len(resp)}")

    face = resp["face_1"]
    score = float(face["score"])
    metrics["face_score"] = score
    if score < t["min_score"]:
        failures.append(f"low_face_score_{score:.3f}<{t['min_score']}")

    x0, y0, x1, y1 = face["facial_area"]
    fw, fh = int(x1 - x0), int(y1 - y0)
    face_min = min(fw, fh)
    area_frac = (fw * fh) / float(w * h)
    metrics["face_min_px"] = face_min
    metrics["face_area_frac"] = round(area_frac, 4)
    if face_min < t["min_face_px"]:
        failures.append(f"face_too_small_{face_min}px<{int(t['min_face_px'])}")
    if area_frac < t["min_face_area_frac"]:
        failures.append(f"face_area_frac_{area_frac:.3f}<{t['min_face_area_frac']}")

    if metrics["sharpness"] < t["min_sharpness"]:
        failures.append(f"blurry_{metrics['sharpness']:.1f}<{t['min_sharpness']}")

    mx0, my0, mx1, my1 = _mouth_box(face["landmarks"], w, h)
    mouth_crop = gray[my0:my1, mx0:mx1]
    if mouth_crop.size > 0:
        ms = _variance_of_laplacian(mouth_crop)
        metrics["mouth_sharpness"] = ms
        if ms < t["min_mouth_sharpness"]:
            failures.append(f"mouth_blurry_{ms:.1f}<{t['min_mouth_sharpness']}")

    valid = not failures
    if valid:
        logger.info("ACCEPT reference %s: %s", image_path, metrics)
    else:
        logger.warning("REJECT reference %s: %s | %s", image_path, failures, metrics)
    return ReferenceCheck(valid, image_path, metrics, failures)
