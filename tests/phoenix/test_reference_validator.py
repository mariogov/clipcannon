"""FSV test for reference_validator over the REAL Santa reference images.

No mocks: runs the real RetinaFace detector on real on-disk references and
asserts the known-correct verdict for each. The test fails if the validator
stops distinguishing usable references from the blurry / occluded / no-face ones.
"""

import os
import pytest

from phoenix.video.reference_validator import validate_reference

REF_DIR = os.path.expanduser("~/.clipcannon/models/santa/reference")

# Known ground truth, grounded in measured sharpness/face data:
#   good (sharp, single clear face): portrait_320 (lap77), portrait (lap55), v5 (lap25)
#   bad: portrait_768 (occluded+lap9), fullframe_600s (lap5), ref_face (no face)
EXPECTED = {
    "santa_portrait_320.jpg": True,
    "santa_portrait.jpg": True,
    "santa_portrait_v5.jpg": True,
    "santa_portrait_768.jpg": False,
    "santa_fullframe_600s.png": False,
    "santa_ref_face.jpg": False,
}


@pytest.mark.parametrize("name,expected_valid", list(EXPECTED.items()))
def test_reference_verdict(name, expected_valid):
    path = os.path.join(REF_DIR, name)
    if not os.path.isfile(path):
        pytest.skip(f"reference not present: {path}")
    res = validate_reference(path)
    assert res.valid is expected_valid, (
        f"{name}: expected valid={expected_valid}, got {res.valid}; "
        f"failures={res.failures}; metrics={res.metrics}"
    )


def test_unreadable_raises():
    with pytest.raises(FileNotFoundError):
        validate_reference("/nonexistent/not_an_image.jpg")
