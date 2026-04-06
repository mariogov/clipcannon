"""LivePortrait integration for real-time face animation.

Wraps KwaiVGI/LivePortrait for efficient single-frame inference.
Does NOT load entire videos into memory — processes frame-by-frame
to prevent WSL2 OOM crashes.

Performance target: <15ms/frame on RTX 5090 (78+ FPS).
VRAM: ~600MB for all models.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from phoenix.video.gpu_guard import check_vram, gpu_cleanup, gpu_operation, log_vram

logger = logging.getLogger(__name__)

LIVEPORTRAIT_ROOT = Path("/home/cabdru/LivePortrait")
LIVEPORTRAIT_SRC = LIVEPORTRAIT_ROOT / "src"
MODELS_DIR = LIVEPORTRAIT_ROOT / "pretrained_weights" / "liveportrait"

# Verify models exist at import time — fail fast
if not MODELS_DIR.exists():
    raise FileNotFoundError(
        f"LivePortrait models not found at {MODELS_DIR}. "
        f"Download with: huggingface-cli download KwaiVGI/LivePortrait"
    )


class LivePortraitAnimator:
    """Single-frame face animator using LivePortrait.

    Loads the reference face once, then produces animated frames
    from driving motion parameters. Thread-safe via gpu_guard lock.

    Usage:
        animator = LivePortraitAnimator()
        animator.load("santa", reference_image_path)
        frame = animator.animate(driving_keypoints)
    """

    def __init__(self) -> None:
        self._wrapper = None
        self._cropper = None
        self._source_features = None  # Cached appearance features
        self._source_motion = None    # Source neutral motion
        self._ready = False
        self._identity_name = ""

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self, identity_name: str, reference_image_path: str | Path) -> None:
        """Load LivePortrait models and prepare source identity.

        Args:
            identity_name: Human-readable name (e.g., "santa").
            reference_image_path: Path to face reference image.
                Must contain a detectable face.

        Raises:
            FileNotFoundError: If reference image or models missing.
            RuntimeError: If VRAM insufficient or face not detected.
        """
        ref_path = Path(reference_image_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

        with gpu_operation("liveportrait_load", estimated_vram_mb=800):
            self._load_models()
            self._prepare_source(ref_path)

        self._identity_name = identity_name
        self._ready = True
        logger.info(
            "LivePortrait ready for '%s' — source features cached",
            identity_name,
        )

    def _load_models(self) -> None:
        """Load LivePortrait model weights."""
        # Add LivePortrait root to path so 'src.*' imports work
        if str(LIVEPORTRAIT_ROOT) not in sys.path:
            sys.path.insert(0, str(LIVEPORTRAIT_ROOT))

        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.cropper import Cropper

        cfg = InferenceConfig()
        cfg.flag_use_half_precision = True  # FP16 for Blackwell tensor cores

        self._wrapper = LivePortraitWrapper(cfg)

        from src.config.crop_config import CropConfig
        self._crop_cfg = CropConfig()
        self._cropper = Cropper(crop_cfg=self._crop_cfg)
        log_vram("liveportrait_models_loaded")

    def _prepare_source(self, ref_path: Path) -> None:
        """Extract and cache source appearance features from reference image."""
        ref_bgr = cv2.imread(str(ref_path))
        if ref_bgr is None:
            raise RuntimeError(f"Failed to read image: {ref_path}")

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

        # Crop face
        crop_info = self._cropper.crop_source_image(ref_rgb, self._crop_cfg)
        if crop_info is None:
            raise RuntimeError(
                f"No face detected in reference image: {ref_path}. "
                f"Ensure the image contains a clearly visible face."
            )

        source_256 = crop_info["img_crop_256x256"]

        # Extract appearance features (cached for all future frames)
        x_s = self._wrapper.prepare_source(source_256).to(self._wrapper.device)
        with torch.no_grad(), self._wrapper.inference_ctx():
            self._source_features = self._wrapper.appearance_feature_extractor(x_s)
            self._source_motion = self._wrapper.motion_extractor(x_s)

        self._source_crop_info = crop_info
        logger.info(
            "Source features: appearance=%s, kp=%s",
            self._source_features.shape,
            self._source_motion["kp"].shape,
        )

    def animate_from_frame(self, driving_frame_rgb: np.ndarray) -> np.ndarray | None:
        """Generate an animated frame by transferring motion from a driving frame.

        Args:
            driving_frame_rgb: RGB uint8 frame containing a face to drive from.

        Returns:
            RGB uint8 256x256 animated frame, or None if face not detected.
        """
        if not self._ready:
            raise RuntimeError("LivePortraitAnimator not loaded. Call load() first.")

        # Crop driving face
        crop_info = self._cropper.crop_source_image(driving_frame_rgb, self._crop_cfg)
        if crop_info is None:
            return None

        drive_256 = crop_info["img_crop_256x256"]
        x_d = self._wrapper.prepare_source(drive_256).to(self._wrapper.device)

        with torch.no_grad(), self._wrapper.inference_ctx():
            x_d_info = self._wrapper.motion_extractor(x_d)

            # Reshape keypoints to (B, N, 3)
            kp_s = self._source_motion["kp"]
            kp_d = x_d_info["kp"]
            if kp_s.dim() == 2:
                kp_s = kp_s.view(kp_s.shape[0], -1, 3)
            if kp_d.dim() == 2:
                kp_d = kp_d.view(kp_d.shape[0], -1, 3)

            generated = self._wrapper.warp_decode(self._source_features, kp_s, kp_d)

        out_img = generated["out"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(out_img * 255, 0, 255).astype(np.uint8)

    def animate_from_motion(
        self,
        pitch: float = 0.0,
        yaw: float = 0.0,
        roll: float = 0.0,
        exp: np.ndarray | None = None,
        kp_delta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate an animated frame from explicit motion parameters.

        This is the method used for embedding-driven animation — our 7
        embeddings are mapped to these motion parameters.

        Args:
            pitch: Head pitch in degrees.
            yaw: Head yaw in degrees.
            roll: Head roll in degrees.
            exp: Expression coefficients (63,). If None, uses source neutral.
            kp_delta: Keypoint deltas (21, 3). If None, computed from pose.

        Returns:
            RGB uint8 256x256 animated frame.
        """
        if not self._ready:
            raise RuntimeError("LivePortraitAnimator not loaded. Call load() first.")

        with torch.no_grad(), self._wrapper.inference_ctx():
            # Source keypoints — reshape to (B, num_kp, 3) if flat
            kp_source = self._source_motion["kp"].clone()
            if kp_source.dim() == 2 and kp_source.shape[-1] % 3 == 0:
                kp_source = kp_source.view(kp_source.shape[0], -1, 3)

            kp = kp_source.clone()

            if kp_delta is not None:
                delta = torch.from_numpy(kp_delta.astype(np.float32)).to(kp.device)
                if delta.dim() == 2:
                    delta = delta.unsqueeze(0)
                if delta.dim() == 2 and delta.shape[-1] % 3 == 0:
                    delta = delta.view(delta.shape[0], -1, 3)
                kp = kp + delta

            # Apply rotation from pose
            if abs(pitch) > 0.1 or abs(yaw) > 0.1 or abs(roll) > 0.1:
                if str(LIVEPORTRAIT_ROOT) not in sys.path:
                    sys.path.insert(0, str(LIVEPORTRAIT_ROOT))
                from src.utils.camera import get_rotation_matrix
                R = get_rotation_matrix(
                    torch.tensor([[pitch]]).to(kp.device),
                    torch.tensor([[yaw]]).to(kp.device),
                    torch.tensor([[roll]]).to(kp.device),
                ).to(dtype=kp.dtype)
                # R is (1, 3, 3), kp is (1, 21, 3)
                kp = torch.bmm(kp, R)

            # Generate frame using wrapper's warp_decode (handles dict properly)
            generated = self._wrapper.warp_decode(
                self._source_features,
                kp_source,
                kp,
            )

        out_img = generated["out"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(out_img * 255, 0, 255).astype(np.uint8)

    def unload(self) -> None:
        """Release all GPU memory."""
        self._source_features = None
        self._source_motion = None
        self._wrapper = None
        self._cropper = None
        self._ready = False
        gpu_cleanup()
        logger.info("LivePortrait unloaded")
