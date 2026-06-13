"""Dimensional speech-emotion analysis pipeline stage for ClipCannon.

This is one of the frozen-embedder *instruments* of the panel (the emotion
axis). Per the differentiation contract it MUST emit grounded, non-degenerate
signal about real emotional state — not a constant or an arbitrary statistic
of an unrelated model.

Segments audio into 5-second windows with 2.5-second stride and predicts
dimensional **arousal** and **valence** (plus a grounded acoustic **energy**)
using a wav2vec2 model fine-tuned for dimensional speech emotion recognition
on MSP-Podcast (audEERING; Wagner et al. 2023, "Dawn of the Transformer Era
in Speech Emotion Recognition: Closing the Valence Gap", IEEE TPAMI). The
model's pooled last-layer hidden states (1024-d) are stored as the emotion
embedding in vec_emotion.

There is NO fallback: if the emotion model cannot be loaded or run, the stage
fails loudly with diagnostics. A silent RMS/zero-vector placeholder would make
this instrument contribute ~0 bits and silently corrupt the constellation.

Model license note: the audEERING model is CC BY-NC-SA 4.0 (research use). A
commercial deployment requires a commercial license from audEERING or a
drop-in dimensional-SER replacement that satisfies the same output contract
(arousal, valence in [0,1] + 1024-d pooled hidden states).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from typing import TYPE_CHECKING

import numpy as np

from clipcannon.db.connection import get_connection
from clipcannon.db.queries import batch_insert
from clipcannon.exceptions import PipelineError
from clipcannon.pipeline.orchestrator import StageResult
from clipcannon.pipeline.source_resolution import resolve_audio_input
from clipcannon.provenance import (
    ExecutionInfo,
    InputInfo,
    ModelInfo,
    OutputInfo,
    record_provenance,
    sha256_file,
    sha256_string,
)

if TYPE_CHECKING:
    from pathlib import Path

    from clipcannon.config import ClipCannonConfig

logger = logging.getLogger(__name__)

OPERATION = "emotion_analysis"
STAGE = "wav2vec2_emotion"
# Dimensional speech-emotion model (arousal/dominance/valence + 1024-d states).
# This is a fine-tuned SER model, NOT an ASR model — its outputs are grounded
# in emotional state, satisfying the instrument differentiation contract.
MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
EMBEDDING_DIM = 1024
WINDOW_S = 5.0
STRIDE_S = 2.5
TARGET_SR = 16000


def _load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio and resample to 16kHz mono.

    Args:
        audio_path: Path to the WAV file.

    Returns:
        Tuple of (mono audio as float32 array, sample rate).
    """
    try:
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)
        audio = waveform.squeeze(0).numpy()
        return audio, TARGET_SR
    except ImportError:
        pass

    # Fallback to scipy
    from scipy.io import wavfile
    from scipy.signal import resample

    sr, data = wavfile.read(str(audio_path))
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        num_samples = int(len(audio) * TARGET_SR / sr)
        audio = resample(audio, num_samples).astype(np.float32)

    return audio, TARGET_SR


def _segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
) -> list[tuple[int, int, np.ndarray]]:
    """Segment audio into overlapping windows.

    Args:
        audio: Mono audio array.
        sample_rate: Audio sample rate.
        window_s: Window duration in seconds.
        stride_s: Stride between windows in seconds.

    Returns:
        List of (start_ms, end_ms, audio_chunk) tuples.
    """
    window_samples = int(window_s * sample_rate)
    stride_samples = int(stride_s * sample_rate)
    segments: list[tuple[int, int, np.ndarray]] = []

    pos = 0
    while pos + window_samples <= len(audio):
        chunk = audio[pos : pos + window_samples]
        start_ms = int(pos / sample_rate * 1000)
        end_ms = int((pos + window_samples) / sample_rate * 1000)
        segments.append((start_ms, end_ms, chunk))
        pos += stride_samples

    # Handle trailing chunk
    if pos < len(audio) and len(audio) - pos > sample_rate:
        chunk = audio[pos:]
        start_ms = int(pos / sample_rate * 1000)
        end_ms = int(len(audio) / sample_rate * 1000)
        segments.append((start_ms, end_ms, chunk))

    return segments


def _build_emotion_model_class() -> tuple:
    """Build the dimensional-SER model class (audEERING architecture).

    Defined lazily so the module imports without torch present. The class
    matches the published audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    architecture: a Wav2Vec2 backbone whose mean-pooled last-layer hidden
    states feed a regression head producing [arousal, dominance, valence].

    Returns:
        Tuple of (EmotionModel class, torch module, nn module).
    """
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        """Regression head for arousal/dominance/valence."""

        def __init__(self, config: object) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features: object) -> object:
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        """wav2vec2 backbone + regression head; returns (pooled_states, logits)."""

        def __init__(self, config: object) -> None:
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            # transformers 5.x: post_init() (not the legacy init_weights()) wires
            # up tied-weight bookkeeping (all_tied_weights_keys) + weight init.
            self.post_init()

        def forward(self, input_values: object) -> tuple:
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            pooled = torch.mean(hidden_states, dim=1)
            logits = self.classifier(pooled)
            return pooled, logits

    return EmotionModel, torch, nn


def _compute_emotion_model(
    segments: list[tuple[int, int, np.ndarray]],
    device: str,
) -> list[dict[str, object]]:
    """Compute GROUNDED emotion features from a dimensional-SER model.

    arousal/valence: regression-head predictions in [0, 1] (grounded in
        emotional state, per Wagner et al. 2023).
    energy: normalized RMS of the raw audio waveform (grounded acoustic loudness).
    embedding: the model's mean-pooled last-layer hidden states (1024-d).

    There is no fallback. If the model cannot be loaded/run, the caller raises.

    Args:
        segments: List of (start_ms, end_ms, audio_chunk) tuples.
        device: Torch device string.

    Returns:
        List of dicts with start_ms, end_ms, energy, arousal, valence, embedding.
    """
    import gc

    from transformers import Wav2Vec2Processor

    EmotionModel, torch, _ = _build_emotion_model_class()  # noqa: N806

    # Clear VRAM left by earlier pipeline stages before loading this model.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # local_files_only=True: models MUST be pre-cached. Never download at
    # runtime (non-deterministic, breaks offline/air-gapped, slow). A missing
    # model raises here with the model id so the operator knows what to fetch.
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, local_files_only=True)
    # FP32: the wav2vec2 Conv1d feature extractor does not support FP16
    # (input/weight dtype mismatch). The 12-layer model is ~1GB — trivial on
    # the RTX 5090's 32GB.
    model = EmotionModel.from_pretrained(MODEL_ID, local_files_only=True)
    model = model.to(device)
    model.eval()

    results: list[dict[str, object]] = []
    try:
        for start_ms, end_ms, chunk in segments:
            inputs = processor(
                chunk,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(device)

            with torch.no_grad():
                pooled, logits = model(input_values)

            # Model output order is arousal, dominance, valence (audEERING).
            adv = logits.squeeze(0).cpu().numpy().astype(np.float64)
            arousal = float(np.clip(adv[0], 0.0, 1.0))
            valence = float(np.clip(adv[2], 0.0, 1.0))

            # Energy from the raw waveform (grounded acoustic loudness), NOT
            # from hidden-state statistics.
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            energy = float(np.clip(rms / (rms + 0.05), 0.0, 1.0))

            embedding = pooled.squeeze(0).cpu().numpy().astype(np.float32)
            if embedding.shape[0] != EMBEDDING_DIM:
                raise PipelineError(
                    f"Emotion embedding dim {embedding.shape[0]} != expected "
                    f"{EMBEDDING_DIM} for model {MODEL_ID}",
                    stage_name=STAGE,
                    operation=OPERATION,
                )

            results.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "energy": round(energy, 4),
                    "arousal": round(arousal, 4),
                    "valence": round(valence, 4),
                    "embedding": embedding,
                }
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _assert_not_degenerate(results)
    return results


def _assert_not_degenerate(results: list[dict[str, object]]) -> None:
    """Guard the instrument differentiation contract at the per-clip level.

    Catches the dead-instrument signatures that previously shipped: an all-zero
    embedding, or a valence that is bit-identical across every window (a real
    neural SER model effectively never does this). Either means the instrument
    is contributing ~0 bits and must fail loudly rather than poison the
    constellation centroids.

    Args:
        results: Computed emotion result dicts.

    Raises:
        PipelineError: If the emotion signal is degenerate.
    """
    if not results:
        return

    embeddings = np.stack([r["embedding"] for r in results])  # type: ignore[index]
    if not np.any(embeddings):
        raise PipelineError(
            "Emotion embeddings are all zero — dead instrument (0 bits). "
            "The SER model produced no signal; refusing to persist a "
            "degenerate emotion axis.",
            stage_name=STAGE,
            operation=OPERATION,
        )

    valences = [float(r["valence"]) for r in results]  # type: ignore[arg-type]
    if len(valences) >= 3 and len(set(valences)) == 1:
        raise PipelineError(
            f"Valence is constant ({valences[0]}) across all "
            f"{len(valences)} windows — degenerate emotion instrument "
            "(contributes 0 bits about emotional state).",
            stage_name=STAGE,
            operation=OPERATION,
        )


def _pack_embedding(embedding: np.ndarray) -> bytes:
    """Pack a float32 embedding into bytes for sqlite-vec.

    Args:
        embedding: 1-D float32 array.

    Returns:
        Packed bytes.
    """
    return struct.pack(f"{len(embedding)}f", *embedding.tolist())


def _insert_results(
    db_path: Path,
    project_id: str,
    results: list[dict[str, object]],
) -> dict[str, int]:
    """Insert emotion results into database tables.

    Args:
        db_path: Path to the project database.
        project_id: Project identifier.
        results: List of emotion result dicts.

    Returns:
        Dict with counts of inserted records per table.
    """
    counts: dict[str, int] = {}

    # Insert into emotion_curve (core table)
    conn = get_connection(db_path, enable_vec=False, dict_rows=False)
    try:
        curve_rows: list[tuple[object, ...]] = [
            (
                project_id,
                int(r["start_ms"]),
                int(r["end_ms"]),
                float(r["arousal"]),
                float(r["valence"]),
                float(r["energy"]),
            )
            for r in results
        ]
        batch_insert(
            conn,
            "emotion_curve",
            ["project_id", "start_ms", "end_ms", "arousal", "valence", "energy"],
            curve_rows,
        )
        conn.commit()
        counts["emotion_curve"] = len(curve_rows)
    except Exception as exc:
        conn.rollback()
        raise PipelineError(
            f"Failed to insert emotion_curve: {exc}",
            stage_name=STAGE,
            operation=OPERATION,
        ) from exc
    finally:
        conn.close()

    # Insert into vec_emotion (vector table). A failure here means the
    # embeddings are LOST — that is a broken state, so we RAISE rather than
    # swallow it and report a misleading count of 0.
    vec_conn = get_connection(db_path, enable_vec=True, dict_rows=False)
    try:
        vec_inserted = 0
        for r in results:
            emb = r.get("embedding")
            if not isinstance(emb, np.ndarray):
                raise PipelineError(
                    f"Missing/invalid emotion embedding for window "
                    f"{r.get('start_ms')}-{r.get('end_ms')} ms; refusing to "
                    "persist an incomplete emotion vector set.",
                    stage_name=STAGE,
                    operation=OPERATION,
                )
            emb_bytes = _pack_embedding(emb)
            vec_conn.execute(
                "INSERT INTO vec_emotion "
                "(project_id, start_ms, end_ms, emotion_embedding) "
                "VALUES (?, ?, ?, ?)",
                (project_id, int(r["start_ms"]), int(r["end_ms"]), emb_bytes),
            )
            vec_inserted += 1
        vec_conn.commit()
        counts["vec_emotion"] = vec_inserted
    except Exception as exc:
        vec_conn.rollback()
        raise PipelineError(
            f"vec_emotion insert FAILED — {len(results)} embeddings would be "
            f"LOST. sqlite-vec may not be loaded, or the embedding dim "
            f"({EMBEDDING_DIM}) does not match the vec0 table. Original error: "
            f"{type(exc).__name__}: {exc}",
            stage_name=STAGE,
            operation=OPERATION,
        ) from exc
    finally:
        vec_conn.close()

    if counts.get("vec_emotion", 0) != counts.get("emotion_curve", 0):
        raise PipelineError(
            f"Row-count mismatch: emotion_curve={counts.get('emotion_curve')} "
            f"vs vec_emotion={counts.get('vec_emotion')}. Every scalar row must "
            "have a matching embedding.",
            stage_name=STAGE,
            operation=OPERATION,
        )

    return counts


async def run_emotion_embed(
    project_id: str,
    db_path: Path,
    project_dir: Path,
    config: ClipCannonConfig,
) -> StageResult:
    """Execute the emotion/energy analysis pipeline stage.

    Segments audio into 5s windows with 2.5s stride and computes
    energy, arousal, and valence using Wav2Vec2 hidden states.
    Falls back to simple RMS energy if model loading fails.

    Args:
        project_id: Project identifier.
        db_path: Path to the project database.
        project_dir: Path to the project directory.
        config: ClipCannon configuration.

    Returns:
        StageResult indicating success or failure.
    """
    start_time = time.monotonic()

    try:
        audio_path = resolve_audio_input(project_dir)
        logger.info("Emotion analysis starting: %s", audio_path)

        # Load audio
        audio, sample_rate = await asyncio.to_thread(_load_audio, audio_path)
        logger.info(
            "Audio loaded: %d samples, %d Hz, %.1f seconds",
            len(audio),
            sample_rate,
            len(audio) / sample_rate,
        )

        # Segment audio
        segments = _segment_audio(audio, sample_rate)
        logger.info("Created %d audio segments for emotion analysis", len(segments))

        if not segments:
            return StageResult(
                success=False,
                operation=OPERATION,
                error_message="No audio segments created (audio too short)",
            )

        # Grounded dimensional-SER analysis. NO fallback: if the model cannot
        # be loaded or run, fail with full diagnostics so the operator can fix
        # the real problem (missing pre-cached model, OOM, etc.). A silent
        # placeholder would make this instrument contribute ~0 bits.
        gpu_device = str(config.get("gpu.device"))
        device = "cuda" if "cuda" in gpu_device else "cpu"
        backend_name = "wav2vec2_msp_dim"

        try:
            results = await asyncio.to_thread(
                _compute_emotion_model,
                segments,
                device,
            )
            logger.info("Dimensional SER emotion analysis succeeded")
        except PipelineError:
            raise
        except Exception as model_err:
            raise PipelineError(
                f"Emotion SER model '{MODEL_ID}' failed on device '{device}': "
                f"{type(model_err).__name__}: {model_err}. "
                "Ensure the model is pre-cached (no runtime downloads) and that "
                "VRAM is available. This stage has no fallback by design.",
                stage_name=STAGE,
                operation=OPERATION,
            ) from model_err

        # Insert results
        counts = await asyncio.to_thread(
            _insert_results,
            db_path,
            project_id,
            results,
        )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Provenance
        input_sha = await asyncio.to_thread(sha256_file, audio_path)
        summary_data = json.dumps(
            [{"s": r["start_ms"], "e": r["energy"], "a": r["arousal"]} for r in results],
            sort_keys=True,
        )
        output_sha = sha256_string(summary_data)

        record_id = record_provenance(
            db_path=db_path,
            project_id=project_id,
            operation=OPERATION,
            stage=STAGE,
            input_info=InputInfo(
                file_path=str(audio_path),
                sha256=input_sha,
                size_bytes=audio_path.stat().st_size,
            ),
            output_info=OutputInfo(
                sha256=output_sha,
                record_count=len(results),
            ),
            model_info=ModelInfo(
                name=backend_name,
                version=MODEL_ID,
                parameters={
                    "window_s": WINDOW_S,
                    "stride_s": STRIDE_S,
                    "embedding_dim": EMBEDDING_DIM,
                },
            ),
            execution_info=ExecutionInfo(
                duration_ms=elapsed_ms,
                gpu_device=device if device == "cuda" else None,
            ),
            parent_record_id=None,
            description=(
                f"Emotion analysis ({backend_name}): {len(results)} windows, "
                f"emotion_curve={counts.get('emotion_curve', 0)}, "
                f"vec_emotion={counts.get('vec_emotion', 0)}"
            ),
        )

        logger.info(
            "Emotion analysis complete in %d ms: %s",
            elapsed_ms,
            counts,
        )

        return StageResult(
            success=True,
            operation=OPERATION,
            duration_ms=elapsed_ms,
            provenance_record_id=record_id,
        )

    except PipelineError:
        raise
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Emotion analysis failed: %s", error_msg)
        return StageResult(
            success=False,
            operation=OPERATION,
            error_message=error_msg,
        )
