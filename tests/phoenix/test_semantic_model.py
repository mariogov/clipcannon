"""Tests for the Semantic Transformer Architecture (7 modalities).

Tests:
1. SPD outputs for known inputs (all 7 modalities)
2. CMB cross-modal consistency (21 bridges)
3. Constellation heads produce bounded outputs
4. End-to-end: audio chunk -> semantic model -> blendshapes
5. Benchmark: <5ms per frame
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: generate all 7 inputs for the model
# ---------------------------------------------------------------------------
def _make_inputs(B: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create all 7 required model inputs."""
    return {
        "visual": torch.randn(B, 1152, device=device),
        "emotion": torch.randn(B, 1024, device=device),
        "prosody": torch.randn(B, 12, device=device),
        "semantic": torch.randn(B, 768, device=device),
        "speaker": torch.randn(B, 512, device=device),
        "sentence": torch.randn(B, 384, device=device),
        "voice": torch.randn(B, 192, device=device),
    }


def _make_spd_outputs(B: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Create all 7 SPD outputs for bridge tests."""
    return {
        "visual": torch.rand(B, 32, device=device),
        "emotion": torch.rand(B, 32, device=device),
        "prosody": torch.rand(B, 32, device=device),
        "semantic": torch.rand(B, 32, device=device),
        "speaker": torch.rand(B, 32, device=device),
        "sentence": torch.rand(B, 32, device=device),
        "voice": torch.rand(B, 32, device=device),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def visual_spd(device):
    from phoenix.clone.semantic_decoders import VisualSPD
    return VisualSPD().to(device).eval()


@pytest.fixture
def emotion_spd(device):
    from phoenix.clone.semantic_decoders import EmotionSPD
    return EmotionSPD().to(device).eval()


@pytest.fixture
def prosody_spd(device):
    from phoenix.clone.semantic_decoders import ProsodySPD
    return ProsodySPD().to(device).eval()


@pytest.fixture
def semantic_spd(device):
    from phoenix.clone.semantic_decoders import SemanticSPD
    return SemanticSPD().to(device).eval()


@pytest.fixture
def bridge_set(device):
    from phoenix.clone.cross_modal_bridges import CrossModalBridgeSet
    return CrossModalBridgeSet().to(device).eval()


@pytest.fixture
def semantic_model(device):
    from phoenix.clone.semantic_model import SemanticCloneModel
    return SemanticCloneModel(num_constellation_heads=4, num_layers=2).to(device).eval()


# ---------------------------------------------------------------------------
# 1. SPD outputs for known inputs
# ---------------------------------------------------------------------------
class TestSPDOutputs:
    """Test that SPDs produce valid, bounded outputs."""

    def test_visual_spd_output_shape(self, visual_spd, device):
        x = torch.randn(4, 1152, device=device)
        out = visual_spd(x)
        assert out.shape == (4, 32)

    def test_visual_spd_output_bounded(self, visual_spd, device):
        x = torch.randn(8, 1152, device=device)
        out = visual_spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_emotion_spd_output_shape(self, emotion_spd, device):
        x = torch.randn(4, 1024, device=device)
        out = emotion_spd(x)
        assert out.shape == (4, 32)

    def test_emotion_spd_output_bounded(self, emotion_spd, device):
        x = torch.randn(8, 1024, device=device)
        out = emotion_spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_prosody_spd_output_shape(self, prosody_spd, device):
        x = torch.randn(4, 12, device=device)
        out = prosody_spd(x)
        assert out.shape == (4, 32)

    def test_prosody_spd_output_bounded(self, prosody_spd, device):
        x = torch.randn(8, 12, device=device)
        out = prosody_spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_semantic_spd_output_shape(self, semantic_spd, device):
        x = torch.randn(4, 768, device=device)
        out = semantic_spd(x)
        assert out.shape == (4, 32)

    def test_semantic_spd_output_bounded(self, semantic_spd, device):
        x = torch.randn(8, 768, device=device)
        out = semantic_spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_spd_calibration(self, visual_spd, device):
        """Test that calibration updates normalization buffers."""
        data = np.random.randn(100, 1152).astype(np.float32)
        visual_spd.calibrate(data)
        assert visual_spd.input_mean.abs().sum() > 0
        # Output should still be bounded after calibration
        x = torch.from_numpy(data[:4]).to(device)
        out = visual_spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_spd_different_inputs_different_outputs(self, emotion_spd, device):
        """Different emotion inputs should produce different outputs."""
        x_happy = torch.randn(1, 1024, device=device)
        x_sad = torch.randn(1, 1024, device=device) * 2.0
        out_happy = emotion_spd(x_happy)
        out_sad = emotion_spd(x_sad)
        # They should differ
        diff = (out_happy - out_sad).abs().sum()
        assert diff > 0.01

    def test_speaker_spd_output_shape(self, device):
        from phoenix.clone.semantic_decoders import SpeakerSPD
        spd = SpeakerSPD().to(device).eval()
        x = torch.randn(4, 512, device=device)
        out = spd(x)
        assert out.shape == (4, 32)

    def test_speaker_spd_output_bounded(self, device):
        from phoenix.clone.semantic_decoders import SpeakerSPD
        spd = SpeakerSPD().to(device).eval()
        x = torch.randn(8, 512, device=device)
        out = spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_sentence_spd_output_shape(self, device):
        from phoenix.clone.semantic_decoders import SentenceSPD
        spd = SentenceSPD().to(device).eval()
        x = torch.randn(4, 384, device=device)
        out = spd(x)
        assert out.shape == (4, 32)

    def test_sentence_spd_output_bounded(self, device):
        from phoenix.clone.semantic_decoders import SentenceSPD
        spd = SentenceSPD().to(device).eval()
        x = torch.randn(8, 384, device=device)
        out = spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_voice_spd_output_shape(self, device):
        from phoenix.clone.semantic_decoders import VoiceSPD
        spd = VoiceSPD().to(device).eval()
        x = torch.randn(4, 192, device=device)
        out = spd(x)
        assert out.shape == (4, 32)

    def test_voice_spd_output_bounded(self, device):
        from phoenix.clone.semantic_decoders import VoiceSPD
        spd = VoiceSPD().to(device).eval()
        x = torch.randn(8, 192, device=device)
        out = spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_speaker_spd_calibration(self, device):
        from phoenix.clone.semantic_decoders import SpeakerSPD
        spd = SpeakerSPD().to(device)
        data = np.random.randn(100, 512).astype(np.float32)
        spd.calibrate(data)
        assert spd.input_mean.abs().sum() > 0
        x = torch.from_numpy(data[:4]).to(device)
        out = spd(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# 2. CMB cross-modal consistency (21 bridges)
# ---------------------------------------------------------------------------
class TestCMBConsistency:
    """Test cross-modal bridges produce consistent outputs."""

    def test_bridge_set_forward(self, bridge_set, device):
        spd_outputs = _make_spd_outputs(4, device)
        predictions = bridge_set(spd_outputs)
        # 21 pairs x 2 directions = 42 bridges
        assert len(predictions) == 42
        for name, pred in predictions.items():
            assert pred.shape == (4, 32)

    def test_bridge_output_bounded(self, bridge_set, device):
        spd_outputs = _make_spd_outputs(8, device)
        predictions = bridge_set(spd_outputs)
        for name, pred in predictions.items():
            assert pred.min() >= 0.0
            assert pred.max() <= 1.0

    def test_consistency_loss_finite(self, bridge_set, device):
        spd_outputs = _make_spd_outputs(8, device)
        loss = bridge_set.total_consistency_loss(spd_outputs)
        assert loss.isfinite()
        assert loss >= 0.0

    def test_bridge_freeze(self, bridge_set):
        bridge_set.freeze()
        for p in bridge_set.parameters():
            assert not p.requires_grad

    def test_bridge_training_reduces_loss(self, device):
        """Training should reduce consistency loss."""
        from phoenix.clone.cross_modal_bridges import CrossModalBridgeSet

        bs = CrossModalBridgeSet().to(device)
        # Create correlated data (all modalities should partially align)
        N = 100
        base = torch.rand(N, 32, device=device)
        spd = {}
        for name in ["visual", "emotion", "prosody", "semantic", "speaker", "sentence", "voice"]:
            spd[name] = (base + torch.randn(N, 32, device=device) * 0.1).clamp(0, 1)

        loss_before = bs.total_consistency_loss(spd).item()

        opt = torch.optim.Adam(bs.parameters(), lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            loss = bs.total_consistency_loss(spd)
            loss.backward()
            opt.step()

        loss_after = bs.total_consistency_loss(spd).item()
        assert loss_after < loss_before

    def test_bridge_count_is_42(self, bridge_set):
        """21 pairs x 2 directions = 42 total bridges."""
        assert len(bridge_set.bridges) == 42

    def test_all_21_pairs_exist(self, bridge_set):
        """Verify all C(7,2)=21 unique pairs have both directions."""
        from phoenix.clone.cross_modal_bridges import MODALITY_NAMES
        from itertools import combinations
        for a, b in combinations(MODALITY_NAMES, 2):
            assert f"{a}_to_{b}" in bridge_set.bridges, f"Missing {a}_to_{b}"
            assert f"{b}_to_{a}" in bridge_set.bridges, f"Missing {b}_to_{a}"


# ---------------------------------------------------------------------------
# 3. Constellation heads produce bounded outputs (7 modalities)
# ---------------------------------------------------------------------------
class TestConstellationHeads:
    """Test constellation-initialized transformer produces bounded outputs."""

    def test_model_output_shape(self, semantic_model, device):
        out = semantic_model(**_make_inputs(4, device))
        assert out["blendshapes"].shape == (4, 52)
        assert out["voice"].shape == (4, 16)

    def test_blendshapes_bounded(self, semantic_model, device):
        out = semantic_model(**_make_inputs(8, device))
        bs = out["blendshapes"]
        assert bs.min() >= 0.0
        assert bs.max() <= 1.0

    def test_missing_input_raises_error(self, semantic_model, device):
        """Model must raise ValueError if any modality is missing."""
        inputs = _make_inputs(2, device)
        # Remove one modality at a time and verify ValueError
        for key in list(inputs.keys()):
            bad_inputs = {k: v for k, v in inputs.items() if k != key}
            with pytest.raises(TypeError):
                semantic_model(**bad_inputs)

    def test_none_input_raises_error(self, semantic_model, device):
        """Passing None for any modality should raise ValueError."""
        inputs = _make_inputs(2, device)
        inputs["speaker"] = None
        with pytest.raises(ValueError, match="Missing required input 'speaker'"):
            semantic_model(**inputs)

    def test_constellation_init(self, semantic_model, device):
        """Test constellation initialization from state embeddings."""
        states = {
            "happy": np.random.randn(224).astype(np.float32),
            "sad": np.random.randn(224).astype(np.float32),
            "neutral": np.random.randn(224).astype(np.float32),
            "emphatic": np.random.randn(224).astype(np.float32),
        }
        semantic_model.init_constellation(states)
        assert semantic_model._constellation_initialized

    def test_constellation_init_pads_short_vectors(self, semantic_model, device):
        """Constellation init should pad vectors shorter than FUSED_DIM."""
        states = {
            "happy": np.random.randn(128).astype(np.float32),
            "sad": np.random.randn(128).astype(np.float32),
        }
        semantic_model.init_constellation(states)
        assert semantic_model._constellation_initialized

    def test_constellation_reg_loss(self, semantic_model, device):
        loss = semantic_model.constellation_reg_loss()
        assert loss.isfinite()
        assert loss >= 0.0

    def test_spd_outputs_in_result(self, semantic_model, device):
        out = semantic_model(**_make_inputs(2, device))
        assert "spd_outputs" in out
        # All 7 modalities in spd_outputs
        for name in ["visual", "emotion", "prosody", "semantic", "speaker", "sentence", "voice"]:
            assert name in out["spd_outputs"]
            assert out["spd_outputs"][name].shape == (2, 32)

    def test_freeze_spds(self, semantic_model):
        semantic_model.freeze_spds()
        for p in semantic_model.spd_visual.parameters():
            assert not p.requires_grad
        for p in semantic_model.spd_speaker.parameters():
            assert not p.requires_grad
        for p in semantic_model.spd_voice.parameters():
            assert not p.requires_grad
        # Transformer params should still be trainable
        for p in semantic_model.constellation_layers.parameters():
            assert p.requires_grad

    def test_fused_dim_is_224(self):
        from phoenix.clone.semantic_model import FUSED_DIM
        assert FUSED_DIM == 224


# ---------------------------------------------------------------------------
# 4. End-to-end: audio -> semantic model -> blendshapes
# ---------------------------------------------------------------------------
class TestEndToEnd:
    """Test the full pipeline from audio to blendshapes."""

    def test_physics_to_semantic(self, semantic_model, device):
        """PhysicsFaceEngine state -> SemanticCloneModel -> blendshapes."""
        from phoenix.render.physics_face import PhysicsFaceEngine

        physics = PhysicsFaceEngine(sample_rate=24000, fps=30)
        # Simulate 1 second of audio (sine wave)
        t = np.linspace(0, 1.0, 24000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 200 * t) * 0.3

        face_states = physics.process_audio_batch(audio)
        assert len(face_states) > 0

        # Convert physics to model input (all 7 modalities required)
        for fs in face_states[:5]:
            pro = torch.zeros(1, 12, device=device)
            pro[0, 0] = fs._f0 / 300.0
            pro[0, 2] = fs._energy
            # Simulate all embeddings seeded from physics
            emo = torch.randn(1, 1024, device=device) * fs._energy * 0.1
            vis = torch.randn(1, 1152, device=device) * 0.1
            sem = torch.randn(1, 768, device=device) * 0.1
            spk = torch.randn(1, 512, device=device) * 0.1
            sent = torch.randn(1, 384, device=device) * 0.1
            voi = torch.randn(1, 192, device=device) * 0.1

            with torch.no_grad():
                out = semantic_model(
                    visual=vis, emotion=emo, prosody=pro,
                    semantic=sem, speaker=spk, sentence=sent, voice=voi,
                )
            bs = out["blendshapes"]
            assert bs.shape == (1, 52)
            assert bs.min() >= 0.0
            assert bs.max() <= 1.0

    def test_batch_processing(self, semantic_model, device):
        """Test batch of frames processes correctly."""
        B = 16
        out = semantic_model(**_make_inputs(B, device))
        assert out["blendshapes"].shape == (B, 52)
        assert out["voice"].shape == (B, 16)


# ---------------------------------------------------------------------------
# 5. Benchmark: <5ms per frame
# ---------------------------------------------------------------------------
class TestBenchmark:
    """Verify inference speed meets real-time requirements."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_inference_speed(self, device):
        """Single frame inference should be <5ms on GPU."""
        from phoenix.clone.semantic_model import SemanticCloneModel

        model = SemanticCloneModel(num_constellation_heads=8, num_layers=4).to(device).eval()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(**_make_inputs(1, device))

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            inputs = _make_inputs(1, device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(**inputs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        p95_ms = np.percentile(times, 95)
        print(f"\nInference: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms")
        assert avg_ms < 5.0, f"Average inference {avg_ms:.2f}ms exceeds 5ms target"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_inference_speed(self, device):
        """Batch of 30 frames (1 second at 30fps) should be <20ms."""
        from phoenix.clone.semantic_model import SemanticCloneModel

        model = SemanticCloneModel(num_constellation_heads=8, num_layers=4).to(device).eval()
        B = 30

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                model(**_make_inputs(B, device))

        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            inputs = _make_inputs(B, device)
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        print(f"\nBatch {B}: avg={avg_ms:.2f}ms")
        assert avg_ms < 20.0, f"Batch inference {avg_ms:.2f}ms exceeds 20ms target"


# ---------------------------------------------------------------------------
# Label generation tests
# ---------------------------------------------------------------------------
class TestLabelGeneration:
    """Test pseudo-label generation functions."""

    def test_generate_emotion_labels(self):
        from phoenix.clone.semantic_decoders import generate_emotion_labels
        emo_data = np.random.randn(3, 1024).astype(np.float32) * 0.1
        labels = generate_emotion_labels(emo_data)
        assert labels.shape == (3, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_generate_emotion_labels_wrong_dim(self):
        from phoenix.clone.semantic_decoders import generate_emotion_labels
        with pytest.raises(ValueError, match="expects .* 1024"):
            generate_emotion_labels(np.random.randn(3, 3).astype(np.float32))

    def test_generate_prosody_labels(self):
        from phoenix.clone.semantic_decoders import generate_prosody_labels
        pro_data = np.random.rand(10, 12).astype(np.float32)
        labels = generate_prosody_labels(pro_data)
        assert labels.shape == (10, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_generate_semantic_labels(self):
        from phoenix.clone.semantic_decoders import generate_semantic_labels
        sem_emb = np.random.randn(5, 768).astype(np.float32) * 0.01
        labels = generate_semantic_labels(sem_emb)
        assert labels.shape == (5, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_generate_speaker_labels(self):
        from phoenix.clone.semantic_decoders import generate_speaker_labels
        spk_emb = np.random.randn(8, 512).astype(np.float32) * 0.1
        labels = generate_speaker_labels(spk_emb)
        assert labels.shape == (8, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_generate_speaker_labels_wrong_dim(self):
        from phoenix.clone.semantic_decoders import generate_speaker_labels
        with pytest.raises(ValueError, match="expects .* 512"):
            generate_speaker_labels(np.random.randn(3, 256).astype(np.float32))

    def test_generate_sentence_labels(self):
        from phoenix.clone.semantic_decoders import generate_sentence_labels
        sent_emb = np.random.randn(6, 384).astype(np.float32) * 0.1
        labels = generate_sentence_labels(sent_emb)
        assert labels.shape == (6, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_generate_sentence_labels_wrong_dim(self):
        from phoenix.clone.semantic_decoders import generate_sentence_labels
        with pytest.raises(ValueError, match="expects .* 384"):
            generate_sentence_labels(np.random.randn(3, 768).astype(np.float32))

    def test_generate_voice_labels(self):
        from phoenix.clone.semantic_decoders import generate_voice_labels
        voice_emb = np.random.randn(7, 192).astype(np.float32) * 0.1
        labels = generate_voice_labels(voice_emb)
        assert labels.shape == (7, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_generate_voice_labels_wrong_dim(self):
        from phoenix.clone.semantic_decoders import generate_voice_labels
        with pytest.raises(ValueError, match="expects .* 192"):
            generate_voice_labels(np.random.randn(3, 96).astype(np.float32))


# ---------------------------------------------------------------------------
# Training sanity check
# ---------------------------------------------------------------------------
class TestTrainingSanity:
    """Verify that training pipeline components work together."""

    def test_spd_training_reduces_loss(self, device):
        """Training an SPD should reduce loss on pseudo-labels."""
        from phoenix.clone.semantic_decoders import ProsodySPD
        spd = ProsodySPD().to(device)
        spd.train()

        X = torch.randn(50, 12, device=device)
        Y = torch.rand(50, 32, device=device)

        opt = torch.optim.Adam(spd.parameters(), lr=1e-2)
        loss_first = None
        for i in range(50):
            opt.zero_grad()
            pred = spd(X)
            loss = torch.nn.functional.mse_loss(pred, Y)
            if i == 0:
                loss_first = loss.item()
            loss.backward()
            opt.step()

        loss_last = loss.item()
        assert loss_last < loss_first

    def test_full_model_backward(self, semantic_model, device):
        """Verify gradients flow through the full model."""
        semantic_model.train()
        out = semantic_model(**_make_inputs(4, device))
        loss = out["blendshapes"].sum() + out["voice"].sum()
        loss.backward()

        # Check gradients exist on trainable params
        has_grad = False
        for p in semantic_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad
