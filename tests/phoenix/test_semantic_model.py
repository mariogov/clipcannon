"""Tests for the Semantic Transformer Architecture.

Tests:
1. SPD outputs for known inputs
2. CMB cross-modal consistency
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
        x = torch.randn(4, 3, device=device)
        out = emotion_spd(x)
        assert out.shape == (4, 32)

    def test_emotion_spd_output_bounded(self, emotion_spd, device):
        x = torch.randn(8, 3, device=device)
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
        x_happy = torch.tensor([[0.2, 0.8, 0.5]], device=device)
        x_sad = torch.tensor([[0.1, 0.2, 0.3]], device=device)
        out_happy = emotion_spd(x_happy)
        out_sad = emotion_spd(x_sad)
        # They should differ
        diff = (out_happy - out_sad).abs().sum()
        assert diff > 0.01


# ---------------------------------------------------------------------------
# 2. CMB cross-modal consistency
# ---------------------------------------------------------------------------
class TestCMBConsistency:
    """Test cross-modal bridges produce consistent outputs."""

    def test_bridge_set_forward(self, bridge_set, device):
        spd_outputs = {
            "visual": torch.rand(4, 32, device=device),
            "emotion": torch.rand(4, 32, device=device),
            "prosody": torch.rand(4, 32, device=device),
            "semantic": torch.rand(4, 32, device=device),
        }
        predictions = bridge_set(spd_outputs)
        assert len(predictions) == 9  # 9 bridges
        for name, pred in predictions.items():
            assert pred.shape == (4, 32)

    def test_bridge_output_bounded(self, bridge_set, device):
        spd_outputs = {
            "visual": torch.rand(8, 32, device=device),
            "emotion": torch.rand(8, 32, device=device),
            "prosody": torch.rand(8, 32, device=device),
            "semantic": torch.rand(8, 32, device=device),
        }
        predictions = bridge_set(spd_outputs)
        for name, pred in predictions.items():
            assert pred.min() >= 0.0
            assert pred.max() <= 1.0

    def test_consistency_loss_finite(self, bridge_set, device):
        spd_outputs = {
            "visual": torch.rand(8, 32, device=device),
            "emotion": torch.rand(8, 32, device=device),
            "prosody": torch.rand(8, 32, device=device),
            "semantic": torch.rand(8, 32, device=device),
        }
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
        # Create correlated data (visual and emotion should align)
        N = 100
        base = torch.rand(N, 32, device=device)
        spd = {
            "visual": base + torch.randn(N, 32, device=device) * 0.1,
            "emotion": base + torch.randn(N, 32, device=device) * 0.1,
            "prosody": base + torch.randn(N, 32, device=device) * 0.1,
            "semantic": base + torch.randn(N, 32, device=device) * 0.1,
        }
        # Clamp to [0,1]
        for k in spd:
            spd[k] = spd[k].clamp(0, 1)

        loss_before = bs.total_consistency_loss(spd).item()

        opt = torch.optim.Adam(bs.parameters(), lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            loss = bs.total_consistency_loss(spd)
            loss.backward()
            opt.step()

        loss_after = bs.total_consistency_loss(spd).item()
        assert loss_after < loss_before


# ---------------------------------------------------------------------------
# 3. Constellation heads produce bounded outputs
# ---------------------------------------------------------------------------
class TestConstellationHeads:
    """Test constellation-initialized transformer produces bounded outputs."""

    def test_model_output_shape(self, semantic_model, device):
        out = semantic_model(
            visual=torch.randn(4, 1152, device=device),
            emotion=torch.randn(4, 3, device=device),
            prosody=torch.randn(4, 12, device=device),
            semantic=torch.randn(4, 768, device=device),
        )
        assert out["blendshapes"].shape == (4, 52)
        assert out["voice"].shape == (4, 16)

    def test_blendshapes_bounded(self, semantic_model, device):
        out = semantic_model(
            visual=torch.randn(8, 1152, device=device),
            emotion=torch.randn(8, 3, device=device),
            prosody=torch.randn(8, 12, device=device),
        )
        bs = out["blendshapes"]
        assert bs.min() >= 0.0
        assert bs.max() <= 1.0

    def test_partial_inputs(self, semantic_model, device):
        """Model should work with only some modalities available."""
        out = semantic_model(prosody=torch.randn(2, 12, device=device))
        assert out["blendshapes"].shape == (2, 52)

    def test_constellation_init(self, semantic_model, device):
        """Test constellation initialization from state embeddings."""
        states = {
            "happy": np.random.randn(128).astype(np.float32),
            "sad": np.random.randn(128).astype(np.float32),
            "neutral": np.random.randn(128).astype(np.float32),
            "emphatic": np.random.randn(128).astype(np.float32),
        }
        semantic_model.init_constellation(states)
        assert semantic_model._constellation_initialized

    def test_constellation_reg_loss(self, semantic_model, device):
        loss = semantic_model.constellation_reg_loss()
        assert loss.isfinite()
        assert loss >= 0.0

    def test_spd_outputs_in_result(self, semantic_model, device):
        out = semantic_model(
            visual=torch.randn(2, 1152, device=device),
            emotion=torch.randn(2, 3, device=device),
        )
        assert "spd_outputs" in out
        assert "visual" in out["spd_outputs"]
        assert out["spd_outputs"]["visual"].shape == (2, 32)

    def test_freeze_spds(self, semantic_model):
        semantic_model.freeze_spds()
        for p in semantic_model.spd_visual.parameters():
            assert not p.requires_grad
        # Transformer params should still be trainable
        for p in semantic_model.constellation_layers.parameters():
            assert p.requires_grad


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

        # Convert physics to model input
        for fs in face_states[:5]:
            pro = torch.zeros(1, 12, device=device)
            pro[0, 0] = fs._f0 / 300.0
            pro[0, 2] = fs._energy
            emo = torch.tensor([[fs.effort, 0.5 + fs.lip_spread * 0.3, fs._energy]], device=device)

            with torch.no_grad():
                out = semantic_model(emotion=emo, prosody=pro)
            bs = out["blendshapes"]
            assert bs.shape == (1, 52)
            assert bs.min() >= 0.0
            assert bs.max() <= 1.0

    def test_batch_processing(self, semantic_model, device):
        """Test batch of frames processes correctly."""
        B = 16
        out = semantic_model(
            visual=torch.randn(B, 1152, device=device),
            emotion=torch.randn(B, 3, device=device),
            prosody=torch.randn(B, 12, device=device),
            semantic=torch.randn(B, 768, device=device),
        )
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
                model(prosody=torch.randn(1, 12, device=device))

        # Benchmark
        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(
                    emotion=torch.randn(1, 3, device=device),
                    prosody=torch.randn(1, 12, device=device),
                )
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
                model(prosody=torch.randn(B, 12, device=device))

        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(
                    emotion=torch.randn(B, 3, device=device),
                    prosody=torch.randn(B, 12, device=device),
                )
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
        emo_data = np.array([
            [0.18, 0.507, 0.32],
            [0.21, 0.508, 0.34],
            [0.15, 0.504, 0.29],
        ], dtype=np.float32)
        labels = generate_emotion_labels(emo_data)
        assert labels.shape == (3, 32)
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

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
        out = semantic_model(
            visual=torch.randn(4, 1152, device=device),
            emotion=torch.randn(4, 3, device=device),
            prosody=torch.randn(4, 12, device=device),
            semantic=torch.randn(4, 768, device=device),
        )
        loss = out["blendshapes"].sum() + out["voice"].sum()
        loss.backward()

        # Check gradients exist on trainable params
        has_grad = False
        for p in semantic_model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad
