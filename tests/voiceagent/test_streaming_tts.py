"""Tests for StreamingTTS chunking + dispatch.

No mocks (issue #58): the TTS boundary is a *real* recording adapter — a small
hand-written class implementing the same `synthesize(text) -> np.ndarray`
contract the production adapters use. It records the exact texts the chunker
dispatched and returns real (deterministic) audio, so we verify the actual
chunking/flush behaviour against a real object, not a library call-spy.
"""
import asyncio

import numpy as np
import pytest

from voiceagent.tts.chunker import SentenceChunker
from voiceagent.tts.streaming import StreamingTTS


async def make_token_stream(tokens: list[str]):
    for token in tokens:
        yield token


def make_dummy_audio(n: int = 2400) -> np.ndarray:
    # Real, non-degenerate audio (a quiet ramp) so downstream type/dtype checks
    # exercise actual array data, not a sentinel.
    return np.linspace(-0.01, 0.01, n, dtype=np.float32)


class RecordingTTSAdapter:
    """Real adapter implementing the synthesize contract; records its inputs."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return make_dummy_audio()


async def collect_chunks(stream_iter) -> list[np.ndarray]:
    chunks = []
    async for chunk in stream_iter:
        chunks.append(chunk)
    return chunks


@pytest.fixture
def adapter():
    return RecordingTTSAdapter()


@pytest.fixture
def chunker():
    return SentenceChunker()


def test_two_sentences(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    tokens = ["I", " am", " good", ".", " You", " are", " too", ".", " "]
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream(tokens)))
    )
    assert len(chunks) == 2
    assert adapter.calls == ["I am good.", "You are too."]


def test_flush_remaining(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    tokens = ["Hi", " there"]
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream(tokens)))
    )
    assert len(chunks) == 1
    assert adapter.calls == ["Hi there"]


def test_empty_flush_skipped(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    tokens = ["I", " am", " good", ".", " "]
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream(tokens)))
    )
    assert len(chunks) == 1
    assert adapter.calls == ["I am good."]


def test_empty_stream(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream([])))
    )
    assert len(chunks) == 0
    assert adapter.calls == []


def test_single_word_flushed(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    tokens = ["Hello"]
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream(tokens)))
    )
    assert len(chunks) == 1
    assert adapter.calls == ["Hello"]


def test_yields_numpy_arrays(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    tokens = ["I", " am", " good", ".", " "]
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream(tokens)))
    )
    for chunk in chunks:
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.float32


def test_hello_how_are_you(adapter, chunker):
    tts = StreamingTTS(adapter, chunker)
    tokens = ["Hello", ".", " How", " are", " you", "?", " "]
    chunks = asyncio.get_event_loop().run_until_complete(
        collect_chunks(tts.stream(make_token_stream(tokens)))
    )
    assert len(chunks) == 1
    assert adapter.calls == ["Hello. How are you?"]
