"""4:2:2 / 10-bit chroma detection and hardware-decode routing (issue #48).

RTX Blackwell has dedicated 4:2:2 NVDEC, but exploiting it safely requires
knowing the source chroma subsampling and whether the *installed* ffmpeg build's
NVDEC actually exposes 4:2:2 pixel formats. The pipeline previously recorded no
pixel format and decoded everything with a generic ``-hwaccel cuda``, which for a
4:2:2 source silently falls back / can subsample chroma to 4:2:0 without notice.

This module:
  * detects the source chroma subsampling + bit depth (via ffprobe),
  * probes whether this ffmpeg's NVDEC decoder supports that chroma, and
  * returns decode args that use the NVDEC 4:2:2 path when available, otherwise a
    colour-preserving software decode with an explicit reason — never a silent
    chroma downgrade.

No model loads, no network.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# pix_fmt substrings -> (subsampling, is_10bit_or_more)
_SUBSAMPLING = {
    "444": "4:4:4",
    "422": "4:2:2",
    "420": "4:2:0",
}


@dataclass
class ChromaInfo:
    pix_fmt: str
    subsampling: str          # "4:2:0" | "4:2:2" | "4:4:4" | "unknown"
    bit_depth: int
    codec: str

    @property
    def is_422(self) -> bool:
        return self.subsampling == "4:2:2"

    @property
    def is_high_bit_depth(self) -> bool:
        return self.bit_depth >= 10


def probe_chroma(source_path: Path) -> ChromaInfo:
    """ffprobe the source and classify chroma subsampling + bit depth."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,bits_per_raw_sample",
            "-of", "json", str(source_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {source_path}: {proc.stderr.strip()} — "
            f"cannot determine chroma; do not guess."
        )
    streams = json.loads(proc.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream in {source_path}")
    st = streams[0]
    pix_fmt = st.get("pix_fmt", "") or ""
    codec = st.get("codec_name", "") or ""
    subsampling = "unknown"
    for key, val in _SUBSAMPLING.items():
        if key in pix_fmt:
            subsampling = val
            break
    # bit depth: prefer bits_per_raw_sample, else infer from pix_fmt suffix.
    bits = st.get("bits_per_raw_sample")
    if bits:
        bit_depth = int(bits)
    elif "p10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt:
        bit_depth = 10
    elif "p12" in pix_fmt or "12le" in pix_fmt:
        bit_depth = 12
    else:
        bit_depth = 8
    info = ChromaInfo(pix_fmt=pix_fmt, subsampling=subsampling, bit_depth=bit_depth, codec=codec)
    logger.info("chroma: %s", info)
    return info


def nvdec_supports_422(codec: str, *, ten_bit: bool = True) -> bool:
    """True iff this ffmpeg's NVDEC decoder for `codec` exposes a 4:2:2 pixel
    format (e.g. p210/p216/yuv422p10). Probes the real ffmpeg build."""
    decoder = f"{codec}_cuvid"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", f"decoder={decoder}"],
        capture_output=True, text=True, timeout=30,
    )
    text = proc.stdout + proc.stderr
    if "Supported pixel formats" not in text:
        return False
    line = next((ln for ln in text.splitlines() if "Supported pixel formats" in ln), "")
    fmts = line.split(":", 1)[1] if ":" in line else ""
    # 4:2:2 NVDEC surfaces: p210le/p216le (10/16-bit 422), yuv422p10le, etc.
    has_422 = any(tok in fmts for tok in ("p210", "p216", "422"))
    logger.info("nvdec %s 4:2:2 support: %s (formats:%s)", decoder, has_422, fmts.strip())
    return has_422


def decode_input_args(source_path: Path) -> tuple[list[str], str]:
    """Return ffmpeg *input* args for a colour-correct decode of this source.

    Returns (args, reason). For 4:2:2 sources we use NVDEC 4:2:2 when the build
    supports it; otherwise we fall back to a colour-preserving SOFTWARE decode
    (no chroma downgrade) and say why — rather than a generic -hwaccel cuda that
    can silently subsample 4:2:2 -> 4:2:0.
    """
    info = probe_chroma(source_path)
    if not info.is_422:
        # 4:2:0 (and 4:4:4) decode fine through the standard CUDA hwaccel.
        return (["-hwaccel", "cuda"], f"{info.subsampling} -> standard CUDA hwaccel")

    if nvdec_supports_422(info.codec, ten_bit=info.is_high_bit_depth):
        return (
            ["-hwaccel", "cuda", "-c:v", f"{info.codec}_cuvid"],
            f"4:2:2 {info.bit_depth}-bit -> NVDEC 4:2:2 decode ({info.codec}_cuvid)",
        )
    # NVDEC build lacks 4:2:2: decode in software to preserve full chroma.
    return (
        [],  # no -hwaccel => software decode, keeps 4:2:2 fidelity
        f"4:2:2 {info.bit_depth}-bit but this ffmpeg NVDEC has no 4:2:2 format; "
        f"using software decode to preserve chroma (upgrade to ffmpeg>=7.1 + "
        f"nv-codec-headers>=12.2 for hardware 4:2:2 on Blackwell).",
    )
