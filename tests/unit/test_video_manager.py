from __future__ import annotations

import pytest

from face_swap.errors import InputError
from face_swap.video_manager import parse_probe_json

_PROBE_WITH_AUDIO = """
{"streams": [
  {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "24/1",
   "nb_frames": "720", "codec_name": "h264", "pix_fmt": "yuv420p"},
  {"codec_type": "audio", "codec_name": "aac"}
], "format": {"duration": "30.0"}}
"""

_PROBE_VFR_NO_NBFRAMES = """
{"streams": [
  {"codec_type": "video", "width": 1280, "height": 720, "r_frame_rate": "30000/1001",
   "codec_name": "h264", "pix_fmt": "yuv420p"}
], "format": {"duration": "10.0"}}
"""

_PROBE_NO_VIDEO = '{"streams": [{"codec_type": "audio", "codec_name": "aac"}], "format": {}}'


def test_parse_basic_with_audio():
    m = parse_probe_json(_PROBE_WITH_AUDIO, "x.mp4")
    assert m.width == 1920 and m.height == 1080
    assert m.fps == 24.0
    assert m.n_frames == 720
    assert m.has_audio and m.audio_codec == "aac"


def test_vfr_frames_derived_from_fps_duration():
    m = parse_probe_json(_PROBE_VFR_NO_NBFRAMES, "x.mp4")
    # 29.97 * 10 ≈ 300
    assert m.n_frames == 300
    assert not m.has_audio


def test_no_video_stream_raises():
    with pytest.raises(InputError):
        parse_probe_json(_PROBE_NO_VIDEO, "x.mp4")
