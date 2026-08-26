import json

import pytest

from protocall.config import HF_TOKEN_ENV, Settings
from protocall.transcribe import (
    MissingToken,
    RecognitionError,
    recognize,
    resolve_device,
    segments_to_json,
)


class FakeBackend:
    """WhisperX без WhisperX: считает вызовы и отдаёт готовые сегменты."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, *, model, language, batch_size, device):
        self.calls.append("transcribe")
        return {"segments": [{"text": "Начнём."}], "language": language}

    def align(self, segments, *, language, audio, device):
        self.calls.append("align")
        return {"segments": [{"text": "Начнём.", "start": 0.0, "end": 2.0}]}

    def diarize(self, audio, aligned, *, token, speakers, device):
        self.calls.append("diarize")
        self.token = token
        self.speakers = speakers
        return {"segments": [{"speaker": "SPEAKER_00", "text": "Начнём.", "start": 0.0, "end": 2.0}]}


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"wav")
    return path


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_test")


def test_auto_device_prefers_the_gpu():
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_explicit_device_wins():
    assert resolve_device("cpu", cuda_available=True) == "cpu"


def test_full_run_goes_through_all_three_stages(audio, with_token):
    backend = FakeBackend()
    result = recognize(audio, Settings(device="cpu"), backend=backend)

    assert backend.calls == ["transcribe", "align", "diarize"]
    assert result.segments[0]["speaker"] == "SPEAKER_00"
    assert result.language == "ru"
    assert result.device == "cpu"


def test_token_reaches_diarization(audio, with_token):
    backend = FakeBackend()
    recognize(audio, Settings(device="cpu", speakers=3), backend=backend)
    assert backend.token == "hf_test"
    assert backend.speakers == 3


def test_missing_token_says_what_to_do(audio, monkeypatch):
    """Это не поломка, а незаконченная настройка: модели pyannote закрытые."""
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    with pytest.raises(MissingToken, match=HF_TOKEN_ENV):
        recognize(audio, Settings(device="cpu"), backend=FakeBackend())


def test_without_diarization_no_token_is_needed(audio, monkeypatch):
    """Годится, чтобы посмотреть, как вообще распознаётся запись."""
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    backend = FakeBackend()
    result = recognize(audio, Settings(device="cpu"), backend=backend, diarize=False)

    assert backend.calls == ["transcribe", "align"]
    assert result.segments and "speaker" not in result.segments[0]


def test_missing_audio_is_caught_before_loading_models(audio, with_token, tmp_path):
    backend = FakeBackend()
    with pytest.raises(RecognitionError, match="не найден"):
        recognize(tmp_path / "нет.wav", Settings(), backend=backend)
    assert backend.calls == []


def test_steps_are_saved_and_reused(audio, with_token, tmp_path):
    """Если диаризация упала на последнем шаге, сорок минут распознавания
    должны остаться при вас."""
    cache = tmp_path / "шаги"
    first = FakeBackend()
    recognize(audio, Settings(device="cpu"), backend=first, cache_dir=cache)
    assert {p.name for p in cache.iterdir()} == {"transcription.json", "aligned.json", "segments.json"}

    second = FakeBackend()
    result = recognize(audio, Settings(device="cpu"), backend=second, cache_dir=cache)
    assert second.calls == [], "ни один шаг не должен считаться заново"
    assert result.reused == ["transcription.json", "aligned.json", "segments.json"]
    assert result.segments[0]["speaker"] == "SPEAKER_00"


def test_partial_cache_continues_from_where_it_stopped(audio, with_token, tmp_path):
    cache = tmp_path / "шаги"
    cache.mkdir()
    (cache / "transcription.json").write_text(
        json.dumps({"segments": [{"text": "Было"}], "language": "ru"}), encoding="utf-8"
    )
    backend = FakeBackend()
    recognize(audio, Settings(device="cpu"), backend=backend, cache_dir=cache)
    assert backend.calls == ["align", "diarize"], "распознавание заново не нужно"


def test_offline_mode_asks_the_hub_to_stay_home(audio, with_token, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    recognize(audio, Settings(device="cpu", offline_models=True), backend=FakeBackend())
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_segments_can_be_saved_for_a_machine_without_a_gpu(audio, with_token, tmp_path):
    """По этому файлу протокол пересобирается без распознавания."""
    result = recognize(audio, Settings(device="cpu"), backend=FakeBackend())
    path = segments_to_json(result, tmp_path / "выгрузка" / "segments.json")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["speaker"] == "SPEAKER_00"
