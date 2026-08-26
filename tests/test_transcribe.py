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


# ------------------------------------------------------- доступ к моделям

class Denied(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def test_access_check_passes_when_both_models_open(with_token):
    from protocall.transcribe import GATED_MODELS, check_model_access

    asked = []
    result = check_model_access("hf_test", fetch=lambda url, token: asked.append(url))
    assert set(result.values()) == {""}
    assert len(asked) == len(GATED_MODELS) == 2


def test_access_check_names_the_model_that_is_closed():
    """Моделей две, и человек обычно подписывает только одну — ту, ссылку на
    которую ему дали. Вторая тянется как зависимость и молча ломает запуск."""
    from protocall.transcribe import check_model_access

    def fetch(url, token):
        if "segmentation" in url:
            raise Denied(403)

    result = check_model_access("hf_test", fetch=fetch)
    open_models = [m for m, why in result.items() if not why]
    closed = {m: why for m, why in result.items() if why}

    assert len(open_models) == 1
    assert "segmentation-3.0" in list(closed)[0]
    assert "условия" in list(closed.values())[0]


def test_missing_token_is_reported_without_going_online():
    from protocall.transcribe import check_model_access

    result = check_model_access(
        None, fetch=lambda url, token: pytest.fail("в сеть ходить незачем")
    )
    assert all(HF_TOKEN_ENV in why for why in result.values())


def test_network_trouble_is_not_disguised_as_a_refusal():
    """Отказ в доступе и отсутствие сети лечатся по-разному."""
    from protocall.transcribe import check_model_access

    result = check_model_access(
        "hf_test", fetch=lambda url, token: (_ for _ in ()).throw(OSError("нет сети"))
    )
    assert all("нет сети" in why for why in result.values())


# ------------------------------------------------------------- ход работы

def test_progress_reports_every_step_in_order(audio, with_token):
    from protocall.transcribe import STEP_TITLES

    seen = []
    recognize(
        audio, Settings(device="cpu"), backend=FakeBackend(),
        progress=lambda step: seen.append((step.name, step.done, step.share)),
    )

    names = [name for name, done, _ in seen if not done]
    assert names == ["transcribe", "align", "diarize"]
    assert all(STEP_TITLES[n] for n in names)


def test_progress_share_only_grows_and_ends_at_one(audio, with_token):
    shares = []
    recognize(
        audio, Settings(device="cpu"), backend=FakeBackend(),
        progress=lambda step: shares.append(step.share),
    )
    assert shares == sorted(shares), "шкала не должна ехать назад"
    assert shares[-1] == 1.0, "к концу работа сделана целиком"


def test_reused_step_is_marked_as_such(audio, with_token, tmp_path):
    """Шаг, взятый из сохранённого, проходит мгновенно. Если не отличать его
    от посчитанного, человек решит, что распознавание подозрительно быстрое.
    """
    cache = tmp_path / "шаги"
    recognize(audio, Settings(device="cpu"), backend=FakeBackend(), cache_dir=cache)

    reused = []
    recognize(
        audio, Settings(device="cpu"), backend=FakeBackend(), cache_dir=cache,
        progress=lambda step: reused.append(step.reused) if step.done else None,
    )
    assert all(reused), "все три шага должны быть отмечены как взятые готовыми"


def test_broken_progress_callback_does_not_kill_the_run(audio, with_token):
    """Сорок минут работы не должны пропасть из-за ошибки в рисовании
    полоски прогресса."""
    def explode(step):
        raise RuntimeError("некуда рисовать")

    result = recognize(audio, Settings(device="cpu"), backend=FakeBackend(), progress=explode)
    assert result.segments, "работа должна закончиться нормально"


def test_finished_step_reports_how_long_it_took(audio, with_token):
    elapsed = []
    recognize(
        audio, Settings(device="cpu"), backend=FakeBackend(),
        progress=lambda step: elapsed.append(step.elapsed_s) if step.done else None,
    )
    assert len(elapsed) == 3
    assert all(value >= 0 for value in elapsed)
