"""scripts/asr_bench.py ставится в окружение GigaAM без minuteforge, поэтому
повторяет пару его правил. Здесь проверяется, что повторяет точно: иначе
minuteforge молча не найдёт расшифровку или найдёт её не там."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minuteforge import transcribe
from minuteforge.config import HF_TOKEN_ENV, Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("asr_bench", ROOT / "scripts" / "asr_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shares_rules_with_minuteforge(bench):
    assert bench.ASR_META == transcribe.ASR_META
    assert bench.EXTERNAL_PREFIX == transcribe.EXTERNAL_PREFIX
    for name in ("large-v3", "gigaam-v3_e2e_ctc", r"D:\models\мой whisper", "", "..x.."):
        assert bench.model_dir(name) == transcribe.model_dir(name)


def test_variants_are_the_punctuated_ones(bench):
    assert bench.VARIANTS == ("v3_e2e_ctc", "v3_e2e_rnnt")
    assert bench.minuteforge_name("v3_e2e_rnnt").startswith(transcribe.EXTERNAL_PREFIX)


def test_gigaam_segments_become_whisperx_transcription(bench):
    segs = [
        SimpleNamespace(text=" Доложите по Ангарску. ", start=12.3456, end=20.0),
        SimpleNamespace(text="", start=20.0, end=21.0),
        SimpleNamespace(text="Принято.", start=21.0, end=22.5),
    ]
    value = bench.to_transcription(segs, variant="v3_e2e_ctc", batch_size=8, extra={"elapsed_s": 1.0})
    assert value["segments"] == [
        {"text": "Доложите по Ангарску.", "start": 12.346, "end": 20.0},
        {"text": "Принято.", "start": 21.0, "end": 22.5},
    ]
    meta = value[bench.ASR_META]
    assert meta["model"] == "gigaam-v3_e2e_ctc"
    assert meta["hints"] == ""
    assert meta["empty_segments"] == 1


def test_minuteforge_picks_up_what_bench_wrote(bench, tmp_path, monkeypatch):
    """Весь стык целиком: что положил скрипт, то основное окружение и берёт —
    без распознавания, с реквизитами GigaAM."""
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_test")
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"wav")
    cache = tmp_path / "шаги"
    value = bench.to_transcription(
        [SimpleNamespace(text="Начнём.", start=0.0, end=2.0)],
        variant="v3_e2e_rnnt", batch_size=8, extra={},
    )
    folder = cache / bench.model_dir(bench.minuteforge_name("v3_e2e_rnnt"))
    folder.mkdir(parents=True)
    (folder / "transcription.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    calls = []

    class Backend:
        def transcribe(self, *a, **k):
            calls.append("transcribe")

        def align(self, segments, **k):
            calls.append("align")
            return {"segments": segments}

        def diarize(self, audio, aligned, **k):
            calls.append("diarize")
            return {"segments": [{**s, "speaker": "SPEAKER_00"} for s in aligned["segments"]]}

    result = transcribe.recognize(
        audio, Settings(device="cpu", asr_model="gigaam-v3_e2e_rnnt"),
        backend=Backend(), cache_dir=cache,
    )
    assert calls == ["align", "diarize"]
    assert result.asr_model == "gigaam-v3_e2e_rnnt"
    assert result.segments[0]["text"] == "Начнём."


def test_finds_the_extracted_audio(bench, tmp_path):
    (tmp_path / "large-v3").mkdir()
    (tmp_path / "запись.wav").write_bytes(b"")
    assert bench.find_audio(tmp_path).name == "запись.wav"
    (tmp_path / "ещё.wav").write_bytes(b"")
    with pytest.raises(SystemExit):
        bench.find_audio(tmp_path)


class _Answer:
    def __init__(self, body=b"{}"):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_unloads_whatever_ollama_holds(bench):
    """Как free_the_card в minuteforge, только без его настроек: что
    загружено — спрашивается у Ollama, и каждое выгружается keep_alive: 0."""
    sent = []

    def opener(request, timeout):
        if isinstance(request, str):
            assert request.endswith("/api/ps")
            return _Answer(json.dumps({"models": [
                {"name": "mistral:latest", "model": "mistral:latest"},
                {"name": "qwen14-protocol:latest", "model": "qwen14-protocol:latest"},
            ]}).encode())
        sent.append((request.full_url, json.loads(request.data)))
        return _Answer()

    freed = bench.unload_ollama("http://localhost:11434/", opener=opener)
    assert freed == ["mistral:latest", "qwen14-protocol:latest"]
    assert sent == [
        ("http://localhost:11434/api/generate", {"model": "mistral:latest", "keep_alive": 0}),
        ("http://localhost:11434/api/generate", {"model": "qwen14-protocol:latest", "keep_alive": 0}),
    ]


def test_no_ollama_is_not_an_error(bench):
    def opener(request, timeout):
        raise OSError("connection refused")

    assert bench.unload_ollama("http://localhost:11434", opener=opener) == []


@pytest.mark.parametrize("name", ["meeting.mp4", "2903kol200181270317940221100EP29102026REC.mp4"])
def test_finds_the_same_cache_folder_as_minuteforge(bench, tmp_path, name):
    """Скрипту дают ту же запись, что и minuteforge, — папка шагов должна
    выйти та же, иначе GigaAM распознает в одну, а minuteforge ищет в другой."""
    from minuteforge.pipeline import cache_dir_for

    record = tmp_path / "I" / name
    record.parent.mkdir()
    record.write_bytes(b"video")
    work = tmp_path / "work"
    assert bench.cache_for_record(record, work) == cache_dir_for(record, work)
