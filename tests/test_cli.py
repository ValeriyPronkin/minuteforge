import json

import pytest

from minuteforge.cli import (
    build_parser,
    load_transcript,
    main,
    parse_names,
    save_transcript,
    settings_from,
)
from minuteforge.blocks import Block, Transcript
from minuteforge.config import Settings

SEGMENTS = [
    {"speaker": "SPEAKER_00", "text": "Подготовьте план до пятницы.", "start": 0, "end": 6},
    {"speaker": "SPEAKER_01", "text": "Принято.", "start": 6, "end": 8},
]

ANSWER = "Поручение: Подготовить план\nКому: Орлов В.П.\nСрок: до пятницы"


@pytest.fixture
def segments_file(tmp_path):
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(SEGMENTS, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def fake_llm(monkeypatch):
    from minuteforge.llm import Reply

    class Client:
        def __init__(self, *a, **kw):
            pass

        def complete(self, system, user, **kwargs):
            return Reply(text=ANSWER)

        def diagnose(self):
            return ""

    monkeypatch.setattr("minuteforge.cli.LLMClient", Client)
    monkeypatch.setattr("minuteforge.pipeline.LLMClient", Client)
    return Client


def test_names_are_parsed_from_pairs():
    assert parse_names(["SPEAKER_00=Орлов В.П."]) == {"SPEAKER_00": "Орлов В.П."}


def test_name_may_contain_equals_sign():
    """Делим по первому знаку равенства: имя — это всё, что после него."""
    assert parse_names(["SPEAKER_00=ООО «Ромашка» = подрядчик"]) == {
        "SPEAKER_00": "ООО «Ромашка» = подрядчик"
    }


def test_broken_pair_says_what_was_expected():
    with pytest.raises(ValueError, match="SPEAKER_00"):
        parse_names(["просто имя"])


def test_command_line_wins_over_the_config():
    args = build_parser().parse_args(["recognize", "meeting.mp4", "--model", "large-v3"])
    settings = settings_from(args, Settings(asr_model="medium", language="ru"))
    assert settings.asr_model == "large-v3"
    assert settings.language == "ru", "незаданное не должно затираться"


def test_transcript_round_trip(tmp_path):
    transcript = Transcript([Block("Иванов", "Раз.", 0, 5)])
    path = save_transcript(transcript, tmp_path / "выгрузка" / "segments.json")
    restored = load_transcript(path)
    assert restored.blocks[0].speaker == "Иванов"
    assert restored.blocks[0].end == 5


def test_whole_recognition_result_is_also_accepted(tmp_path):
    """Стенограмму сохраняют по-разному: списком сегментов и целым ответом
    распознавания. Читаться должны оба."""
    path = tmp_path / "segments.json"
    path.write_text(json.dumps({"segments": SEGMENTS}, ensure_ascii=False), encoding="utf-8")
    assert len(load_transcript(path).blocks) == 2


def test_protocol_command_writes_both_files(segments_file, tmp_path, fake_llm, capsys):
    code = main([
        "protocol", str(segments_file),
        "--out", str(tmp_path / "out"),
        "--name", "SPEAKER_00=Орлов В.П.",
        "--date", "05.06.2025",
    ])
    assert code == 0
    assert (tmp_path / "out" / "protocol.md").exists()
    assert (tmp_path / "out" / "protocol_tasks.csv").exists()

    document = (tmp_path / "out" / "protocol.md").read_text(encoding="utf-8")
    assert "Орлов В.П." in document
    assert "05.06.2025" in document
    assert "Поручений: 1" in capsys.readouterr().out


def test_missing_transcript_is_an_error_not_a_traceback(tmp_path, capsys):
    code = main(["protocol", str(tmp_path / "нет.json"), "--out", str(tmp_path)])
    assert code == 1
    assert "не найден" in capsys.readouterr().err


def test_broken_name_pair_stops_before_calling_the_model(segments_file, tmp_path, capsys):
    code = main([
        "protocol", str(segments_file), "--out", str(tmp_path), "--name", "ерунда"
    ])
    assert code == 1
    assert "Ошибка" in capsys.readouterr().err


def test_recognize_reports_labels_for_the_next_step(tmp_path, monkeypatch, capsys):
    """Распознавание должно закончиться подсказкой, что делать дальше:
    метки без имён — это ровно то, чего человек не знает наизусть."""
    transcript = Transcript([Block("SPEAKER_00", "Раз.", 0, 5), Block("SPEAKER_01", "Два.", 5, 9)])
    monkeypatch.setattr("minuteforge.cli.transcribe_meeting", lambda *a, **kw: transcript)

    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    code = main(["recognize", str(source), "--out", str(tmp_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "SPEAKER_00, SPEAKER_01" in out
    assert "minuteforge protocol" in out, "должна быть подсказка про следующий шаг"
    assert (tmp_path / "meeting_segments.json").exists()


def test_missing_token_exits_without_a_traceback(tmp_path, monkeypatch, capsys):
    from minuteforge.transcribe import MissingToken

    def refuse(*args, **kwargs):
        raise MissingToken("нужен токен")

    monkeypatch.setattr("minuteforge.cli.transcribe_meeting", refuse)
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")

    assert main(["recognize", str(source), "--out", str(tmp_path)]) == 2
    assert "нужен токен" in capsys.readouterr().err


def test_recognize_prints_progress(tmp_path, monkeypatch, capsys):
    """В терминале ход работы — единственный признак жизни: распознавание
    часовой записи идёт десятки минут молча."""
    from minuteforge.transcribe import Step

    def fake(source, settings=None, *, progress=None, **kwargs):
        progress(Step(name="transcribe", title="Распознаю речь", share=0.05))
        progress(Step(
            name="transcribe", title="Распознаю речь", done=True, elapsed_s=12.3, share=0.65
        ))
        return Transcript([Block("SPEAKER_00", "Раз.", 0, 5)])

    monkeypatch.setattr("minuteforge.cli.transcribe_meeting", fake)
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")

    main(["recognize", str(source), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Распознаю речь — 12.3 с" in out
    assert "65%" in out


def test_compare_runs_the_same_transcript_through_each_model(segments_file, monkeypatch, capsys):
    """Стенограмма одна, чтобы разница была только в модели."""
    from minuteforge.llm import Reply

    asked = []

    class Client:
        def __init__(self, settings=None, *a, **kw):
            self.model = settings.llm_model if settings else "?"

        def complete(self, system, user, **kwargs):
            asked.append(self.model)
            # Одна модель находит поручение, другая нет.
            text = ANSWER if self.model == "qwen2.5:14b" else "НЕТ ПОРУЧЕНИЙ"
            return Reply(text=text)

        def diagnose(self):
            return ""

    monkeypatch.setattr("minuteforge.cli.LLMClient", Client)

    code = main([
        "compare", str(segments_file), "--model", "mistral", "--model", "qwen2.5:14b"
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert set(asked) == {"mistral", "qwen2.5:14b"}
    assert "mistral" in out and "qwen2.5:14b" in out
    assert "не оценка" in out, "числа нельзя подавать как приговор"


def test_unavailable_model_is_skipped_not_fatal(segments_file, monkeypatch, capsys):
    """Одна модель не скачана — это не повод бросать сравнение остальных."""
    from minuteforge.llm import Reply

    class Client:
        def __init__(self, settings=None, *a, **kw):
            self.model = settings.llm_model if settings else "?"

        def complete(self, system, user, **kwargs):
            return Reply(text=ANSWER)

        def diagnose(self):
            return "" if self.model == "mistral" else "модель не скачана"

    monkeypatch.setattr("minuteforge.cli.LLMClient", Client)

    code = main(["compare", str(segments_file), "--model", "mistral", "--model", "нетакой"])
    out = capsys.readouterr().out
    assert code == 0
    assert "нетакой: пропущена" in out


def test_compare_fails_when_nobody_answered(segments_file, monkeypatch, capsys):
    class Client:
        def __init__(self, *a, **kw):
            pass

        def diagnose(self):
            return "сервер не поднят"

    monkeypatch.setattr("minuteforge.cli.LLMClient", Client)
    assert main(["compare", str(segments_file), "--model", "mistral"]) == 1
    assert "Ни одна модель не ответила" in capsys.readouterr().err


def test_check_reports_the_environment(monkeypatch, capsys):
    """Без CUDA всё выглядит исправным и просто считает в двадцать раз
    дольше — человек решает, что так и должно быть, и ждёт часами."""
    from minuteforge.cli import describe_environment

    monkeypatch.setattr("minuteforge.cli.ffmpeg_available", lambda: False)
    notes = [note for _, note in describe_environment()]

    assert any("ffmpeg не найден" in note for note in notes)
    assert any("winget install ffmpeg" in note for note in notes), "надо сказать, как ставить"


def test_check_explains_a_cpu_only_torch(monkeypatch, capsys):
    """Самая частая ловушка: pip install torch ставит сборку без CUDA."""
    import sys as _sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(_sys.modules, "torch", fake_torch)
    monkeypatch.setattr("minuteforge.cli.ffmpeg_available", lambda: True)

    from minuteforge.cli import describe_environment

    notes = [note for good, note in describe_environment() if not good]
    assert any("download.pytorch.org" in note for note in notes)


def test_check_returns_nonzero_when_something_is_missing(monkeypatch, capsys):
    """Годится первой строкой пакетного скрипта: не настроено — не считаем."""
    monkeypatch.setattr("minuteforge.cli.describe_environment", lambda: [(False, "нет ffmpeg")])
    monkeypatch.setattr("minuteforge.cli.check_model_access", lambda *a, **kw: {})

    class Client:
        def __init__(self, *a, **kw):
            pass

        def diagnose(self):
            return ""

        def available_models(self):
            return []

    monkeypatch.setattr("minuteforge.cli.LLMClient", Client)
    assert main(["check"]) == 1
    assert "нет ffmpeg" in capsys.readouterr().out
