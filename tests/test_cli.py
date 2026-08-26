import json

import pytest

from protocall.cli import (
    build_parser,
    load_transcript,
    main,
    parse_names,
    save_transcript,
    settings_from,
)
from protocall.blocks import Block, Transcript
from protocall.config import Settings

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
    from protocall.llm import Reply

    class Client:
        def __init__(self, *a, **kw):
            pass

        def complete(self, system, user, **kwargs):
            return Reply(text=ANSWER)

        def diagnose(self):
            return ""

    monkeypatch.setattr("protocall.cli.LLMClient", Client)
    monkeypatch.setattr("protocall.pipeline.LLMClient", Client)
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
    monkeypatch.setattr("protocall.cli.transcribe_meeting", lambda *a, **kw: transcript)

    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")
    code = main(["recognize", str(source), "--out", str(tmp_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "SPEAKER_00, SPEAKER_01" in out
    assert "protocall protocol" in out, "должна быть подсказка про следующий шаг"
    assert (tmp_path / "meeting_segments.json").exists()


def test_missing_token_exits_without_a_traceback(tmp_path, monkeypatch, capsys):
    from protocall.transcribe import MissingToken

    def refuse(*args, **kwargs):
        raise MissingToken("нужен токен")

    monkeypatch.setattr("protocall.cli.transcribe_meeting", refuse)
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")

    assert main(["recognize", str(source), "--out", str(tmp_path)]) == 2
    assert "нужен токен" in capsys.readouterr().err
