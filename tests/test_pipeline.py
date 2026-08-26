import json

import pytest

from protocall.blocks import Block, Transcript
from protocall.config import HF_TOKEN_ENV, Settings
from protocall.llm import Reply
from protocall.pipeline import (
    Meeting,
    process,
    protocol_from_transcript,
    save,
    transcribe_meeting,
)
from protocall.tasks import NOTHING_FOUND

SEGMENTS = [
    {"speaker": "SPEAKER_00", "text": "Начинаем. Сроки по интеграции?", "start": 0, "end": 6},
    {"speaker": "SPEAKER_01", "text": "Осталась выгрузка справочников.", "start": 6, "end": 12},
    {"speaker": "SPEAKER_00", "text": "Подготовьте план до пятницы.", "start": 12, "end": 18},
]

ANSWER = """Поручение: Подготовить план работ по выгрузке справочников
Кому: Сергей Ким
Срок: до пятницы"""


class FakeBackend:
    def transcribe(self, audio, **kwargs):
        return {"segments": SEGMENTS, "language": "ru"}

    def align(self, segments, **kwargs):
        return {"segments": SEGMENTS}

    def diarize(self, audio, aligned, **kwargs):
        return {"segments": SEGMENTS}


class FakeClient:
    def __init__(self, *answers):
        self.answers = list(answers) or [ANSWER]
        self.prompts = []

    def complete(self, system, user, **kwargs):
        self.prompts.append(user)
        return Reply(text=self.answers.pop(0) if self.answers else NOTHING_FOUND)


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"wav")
    return path


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_test")


def transcript():
    return Transcript(
        [
            Block("SPEAKER_00", "Начинаем. Сроки по интеграции?", 0, 6),
            Block("SPEAKER_01", "Осталась выгрузка справочников.", 6, 12),
            Block("SPEAKER_00", "Подготовьте план до пятницы.", 12, 18),
        ]
    )


def test_transcription_merges_segments_into_replies(audio, with_token):
    result = transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend())
    assert [b.speaker for b in result.blocks] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert result.duration_min == 0.3


def test_protocol_is_built_from_a_transcript_without_a_gpu():
    """Ради этого шаг и отделён: пересобрать протокол можно на ноутбуке."""
    client = FakeClient()
    protocol = protocol_from_transcript(transcript(), Settings(), client=client)

    assert len(protocol.tasks) == 1
    assert protocol.tasks[0].who == "Сергей Ким"
    assert client.prompts, "модель должна быть спрошена"


def test_names_replace_labels_everywhere():
    protocol = protocol_from_transcript(
        transcript(),
        Settings(),
        meeting=Meeting(names={"SPEAKER_00": "Орлов В.П.", "SPEAKER_01": "Сергей Ким"}),
        client=FakeClient(),
    )
    assert protocol.attendees == ["Орлов В.П.", "Сергей Ким"]
    assert "Орлов В.П.:" in protocol.as_markdown(with_transcript=True)
    assert "SPEAKER_00" not in protocol.as_markdown(with_transcript=True)


def test_meeting_details_are_taken_as_given_not_invented():
    """Дату и место знает только тот, кто запускает расчёт."""
    protocol = protocol_from_transcript(
        transcript(),
        Settings(),
        meeting=Meeting(title="Протокол по инциденту", date="05.06.2025", place="Зал заседаний"),
        client=FakeClient(),
    )
    text = protocol.as_markdown()
    assert text.startswith("# Протокол по инциденту")
    assert "**Дата:** 05.06.2025" in text
    assert "**Место:** Зал заседаний" in text


def test_attendees_can_include_those_who_kept_silent():
    """В стенограмме их нет, а на совещании они были."""
    protocol = protocol_from_transcript(
        transcript(),
        Settings(),
        meeting=Meeting(attendees=["Орлов В.П.", "Сергей Ким", "Молчаливый участник"]),
        client=FakeClient(),
    )
    assert "Молчаливый участник" in protocol.attendees


def test_full_path_from_recording_to_protocol(audio, with_token):
    protocol = process(
        audio,
        Settings(device="cpu"),
        meeting=Meeting(names={"SPEAKER_01": "Сергей Ким"}),
        backend=FakeBackend(),
        client=FakeClient(),
    )
    assert protocol.tasks
    assert "Сергей Ким" in protocol.attendees


def test_video_gets_its_audio_extracted_first(tmp_path, with_token, monkeypatch):
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_extract(source, target=None, **kwargs):
        calls.append(source)
        path = tmp_path / "meeting.wav"
        path.write_bytes(b"wav")
        return path

    monkeypatch.setattr("protocall.pipeline.extract_audio", fake_extract)
    transcribe_meeting(video, Settings(device="cpu"), backend=FakeBackend(), work_dir=tmp_path)
    assert calls == [video]


def test_audio_input_skips_extraction(audio, with_token, monkeypatch):
    monkeypatch.setattr(
        "protocall.pipeline.extract_audio",
        lambda *a, **k: pytest.fail("извлекать звук из wav не нужно"),
    )
    transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend())


def test_saving_gives_a_document_and_a_table(tmp_path):
    protocol = protocol_from_transcript(transcript(), Settings(), client=FakeClient())
    paths = save(protocol, tmp_path / "выгрузка")

    assert paths["protocol"].read_text(encoding="utf-8").startswith("# Протокол")
    assert "Сергей Ким" in paths["tasks"].read_text(encoding="utf-8-sig")


def test_table_is_written_with_bom_for_excel(tmp_path):
    """Без BOM Excel в русской локали открывает таблицу кракозябрами."""
    protocol = protocol_from_transcript(transcript(), Settings(), client=FakeClient())
    paths = save(protocol, tmp_path)
    assert paths["tasks"].read_bytes().startswith(b"\xef\xbb\xbf")


def test_steps_are_cached_between_runs(audio, with_token, tmp_path):
    settings = Settings(device="cpu")
    transcribe_meeting(audio, settings, backend=FakeBackend(), work_dir=tmp_path)
    assert (tmp_path / "transcription.json").exists()

    saved = json.loads((tmp_path / "segments.json").read_text(encoding="utf-8"))
    assert saved["segments"][0]["speaker"] == "SPEAKER_00"
