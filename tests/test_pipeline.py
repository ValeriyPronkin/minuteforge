import json
from pathlib import Path

import pytest

from minuteforge.blocks import Block, Transcript
from minuteforge.config import HF_TOKEN_ENV, Settings
from minuteforge.llm import Reply
from minuteforge.pipeline import (
    Meeting,
    process,
    protocol_from_transcript,
    save,
    save_transcript,
    transcribe_meeting,
)
from minuteforge.protocol import Protocol
from minuteforge.tasks import NOTHING_FOUND

SEGMENTS = [
    {"speaker": "SPEAKER_00", "text": "Начинаем. Сроки по интеграции?", "start": 0, "end": 6},
    {"speaker": "SPEAKER_01", "text": "Осталась выгрузка справочников, нужен план работ.", "start": 6, "end": 12},
    {"speaker": "SPEAKER_00", "text": "Сергей Ким, подготовьте план до пятницы.", "start": 12, "end": 18},
]

ANSWER = """Поручение: Подготовить план работ по выгрузке справочников
Кому: Сергей Ким
Срок: до пятницы"""


class FakeBackend:
    def __init__(self):
        self.calls: list[str] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append("transcribe")
        return {"segments": SEGMENTS, "language": "ru"}

    def align(self, segments, **kwargs):
        self.calls.append("align")
        return {"segments": SEGMENTS}

    def diarize(self, audio, aligned, **kwargs):
        self.calls.append("diarize")
        return {"segments": SEGMENTS}


class FakeClient:
    def __init__(self, *answers):
        self.answers = list(answers) or [ANSWER]
        self.prompts = []
        self.warmed = False

    def complete(self, system, user, **kwargs):
        # Прогрев — не разбор: он идёт до фрагментов, чтобы загрузка модели в
        # видеопамять не выглядела отказом сервера. В очередь ответов он не
        # лезет, иначе первый фрагмент остался бы без ответа.
        if user == "Готов?":
            self.warmed = True
            return Reply(text="Да")
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
            Block("SPEAKER_01", "Осталась выгрузка справочников, нужен план работ.", 6, 12),
            Block("SPEAKER_00", "Сергей Ким, подготовьте план до пятницы.", 12, 18),
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

    monkeypatch.setattr("minuteforge.pipeline.extract_audio", fake_extract)
    transcribe_meeting(video, Settings(device="cpu"), backend=FakeBackend(), work_dir=tmp_path)
    assert calls == [video]


def test_audio_input_skips_extraction(audio, with_token, monkeypatch):
    monkeypatch.setattr(
        "minuteforge.pipeline.extract_audio",
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
    from minuteforge.pipeline import cache_dir_for

    settings = Settings(device="cpu")
    transcribe_meeting(audio, settings, backend=FakeBackend(), work_dir=tmp_path)

    cache = cache_dir_for(audio, tmp_path)
    assert (cache / "transcription.json").exists()
    assert cache.name.startswith("meeting-"), "у каждой записи своя папка"

    saved = json.loads((cache / "segments.json").read_text(encoding="utf-8"))
    assert saved["segments"][0]["speaker"] == "SPEAKER_00"


def test_transcript_is_saved_apart_from_the_protocol(tmp_path):
    """Протокол — документ, который подписывают. Стенограмма — сырьё для
    проверки спорного места. Подшитая внутрь, она превращает документ в
    расшифровку на сорок страниц, которую никто не читает.
    """
    protocol = protocol_from_transcript(transcript(), Settings(), client=FakeClient())
    paths = save(protocol, tmp_path)

    document = paths["protocol"].read_text(encoding="utf-8")
    assert "Начинаем" not in document, "стенограммы в протоколе быть не должно"
    assert "Начинаем" in paths["transcript"].read_text(encoding="utf-8")


def test_protocol_is_built_by_the_given_form(tmp_path):
    """Форма протокола в каждой организации своя, и угадать её нельзя."""
    template = tmp_path / "форма.md"
    template.write_text(
        "ПРОТОКОЛ № {{number}} от {{date}}\n"
        "Председательствующий: {{chair}}\n"
        "Секретарь: {{secretary}}\n"
        "РЕШИЛИ:\n{{tasks}}\n",
        encoding="utf-8",
    )
    protocol = protocol_from_transcript(
        transcript(),
        Settings(),
        meeting=Meeting(date="05.06.2025", chair="Орлов В.П.", secretary="Ким С.А.", number="17"),
        client=FakeClient(),
    )
    paths = save(protocol, tmp_path / "out", template=template)
    document = paths["protocol"].read_text(encoding="utf-8")

    assert document.startswith("ПРОТОКОЛ № 17 от 05.06.2025")
    assert "Секретарь: Ким С.А." in document
    assert "Сергей Ким: Подготовить план" in document


def test_unknown_placeholder_is_left_for_a_human(tmp_path):
    """Незаполненное место — не ошибка, а подсказка тому, кто дописывает
    документ руками."""
    template = tmp_path / "форма.md"
    template.write_text("Гриф: {{гриф_секретности}}\nДата: {{date}}\n", encoding="utf-8")
    protocol = protocol_from_transcript(
        transcript(), Settings(), meeting=Meeting(date="05.06.2025"), client=FakeClient()
    )
    document = save(protocol, tmp_path / "out", template=template)["protocol"].read_text(
        encoding="utf-8"
    )
    assert "{{гриф_секретности}}" in document
    assert "Дата: 05.06.2025" in document


# --------------------------------------------- своя папка на каждую запись

def make_audio(tmp_path, name, content=b"wav"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_another_recording_is_not_given_a_previous_transcript(tmp_path, with_token):
    """Самая опасная ошибка из возможных: расчёт молча выдаёт чужую
    расшифровку за свою — за секунду и с полной уверенностью.
    """
    first = make_audio(tmp_path, "первое.wav")
    second = make_audio(tmp_path, "второе.wav", b"another recording entirely")
    work = tmp_path / "work"

    transcribe_meeting(first, Settings(device="cpu"), backend=FakeBackend(), work_dir=work)

    backend = FakeBackend()
    transcribe_meeting(second, Settings(device="cpu"), backend=backend, work_dir=work)
    assert backend.calls == ["transcribe", "align", "diarize"], "вторую запись надо считать"


def test_the_same_recording_is_not_recomputed(tmp_path, with_token):
    """Ради этого шаги и сохраняются: сбой на последнем не должен стоить
    сорока минут работы."""
    audio = make_audio(tmp_path, "meeting.wav")
    work = tmp_path / "work"

    transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend(), work_dir=work)
    backend = FakeBackend()
    transcribe_meeting(audio, Settings(device="cpu"), backend=backend, work_dir=work)
    assert backend.calls == []


def test_changed_file_under_the_same_name_is_recomputed(tmp_path, with_token):
    """Файл переписали, имя осталось — это другая запись."""
    import os
    import time

    audio = make_audio(tmp_path, "meeting.wav")
    work = tmp_path / "work"
    transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend(), work_dir=work)

    audio.write_bytes(b"different content, different size")
    os.utime(audio, (time.time() + 10, time.time() + 10))

    backend = FakeBackend()
    transcribe_meeting(audio, Settings(device="cpu"), backend=backend, work_dir=work)
    assert backend.calls == ["transcribe", "align", "diarize"]


def test_fresh_forces_a_recount(tmp_path, with_token):
    audio = make_audio(tmp_path, "meeting.wav")
    work = tmp_path / "work"
    transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend(), work_dir=work)

    backend = FakeBackend()
    transcribe_meeting(
        audio, Settings(device="cpu"), backend=backend, work_dir=work, fresh=True
    )
    assert backend.calls == ["transcribe", "align", "diarize"]


def test_extracted_audio_lands_in_the_recording_own_folder(tmp_path, with_token, monkeypatch):
    """Иначе звук от одного видео достанется другому с тем же именем."""
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"video")
    work = tmp_path / "work"
    seen = {}

    def fake_extract(source, target=None, **kwargs):
        seen["target"] = Path(target)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"wav")
        return Path(target)

    monkeypatch.setattr("minuteforge.pipeline.extract_audio", fake_extract)
    transcribe_meeting(video, Settings(device="cpu"), backend=FakeBackend(), work_dir=work)

    assert seen["target"].parent != work, "звук не должен ложиться в общую папку"
    assert seen["target"].parent.name.startswith("meeting-")


def test_roll_call_is_dropped_only_on_the_way_to_the_model(tmp_path):
    """Стенограмма остаётся полной: расшифровка должна быть точной, а что
    войдёт в документ, решает тот, кто ведёт протокол, а не инструмент.
    """
    transcript_with_roll_call = Transcript([
        Block("SPEAKER_00", "Саратов, как слышно, видно?", 0, 5),
        Block("SPEAKER_01", "Да, видим, слышим.", 5, 9),
        Block("SPEAKER_02", "Коллеги, начинаем. Подготовьте план до пятницы.", 10, 60),
    ])
    client = FakeClient()
    protocol = protocol_from_transcript(transcript_with_roll_call, Settings(), client=client)

    asked = client.prompts[0]
    assert "как слышно" not in asked, "перекличка в модель идти не должна"
    assert "Подготовьте план" in asked

    saved = save(protocol, tmp_path)
    assert "как слышно" in saved["transcript"].read_text(encoding="utf-8")


def test_roll_call_filter_can_be_turned_off():
    transcript_with_roll_call = Transcript([
        Block("SPEAKER_00", "Саратов, как слышно, видно?", 0, 5),
        Block("SPEAKER_02", "Коллеги, начинаем заседание.", 10, 60),
    ])
    client = FakeClient()
    protocol_from_transcript(
        transcript_with_roll_call, Settings(drop_soundcheck=False), client=client
    )
    assert "как слышно" in client.prompts[0]


def test_only_the_asked_part_is_extracted(tmp_path, with_token, monkeypatch):
    """Первый час совещания уходит на перекличку — распознавать его значит
    час работы видеокарты впустую."""
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"video")
    seen = {}

    def fake_extract(source, target=None, **kwargs):
        seen.update(kwargs)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"wav")
        return Path(target)

    monkeypatch.setattr("minuteforge.pipeline.extract_audio", fake_extract)
    transcribe_meeting(
        video, Settings(device="cpu"), backend=FakeBackend(),
        work_dir=tmp_path, start="1:00:00", end="1:30:00",
    )
    assert seen["start"] == 3600
    assert seen["duration"] == 1800


def test_timestamps_stay_relative_to_the_whole_recording(tmp_path, with_token):
    """По времени возвращаются к спорному месту в исходном видео: «12:30»
    должно значить 12:30 записи, а не отрезка."""
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"wav")

    transcript = transcribe_meeting(
        audio, Settings(device="cpu"), backend=FakeBackend(),
        work_dir=tmp_path, start="1:00:00",
    )
    assert transcript.blocks[0].start == 3600.0


def test_different_parts_of_one_file_do_not_share_a_cache(tmp_path, with_token):
    """Распознав первый час, нельзя выдать его за второй только потому, что
    файл тот же."""
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"wav")

    transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend(),
                       work_dir=tmp_path, start="0:00")
    backend = FakeBackend()
    transcribe_meeting(audio, Settings(device="cpu"), backend=backend,
                       work_dir=tmp_path, start="1:00:00")
    assert backend.calls == ["transcribe", "align", "diarize"]


def test_backwards_range_is_an_error(tmp_path, with_token):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"wav")
    with pytest.raises(Exception, match="распознавать нечего"):
        transcribe_meeting(audio, Settings(device="cpu"), backend=FakeBackend(),
                           work_dir=tmp_path, start="10:00", end="5:00")


def test_saved_paths_are_absolute(tmp_path, monkeypatch):
    """Относительный путь ничего не говорит человеку, который потом ищет
    файлы в проводнике: показывать надо тот, по которому их найдут."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "вывод").mkdir()

    paths = save_transcript(Transcript([Block("И", "Слово.", 0, 1)]), "вывод")
    assert all(p.is_absolute() for p in paths.values())

    written = save(Protocol(tasks=[]), "вывод", with_transcript=False)
    assert all(p.is_absolute() for p in written.values())


def test_each_run_gets_its_own_folder(tmp_path):
    """Второй прогон той же записи не должен затирать первый: без истории
    сравнить настройки не с чем, а «стало лучше» на глаз не определяется."""
    from datetime import datetime

    from minuteforge.pipeline import run_dir

    folder = run_dir(tmp_path, "совещание", stamp=datetime(2026, 8, 27, 16, 50))
    assert folder.name == "2026-08-27_1650_совещание"
    assert folder.parent == tmp_path.resolve()


def test_second_protocol_does_not_overwrite_the_first(tmp_path):
    """Внутри одного разбора протокол пересобирают по нескольку раз — с
    другой моделью, с другим окном. Какой из них лучше, решает человек."""
    first = save(Protocol(tasks=[]), tmp_path, stem="протокол", with_transcript=False)
    second = save(Protocol(tasks=[]), tmp_path, stem="протокол", with_transcript=False)

    assert first["protocol"].name == "протокол.md"
    assert second["protocol"].name == "протокол_2.md"
    assert first["protocol"].exists()


def test_model_is_warmed_up_before_the_run():
    """Загрузка модели в видеопамять занимает десятки секунд, и раньше в это
    окно попадал первый рабочий запрос: человек видел «модель не отвечает» на
    записи, которую только что сорок минут распознавали."""
    from minuteforge.pipeline import warm_up

    asked = []

    class Model:
        def complete(self, system, user, **kwargs):
            asked.append(user)
            return Reply(text="Готов")

    warm_up(Model())
    assert asked == ["Готов?"]


def test_failed_warm_up_does_not_stop_the_run():
    """Настоящие запросы всё равно повторяются — сдаваться на прогреве не за
    что."""
    from minuteforge.llm import LLMUnavailable
    from minuteforge.pipeline import warm_up

    class Dead:
        def complete(self, system, user, **kwargs):
            raise LLMUnavailable("сервер молчит")

    warm_up(Dead())


def test_unwritable_folder_is_reported_before_the_run(tmp_path):
    """Папка для результатов часто сетевая, а сетевой диск отваливается.
    Узнавать об этом после сорока минут распознавания — терять сорок минут."""
    from minuteforge.pipeline import check_writable

    assert check_writable(tmp_path / "новая") == ""

    busy = tmp_path / "занято"
    busy.write_text("это файл, а не папка", encoding="utf-8")
    trouble = check_writable(busy)
    assert "не получается" in trouble
    assert "сетевой диск" in trouble, "подсказка нужна: чаще всего папка сетевая"


def test_writability_check_leaves_nothing_behind(tmp_path):
    from minuteforge.pipeline import check_writable

    check_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []
