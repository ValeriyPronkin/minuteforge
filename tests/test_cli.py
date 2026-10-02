import json

import pytest

from minuteforge.cli import (
    build_parser,
    load_transcript,
    main,
    parse_names,
    settings_from,
)
from minuteforge.pipeline import save_transcript
from minuteforge.blocks import Block, Transcript
from minuteforge.config import Settings

SEGMENTS = [
    {"speaker": "SPEAKER_00", "text": "Сергей Ким, подготовьте план до пятницы.", "start": 0, "end": 6},
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
    """Стенограмма сохраняется дважды: текстом для людей, json для расчёта."""
    transcript = Transcript([Block("Иванов", "Раз.", 0, 5)])
    paths = save_transcript(transcript, tmp_path / "выгрузка")

    assert "Иванов: Раз." in paths["text"].read_text(encoding="utf-8")
    restored = load_transcript(paths["json"])
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
    written = sorted((tmp_path / "out").rglob("*протокол.md"))
    assert written, "протокол"
    assert sorted((tmp_path / "out").rglob("*протокол_поручения.csv")), "поручения"

    document = written[0].read_text(encoding="utf-8")
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
    assert sorted(tmp_path.rglob("*стенограмма.txt")), "стенограмма текстом"
    assert sorted(tmp_path.rglob("*стенограмма.json")), "стенограмма для расчёта"


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

        def context_window(self):
            return None, None

    monkeypatch.setattr("minuteforge.cli.LLMClient", Client)
    assert main(["check"]) == 1
    assert "нет ffmpeg" in capsys.readouterr().out


def test_the_transcript_remembers_what_recognised_it(tmp_path):
    """Модель распознавания переживает сохранение в json.

    Протокол пересобирают из готовой стенограммы, не распознавая заново, — и
    на этом пути реквизит «Распознано моделью» терялся: документ выходил без
    строки, которой решается спор, «Ессентуки» или «Исинтуки» сказал
    докладчик.
    """
    from minuteforge.blocks import Block, Transcript
    from minuteforge.cli import load_transcript
    from minuteforge.pipeline import save_transcript

    transcript = Transcript([Block("SPEAKER_00", "Начинаем.", 0, 5)])
    transcript.model = "large-v3"
    transcript.notes = ["модель уменьшена: не хватило видеопамяти"]

    paths = save_transcript(transcript, tmp_path, stem="стенограмма")
    restored = load_transcript(paths["json"])

    assert restored.model == "large-v3"
    assert restored.notes == ["модель уменьшена: не хватило видеопамяти"]
    assert [b.text for b in restored.blocks] == ["Начинаем."]


def test_a_transcript_from_an_older_version_still_reads(tmp_path):
    """Голый список реплик — файлы прежних версий, и они никуда не делись."""
    import json

    from minuteforge.cli import load_transcript

    path = tmp_path / "старая.json"
    path.write_text(
        json.dumps([{"speaker": "SPEAKER_00", "text": "Начинаем.", "start": 0, "end": 5}]),
        encoding="utf-8",
    )
    restored = load_transcript(path)

    assert [b.text for b in restored.blocks] == ["Начинаем."]
    assert restored.model == ""


def test_run_takes_the_meeting_date_from_the_recording(tmp_path, monkeypatch, capsys, fake_llm):
    """В команде run дату не спрашивают отдельным ключом там, где она лежит
    в имени файла: иначе протоколы уходят без сроков."""
    video = tmp_path / "Совещание 07.09.2026.mp4"
    video.write_bytes(b"")
    transcript = Transcript([Block("SPEAKER_00", "Подготовьте план через две недели.", 0.0, 6.0)])
    monkeypatch.setattr("minuteforge.cli.transcribe_meeting", lambda *a, **kw: transcript)
    # Чтобы проверка не зависела от того, стоит ли ffprobe на этой машине.
    monkeypatch.setattr("minuteforge.pipeline.recorded_at", lambda source: None)

    seen = {}

    def remember(transcript, settings, *, meeting=None, **kwargs):
        seen["meeting"] = meeting
        from minuteforge.protocol import Protocol
        return Protocol(date=meeting.date)

    monkeypatch.setattr("minuteforge.cli.protocol_from_transcript", remember)
    monkeypatch.setattr("minuteforge.cli.save", lambda *a, **kw: {})

    assert main(["run", str(video), "--out", str(tmp_path / "out")]) == 0
    assert seen["meeting"].date == "07.09.2026"
    assert "из имени файла" in capsys.readouterr().out


def test_a_date_given_by_hand_is_not_overridden(tmp_path, monkeypatch, fake_llm):
    """Человек знает про совещание больше, чем имя файла."""
    video = tmp_path / "Совещание 07.09.2026.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(
        "minuteforge.cli.transcribe_meeting",
        lambda *a, **kw: Transcript([Block("SPEAKER_00", "Готово.", 0.0, 2.0)]),
    )

    seen = {}

    def remember(transcript, settings, *, meeting=None, **kwargs):
        seen["meeting"] = meeting
        from minuteforge.protocol import Protocol
        return Protocol(date=meeting.date)

    monkeypatch.setattr("minuteforge.cli.protocol_from_transcript", remember)
    monkeypatch.setattr("minuteforge.cli.save", lambda *a, **kw: {})

    main(["run", str(video), "--out", str(tmp_path / "out"), "--date", "08.09.2026, 11:00"])
    assert seen["meeting"].date == "08.09.2026, 11:00"


@pytest.fixture
def anywhere_but_the_project(tmp_path, monkeypatch):
    """Папки приложения — в tmp_path, а не в проекте.

    Настройки читаются из настоящего config.yaml, и без этого расчёт писал бы
    в рабочие папки разработчика.
    """
    places = {
        "meetings_dir": tmp_path / "заседания",
        "input_dir": tmp_path / "входящие",
        "output_dir": tmp_path / "готовое",
        "work_dir": tmp_path / "промежуточное",
    }
    for name, where in places.items():
        monkeypatch.setenv(f"MINUTEFORGE_{name.upper()}", str(where))
    return places


@pytest.fixture
def heard(monkeypatch):
    """Распознавание подменено: проверяем раскладку, а не расшифровку."""
    seen = {}

    def remember(source, settings, **kwargs):
        seen.update(kwargs, source=source)
        return Transcript([Block("SPEAKER_00", "Раз.", 0, 5)])

    monkeypatch.setattr("minuteforge.cli.transcribe_meeting", remember)
    return seen


def test_result_lands_in_the_meeting_folder_of_the_recording(
    anywhere_but_the_project, heard, tmp_path
):
    """Результат ложится туда, откуда взяли запись.

    Папку под заседание заводит человек и кладёт в неё присланное
    секретарём; там же он ждёт и стенограмму. Прежде ключ --out молча
    указывал на data/output, и расшифровка уходила в перевалочную папку
    приложения — а папку заседания заводили заранее.

    Папка с присланным названа по-русски намеренно: её имя придумывает
    человек, и требовать «secretary» значит не узнать заседание.
    """
    meeting = anywhere_but_the_project["meetings_dir"] / "2026-10-01"
    (meeting / "Секретарь").mkdir(parents=True)
    (meeting / "Секретарь" / "повестка.docx").write_bytes(b"docx")
    source = meeting / "запись.mp4"
    source.write_bytes(b"video")

    assert main(["recognize", str(source)]) == 0

    assert sorted((meeting / "runs").glob("*/*стенограмма.txt")), "прогон в runs"
    assert (meeting / "стенограмма.txt").exists(), "стенограмма заседания в корне"
    assert not anywhere_but_the_project["output_dir"].exists(), (
        "в перевалочную папку приложения не должно лечь ничего"
    )


def test_a_recording_from_the_staging_folder_goes_to_the_output_folder(
    anywhere_but_the_project, heard
):
    """Перевалочная папка заседанием не бывает.

    В input_dir запись кладут на время, и складывать туда протоколы незачем:
    оттуда их никто не забирает.
    """
    inbox = anywhere_but_the_project["input_dir"]
    inbox.mkdir(parents=True)
    source = inbox / "запись.mp4"
    source.write_bytes(b"video")

    assert main(["recognize", str(source)]) == 0

    out = anywhere_but_the_project["output_dir"]
    assert sorted(out.rglob("*стенограмма.txt")), "результат в папке вывода"
    assert not sorted(inbox.rglob("*стенограмма.txt")), "во входящих ничего не осталось"


def test_intermediate_files_stay_in_the_work_folder(
    anywhere_but_the_project, heard, tmp_path
):
    """Звук из записи — промежуточное, и в папке заседания ему не место.

    Прежде рабочей папкой служила папка вывода: из часовой записи туда
    ложилось больше сотни мегабайт wav, и убирать его приходилось руками.
    """
    meeting = anywhere_but_the_project["meetings_dir"] / "2026-10-01"
    (meeting / "secretary").mkdir(parents=True)
    source = meeting / "запись.mp4"
    source.write_bytes(b"video")

    assert main(["recognize", str(source)]) == 0
    assert heard["work_dir"] == anywhere_but_the_project["work_dir"]


def test_the_out_key_still_wins(anywhere_but_the_project, heard, tmp_path):
    """Сказанное явно важнее угаданного: --out указывает место сам."""
    meeting = anywhere_but_the_project["meetings_dir"] / "2026-10-01"
    (meeting / "secretary").mkdir(parents=True)
    source = meeting / "запись.mp4"
    source.write_bytes(b"video")
    asked = tmp_path / "сетевая папка"

    assert main(["recognize", str(source), "--out", str(asked)]) == 0

    assert sorted(asked.rglob("*стенограмма.txt")), "результат там, где велели"
    assert not (meeting / "стенограмма.txt").exists(), "в заседание ничего не легло"


def test_the_run_folder_is_named_by_the_day_of_the_meeting(
    anywhere_but_the_project, heard
):
    """Дата совещания в имени папки прогона, а не день расчёта.

    Проводник сортирует прогоны по ней, а своего ключа --date у recognize
    нет: дату знает сама запись — из имени файла или из свойств контейнера.
    """
    meeting = anywhere_but_the_project["meetings_dir"] / "2026-10-01"
    (meeting / "secretary").mkdir(parents=True)
    source = meeting / "Совещание 01.10.2026.mp4"
    source.write_bytes(b"video")

    assert main(["recognize", str(source)]) == 0

    runs = [path.name for path in (meeting / "runs").iterdir()]
    assert len(runs) == 1
    assert runs[0].startswith("2026-10-01_"), runs[0]
