# -*- coding: utf-8 -*-
"""Папка заседания узнаётся по приметам, а не по точному имени.

Раскладку заводит человек, и ошибка здесь тихая: заседание не узналось —
результат ушёл в перевалочную папку, а присланное секретарём не нашлось
вовсе. Поэтому приметы проверяются по одной.
"""

from minuteforge.meetings import (
    is_meeting,
    materials_in,
    meeting_folders,
    meeting_for,
    meeting_near,
    recordings_in,
    runs_root,
)


def test_a_folder_of_ours_marks_a_meeting(tmp_path):
    """Свои подпапки — имена наши, их пишет код, и спрашивать о них некого."""
    (tmp_path / "runs").mkdir()
    assert is_meeting(tmp_path)


def test_the_folder_with_the_sent_files_may_be_named_anyhow(tmp_path):
    """«Секретарь», «от секретаря», «Материалы» — как придётся.

    Требовать точного «secretary» значит не узнать заседание молча: прогон
    ляжет россыпью в папку, а стенограмма заседания в её корень не попадёт.
    """
    for name in ("Секретарь", "от секретаря", "Материалы"):
        folder = tmp_path / name.replace(" ", "_")
        (folder / name).mkdir(parents=True)
        (folder / name / "повестка.docx").write_bytes(b"docx")
        assert is_meeting(folder), name


def test_a_folder_in_the_meetings_folder_is_a_meeting_by_the_layout(tmp_path):
    """Присланное бывает и не docx — тогда заседание узнаётся по месту."""
    meeting = tmp_path / "2026-10-01"
    (meeting / "от секретаря").mkdir(parents=True)
    (meeting / "от секретаря" / "повестка.txt").write_text("план", encoding="utf-8")

    assert not is_meeting(meeting), "без папки заседаний примет нет"
    assert is_meeting(meeting, meetings_dir=tmp_path)


def test_the_meetings_folder_itself_is_not_a_meeting(tmp_path):
    """Иначе результат ляжет в корень общего диска.

    Примет в ней полно: материалы находятся в соседнем заседании этажом ниже.
    """
    meeting = tmp_path / "2026-10-01"
    meeting.mkdir()
    (meeting / "участники.pdf").write_bytes(b"pdf")
    assert materials_in(tmp_path), "примета есть"
    assert not is_meeting(tmp_path, meetings_dir=tmp_path), "а заседанием не бывает"


def test_an_ordinary_folder_is_not_a_meeting(tmp_path):
    (tmp_path / "просто папка").mkdir()
    assert not is_meeting(tmp_path / "просто папка")


def test_the_signed_protocol_is_not_a_material(tmp_path):
    """Протокол — выход, а не вход: попав в материалы, он уходил моделью в
    тезисы, то есть в подсказку распознаванию попадал готовый ответ."""
    (tmp_path / "протокол.pdf").write_bytes(b"pdf")
    assert materials_in(tmp_path) == []


def test_our_subfolders_are_not_searched_for_materials(tmp_path):
    """В «runs» лежит протокол прошлого прогона — он вернулся бы материалом."""
    (tmp_path / "runs" / "прогон").mkdir(parents=True)
    (tmp_path / "runs" / "прогон" / "повестка.docx").write_bytes(b"docx")
    assert materials_in(tmp_path) == []


def test_the_meeting_is_found_from_the_recording_upwards(tmp_path):
    """Запись кладут и в корень папки заседания, и вместе с присланным."""
    meeting = tmp_path / "2026-10-01"
    (meeting / "Секретарь").mkdir(parents=True)
    (meeting / "Секретарь" / "повестка.docx").write_bytes(b"docx")
    inside = meeting / "Секретарь" / "запись.mp4"
    inside.write_bytes(b"video")

    found, materials = meeting_near(inside, meetings_dir=tmp_path)
    assert found == meeting
    assert [path.name for path in materials] == ["повестка.docx"]


def test_a_transcript_from_a_previous_run_still_finds_its_meeting(tmp_path):
    """Стенограмма прошлого прогона лежит в «<заседание>/runs/<прогон>/» —
    это два этажа, а не один."""
    run = tmp_path / "2026-10-01" / "runs" / "2026-10-01_1200_запись"
    run.mkdir(parents=True)
    segments = run / "стенограмма.json"
    segments.write_text("[]", encoding="utf-8")

    found, _ = meeting_near(segments, meetings_dir=tmp_path)
    assert found == tmp_path / "2026-10-01"


def test_nothing_like_a_meeting_means_the_folder_of_the_file(tmp_path):
    """Результат кладут рядом с записью, а не в папку приложения."""
    source = tmp_path / "запись.mp4"
    source.write_bytes(b"video")
    found, _ = meeting_near(source)
    assert found == tmp_path


def test_the_staging_and_work_folders_are_never_a_meeting(tmp_path):
    """В одной запись лежит на время, в другой — наше промежуточное."""
    inbox = tmp_path / "входящие"
    inbox.mkdir()
    source = inbox / "запись.mp4"
    source.write_bytes(b"video")

    assert meeting_for(source, not_meetings=(inbox,)) == (None, [])
    assert meeting_for(None) == (None, [])


def test_runs_live_in_runs_of_a_meeting_and_alone_elsewhere(tmp_path):
    """Заседание — документы секретаря и подписанный протокол, прогон —
    черновик машины, и путать их нельзя."""
    meeting = tmp_path / "2026-10-01"
    (meeting / "secretary").mkdir(parents=True)
    plain = tmp_path / "перевалочная"
    plain.mkdir()

    assert runs_root(meeting) == meeting / "runs"
    assert runs_root(plain) == plain


def test_recordings_are_listed_by_meeting_newest_first(tmp_path):
    """Нужно там, где системным диалогом записи не выбрать: он открывается
    на машине с приложением, и пришедший по ссылке его не видит."""
    for day in ("2026-10-01", "2026-10-02"):
        (tmp_path / day / "Секретарь").mkdir(parents=True)
        (tmp_path / day / "запись.mp4").write_bytes(b"video")
    (tmp_path / "2026-10-02" / "Секретарь" / "вторая.m4a").write_bytes(b"audio")
    # Вытащенный звук прошлого прогона записью совещания не является.
    (tmp_path / "2026-10-01" / "runs" / "прогон").mkdir(parents=True)
    (tmp_path / "2026-10-01" / "runs" / "прогон" / "звук.wav").write_bytes(b"wav")

    found = [str(path.relative_to(tmp_path)) for path in recordings_in(tmp_path)]
    assert found == [
        "2026-10-02/запись.mp4",
        "2026-10-02/Секретарь/вторая.m4a",
        "2026-10-01/запись.mp4",
    ]
    assert [path.name for path in meeting_folders(tmp_path)] == [
        "2026-10-02", "2026-10-01",
    ]


def test_a_missing_meetings_folder_is_not_a_crash(tmp_path):
    """Общий диск отваливается молча — расчёт из-за этого падать не должен."""
    assert meeting_folders(tmp_path / "нет такой") == []
    assert recordings_in(tmp_path / "нет такой") == []
    assert materials_in(tmp_path / "нет такой") == []


def test_a_strong_sign_above_beats_a_weak_sign_below(tmp_path):
    """Запись лежит внутри присланного — результат всё равно в заседание.

    Присланное видно и из папки, в которой оно лежит, и слабая примета
    нижнего этажа побеждала сильную верхнего: поиск останавливался на
    «Секретарь», и стенограмма ложилась внутрь материалов.
    """
    meeting = tmp_path / "2026-10-01"
    (meeting / "Секретарь").mkdir(parents=True)
    (meeting / "Секретарь" / "повестка.docx").write_bytes(b"docx")
    inside = meeting / "Секретарь" / "запись.mp4"
    inside.write_bytes(b"video")

    found, materials = meeting_near(inside, meetings_dir=tmp_path)
    assert found == meeting
    assert [path.name for path in materials] == ["повестка.docx"]


def test_the_old_layout_keeps_the_sent_files_in_the_meeting_folder(tmp_path):
    """Заседания, разложенные до «secretary»: присланное лежит прямо в папке."""
    meeting = tmp_path / "2026-08-27"
    meeting.mkdir()
    (meeting / "повестка.docx").write_bytes(b"docx")
    source = meeting / "запись.mp4"
    source.write_bytes(b"video")

    found, materials = meeting_near(source, meetings_dir=tmp_path)
    assert found == meeting
    assert [path.name for path in materials] == ["повестка.docx"]


def test_without_the_meetings_folder_the_search_stays_where_the_file_is(tmp_path):
    """Нет папки заседаний — выше папки с файлом не поднимаемся.

    «Папку заседания прежней раскладки» от «папки с присланным» без папки
    заседаний отличить не из чего: раскладка у них одна. Из двух ошибок
    выбирается та, что кладёт результат рядом с записью, — этажом выше лежат
    соседние заседания, и результат встал бы рядом с ними.
    """
    meeting = tmp_path / "2026-08-27"
    meeting.mkdir()
    (meeting / "повестка.docx").write_bytes(b"docx")
    source = meeting / "запись.mp4"
    source.write_bytes(b"video")

    found, _ = meeting_near(source)
    assert found == meeting


def test_our_own_protocol_is_not_a_material(tmp_path):
    """Протокол прогона в корне заседания материалом не становится.

    Имя его начинается с метки прогона, а не со слова «протокол», и прежняя
    проверка — по началу имени — его пропускала. Попав в материалы, он
    уходит моделью в тезисы: в подсказку распознаванию приходит готовый
    ответ.
    """
    (tmp_path / "2026-10-01_1150_qwen3_протокол.docx").write_bytes(b"docx")
    (tmp_path / "последний-протокол.docx").write_bytes(b"docx")
    (tmp_path / "протокол.pdf").write_bytes(b"pdf")
    assert materials_in(tmp_path) == []


def test_an_extract_from_a_protocol_is_still_a_material(tmp_path):
    """«Выписка из протокола» — присланное к совещанию, и терять её нельзя.

    Поэтому проверяются концы имени, а не вхождение слова где угодно:
    пропажа материала выглядит как «файлов не прислали».
    """
    (tmp_path / "Выписка из протокола № 5.docx").write_bytes(b"docx")
    (tmp_path / "Доклад к протоколу.docx").write_bytes(b"docx")
    assert [path.name for path in materials_in(tmp_path)] == [
        "Выписка из протокола № 5.docx", "Доклад к протоколу.docx",
    ]


def test_hints_in_prefers_prep_and_skips_empty(tmp_path):
    from minuteforge.meetings import hints_in

    assert hints_in(None) is None
    assert hints_in(tmp_path) is None
    (tmp_path / "asr_hints.txt").write_text("Прежняя раскладка.\n", encoding="utf-8")
    assert hints_in(tmp_path) == tmp_path / "asr_hints.txt"
    (tmp_path / "prep").mkdir()
    (tmp_path / "prep" / "asr_hints.txt").write_text("", encoding="utf-8")
    assert hints_in(tmp_path) == tmp_path / "asr_hints.txt", "пустой в prep не в счёт"
    (tmp_path / "prep" / "asr_hints.txt").write_text("Новая раскладка.\n", encoding="utf-8")
    assert hints_in(tmp_path) == tmp_path / "prep" / "asr_hints.txt"
