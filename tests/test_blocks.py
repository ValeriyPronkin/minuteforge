import pytest

from minuteforge.blocks import (
    UNKNOWN,
    Block,
    Transcript,
    blocks_from_segments,
    consolidate,
    normalize_speaker,
    rename_speakers,
    speakers,
)


def segment(speaker, text, start=None, end=None):
    return {"speaker": speaker, "text": text, "start": start, "end": end}


@pytest.mark.parametrize(
    "label, expected",
    [
        ("SPEAKER_1", "SPEAKER_01"),
        ("SPEAKER 1", "SPEAKER_01"),
        ("speaker-1", "SPEAKER_01"),
        ("SPEAKER_01", "SPEAKER_01"),
        ("SPEAKER10", "SPEAKER_10"),
    ],
)
def test_speaker_label_variants_become_one(label, expected):
    """Одна метка в разном написании — один человек.

    Иначе его реплики не склеятся, а в протоколе появятся участники-призраки.
    """
    assert normalize_speaker(label) == expected


def test_numbers_are_padded_so_sorting_works():
    labels = sorted(normalize_speaker(f"SPEAKER_{i}") for i in (1, 2, 10))
    assert labels == ["SPEAKER_01", "SPEAKER_02", "SPEAKER_10"]


def test_missing_speaker_becomes_unknown_not_dropped():
    """Диаризация приписывает спикера не всегда: на наложении голосов метки
    может не быть. Текст при этом терять нельзя — в нём бывает поручение."""
    assert normalize_speaker(None) == UNKNOWN
    assert normalize_speaker("  ") == UNKNOWN
    blocks = blocks_from_segments([segment(None, "Слышно плохо")])
    assert [b.speaker for b in blocks] == [UNKNOWN]


def test_real_name_survives_normalization():
    assert normalize_speaker("Иванов И.И.") == "Иванов И.И."


def test_typo_in_label_stays_a_separate_speaker():
    """Ловушка, о которой надо знать.

    Опечатка внутри самого слова (``SPE_KER_1``) не сводится к ``SPEAKER_01``:
    угадывать это значило бы склеивать людей по догадке, а перепутать двоих
    хуже, чем показать лишнюю метку. Такая метка видна на шаге сопоставления
    с участниками, и там же лечится — человек назначает обеим одно имя.
    """
    assert normalize_speaker("SPE_KER_1") == "SPE_KER_1"

    blocks = consolidate(
        blocks_from_segments(
            [segment("SPEAKER_1", "Начнём."), segment("SPE_KER_1", "Первый вопрос.")]
        )
    )
    assert len(blocks) == 2

    fixed = rename_speakers(blocks, {"SPEAKER_01": "Иванов", "SPE_KER_1": "Иванов"})
    assert len(fixed) == 1
    assert fixed[0].text == "Начнём. Первый вопрос."


def test_consecutive_replies_are_merged():
    blocks = consolidate(
        blocks_from_segments(
            [
                segment("SPEAKER_1", "Привет.", 0, 2),
                segment("SPEAKER_1", "Начнём совещание.", 2, 5),
                segment("SPEAKER_2", "Доброе утро.", 5, 7),
            ]
        )
    )
    assert [b.speaker for b in blocks] == ["SPEAKER_01", "SPEAKER_02"]
    assert blocks[0].text == "Привет. Начнём совещание."


def test_merged_block_keeps_the_whole_time_range():
    """Начало у первой реплики, конец у последней: по этим секундам потом
    возвращаются к месту в записи."""
    blocks = consolidate(
        blocks_from_segments(
            [
                segment("SPEAKER_1", "Раз.", 10, 12),
                segment("SPEAKER_1", "Два.", 12, 19),
            ]
        )
    )
    assert (blocks[0].start, blocks[0].end) == (10.0, 19.0)
    assert blocks[0].duration == 9.0


def test_empty_segments_are_skipped():
    blocks = blocks_from_segments(
        [segment("SPEAKER_1", "   "), segment("SPEAKER_1", ""), segment("SPEAKER_1", "Да.")]
    )
    assert len(blocks) == 1


def test_broken_timings_do_not_break_reading():
    blocks = blocks_from_segments([{"speaker": "SPEAKER_1", "text": "Да.", "start": "нет"}])
    assert blocks[0].start is None
    assert blocks[0].duration is None


def test_speakers_are_listed_in_order_of_appearance():
    """Порядок по ходу совещания, а не алфавитный: сопоставлять метки с
    участниками проще, идя сверху вниз по стенограмме."""
    blocks = blocks_from_segments(
        [segment("SPEAKER_3", "Я начну."), segment("SPEAKER_1", "Продолжу."), segment("SPEAKER_3", "И ещё.")]
    )
    assert speakers(blocks) == ["SPEAKER_03", "SPEAKER_01"]


def test_unmapped_label_keeps_its_name():
    """Лучше SPEAKER_03 в протоколе, чем чужая фамилия под чужими словами."""
    blocks = [Block("SPEAKER_01", "Раз."), Block("SPEAKER_03", "Два.")]
    renamed = rename_speakers(blocks, {"SPEAKER_01": "Иванов"})
    assert [b.speaker for b in renamed] == ["Иванов", "SPEAKER_03"]


def test_transcript_reports_duration_and_text():
    transcript = Transcript(
        consolidate(
            blocks_from_segments(
                [segment("SPEAKER_1", "Раз.", 0, 90), segment("SPEAKER_2", "Два.", 90, 150)]
            )
        )
    )
    assert transcript.duration_min == 2.5
    assert transcript.as_text() == "SPEAKER_01: Раз.\nSPEAKER_02: Два."
    assert "[00:01:30]" in transcript.as_text(with_time=True)


def test_empty_transcript_is_not_an_error():
    empty = Transcript([])
    assert empty.duration_min == 0.0
    assert empty.speakers == []
    assert empty.as_text() == ""
