import pytest

from protocall.blocks import Block
from protocall.chunking import Chunk, estimate_tokens, split_into_chunks


def reply(speaker="SPEAKER_01", text="Слово."):
    return Block(speaker, text)


def words(count, speaker="SPEAKER_01", mark=""):
    """Реплика нужной длины. ``mark`` делает её отличимой от соседних."""
    return Block(speaker, " ".join([mark or "слово"] * count))


def test_short_meeting_stays_one_chunk():
    chunks = split_into_chunks([reply(), reply("SPEAKER_02")], max_tokens=1000)
    assert len(chunks) == 1
    assert chunks[0].index == 1 and chunks[0].total == 1


def test_chunk_never_exceeds_the_budget():
    blocks = [words(40, mark=f"реплика{i}") for i in range(20)]
    chunks = split_into_chunks(blocks, max_tokens=200, overlap_blocks=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert estimate_tokens(chunk.text) <= 200 + estimate_tokens("SPEAKER_01: ")


def test_replies_are_not_cut_in_the_middle():
    """Кусок собирается из целых реплик.

    Разорванная реплика теряет либо исполнителя, либо срок — и поручение из
    протокола пропадает.
    """
    blocks = [words(30, f"SPEAKER_0{i}", mark=f"реплика{i}") for i in range(1, 6)]
    chunks = split_into_chunks(blocks, max_tokens=150, overlap_blocks=0)
    restored = [b.text for chunk in chunks for b in chunk.blocks]
    assert restored == [b.text for b in blocks]


def test_overlap_carries_the_boundary_reply_forward():
    """Поручение часто разложено на две реплики: «подготовьте отчёт» и «до
    среды». Без нахлёста срок остаётся в предыдущем куске и теряется."""
    blocks = [words(30, mark=f"реплика{i}") for i in range(6)]
    with_overlap = split_into_chunks(blocks, max_tokens=150, overlap_blocks=1)
    without = split_into_chunks(blocks, max_tokens=150, overlap_blocks=0)

    assert len(with_overlap) >= 2
    tail = with_overlap[0].blocks[-1].text
    assert with_overlap[1].blocks[0].text == tail
    assert without[1].blocks[0].text != tail


def test_chunks_know_their_place_in_the_meeting():
    chunks = split_into_chunks(
        [words(30, mark=f"реплика{i}") for i in range(6)], max_tokens=150
    )
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    assert {c.total for c in chunks} == {len(chunks)}


def test_reply_longer_than_the_window_is_split_by_sentences():
    long_speech = Block("SPEAKER_01", " ".join(f"Предложение номер {i}." for i in range(60)))
    chunks = split_into_chunks([long_speech], max_tokens=120, overlap_blocks=0)
    assert len(chunks) > 1
    for chunk in chunks:
        for block in chunk.blocks:
            assert block.text.endswith(".")
    joined = " ".join(b.text for c in chunks for b in c.blocks)
    assert "Предложение номер 0." in joined and "Предложение номер 59." in joined


def test_speaker_stays_with_every_piece_of_a_split_reply():
    """При делении длинной речи имя говорящего должно остаться на каждой
    части, иначе половина монолога уйдёт в модель без автора."""
    long_speech = Block("Иванов", " ".join(f"Мысль {i}." for i in range(60)))
    chunks = split_into_chunks([long_speech], max_tokens=120, overlap_blocks=0)
    assert all(b.speaker == "Иванов" for c in chunks for b in c.blocks)


def test_chunk_text_and_speakers_are_ready_for_the_prompt():
    chunk = Chunk([Block("Иванов", "Раз."), Block("Петров", "Два."), Block("Иванов", "Три.")])
    assert chunk.text == "Иванов: Раз.\nПетров: Два.\nИванов: Три."
    assert chunk.speakers == ["Иванов", "Петров"]


def test_empty_transcript_gives_no_chunks():
    assert split_into_chunks([], max_tokens=100) == []


def test_zero_budget_is_an_error_not_an_endless_loop():
    with pytest.raises(ValueError, match="max_tokens"):
        split_into_chunks([reply()], max_tokens=0)


def test_custom_counter_is_used():
    """Оценка по символам — умолчание; токенизатор модели подставляется."""
    calls = []

    def counter(text: str) -> int:
        calls.append(text)
        return 1

    split_into_chunks([reply(), reply()], max_tokens=1, count_tokens=counter)
    assert calls, "переданный счётчик должен использоваться"
