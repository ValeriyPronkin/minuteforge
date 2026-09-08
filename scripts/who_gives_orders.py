"""Кто на совещании раздаёт поручения — и сколько текста лишнее.

    python scripts/who_gives_orders.py data/output/запись_стенограмма.json

Скрипт отвечает на два вопроса разом, и оба — про то, стоит ли отсеивать
реплики до модели, а не после.

Первый: правда ли поручения даёт один человек. Если да, в модель можно
отправлять реплики председателя, а не всё совещание. Скрипт считает не
догадками, а тем же правилом :func:`minuteforge.tasks.is_directive`, которым
сейчас отсеиваются готовые поручения, — и показывает, что фильтр по одному
голосу потерял бы: обещания докладчиков «направим сегодня письмо» звучат
чужими голосами, и на живой записи их бывает больше, чем кажется.

Второй: насколько вообще похудеет запрос, если в модель пойдут только
реплики, в которых поручение слышно. Это и есть цена нынешнего порядка —
сейчас мелкая модель читает два часа докладов, чтобы правило потом
выбросило почти всё прочитанное.

Модель для этого не нужна и видеокарта тоже: правило работает на тексте.
Значит, проверять гипотезу можно на любой уже готовой стенограмме.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.blocks import (  # noqa: E402
    Block,
    Transcript,
    blocks_from_segments,
    consolidate,
    drop_soundcheck,
    is_soundcheck,
    normalize_speaker,
    timestamp,
)
# Дробление на предложения берётся у разбора, а не пишется заново: правило
# смотрит на отдельную фразу, и проверять его надо ровно теми фразами,
# какие оно увидит в работе.
from minuteforge.tasks import _sentences, is_directive  # noqa: E402

#: Строка стенограммы в текстовом виде: «[00:12:30] SPEAKER_00: текст».
#: Отметка времени необязательна — стенограмму приносят и руками набранной.
_LINE = re.compile(r"^(?:\[(\d{1,2}):(\d{2}):(\d{2})\]\s*)?([^:]{1,60}?)\s*:\s*(.+)$")


@dataclass
class Voice:
    """Что один голос наговорил и сколько из этого — поручения."""

    label: str
    blocks: int = 0
    words: int = 0
    #: Реплики, в которых поручение слышно хотя бы в одной фразе. Именно они
    #: уехали бы в модель при отборе до запроса.
    directive_blocks: int = 0
    #: Отдельные фразы с поручением — их больше, чем реплик: в одном
    #: выступлении председатель поручает по десять раз.
    directive_sentences: int = 0
    #: Слова в директивных репликах — из них и складывается размер запроса.
    directive_words: int = 0


def read_blocks(path: Path) -> Transcript:
    """Читает стенограмму: json со сегментами или текст построчно."""
    if path.suffix.lower() == ".json":
        body = json.loads(path.read_text(encoding="utf-8-sig"))
        # WhisperX отдаёт {"segments": [...]}, а наш save_transcript — список.
        segments = body.get("segments", []) if isinstance(body, dict) else body
        return Transcript(consolidate(blocks_from_segments(segments)))
    return Transcript(consolidate(_from_text(path.read_text(encoding="utf-8-sig"))))


def _from_text(text: str) -> list[Block]:
    """Разбирает стенограмму, сохранённую текстом.

    Шапка с моделью и пометками «ПРОВЕРИТЬ» начинается с решётки и
    пропускается: это наши примечания, а не чья-то речь.
    """
    blocks: list[Block] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            # Продолжение реплики без метки — дописывается к предыдущей, а не
            # теряется: перенос строки посреди выступления это обычное дело.
            if blocks:
                blocks[-1].text = f"{blocks[-1].text} {line}".strip()
            continue
        hours, minutes, seconds, speaker, said = match.groups()
        start = None
        if hours is not None:
            start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        blocks.append(Block(normalize_speaker(speaker), said.strip(), start, start))
    return blocks


def count_voices(blocks: Sequence[Block]) -> dict[str, Voice]:
    """Считает по каждому голосу реплики, слова и поручения."""
    voices: dict[str, Voice] = {}
    for block in blocks:
        voice = voices.setdefault(block.speaker, Voice(block.speaker))
        words = len(re.findall(r"\w+", block.text))
        voice.blocks += 1
        voice.words += words
        heard = sum(1 for sentence in _sentences(block.text) if is_directive(sentence))
        if heard:
            voice.directive_blocks += 1
            voice.directive_sentences += heard
            voice.directive_words += words
    return voices


def directive_blocks(blocks: Sequence[Block]) -> list[Block]:
    """Реплики, в которых поручение слышно, — то, что уехало бы в модель."""
    return [b for b in blocks if any(is_directive(s) for s in _sentences(b.text))]


def window_words(blocks: Sequence[Block], *, around: int = 1) -> int:
    """Сколько слов останется, если брать фразу с поручением и её соседей.

    Отбор по реплике на живой записи почти ничего не экономит: выступление
    председателя — это десять минут речи, из которых поручением сказаны две
    фразы, а в модель уезжает всё выступление. Отбор по фразе экономит вдвое
    больше, и соседняя фраза при этом сохраняется — в ней стоит срок
    («…до среды») и адресат.
    """
    kept = 0
    for block in blocks:
        sentences = _sentences(block.text)
        take: set[int] = set()
        for position, sentence in enumerate(sentences):
            if is_directive(sentence):
                take.update(range(position - around, position + around + 1))
        kept += sum(
            len(re.findall(r"\w+", sentences[i]))
            for i in take
            if 0 <= i < len(sentences)
        )
    return kept


def _share(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "—"


def _plural(count: int, one: str, few: str, many: str) -> str:
    """«1 голос», «3 голоса», «11 голосов» — иначе отчёт читается как черновик."""
    if count % 100 // 10 == 1:
        return f"{count} {many}"
    last = count % 10
    if last == 1:
        return f"{count} {one}"
    return f"{count} {few}" if 2 <= last <= 4 else f"{count} {many}"


def report(transcript: Transcript, *, examples: int, top: int) -> None:
    spoken = sum(len(re.findall(r"\w+", b.text)) for b in transcript.blocks)
    print(
        f"Стенограмма: реплик {len(transcript.blocks)}, "
        f"голосов {len(transcript.speakers)}, слов {spoken}, "
        f"длительность {transcript.duration_min} мин"
    )
    blocks = _without_soundcheck(transcript.blocks)
    print()
    voices = count_voices(blocks)
    # Порядок по числу поручений, а не по времени: вопрос ровно в том, кто
    # поручает, а не кто дольше говорит.
    ranked = sorted(
        voices.values(),
        key=lambda v: (v.directive_sentences, v.directive_blocks, v.words),
        reverse=True,
    )
    total_sentences = sum(v.directive_sentences for v in ranked)
    total_blocks = len(blocks)
    total_words = sum(v.words for v in ranked)
    heard_blocks = sum(v.directive_blocks for v in ranked)
    heard_words = sum(v.directive_words for v in ranked)

    if not total_sentences:
        print(
            "Поручений не слышно ни в одной реплике. Либо это не то совещание, "
            "либо правило is_directive не знает здешних слов — второе важнее "
            "и чинится списком DIRECTIVE_WORDS."
        )
        return

    print("Кто поручает")
    print(f"{'голос':<22}{'реплик':>8}{'из них с поручением':>22}{'фраз':>7}{'слов':>9}")
    for voice in ranked[:top]:
        print(
            f"{voice.label:<22}{voice.blocks:>8}"
            f"{voice.directive_blocks:>15} {_share(voice.directive_blocks, voice.blocks):>6}"
            f"{voice.directive_sentences:>7}{voice.words:>9}"
        )
    if len(ranked) > top:
        rest = ranked[top:]
        print(
            f"{'остальные ' + _plural(len(rest), 'голос', 'голоса', 'голосов'):<22}"
            f"{sum(v.blocks for v in rest):>8}"
            f"{sum(v.directive_blocks for v in rest):>15}"
            f"{'':>7}{sum(v.directive_sentences for v in rest):>7}"
            f"{sum(v.words for v in rest):>9}"
        )
    print()

    first = ranked[0]
    two = ranked[:2]
    print("Гипотеза «поручает один»")
    print(
        f"  первый голос ({first.label}): {first.directive_sentences} "
        f"поручительных фраз из {total_sentences} — "
        f"{_share(first.directive_sentences, total_sentences)}"
    )
    print(
        f"  первые два голоса: {sum(v.directive_sentences for v in two)} из "
        f"{total_sentences} — "
        f"{_share(sum(v.directive_sentences for v in two), total_sentences)}"
    )
    print()

    print("Что уедет в модель при отборе до запроса")
    print(
        f"  по репликам: {heard_blocks} из {total_blocks} — "
        f"{_share(heard_blocks, total_blocks)}, "
        f"слов {heard_words} из {total_words} — {_share(heard_words, total_words)}"
    )
    windowed = window_words(blocks)
    print(
        f"  по фразам с соседями: слов {windowed} из {total_words} — "
        f"{_share(windowed, total_words)}"
    )
    print()

    if examples:
        _print_losses(blocks, first.label, examples)


def _without_soundcheck(blocks: Sequence[Block]) -> list[Block]:
    """Убирает перекличку и говорит, сколько её было.

    Перекличка на штабе идёт все три часа — регионы подключаются по ходу, —
    и поручений в ней нет. Считается тем же, чем в работе: сначала правилом
    по каждой реплике, потом :func:`drop_soundcheck` с той же проверкой на
    поручение, какую делает конвейер. Расхождение двух счётов — это реплики,
    которые похожи на перекличку, но что-то поручают; их и стоит увидеть.
    """
    marked = [b for b in blocks if is_soundcheck(b.text)]
    if not marked:
        return list(blocks)
    kept, cut = drop_soundcheck(blocks, keep=is_directive)
    print(
        f"Похоже на перекличку: {_plural(len(marked), 'реплика', 'реплики', 'реплик')}, "
        f"последняя на {timestamp(marked[-1].start)}; "
        f"конвейер срезает {cut}. Дальше считаю без них."
    )
    if len(marked) > cut:
        print(
            f"  {_plural(len(marked) - cut, 'реплика', 'реплики', 'реплик')} "
            "оставлено: похоже на перекличку, но поручение в ней слышно."
        )
    return kept


def _print_losses(blocks: Sequence[Block], chair: str, examples: int) -> None:
    """Показывает, что потерял бы отбор по одному голосу.

    Это главная строка отчёта. Доли говорят, что гипотеза скорее верна, а
    здесь видно её цену: живые фразы, которые при фильтре по председателю
    в протокол уже не попадут. Их читают глазами — правило не решит, обидная
    это потеря или пустая.
    """
    others = [b for b in directive_blocks(blocks) if b.speaker != chair]
    print(f"Что потерял бы отбор по одному голосу — {len(others)} реплик:")
    if not others:
        print("  ничего: поручения звучат только голосом " + chair)
        return
    for block in others[:examples]:
        said = " ".join(block.text.split())
        if len(said) > 160:
            said = said[:160].rsplit(" ", 1)[0] + "…"
        print(f"  [{timestamp(block.start)}] {block.speaker}: {said}")
    if len(others) > examples:
        print(f"  … и ещё {len(others) - examples}. Показать все: --примеры 0")


def main() -> None:
    # Русский текст в консоли Windows: без этого отчёт падает на первой же
    # фамилии, а гоняют скрипт именно там.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path, help="стенограмма: json или txt")
    parser.add_argument(
        "--примеры", dest="examples", type=int, default=15,
        help="сколько чужих поручений показать; 0 — все",
    )
    parser.add_argument(
        "--голосов", dest="top", type=int, default=12,
        help="сколько голосов показать в таблице",
    )
    args = parser.parse_args()

    if not args.transcript.exists():
        parser.error(f"нет такого файла: {args.transcript}")

    transcript = read_blocks(args.transcript)
    if not transcript.blocks:
        parser.error(f"в файле нет реплик: {args.transcript}")

    examples = args.examples if args.examples > 0 else 10 ** 6
    report(transcript, examples=examples, top=max(1, args.top))


if __name__ == "__main__":
    main()
