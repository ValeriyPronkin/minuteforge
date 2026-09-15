"""Отметили: что доложили по каждому направлению.

В подписанном протоколе два раздела, и собираются они по-разному.

«Решили» складывается из поручений. Поручение — иголка: одна фраза в
двухчасовой записи, и всё устройство разбора — окна вокруг фразы,
проверочный проход, отсев докладов — нужно ровно затем, чтобы её найти и не
принять за неё соседнюю.

«Отметили» устроено иначе. Один пункт — один доклад: «Информацию такого-то о
том-то», и пункты идут по кругу разбора. Единица здесь не фраза, а связный
отрезок записи, и границы у него берутся не из суждения модели, а из двух
готовых вещей: смены голоса и объявления очередного направления по
справочнику. Искать нечего — отрезок уже отобран устройством совещания, и
модели остаётся пересказать короткий кусок, а это она делает заметно лучше,
чем выписывает из длинного.

Канцелярскую форму — «Информацию заместителя председателя правительства
такого-то о завершении строительства…» — дописывает секретарь. Инструмент
даёт материал и привязку: чей вопрос разбирали, когда и что при этом
прозвучало. Подгонять пересказ под оборот подписанного документа значило бы
переобучаться под одного делопроизводителя.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from loguru import logger

from .blocks import Block
from .chunking import Chunk, piece, split_into_chunks
from .directory import Directory, at
from .llm import LLMClient, LLMError
from .progress import Progress, Step, report
# Дробление на фразы, значащие слова и правило «это поручение» берутся у
# разбора, а не пишутся заново: разделы протокола должны делить материал по
# одной и той же границе, иначе один и тот же доклад попадёт в оба.
from .tasks import _moment, _sentences, _significant, is_directive, is_russian

#: Сколько речи должно набраться, чтобы счесть отрезок докладом. Полминуты —
#: это «подключились, слышно, доложим письмом»; пункта протокола из такого не
#: напишешь. Порог намеренно низкий: пропущенный доклад дороже лишнего
#: пункта, который вычёркивают при вычитке.
REPORT_SECONDS = 90

#: Какая доля значащих слов тезиса должна найтись в куске.
#:
#: Мерка грубая, и точнее её не сделать: в тезисе из пяти значащих слов шаг
#: между «четыре из пяти» и «три из пяти» — двадцать процентов, а пересказ
#: законно меняет слова. На пороге 70 % отсеивалось «Суд по лишению лицензии
#: перенесён на октябрь, вероятность оценена как 90 %» — тезис верный и
#: целиком из записи. Поэтому порог 60 %, а настоящую защиту от выдумки
#: держат числа: см. :func:`_invented_numbers`.
FROM_THE_CHUNK = 0.6

#: Чем переспросить, когда модель ответила не по-русски. Указания у неё
#: русские, и кусок русский, а `mistral` всё равно срывается на английский —
#: на записи 03.09 так потерялись Якутия и Калмыкия целиком, по два десятка
#: верных тезисов. Дешевле переспросить, чем потерять доклад.
SAY_IT_IN_RUSSIAN = (
    "\n\nПредыдущий ответ был не по-русски. Отвечай по-русски, "
    "тем же языком, каким говорят в куске."
)

#: Сколько знаков позволено тезису. Модель, которой велели писать полнее,
#: иногда пересказывает кусок целиком.
LONGEST = 400

#: Короче этого — не тезис, а обрывок: «Даты выполнения», «Показатели
#: мощности», «У нас есть связь». Модель так пересказывает слайд, с которого
#: докладывают, — заголовками граф, а не тем, что в них стоит.
#:
#: Порог низкий нарочно. Короткий тезис бывает и настоящим — «Фотоотчёт
#: отправлен сегодня», — а отличить его от заголовка длиной не выходит:
#: «Проценты исполнения мероприятий на сегодня» длиннее обоих. Поэтому здесь
#: отсекается только заведомое, а остальное правится указаниями модели.
SHORTEST = 25


@dataclass
class Report:
    """Отрезок записи, отданный одному направлению."""

    unit: str
    blocks: list[Block] = field(default_factory=list)
    #: Сколько раз к направлению возвращались. Возврат — обычное дело:
    #: разбирают, уходят к общей части, досматривают позже. В протоколе это
    #: всё равно один пункт.
    runs: int = 1
    #: Направление не объявлено, а узнано по содержанию отрезка.
    guessed: bool = False

    @property
    def start(self) -> float | None:
        moments = [b.start for b in self.blocks if b.start is not None]
        return min(moments) if moments else None

    @property
    def end(self) -> float | None:
        moments = [b.end for b in self.blocks if b.end is not None]
        return max(moments) if moments else None

    @property
    def seconds(self) -> float:
        return sum(b.duration or 0.0 for b in self.blocks)

    @property
    def voices(self) -> list[str]:
        seen: list[str] = []
        for block in self.blocks:
            if block.speaker and block.speaker not in seen:
                seen.append(block.speaker)
        return seen

    @property
    def text(self) -> str:
        return "\n".join(f"{b.speaker}: {b.text}" for b in self.blocks)


@dataclass
class Note:
    """Пункт раздела «Отметили»: что доложили по одному направлению."""

    unit: str = ""
    theses: list[str] = field(default_factory=list)
    #: Начало доклада в записи — чтобы вернуться к месту в видео.
    at: float | None = None
    #: Чьими голосами доложено. Метки диаризации, пока имена не сопоставлены.
    voices: list[str] = field(default_factory=list)


def split_into_reports(
    blocks: Sequence[Block],
    directory: Directory,
    *,
    hosts: Iterable[str] = (),
    floor: float = REPORT_SECONDS,
) -> tuple[list[Report], Report]:
    """Раскладывает запись по направлениям.

    Возвращает доклады и отдельно всё остальное: повестку, перекличку и
    общие доклады, у которых направления нет вовсе. У них ключ другой — не
    направление, а докладчик, — и собирать их по кругу нельзя.

    Отрезки короче ``floor`` докладами не считаются и уходят в остаток:
    подключились, отчитались одной фразой, название прозвучало мимоходом.
    """
    if not directory:
        return [], Report("", list(blocks))

    kept: dict[str, Report] = {}
    outside = Report("")
    for name, run in _runs(blocks, directory, hosts=hosts):
        heard = name
        if not name:
            name = _recognize(run, directory)
        if not name:
            outside.blocks.extend(run)
            continue
        found = kept.get(name)
        if found is None:
            kept[name] = Report(name, list(run), guessed=not heard)
        else:
            found.runs += 1
            found.blocks.extend(run)
            found.guessed = found.guessed and not heard

    reports: list[Report] = []
    for found in kept.values():
        if found.seconds >= floor:
            reports.append(found)
        else:
            outside.blocks.extend(found.blocks)
    # Порядок — по ходу совещания, а не по алфавиту: человек, слушавший
    # разбор, ищет пункт там же, где он прозвучал.
    reports.sort(key=lambda r: (r.start is None, r.start or 0))
    outside.blocks.sort(key=lambda b: (b.start is None, b.start or 0))
    return reports, outside


def _runs(
    blocks: Sequence[Block],
    directory: Directory,
    *,
    hosts: Iterable[str] = (),
) -> list[tuple[str, list[Block]]]:
    """Режет запись на отрезки: подряд идущая речь под одним объявлением.

    Режется по фразам, а не по репликам, и это не педантизм. Очередное
    направление объявляют, не дожидаясь своей реплики: «Задача понятна.
    Желаем успехов. Следующий регион.» — и тут же, в той же реплике,
    начинается доклад. По целым репликам он достался бы предыдущему
    направлению целиком.
    """
    marks = directory.follow(blocks, hosts=hosts)
    runs: list[tuple[str, list[Block]]] = []

    def add(name: str, part: Block) -> None:
        if runs and runs[-1][0] == name:
            runs[-1][1].append(part)
        else:
            runs.append((name, [part]))

    for block in blocks:
        sentences = _sentences(block.text)
        if not sentences:
            continue
        # Время фразы считается ровно так же, как его считает разметка
        # переходов: по месту фразы в тексте реплики. Считать двумя способами
        # нельзя — объявление оказывается на долю секунды раньше собственной
        # отметки и достаётся предыдущему направлению вместе со всем, что за
        # ним следует.
        moments = _moments(block, sentences)
        first = 0
        name = at(marks, moments[0])
        for position in range(1, len(sentences) + 1):
            here = at(marks, moments[position]) if position < len(sentences) else None
            if here == name:
                continue
            add(name, piece(block, sentences, first, position - 1))
            first, name = position, here or ""
    return runs


def _moments(block: Block, sentences: Sequence[str]) -> list[float | None]:
    """Когда в записи звучит каждая фраза реплики."""
    text = block.text or ""
    moments: list[float | None] = []
    offset = 0
    for sentence in sentences:
        at_char = text.find(sentence, offset)
        if at_char < 0:
            at_char = offset
        offset = at_char + len(sentence)
        moments.append(_moment(block, at_char))
    return moments


def _recognize(run: Sequence[Block], directory: Directory) -> str:
    """Чьё это выступление, если объявление прозвучало без названия.

    «Следующий регион.» — и дальше две минуты о площадках и контейнерах;
    название звучит только в напутствии, которым разбор и закрывают. Вперёд
    на две фразы, как ищет :meth:`Directory.follow`, его не достать.

    Для поручения смотреть дальше нельзя: упомянутое мимоходом направление
    подпишется под чужим поручением, а неверный адресат хуже пустого. Здесь
    другое дело — отрезок закрыт с обеих сторон объявлениями, и смотреть
    внутрь него безопасно.

    Названо одно направление — оно и разбирается. Названо несколько — это
    общий доклад, где перечисляют всех подряд, и своего направления у него
    нет. Так эти два случая и различаются, без порогов и подсчёта частот.
    """
    found = directory.find_all(" ".join(block.text for block in run))
    return found[0] if len(found) == 1 else ""


NOTES_SYSTEM = """Ты — секретарь совещания. Тебе дан кусок стенограммы: доклад по
одному вопросу. Выпиши, о чём доложили, — это пойдёт в раздел протокола
«Отметили».

Правила:
— один тезис — одно предложение;
— бери числа, доли, сроки и названия как сказано: готовность 93,5 %, ввод
  15 декабря, шесть соглашений, комплекс «Черемушское»;
— не выписывай поручений и требований — «обеспечить», «представить»,
  «усилить», «в срок до»: это другой раздел протокола;
— не добавляй того, чего в куске нет: ни оценок, ни выводов, ни причин;
— не пересказывай перекличку, приветствия и благодарности;
— пиши по-русски, каким бы ни был язык указаний;
— тезис должен быть законченным утверждением, а не заголовком графы:
  «Готовность цеха 93,5 %» — да, «Показатели мощности» — нет;
— не больше трёх тезисов; доложено не о чем — верни пустой список.

Ответь строго так, без единого слова вокруг:

{"theses": ["…", "…"]}"""

#: Схема ответа. Одно поле: дай мелкой модели свободу — и она вернёт тезисы
#: вместе с рассуждением о том, почему выбрала именно их.
NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "theses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["theses"],
    "additionalProperties": False,
}


def take_notes(
    reports: Sequence[Report],
    client: LLMClient,
    *,
    budget: int = 1200,
    json_mode: bool = True,
    extra: str = "",
    progress: Progress | None = None,
    journal: object | None = None,
) -> list[Note]:
    """Пересказывает каждый доклад тезисами.

    Доклад в пятнадцать минут — это три-четыре тысячи токенов, и целиком он
    мелкой модели не по силам ровно так же, как кусок стенограммы: она
    прочтёт начало. Поэтому доклад делится по бюджету запроса, тезисы
    собираются по частям и сводятся в один пункт.

    Сбой запроса — часть доклада остаётся без тезисов, а остальные
    собираются: молчание сервера не повод бросить весь раздел.
    """
    if not reports:
        return []

    system = _with_extra(NOTES_SYSTEM, extra)
    notes: list[Note] = []
    for position, found in enumerate(reports, 1):
        report(progress, Step(
            name="notes",
            title=f"Собираю отмеченное: {position} из {len(reports)}",
            share=(position - 1) / len(reports),
        ))
        theses: list[str] = []
        for chunk in split_into_chunks(
            found.blocks, max_tokens=budget, overlap_blocks=0,
        ):
            theses.extend(_ask(
                client, system, chunk,
                json_mode=json_mode, unit=found.unit, journal=journal,
            ))
        note = Note(
            unit=found.unit,
            theses=_without_repeats(theses),
            at=found.start,
            voices=found.voices,
        )
        if note.theses:
            notes.append(note)
        else:
            logger.info("По направлению «{}» тезисов не набралось", found.unit)
    logger.info(
        "Отмечено: {} пунктов, тезисов {}",
        len(notes), sum(len(n.theses) for n in notes),
    )
    return notes


def _ask(
    client: LLMClient,
    system: str,
    chunk: Chunk,
    *,
    json_mode: bool,
    unit: str = "",
    journal: object | None = None,
) -> list[str]:
    """Тезисы по одной части доклада — те, что выдержали проверку."""
    theses = _answered(client, system, chunk, json_mode=json_mode)
    if theses and all(not is_russian(thesis) for thesis in theses):
        # Не по-русски ответила не половина тезисов, а весь кусок целиком —
        # значит сорвалась модель, а не тезис не удался. Переспрашиваем.
        again = _answered(
            client, system + SAY_IT_IN_RUSSIAN, chunk, json_mode=json_mode,
        )
        if any(is_russian(thesis) for thesis in again):
            logger.info("Кусок пересказан заново по-русски: {}", unit or "—")
            theses = again

    at = min(
        (b.start for b in getattr(chunk, "blocks", []) if b.start is not None),
        default=None,
    )
    kept: list[str] = []
    for thesis in theses:
        why = _why_dropped(thesis, chunk.text)
        if journal is not None:
            journal.thesis(unit, at, thesis, why)
        if not why:
            kept.append(thesis)
    return kept


def _answered(
    client: LLMClient, system: str, chunk: Chunk, *, json_mode: bool,
) -> list[str]:
    """Что модель ответила на кусок. Сбой запроса — пустой список."""
    try:
        reply = client.complete(
            system, chunk.text,
            json_mode=json_mode,
            schema=NOTES_SCHEMA if json_mode else None,
        )
    except LLMError as exc:
        logger.warning("Часть доклада не пересказана: {}", exc)
        return []
    return _read(reply.text)


def _read(answer: str) -> list[str]:
    """Тезисы из ответа модели."""
    text = (answer or "").strip()
    if not text:
        return []
    # Модель оборачивает ответ в ```json — с этим ничего не поделать, кроме
    # как снять обёртку.
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    found = re.search(r"\{.*\}", text, re.S)
    if not found:
        return []
    try:
        body = json.loads(found.group(0))
    except ValueError:
        return []
    theses = body.get("theses") if isinstance(body, dict) else None
    if not isinstance(theses, list):
        return []
    # Каждое предложение — свой пункт. Модели сказано «одно предложение»,
    # но она складывает в один тезис три, и в документе они стоят одной
    # строкой на двести знаков. Порознь их и читать легче, и проверять:
    # выдуманное предложение внутри верного тезиса отсеется само.
    return [
        part
        for item in theses
        for part in _sentences(_without_speaker(str(item)))
        if part.strip()
    ]


#: Метка говорящего в начале тезиса: модель иногда переписывает строку куска
#: целиком, вместе с тем, кто её сказал. В документе это лишнее — чей доклад,
#: стоит в заголовке пункта.
#:
#: Двоеточие само по себе меткой не считается: «Готовность: 93 %» и «Ввод
#: 15 декабря: срок подтверждён» — тезисы, а не подписи. Поэтому метка это
#: либо `SPEAKER_00`, либо два-три слова с заглавной: «Иванов И.И.»,
#: «Максим Вячеславович».
_SPEAKER_PREFIX = re.compile(
    r"^\s*(?:SPEAKER[_ -]*\d+"
    r"|[А-ЯЁ][а-яё]+(?:\s*[А-ЯЁ][а-яё]*\.?){1,3})\s*:\s*"
)


def _without_speaker(thesis: str) -> str:
    """Убирает метку говорящего, если модель переписала строку куска целиком."""
    return _SPEAKER_PREFIX.sub("", (thesis or "").strip()).strip()


def _why_dropped(thesis: str, source: str) -> str:
    """Чем тезис не годится для документа. Пусто — годится.

    Причина, а не «да/нет», и это важно: отсев здесь идёт правилами, и по
    документу его не видно — направление просто не попадает в протокол.
    Причина уходит в записку разбора, и по ней настраивают пороги.

    Проверки те же, какими держится выписка поручений, и по той же причине:
    мелкая модель отвечает не на том языке, сочиняет подробности, которых в
    куске не было, и путает разделы протокола.
    """
    # Язык проверяется первым не из педантизма: английский ответ это сбой
    # модели, а не плохой тезис, и в записке их надо различать. По длине он
    # сошёл бы за обрывок, и сбой остался бы незамеченным.
    if not is_russian(thesis):
        return "не по-русски"
    if len(thesis) < SHORTEST:
        return "короче обрывка"
    if len(thesis) > LONGEST:
        return "длиннее пункта"
    if is_directive(thesis) or _demanded(thesis):
        # Требование — это «Решили», а не «Отметили». Пункт, попавший не в
        # свой раздел, хуже пропущенного: его исполнят дважды или не
        # исполнят вовсе.
        return "звучит требованием — это «Решили»"
    invented = _invented_numbers(thesis, source)
    if invented:
        return f"в куске не звучало: {', '.join(sorted(invented))}"
    words = _significant(thesis)
    if not words:
        return "нет значащих слов"
    share = len(words & _significant(source)) / len(words)
    if share < FROM_THE_CHUNK:
        return f"в куске найдено: {share:.0%} слов из {len(words)}"
    return ""


#: Число в тексте: 93,5 · 42 · 2026 · 15.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _invented_numbers(thesis: str, source: str) -> set[str]:
    """Числа тезиса, которых в куске не звучало.

    Доля общих слов ловит выдумку плохо: пересказ законно меняет слова, а
    подставленное число слов почти не меняет. На записи 03.09 в доклад
    Амурской области переехали «готовность 93,5 %» и «ввод 15 декабря» —
    показатели Архангельской, с которой модель их и взяла.

    Выдуманное число — худшее, что может случиться с протоколом: словами
    спорят, а цифру переносят в отчёт как есть. Поэтому здесь не доля, а
    строгое требование: не прозвучало — тезиса нет.
    """
    said = {_as_number(found) for found in _NUMBER.findall(source or "")}
    return {
        found for found in _NUMBER.findall(thesis or "")
        if _as_number(found) not in said
    }


def _as_number(text: str) -> str:
    """«93.5» и «93,5» — одно и то же число."""
    return text.replace(".", ",")


#: Чем требование притворяется существительным: «Необходимость провести
#: мониторинг», «Требуется подтвердить сроки». Глагола в повелительном
#: наклонении здесь нет, и правило :func:`is_directive` такое не берёт — а
#: это всё равно «Решили», а не «Отметили».
_DEMAND = ("необходимость ", "необходимо ", "требуется ", "нужно ", "следует ")


def _demanded(thesis: str) -> bool:
    """Требование, названное существительным, — тоже требование."""
    lowered = (thesis or "").strip().lower()
    return any(lowered.startswith(word) for word in _DEMAND)


def _without_repeats(theses: Sequence[str]) -> list[str]:
    """Убирает повторы: соседние части доклада пересказывают одно и то же."""
    kept: list[str] = []
    for thesis in theses:
        words = _significant(thesis)
        if not words:
            continue
        same = any(
            len(words & _significant(known)) / len(words | _significant(known)) > 0.6
            for known in kept
        )
        if not same:
            kept.append(thesis)
    return kept


def _with_extra(system: str, extra: str) -> str:
    extra = (extra or "").strip()
    return f"{system}\n\n{extra}" if extra else system
