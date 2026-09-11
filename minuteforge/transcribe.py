"""Распознавание речи с разделением по говорящим.

Обёртка над WhisperX. Сам он тяжёлый — тянет torch, модели на гигабайты и
требует видеокарту, — поэтому ввозится он лениво, внутри функций: пакет
должен ставиться и проверяться там, где ничего этого нет.

Работа идёт в три приёма: распознать, выровнять по времени, разделить по
говорящим. Каждый приём сохраняется на диск. Это не забота о дисковом
пространстве, а единственный способ работать с часовыми записями: если
диаризация упала на последнем шаге, сорок минут распознавания должны
остаться при вас.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from loguru import logger

from .chunking import estimate_tokens
from .config import HF_TOKEN_ENV, Settings
from .progress import Progress, Step, report


#: Закрытые модели, без согласия на условия которых диаризация не работает.
#: Их две, и это неочевидно: вторая тянется как зависимость первой, а человек
#: обычно подписывает только ту страницу, ссылку на которую ему дали.
GATED_MODELS = ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0")


class RecognitionError(RuntimeError):
    """Распознать не удалось."""


class MissingToken(RecognitionError):
    """Нет токена HuggingFace.

    Отдельно, потому что это не поломка, а незаконченная настройка: модели
    диаризации закрытые, и без токена их не отдадут.
    """


class Backend(Protocol):
    """То, что нужно от WhisperX.

    Объявлено протоколом, чтобы разбор и сохранение шагов проверялись без
    видеокарты и без моделей: подставляется заглушка.
    """

    def transcribe(
        self, audio: str, *, model: str, language: str, batch_size: int,
        device: str, hints: str = "",
    ) -> dict: ...

    def align(self, segments: list[dict], *, language: str, audio: str, device: str) -> dict: ...

    def diarize(
        self, audio: str, aligned: dict, *, token: str, speakers: int | None, device: str
    ) -> dict: ...


#: Модели Whisper от крупной к мелкой. По этой лестнице расчёт спускается,
#: когда видеопамяти не хватило: лучше распознать моделью поменьше, чем не
#: распознать вовсе. Качество при этом падает, и об этом говорится вслух.
MODEL_LADDER = ("large-v3", "large-v2", "large", "medium", "small", "base", "tiny")

#: Признак того, что кончилась видеопамять. Torch сообщает об этом
#: по-разному в зависимости от версии, поэтому смотрим и на тип, и на текст.
_OUT_OF_MEMORY = ("out of memory", "cuda oom", "не хватает памяти")


def is_out_of_memory(error: BaseException) -> bool:
    return any(mark in str(error).lower() for mark in _OUT_OF_MEMORY) or type(
        error
    ).__name__ == "OutOfMemoryError"


#: Сколько времени занимает каждый шаг — доли от общего, по наблюдениям.
#: Точность тут недостижима: она зависит от записи, модели и видеокарты.
#: Но даже грубая шкала честнее крутилки, по которой не понять, идёт работа
#: или всё повисло.
STEP_SHARES = {"audio": 0.05, "transcribe": 0.60, "align": 0.15, "diarize": 0.20}

STEP_TITLES = {
    "audio": "Извлекаю звук",
    "transcribe": "Распознаю речь",
    "align": "Выравниваю по времени",
    "diarize": "Разделяю по говорящим",
}


@dataclass
class Recognition:
    """Итог распознавания."""

    segments: list[dict] = field(default_factory=list)
    language: str = ""
    device: str = ""
    #: Какие шаги взяты из сохранённых файлов, а не посчитаны заново.
    reused: list[str] = field(default_factory=list)
    #: Какой моделью и какой порцией распознавание прошло на самом деле.
    #: Может отличаться от заказанного: при нехватке видеопамяти расчёт
    #: спускается на модель поменьше, и знать об этом нужно — качество
    #: расшифровки другое.
    asr_model: str = ""
    batch_size: int = 0
    #: Что пришлось уступить. Пусто — прошло как заказано.
    fallbacks: list[str] = field(default_factory=list)
    #: С какими подсказками распознавали. Реквизит наравне с моделью:
    #: расшифровка с подсказками — другая расшифровка, и сравнивать её с
    #: прежней можно, только зная об этом.
    hints: str = ""


def free_vram(torch_module: Any | None = None) -> bool:
    """Возвращает видеопамять, занятую предыдущим шагом.

    Torch не отдаёт память сразу: он держит её в своём распределителе, чтобы
    не просить у драйвера заново. Обычно это правильно, но здесь шаги идут
    один за другим и каждый грузит свою модель — распознавание, выравнивание,
    диаризацию. На карте с 8 ГБ они втроём не помещаются, и без явной чистки
    третий шаг падает с нехваткой памяти на записи, которую первые два
    прошли без запинки.

    Возвращает True, если чистка была: без видеокарты и без torch делать
    нечего, но и ошибкой это не является.
    """
    gc.collect()
    module = torch_module
    if module is None:
        try:
            import torch as module  # noqa: PLC0415
        except ImportError:
            return False
    try:
        if not module.cuda.is_available():
            return False
        module.cuda.empty_cache()
    except Exception as exc:  # чистка памяти не повод ронять расчёт
        logger.warning("Не удалось освободить видеопамять: {}", exc)
        return False
    logger.debug("Видеопамять освобождена")
    return True


def resolve_device(requested: str, cuda_available: bool | None = None) -> str:
    """Выбирает устройство.

    ``auto`` — видеокарта, если она есть. Отдельного разговора стоит случай,
    когда её нет: работать будет, но час записи распознаётся часами, поэтому
    выбор пишется в журнал, а не делается молча.
    """
    if requested and requested != "auto":
        return requested
    if cuda_available is None:
        cuda_available = _cuda_available()
    device = "cuda" if cuda_available else "cpu"
    if device == "cpu":
        logger.warning(
            "Видеокарта не найдена, распознавание пойдёт на процессоре — "
            "это в разы дольше. Двухчасовую запись целиком лучше не брать: "
            "модель small вместо medium и временной интервал на нужный кусок "
            "дают разницу в часы."
        )
    return device


def torch_device(device: str, mps_available: bool | None = None) -> str:
    """Устройство для шагов, которые считает torch.

    Распознавание и всё остальное здесь расходятся, и это не прихоть.
    Распознавание идёт через CTranslate2 — библиотеку, которая графическое
    ядро Apple не умеет вовсе и считает на процессоре. А выравнивание и
    разметка голосов — обычный torch, и вот он в Apple Silicon умеет.

    Поэтому на маке распознавание остаётся на процессоре, а два оставшихся
    шага уходят на графическое ядро. Половина работы ускоряется — это не
    полная победа, но и не «ничего нельзя».
    """
    if device != "cpu":
        return device
    if mps_available is None:
        mps_available = _mps_available()
    return "mps" if mps_available else "cpu"


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover — без torch распознавания нет вовсе
        return False
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def compute_type_for(device: str) -> str:
    """В каких числах считать.

    WhisperX по умолчанию просит float16 — на видеокарте это правильно, а на
    процессоре такого счёта нет: библиотека молча переходит на float32, вдвое
    больше памяти и заметно медленнее. На процессоре считают в int8, и
    разборчивость от этого страдает мало, а время — в разы.
    """
    return "float16" if device == "cuda" else "int8"


def recognize(
    audio: str | Path,
    settings: Settings | None = None,
    *,
    backend: Backend | None = None,
    cache_dir: str | Path | None = None,
    diarize: bool = True,
    progress: Progress | None = None,
) -> Recognition:
    """Распознаёт запись и размечает её по говорящим.

    :param cache_dir: куда складывать промежуточные шаги. Готовый шаг при
        повторном запуске читается с диска, а не считается заново.
    :param diarize: делить ли по говорящим. Без разделения протокол выйдет
        сплошным текстом без авторов, но зато не нужен токен HuggingFace —
        годится, чтобы посмотреть, как вообще распознаётся запись.
    :param progress: куда сообщать о ходе работы. Часовая запись считается
        десятки минут, и человеку нужно видеть, что происходит.
    """
    settings = settings or Settings()
    audio = Path(audio)
    if not audio.exists():
        raise RecognitionError(f"Файл не найден: {audio}")

    if settings.offline_models:
        # Модели уже скачаны: не ходить в сеть вовсе. Быстрее на повторных
        # запусках и не отмечает лишний раз использование токена.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    token = settings.hf_token
    if diarize and not token:
        # Сообщение длиннее обычного намеренно: сюда упирается каждый, кто
        # запускает инструмент впервые, и почти никто в этот момент не идёт
        # читать документацию. Порядок шагов тут неочевиден — согласие на
        # условия важнее самого токена, без него он бесполезен.
        raise MissingToken(
            "Для разделения по говорящим нужен токен HuggingFace: модели "
            "pyannote закрытые. Токен личный, у каждого свой — он привязан "
            "к аккаунту и заменяет пароль.\n"
            "1. Примите условия на huggingface.co/pyannote/speaker-diarization-3.1 "
            "и huggingface.co/pyannote/segmentation-3.0 — нужны обе.\n"
            "2. Создайте токен типа Read на huggingface.co/settings/tokens.\n"
            f"3. Положите его в переменную окружения {HF_TOKEN_ENV}. "
            "В файл настроек и тем более в код — нельзя.\n"
            f'На Windows: setx {HF_TOKEN_ENV} "hf_..." — и обязательно '
            "закройте окно терминала и откройте новое. В том окне, где "
            "выполнили setx, переменной ещё нет: она достаётся только тем "
            "программам, которые запущены после.\n"
            "Разделение по говорящим можно и отключить — стенограмма будет "
            "без имён, но поручения извлекутся: токен нужен только этому шагу."
        )

    backend = backend or _whisperx_backend()
    device = resolve_device(settings.device)
    cache = Path(cache_dir) if cache_dir else None
    result = Recognition(device=device)

    # Часть шагов идёт на другом устройстве — см. torch_device.
    helper = torch_device(device)
    if helper != device:
        # Некоторые операции pyannote в Apple Silicon не реализованы. Без
        # этого разрешения torch падает вместо того, чтобы посчитать их на
        # процессоре, и разметка голосов на маке не работает вовсе.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("Выравнивание и разметка голосов пойдут на {}", helper)

    logger.info("Распознавание: модель {}, устройство {}", settings.asr_model, device)
    raw = _step(
        cache, "transcription.json", result, progress, "transcribe",
        lambda: _transcribe_with_fallback(backend, str(audio), settings, device, result),
    )
    result.language = raw.get("language") or settings.language

    aligned = _step(
        cache, "aligned.json", result, progress, "align",
        lambda: backend.align(
            raw.get("segments", []),
            language=result.language,
            audio=str(audio),
            device=helper,
        ),
    )

    if not diarize:
        result.segments = aligned.get("segments", [])
        return result

    assigned = _step(
        cache, "segments.json", result, progress, "diarize",
        lambda: backend.diarize(
            str(audio),
            aligned,
            token=token or "",
            speakers=settings.speakers,
            device=helper,
        ),
    )
    result.segments = assigned.get("segments", [])
    logger.info("Готово: сегментов {}", len(result.segments))
    return result


def _transcribe_with_fallback(
    backend: Backend,
    audio: str,
    settings: Settings,
    device: str,
    result: Recognition,
) -> dict:
    """Распознаёт, уступая по мере нехватки видеопамяти.

    Сначала уменьшается порция: она влияет на память сильнее всего, а на
    качество не влияет вовсе — только на скорость. Когда уменьшать уже
    некуда, расчёт спускается на модель поменьше. Это уже размен качества
    на возможность закончить, и потому он записывается в результат и
    говорится вслух: расшифровка, сделанная моделью tiny, — это другая
    расшифровка, и человек должен знать, что получил её.

    Заранее угадать, что поместится, нельзя: видеопамять делят и другие
    программы, а длинная запись просит больше короткой.
    """
    ladder = _ladder_from(settings.asr_model)
    last_error: BaseException | None = None
    result.hints = read_hints(settings.asr_hints_file)

    for model in ladder:
        batch = max(1, settings.batch_size)
        while batch >= 1:
            try:
                value = backend.transcribe(
                    audio,
                    model=model,
                    language=settings.language,
                    batch_size=batch,
                    device=device,
                    hints=result.hints,
                )
            except Exception as exc:
                if not is_out_of_memory(exc):
                    raise
                last_error = exc
                free_vram()
                if batch > 1:
                    batch = max(1, batch // 2)
                    note = f"порция уменьшена до {batch}"
                    logger.warning("Не хватило видеопамяти: {}", note)
                    result.fallbacks.append(note)
                    continue
                break  # порцию уменьшать больше некуда — меняем модель

            result.asr_model = model
            result.batch_size = batch
            if model != settings.asr_model:
                note = f"модель {settings.asr_model} не поместилась, распознано моделью {model}"
                logger.warning(note)
                result.fallbacks.append(note)
            return value

    raise RecognitionError(
        "Не хватило видеопамяти даже для самой мелкой модели. "
        "Освободите её: закройте Ollama или задайте OLLAMA_KEEP_ALIVE=0, "
        f"либо считайте на процессоре — device=cpu. Последняя ошибка: {last_error}"
    )


def _ladder_from(model: str) -> list[str]:
    """Модели от заказанной и ниже.

    Незнакомое имя — например, дообученная своя модель — пробуется первым, а
    дальше идёт обычная лестница: вдруг не поместится именно она.
    """
    if model in MODEL_LADDER:
        return list(MODEL_LADDER[MODEL_LADDER.index(model):])
    return [model, *MODEL_LADDER[MODEL_LADDER.index("medium"):]]


def _step(
    cache: Path | None,
    name: str,
    result: Recognition,
    progress: Progress | None,
    key: str,
    work,
) -> dict:
    """Считает шаг или берёт готовый с диска, сообщая о ходе работы."""
    title = STEP_TITLES.get(key, key)
    report(progress, Step(name=key, title=title, share=_share_before(key)))

    started = time.perf_counter()
    reused = False
    value = None

    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / name
        if path.exists():
            logger.info("Шаг {} взят из {}", name, path)
            result.reused.append(name)
            value = json.loads(path.read_text(encoding="utf-8"))
            reused = True

    if value is None:
        value = work()
        if cache is not None:
            (cache / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )

    if not reused:
        # Именно здесь, а не в конце работы: следующий шаг грузит свою
        # модель, и память ему нужна до того, как он начнёт.
        free_vram()

    report(progress, Step(
        name=key,
        title=title,
        done=True,
        reused=reused,
        elapsed_s=round(time.perf_counter() - started, 1),
        share=_share_after(key),
    ))
    return value


def _share_before(key: str) -> float:
    """Сколько сделано к началу шага."""
    total = 0.0
    for name, share in STEP_SHARES.items():
        if name == key:
            break
        total += share
    return round(total, 3)


def _share_after(key: str) -> float:
    return round(min(1.0, _share_before(key) + STEP_SHARES.get(key, 0.0)), 3)


#: Сколько знаков должно быть в сегменте, чтобы его можно было выровнять.
#: Выравнивание идёт по переходам между соседними символами: WhisperX строит
#: решётку, где каждый шаг — «остаться на этом символе или перейти к
#: следующему». В сегменте из одного знака следующего символа нет, список
#: переходов оказывается пустым, torch считает пустой список вещественным и
#: отказывается индексировать им — с сообщением «tensors used as indices must
#: be long, int, byte or bool tensors». Whisper такие сегменты выдаёт
#: постоянно: «А», «И», «—» — междометия и обрывки на паузах. Один такой
#: сегмент в часовой записи ронял весь разбор на последнем шаге.
MIN_ALIGNABLE_CHARS = 2


def is_alignable(segment: dict) -> bool:
    """Можно ли выравнивать этот сегмент, не роняя WhisperX."""
    return len(str(segment.get("text") or "").strip()) >= MIN_ALIGNABLE_CHARS


def _unaligned(segment: dict) -> dict:
    """Сегмент в том виде, в каком его отдаёт WhisperX, когда выровнять не смог.

    Времена остаются от распознавания — секунда-другая неточности. Выбросить
    такой сегмент было бы хуже: «Да» и «Нет» в ответ на вопрос — это решение
    совещания, и оно должно остаться в стенограмме.
    """
    return {
        "text": segment.get("text", ""),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "words": [],
    }


def align_alignable(segments: Sequence[dict], align: Callable[[list[dict]], dict]) -> dict:
    """Выравнивает всё, кроме сегментов, на которых WhisperX падает.

    Короткие сегменты изымаются перед вызовом и возвращаются на своё место по
    времени начала — с временами от распознавания. Порядок сегментов важен:
    из них потом собираются реплики, а реплика, приехавшая не туда, меняет
    смысл разговора.

    :param align: чем выравнивать. Вынесено параметром, чтобы разбор
        проверялся без видеокарты и без моделей.
    """
    short = [s for s in segments if not is_alignable(s)]
    if not short:
        return align(list(segments))

    logger.info(
        "Выравниваю без {} коротких сегментов из {}: WhisperX на них падает",
        len(short), len(segments),
    )
    rest = [s for s in segments if is_alignable(s)]
    aligned = align(rest) if rest else {"segments": [], "word_segments": []}

    merged = list(aligned.get("segments", [])) + [_unaligned(s) for s in short]
    # Сортировка устойчивая: при совпадении времён выровненный сегмент
    # остаётся впереди возвращённого.
    merged.sort(key=lambda s: (s.get("start") is None, s.get("start") or 0.0))
    return {**aligned, "segments": merged}


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


#: Названия колонок, которыми открывается файл-справочник. Подсказкой они
#: не являются: «ФИО» на совещании не произносят.
HEADINGS = frozenset({
    "фио", "фамилия", "имя", "участник", "участники", "слово", "слова",
    "термин", "термины", "название",
})

#: Сколько подсказки принимает Whisper. Окно модели — 448 токенов, половина
#: отведена под предыдущий текст, и всё сверх этого она отбрасывает молча.
#: Поэтому режем сами и говорим, что не поместилось.
HINT_TOKENS = 200


def read_hints(path: "str | Path | None") -> str:
    """Свои слова для распознавания — как их ждёт Whisper.

    Подсказка у Whisper — это **предыдущий текст**: модель считает, что перед
    записью прозвучало вот это, и продолжает в том же духе. Отсюда две формы
    файла, и они не равноценны.

    **Связной фразой** — если первая строка кончается точкой. Берётся как
    есть, до конца последнего поместившегося предложения. Это родная для
    модели форма: она получает не перечень, а речь того же склада, что
    ожидается дальше.

    **Списком** — по слову в строке, или csv, тогда берётся первая колонка,
    и годится тот же `участники.csv`, что разобран из списка приглашённых.
    Слова сшиваются через запятую. Форма простая, но чужеродная: на записи
    штаба список из 54 слов не дал ничего, а расшифровка вышла на 5 % короче
    и с лишними галлюцинациями на тишине. Похоже, перечень уводит декодер в
    перечислительный лад и занимает весь бюджет подсказки.

    Не поместившееся отбрасывается здесь, а не в модели, и попадает в
    журнал: молча укоротить подсказку значит оставить человека гадать,
    почему половина фамилий по-прежнему перевирается.
    """
    if not path:
        logger.info(
            "Подсказки распознаванию не заданы (asr_hints_file) — "
            "фамилии и термины разбираются на слух."
        )
        return ""
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        logger.warning("Не прочитать подсказки {}, распознаю без них: {}", path, exc)
        return ""

    meaningful = [line.strip() for line in lines if line.strip()]
    if meaningful and _looks_like_speech(meaningful[0]):
        return _fit_speech(meaningful)

    words: list[str] = []
    for line in lines:
        word = line.split(";")[0].split(",")[0].strip()
        if word and word not in words:
            words.append(word)
    if words and words[0].lower() in HEADINGS:
        # Первая строка csv — название колонки, а не слово. «ФИО» в
        # подсказке сбивает модель: она такого на совещании не услышит.
        words.pop(0)
    if not words:
        return ""

    kept: list[str] = []
    while words and estimate_tokens(", ".join(kept + words[:1])) <= HINT_TOKENS:
        kept.append(words.pop(0))
    if words:
        logger.warning(
            "Подсказок больше, чем принимает Whisper: взяты первые {}, "
            "отброшено {} ({}…). Важное ставьте в начало файла. Если это "
            "список участников — оставьте в нём одни фамилии: их влезет "
            "втрое больше, а коверкает распознавание прежде всего их.",
            len(kept), len(words), ", ".join(words[:3]),
        )
    logger.info("Подсказки распознаванию: {} слов списком", len(kept))
    return ", ".join(kept) + "."


def _looks_like_speech(line: str) -> bool:
    """Фраза это или строка списка.

    Различает точка в конце: в списке фамилий и терминов её не ставят, а
    предложение без неё не кончается. Внутренняя точка не в счёт — «ВЭБ.РФ»
    остаётся словом списка.
    """
    return " " in line and line.rstrip().endswith((".", "!", "?", "…"))


def _fit_speech(lines: list[str]) -> str:
    """Подсказка-фраза, урезанная по целым предложениям.

    По целым намеренно: оборванная на полуслове подсказка хуже никакой —
    модель продолжит незаконченную мысль вместо того, чтобы слушать запись.
    """
    text = " ".join(lines)
    sentences = re.findall(r"[^.!?…]+[.!?…]+|\S+$", text)
    kept: list[str] = []
    for sentence in sentences:
        if estimate_tokens("".join(kept + [sentence])) > HINT_TOKENS:
            break
        kept.append(sentence)
    if not kept:
        logger.warning(
            "Первое же предложение подсказки длиннее, чем принимает Whisper "
            "(~{} токенов) — распознаю без подсказок. Разбейте его короче.",
            HINT_TOKENS,
        )
        return ""
    if len(kept) < len(sentences):
        logger.warning(
            "Подсказка длиннее, чем принимает Whisper: взяты первые {} "
            "предложений из {}, остальное отброшено.",
            len(kept), len(sentences),
        )
    hint = " ".join(s.strip() for s in kept)
    logger.info("Подсказки распознаванию: фраза в {} слов", len(hint.split()))
    return hint


def _whisperx_backend() -> Backend:
    """Настоящий WhisperX. Ввозится лениво — его может не быть вовсе."""
    try:
        import whisperx  # noqa: F401
    except ImportError as exc:  # pragma: no cover — проверяется вручную
        raise RecognitionError(
            "Не установлен whisperx. Он нужен только для распознавания; "
            "поставьте вместе с зависимостями: pip install -e '.[asr]'"
        ) from exc
    return _WhisperX()


class _WhisperX:  # pragma: no cover — требует моделей и видеокарты
    """Тонкая обёртка: вызовы WhisperX и ничего больше."""

    def transcribe(
        self, audio: str, *, model: str, language: str, batch_size: int,
        device: str, hints: str = "",
    ) -> dict:
        import whisperx

        common = dict(language=language, compute_type=compute_type_for(device))
        try:
            asr = whisperx.load_model(
                model, device,
                asr_options={"initial_prompt": hints} if hints else None,
                **common,
            )
        except TypeError as exc:
            # Ранние whisperx не знают ``asr_options``. Молча считать без
            # подсказок нельзя: распознавание без них другое, и человек
            # должен знать, что получил именно его.
            logger.warning(
                "Распознавание без подсказок: whisperx их не принимает ({}). "
                "Расшифровка выйдет прежней, но фамилии и термины в ней "
                "будут коверкаться чаще.", exc,
            )
            asr = whisperx.load_model(model, device, **common)
        sound = whisperx.load_audio(audio)
        try:
            return asr.transcribe(sound, batch_size=batch_size)
        finally:
            # Модель больше не нужна, а следующему шагу нужна её память.
            del asr
            free_vram()

    def align(self, segments: list[dict], *, language: str, audio: str, device: str) -> dict:
        import whisperx

        model, metadata = whisperx.load_align_model(language_code=language, device=device)
        sound = whisperx.load_audio(audio)
        try:
            return align_alignable(
                segments,
                lambda batch: whisperx.align(
                    batch, model, metadata, sound, device, return_char_alignments=False
                ),
            )
        finally:
            del model, metadata
            free_vram()

    def diarize(
        self, audio: str, aligned: dict, *, token: str, speakers: int | None, device: str
    ) -> dict:
        import whisperx

        pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
        sound = whisperx.load_audio(audio)
        try:
            diarized = pipeline(sound, min_speakers=speakers, max_speakers=speakers)
            return whisperx.assign_word_speakers(diarized, aligned)
        finally:
            del pipeline
            free_vram()


def check_model_access(
    token: str | None,
    *,
    models: Sequence[str] = GATED_MODELS,
    fetch: Callable[[str, str], Any] | None = None,
) -> dict[str, str]:
    """Пускают ли к закрытым моделям с этим токеном.

    Возвращает ``{модель: причина отказа}``; пустая строка означает, что
    доступ есть. Проверка занимает секунду и стоит того: иначе отказ
    вскроется после сорока минут распознавания, на последнем шаге.

    :param fetch: чем ходить в сеть. Вынесено параметром, чтобы проверять
        разбор ответов, не обращаясь к HuggingFace.
    """
    if not token:
        return {model: f"нет токена в {HF_TOKEN_ENV}" for model in models}

    fetch = fetch or _http_head
    result: dict[str, str] = {}
    for model in models:
        try:
            fetch(f"https://huggingface.co/api/models/{model}", token)
            result[model] = ""
        except Exception as exc:  # сеть, отказ, что угодно — всё это «нельзя»
            result[model] = _access_reason(exc)
    return result


def _access_reason(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code in (401, 403):
        return "условия модели не приняты либо токен без прав на чтение"
    if code == 404:
        return "модель не найдена: возможно, у вашей версии whisperx другая"
    return str(exc)


def _http_head(url: str, token: str) -> Any:
    import urllib.request

    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return urllib.request.urlopen(request, timeout=10)


def segments_to_json(recognition: Recognition, path: str | Path) -> Path:
    """Сохраняет сегменты отдельным файлом.

    Пригождается чаще, чем кажется: по этому файлу протокол пересобирается
    без распознавания, а значит без видеокарты — например, на другой машине.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(recognition.segments, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
