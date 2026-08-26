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

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from loguru import logger

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

    def transcribe(self, audio: str, *, model: str, language: str, batch_size: int, device: str) -> dict: ...

    def align(self, segments: list[dict], *, language: str, audio: str, device: str) -> dict: ...

    def diarize(
        self, audio: str, aligned: dict, *, token: str, speakers: int | None, device: str
    ) -> dict: ...


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
            "это в разы дольше."
        )
    return device


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
            "В файл настроек и тем более в код — нельзя."
        )

    backend = backend or _whisperx_backend()
    device = resolve_device(settings.device)
    cache = Path(cache_dir) if cache_dir else None
    result = Recognition(device=device)

    logger.info("Распознавание: модель {}, устройство {}", settings.asr_model, device)
    raw = _step(
        cache, "transcription.json", result, progress, "transcribe",
        lambda: backend.transcribe(
            str(audio),
            model=settings.asr_model,
            language=settings.language,
            batch_size=settings.batch_size,
            device=device,
        ),
    )
    result.language = raw.get("language") or settings.language

    aligned = _step(
        cache, "aligned.json", result, progress, "align",
        lambda: backend.align(
            raw.get("segments", []),
            language=result.language,
            audio=str(audio),
            device=device,
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
            device=device,
        ),
    )
    result.segments = assigned.get("segments", [])
    logger.info("Готово: сегментов {}", len(result.segments))
    return result


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


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


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

    def transcribe(self, audio: str, *, model: str, language: str, batch_size: int, device: str) -> dict:
        import whisperx

        asr = whisperx.load_model(model, device, language=language)
        sound = whisperx.load_audio(audio)
        return asr.transcribe(sound, batch_size=batch_size)

    def align(self, segments: list[dict], *, language: str, audio: str, device: str) -> dict:
        import whisperx

        model, metadata = whisperx.load_align_model(language_code=language, device=device)
        sound = whisperx.load_audio(audio)
        return whisperx.align(segments, model, metadata, sound, device, return_char_alignments=False)

    def diarize(
        self, audio: str, aligned: dict, *, token: str, speakers: int | None, device: str
    ) -> dict:
        import whisperx

        pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
        sound = whisperx.load_audio(audio)
        diarized = pipeline(sound, min_speakers=speakers, max_speakers=speakers)
        return whisperx.assign_word_speakers(diarized, aligned)


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
