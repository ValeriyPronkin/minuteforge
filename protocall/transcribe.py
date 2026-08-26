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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from .config import HF_TOKEN_ENV, Settings


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
) -> Recognition:
    """Распознаёт запись и размечает её по говорящим.

    :param cache_dir: куда складывать промежуточные шаги. Готовый шаг при
        повторном запуске читается с диска, а не считается заново.
    :param diarize: делить ли по говорящим. Без разделения протокол выйдет
        сплошным текстом без авторов, но зато не нужен токен HuggingFace —
        годится, чтобы посмотреть, как вообще распознаётся запись.
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
        raise MissingToken(
            "Для разделения по говорящим нужен токен HuggingFace: модели "
            f"pyannote закрытые. Положите его в переменную {HF_TOKEN_ENV} "
            "(в файл настроек и тем более в код — нельзя) и примите условия "
            "модели pyannote/speaker-diarization на huggingface.co."
        )

    backend = backend or _whisperx_backend()
    device = resolve_device(settings.device)
    cache = Path(cache_dir) if cache_dir else None
    result = Recognition(device=device)

    logger.info("Распознавание: модель {}, устройство {}", settings.asr_model, device)
    raw = _step(
        cache, "transcription.json", result,
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
        cache, "aligned.json", result,
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
        cache, "segments.json", result,
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


def _step(cache: Path | None, name: str, result: Recognition, work) -> dict:
    """Считает шаг или берёт готовый с диска."""
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / name
        if path.exists():
            logger.info("Шаг {} взят из {}", name, path)
            result.reused.append(name)
            return json.loads(path.read_text(encoding="utf-8"))

    value = work()

    if cache is not None:
        (cache / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return value


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
