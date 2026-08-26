"""Параметры расчёта: значения по умолчанию, файл, переменные окружения.

Приоритет — от низкого к высокому: умолчания, ``config.yaml``, переменные
окружения с префиксом ``PROTOCALL_``. Секреты живут только в окружении: токен
HuggingFace однажды уже был вписан в код и в файл рядом с ним, и это стоило
отзыва токена.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

#: Переменные окружения, откуда берутся секреты. В ``config.yaml`` им не место.
HF_TOKEN_ENV = "HF_TOKEN"


@dataclass
class Settings:
    """Параметры распознавания, нарезки и обращения к модели."""

    # --- распознавание -------------------------------------------------
    #: Размер модели Whisper: ``tiny``…``large-v3``. Крупнее — точнее и
    #: заметно дольше; на совещании с плохим микрофоном разница решающая.
    asr_model: str = "medium"
    language: str = "ru"
    #: Сколько участников говорит. Диаризация работает точнее, когда число
    #: известно: иначе она угадывает и склонна дробить одного на нескольких.
    speakers: int | None = None
    #: ``cuda``, ``cpu`` или ``auto``.
    device: str = "auto"
    batch_size: int = 8
    #: Не ходить в HuggingFace, если модели уже скачаны. Быстрее на повторных
    #: запусках и не отмечает лишний раз использование токена.
    offline_models: bool = False

    # --- нарезка -------------------------------------------------------
    #: Окно модели в токенах. Отсюда считается бюджет куска — нарезка нужна
    #: ровно потому, что стенограмма не влезает целиком. У модели с большим
    #: окном кусок получится один: часовое совещание это тысяч двадцать
    #: токенов, и при окне в миллион делить нечего.
    llm_context_tokens: int = 8192
    #: Бюджет куска в токенах. Пусто — считать от окна. Задавать вручную
    #: имеет смысл, чтобы сделать куски **меньше** расчётных: малая модель
    #: на длинном фрагменте начинает терять поручения из середины.
    chunk_max_tokens: int | None = None
    #: Какую долю окна отдать тексту. Остальное — промпту и ответу: если
    #: занять окно целиком, ответ обрежется на середине последнего поручения.
    chunk_context_share: float = 0.65
    #: Сколько реплик повторить в начале следующего куска.
    chunk_overlap_blocks: int = 1

    # --- локальная модель ----------------------------------------------
    #: Адрес OpenAI-совместимого сервера: Ollama — 11434, LM Studio — 1234.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "mistral"
    #: Низкая температура намеренно: задача извлекающая, а не сочинительская.
    #: Поручение надо найти в тексте, а не придумать похожее.
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    #: Местная модель отвечает медленно, особенно первый запрос после
    #: загрузки весов. Короткий таймаут здесь ломает работу чаще, чем спасает.
    #: Просить ли у сервера строгий JSON. Для мелких моделей это решающая
    #: разница: уговорами формат от них не добиться, они просто продолжают
    #: текст, который видят. Сервер, который такого не умеет, ответит
    #: отказом, и запрос повторится без принуждения.
    llm_json_mode: bool = True
    llm_timeout_s: int = 180
    llm_retries: int = 2

    # --- пути ----------------------------------------------------------
    input_dir: Path = field(default_factory=lambda: Path("data/input"))
    output_dir: Path = field(default_factory=lambda: Path("data/output"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    log_level: str = "INFO"

    @property
    def chunk_budget(self) -> int:
        """Сколько токенов стенограммы уходит в один запрос."""
        if self.chunk_max_tokens:
            return int(self.chunk_max_tokens)
        return max(500, int(self.llm_context_tokens * self.chunk_context_share))

    @property
    def hf_token(self) -> str | None:
        """Токен HuggingFace — только из окружения.

        Свойством, а не полем: поле попало бы в ``asdict`` и утекло бы в
        журнал или в сохранённые настройки вместе со всем остальным.
        """
        return os.environ.get(HF_TOKEN_ENV) or None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """Собирает настройки из файла и окружения."""
        values: dict[str, Any] = {}
        path = Path(path) if path else Path("config.yaml")
        if path.exists() and yaml is not None:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            values.update({k: v for k, v in loaded.items() if k in _names()})

        for name, kind in _kinds().items():
            raw = os.environ.get(f"PROTOCALL_{name.upper()}")
            if raw is not None:
                values[name] = _cast(raw, kind)

        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Настройки для журнала. Секретов здесь нет и быть не должно."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def _names() -> set[str]:
    return {f.name for f in fields(Settings)}


def _kinds() -> dict[str, Any]:
    return {f.name: f.type for f in fields(Settings)}


def _cast(raw: str, kind: Any) -> Any:
    text = str(kind)
    if "Path" in text:
        return Path(raw)
    if "bool" in text:
        return raw.strip().lower() in {"1", "true", "yes", "да"}
    if "int" in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    return raw
