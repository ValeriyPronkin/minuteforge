"""Обращение к локальной модели по OpenAI-совместимому протоколу.

Модель поднимается рядом — Ollama, LM Studio, vLLM, любой сервер с
``/v1/chat/completions``. Смысл именно в этом: стенограмма совещания уходит
в модель целиком, и отправлять её наружу нельзя. Ключей здесь нет, потому
что и сервиса снаружи нет.

Модуль знает про HTTP и ничего не знает про протоколы и поручения: что
спрашивать, решает :mod:`minuteforge.tasks`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import requests
from loguru import logger

from .config import Settings


#: Адреса, которые заведомо на этой же машине.
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class LLMError(RuntimeError):
    """Модель не ответила или ответила не тем."""


class LLMUnavailable(LLMError):
    """Сервер модели недоступен.

    Отдельный класс, потому что лечится это иначе: не повтором запроса, а
    запуском сервера. Самая частая ошибка при первом знакомстве.
    """


class _Session(Protocol):
    """То, что нужно клиенту от ``requests.Session``.

    Протокол объявлен явно, чтобы в тестах подставлялась заглушка, а не
    поднимался настоящий сервер: проверять разбор ответа на живой модели
    долго и невоспроизводимо.
    """

    def post(self, url: str, *, json: dict, timeout: float) -> Any: ...

    def get(self, url: str, *, timeout: float) -> Any: ...


@dataclass
class Reply:
    """Ответ модели."""

    text: str
    #: Сколько токенов сообщил сервер. Ollama и LM Studio отдают это
    #: по-разному, а иногда не отдают вовсе — тогда ноль.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Сколько секунд занял запрос. Нужно не для красоты: по этому числу
    #: видно, что модель считает на процессоре вместо видеокарты.
    elapsed_s: float = 0.0


class LLMClient:
    """Клиент к локальному серверу модели."""

    def __init__(self, settings: Settings | None = None, session: _Session | None = None):
        self.settings = settings or Settings()
        self._session = session or self._local_session()

    def _local_session(self) -> requests.Session:
        """Сессия, которая не пойдёт к соседу через проходную.

        На корпоративной машине почти всегда заданы HTTP_PROXY и HTTPS_PROXY,
        и requests послушно гонит через прокси даже обращение к localhost.
        Прокси про localhost ничего не знает и отвечает отказом — со стороны
        это выглядит как «Ollama не запущена», хотя она работает.
        """
        session = requests.Session()
        if urlparse(self.settings.llm_base_url).hostname in LOCAL_HOSTS:
            session.trust_env = False
        return session

    @property
    def base_url(self) -> str:
        """Адрес API, дополненный, если человек ввёл его без пути.

        В поле адреса естественно вписать «http://localhost:11434» — так
        Ollama и представляется. Но запросы идут по OpenAI-совместимому
        протоколу, где путь начинается с /v1, и без него сервер отвечает
        404, неотличимым от «модель не найдена».
        """
        url = self.settings.llm_base_url.strip().rstrip("/")
        path = urlparse(url).path.strip("/")
        return url if path else f"{url}/v1"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Reply:
        """Задаёт модели один вопрос и возвращает ответ.

        Повтор делается только для сбоев, которые бывают временными:
        таймаут и ошибка сервера. Отказ вида «нет такой модели» повторять
        бессмысленно — он повторится ровно так же, только время уйдёт.
        """
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "stream": False,
        }
        if json_mode:
            # Сервер принуждает модель выдавать разбираемый JSON, отсекая на
            # каждом шаге всё, что его сломает. Для мелкой модели это
            # решающая разница: уговорами формат от неё не добиться, она
            # просто продолжает текст, который видит.
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(1, self.settings.llm_retries + 2):
            started = time.perf_counter()
            try:
                response = self._session.post(
                    self.endpoint, json=payload, timeout=self.settings.llm_timeout_s
                )
            except requests.exceptions.ConnectionError as exc:
                # Повторять нечего: сервер не запущен, и сам он не запустится.
                raise LLMUnavailable(
                    f"Модель не отвечает по адресу {self.settings.llm_base_url}. "
                    "Запущен ли Ollama или LM Studio и слушает ли он этот порт?"
                ) from exc
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning(
                    "Модель не ответила за {} с (попытка {})",
                    self.settings.llm_timeout_s, attempt,
                )
                continue

            if response.status_code >= 500:
                last_error = LLMError(f"Сервер модели вернул {response.status_code}")
                logger.warning("Сервер модели вернул {} (попытка {})", response.status_code, attempt)
                time.sleep(min(2 ** attempt, 10))
                continue
            if response.status_code != 200:
                if json_mode and response.status_code in (400, 422):
                    # Строгий JSON понимают не все серверы. Отказ от него —
                    # не беда: разбор ответа умеет и строки, просто мелкая
                    # модель справится хуже.
                    logger.info("Сервер не понял строгий JSON, повторяю без него")
                    # Копией, а не правкой на месте: тот словарь уже ушёл в
                    # запрос, и менять его задним числом — верный способ
                    # однажды получить в журнале не то, что было отправлено.
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                    json_mode = False
                    continue
                raise LLMError(
                    f"Сервер модели вернул {response.status_code}: "
                    f"{_short(getattr(response, 'text', ''))}"
                )

            return _parse(response, time.perf_counter() - started)

        raise LLMError(
            f"Модель не ответила за {self.settings.llm_retries + 1} попыток: {last_error}"
        )

    def available_models(self) -> list[str]:
        """Какие модели подняты на сервере.

        Нужно, чтобы человек выбирал из установленного, а не вспоминал
        точное написание имени: ошибиться в «qwen2.5:14b» проще простого, а
        сервер на такое отвечает сухим отказом.

        Пустой список означает «спросить не удалось» — сервер не поднят,
        отвечает иначе или это вовсе не Ollama. Это не ошибка: имя модели
        всегда можно вписать руками.
        """
        url = f"{self.base_url}/models"
        try:
            response = self._session.get(url, timeout=10)
            if getattr(response, "status_code", 0) != 200:
                return []
            body = response.json()
        except Exception as exc:
            logger.info("Список моделей получить не удалось: {}", exc)
            return []

        items = body.get("data") if isinstance(body, dict) else None
        if not isinstance(items, list):
            return []
        return sorted(
            str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")
        )

    def health(self) -> bool:
        """Отвечает ли сервер и есть ли на нём заданная модель."""
        return not self.diagnose()

    def diagnose(self) -> str:
        """Что мешает работать с моделью. Пустая строка — всё в порядке.

        Проверять стоит до расчёта, а не после: распознавание часового
        совещания идёт десятки минут, и упереться в незапущенный сервер на
        последнем шаге особенно обидно.

        Причин две, и они лечатся разным. Сервер не поднят — его надо
        запустить. Сервер отвечает, но отказывает — почти всегда это значит,
        что модель не скачана, и говорить в таком случае «сервер не
        отвечает» значит отправить человека чинить исправное.
        """
        try:
            self.complete("Ответь одним словом.", "Проверка связи.", max_tokens=8)
        except LLMUnavailable as exc:
            logger.warning("Сервер модели недоступен: {}", exc)
            return str(exc)
        except LLMError as exc:
            logger.warning("Модель не отвечает как надо: {}", exc)
            return (
                f"Сервер по адресу {self.settings.llm_base_url} отвечает, но "
                f"работать отказывается: {exc}\n"
                f"Чаще всего это значит, что модель «{self.settings.llm_model}» "
                "не скачана. Для Ollama: ollama pull "
                f"{self.settings.llm_model}"
            )
        return ""


def _parse(response: Any, elapsed: float) -> Reply:
    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(
            f"Не разобрать ответ модели: {_short(getattr(response, 'text', ''))}"
        ) from exc

    usage = body.get("usage") or {}
    return Reply(
        text=(text or "").strip(),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        elapsed_s=round(elapsed, 2),
    )


def _short(text: str, limit: int = 200) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"
