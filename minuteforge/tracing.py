"""Трассировка обращений к модели — если её включили и есть куда слать.

Разбор уже рассказывает о себе сам: журнал пишет воронку по стадиям, а
«разбор.md» — сколько пунктов ушло на каждой и почему. Чего в нём нет, так
это отдельного запроса: с каким указанием пошло окно, что ответила модель,
сколько это заняло. И нет сравнения прогонов между собой — а сравнивать
приходится постоянно, потому что при температуре 0,1 одна и та же запись
даёт разные списки.

Это и берёт на себя Langfuse. Прогон — трасса, запрос к модели — наблюдение
внутри неё, стадия — имя наблюдения. Воронку он не заменяет и заменить не
может: «было 238, стало 134, ушло 104, потому что ни просьбы, ни
повелительного наклонения» — знание о предметной области, и журнал пишет его
лучше.

Три правила, из которых сделан этот модуль.

**Молчание при любой беде.** Сорок минут распознавания и двадцать разбора не
должны пропасть оттого, что лёг контейнер с трассировкой или кто-то забыл
ключи. Поэтому здесь нет ни одного места, где ошибка уходила бы наверх: не
получилось — не получилось, работа идёт дальше.

**Выключено по умолчанию.** Пакета может не быть вовсе, и это нормально:
инструмент работает без него, как работал.

**Ключи только из окружения.** В файл настроек они не идут: настройки правят
и показывают, а секрет показывать нельзя.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from loguru import logger

#: Ключи доступа. Langfuse читает эти же имена сам, так что задавать их
#: дважды не придётся.
PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"

#: Сколько ждать ответа от Langfuse при закрытии прогона. Он собирает
#: события пачками и досылает их в конце; ждать этого дольше нескольких
#: секунд незачем — протокол уже собран, и трасса его не держит.
FLUSH_SECONDS = 5

_client = None
_broken = False


def _langfuse(settings):
    """Клиент Langfuse — один на всё время работы, или ничего.

    Сломалось однажды — больше не пробуем: если нет пакета или нет ключей,
    следующая попытка кончится тем же, а в журнал за прогон набежит три
    сотни одинаковых предупреждений.
    """
    global _client, _broken
    if _client is not None or _broken:
        return _client
    if not getattr(settings, "langfuse_enabled", False):
        _broken = True
        return None
    if not (os.environ.get(PUBLIC_KEY_ENV) and os.environ.get(SECRET_KEY_ENV)):
        logger.warning(
            "Трассировка включена, но ключей нет: задайте {} и {} в окружении. "
            "Разбор идёт без неё.",
            PUBLIC_KEY_ENV, SECRET_KEY_ENV,
        )
        _broken = True
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(host=getattr(settings, "langfuse_host", "") or None)
    except Exception as exc:  # пакета нет, адрес не тот, что угодно ещё
        logger.warning("Трассировка не поднялась, иду без неё: {}", exc)
        _broken = True
        return None
    logger.info("Трассировка включена: {}", getattr(settings, "langfuse_host", ""))
    return _client


def forget() -> None:
    """Забыть клиента. Нужно проверкам и смене настроек на ходу."""
    global _client, _broken
    _client = None
    _broken = False


class _Nothing:
    """Заглушка на случай, когда трассировки нет.

    Ею возвращается всё, чем пользуются снаружи, — тогда у вызывающего не
    появляется ни одной проверки «а включено ли»: код разбора одинаков в
    обоих случаях.
    """

    def update(self, **_) -> None:
        pass

    def done(self, _reply=None) -> None:
        pass


NOTHING = _Nothing()


@contextmanager
def run(name: str, settings, **about):
    """Прогон целиком — одна трасса.

    Имя берётся то же, что у папки прогона: день ВКС, час расчёта, модели.
    Тогда трасса и папка ищутся одна по другой, а прогоны одной записи
    ложатся рядом — ради этого всё и затевалось.
    """
    client = _langfuse(settings)
    if client is None:
        yield NOTHING
        return
    try:
        with client.start_as_current_observation(as_type="span", name=name) as span:
            span.update(metadata={k: v for k, v in about.items() if v})
            yield span
    except Exception as exc:  # pragma: no cover — сеть, версия, что угодно
        logger.warning("Трасса прогона не завелась: {}", exc)
        yield NOTHING
    finally:
        try:
            client.flush()
        except Exception:  # pragma: no cover
            pass


class _Answer:
    """Наблюдение за одним запросом: дождаться ответа и записать его."""

    def __init__(self, span) -> None:
        self._span = span

    def update(self, **fields) -> None:
        try:
            self._span.update(**fields)
        except Exception:  # pragma: no cover
            pass

    def done(self, reply=None) -> None:
        if reply is None:
            return
        usage = {}
        if getattr(reply, "prompt_tokens", 0) or getattr(reply, "completion_tokens", 0):
            usage = {
                "input": reply.prompt_tokens,
                "output": reply.completion_tokens,
            }
        self.update(
            output=getattr(reply, "text", ""),
            usage_details=usage or None,
            metadata={
                "секунд": getattr(reply, "elapsed_s", 0.0),
                "оборван": getattr(reply, "truncated", False),
            },
        )


@contextmanager
def answer(settings, *, name: str, model: str, system: str, user: str, **about):
    """Один запрос к модели.

    Пишется и указание, и текст запроса: спор о том, почему из этого окна
    ничего не вышло, решается только ими. Молчит, если трассировки нет.
    """
    client = _langfuse(settings)
    if client is None:
        yield NOTHING
        return
    try:
        with client.start_as_current_observation(
            as_type="generation", name=name or "запрос", model=model,
        ) as span:
            span.update(
                input={"система": system, "запрос": user},
                metadata={k: v for k, v in about.items() if v not in (None, "")},
            )
            yield _Answer(span)
    except Exception as exc:  # pragma: no cover
        logger.warning("Запрос не записался в трассу: {}", exc)
        yield NOTHING
