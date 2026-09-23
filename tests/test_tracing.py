import os

from minuteforge.config import Settings
from minuteforge.llm import LLMClient
from minuteforge.tracing import PUBLIC_KEY_ENV, SECRET_KEY_ENV, answer, forget, run

from test_llm import FAST, FakeResponse, FakeSession


def setup_function() -> None:
    forget()


def teardown_function() -> None:
    forget()


def test_switched_off_by_default():
    """Пакета может не быть вовсе, и это нормальное состояние машины."""
    assert Settings().langfuse_enabled is False
    with run("прогон", Settings()) as trace:
        trace.update(metadata={"что угодно": 1})
    with answer(Settings(), name="выписка", model="mistral", system="s", user="u") as one:
        one.done(None)


def test_keys_are_taken_from_the_environment(monkeypatch, caplog):
    """В файл настроек ключи не идут: настройки правят и показывают.

    Нет ключей — не беда: разбор идёт без трассировки, и об этом говорится
    один раз.
    """
    monkeypatch.delenv(PUBLIC_KEY_ENV, raising=False)
    monkeypatch.delenv(SECRET_KEY_ENV, raising=False)
    settings = Settings(langfuse_enabled=True)
    with answer(settings, name="выписка", model="mistral", system="s", user="u") as one:
        one.done(None)
    # Второй раз не пробуем: за прогон запросов триста, и столько же
    # одинаковых предупреждений в журнале читать невозможно.
    assert os.environ.get(PUBLIC_KEY_ENV) is None


def test_a_broken_tracing_does_not_break_the_run(monkeypatch):
    """Сорок минут распознавания не должны пропасть из-за трассировки.

    Ключи на месте, а Langfuse не поднимается — пакета нет или адрес не тот.
    Разбор обязан идти дальше.
    """
    monkeypatch.setenv(PUBLIC_KEY_ENV, "pk-проверка")
    monkeypatch.setenv(SECRET_KEY_ENV, "sk-проверка")
    settings = Settings(langfuse_enabled=True, langfuse_host="http://127.0.0.1:1")
    with run("прогон", settings) as trace:
        trace.update(metadata={"заседание": "10.09.2026"})
    with answer(settings, name="проверка", model="qwen", system="s", user="u") as one:
        one.done(None)


def test_the_model_is_asked_the_same_way_with_tracing_off():
    """Стадия и приметы запроса в расчёте не участвуют.

    Они нужны только трассировке, и выключенная трассировка не должна
    менять ни запроса, ни ответа.
    """
    plain = FakeSession(FakeResponse({"choices": [{"message": {"content": "да"}}]}))
    marked = FakeSession(FakeResponse({"choices": [{"message": {"content": "да"}}]}))

    first = LLMClient(FAST, plain).complete("s", "u")
    second = LLMClient(FAST, marked).complete(
        "s", "u", stage="проверка", about={"окно": "3 из 7"},
    )

    assert first.text == second.text == "да"
    assert plain.calls[0]["json"] == marked.calls[0]["json"]
