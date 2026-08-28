from unittest.mock import patch

import pytest
import requests

from minuteforge.config import Settings
from minuteforge.llm import LLMClient, LLMError, LLMUnavailable


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or ""

    def json(self):
        if self._payload is None:
            raise ValueError("не json")
        return self._payload


class FakeSession:
    """Заглушка сервера модели: отдаёт заранее заданные ответы по очереди."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, json, timeout, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.responses.pop(0) if self.responses else self.responses
        if isinstance(item, Exception):
            raise item
        return item


def answer(text, prompt_tokens=10, completion_tokens=5):
    return FakeResponse(
        {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    )


FAST = Settings(llm_retries=1, llm_timeout_s=1)


def test_request_goes_to_the_chat_endpoint():
    session = FakeSession(answer("Готово."))
    client = LLMClient(Settings(llm_base_url="http://localhost:1234/v1/"), session)

    client.complete("Ты секретарь.", "Стенограмма.")

    call = session.calls[0]
    assert call["url"] == "http://localhost:1234/v1/chat/completions"
    assert [m["role"] for m in call["json"]["messages"]] == ["system", "user"]
    assert call["json"]["stream"] is False


def test_temperature_is_low_because_the_task_is_extractive():
    """Поручение надо найти в тексте, а не сочинить похожее."""
    session = FakeSession(answer("Готово."))
    LLMClient(Settings(), session).complete("s", "u")
    assert session.calls[0]["json"]["temperature"] <= 0.2


def test_reply_carries_text_and_usage():
    session = FakeSession(answer("  Поручение: отчёт  ", 120, 30))
    reply = LLMClient(FAST, session).complete("s", "u")
    assert reply.text == "Поручение: отчёт"
    assert (reply.prompt_tokens, reply.completion_tokens) == (120, 30)
    assert reply.elapsed_s >= 0


def test_missing_usage_is_not_an_error():
    """Ollama и LM Studio считают токены по-разному, а иногда не считают."""
    session = FakeSession(FakeResponse({"choices": [{"message": {"content": "Да."}}]}))
    reply = LLMClient(FAST, session).complete("s", "u")
    assert reply.text == "Да."
    assert reply.prompt_tokens == 0


def test_server_not_running_says_so_plainly():
    """Самая частая ошибка при первом запуске — не поднят сервер модели.

    Повторы делаются: Ollama перезапускает рантайм, когда модель не влезла в
    видеопамять, и на эти секунды отказывает в соединении. Но если сервера
    нет вовсе, сказать надо прямо и с адресом в тексте.
    """
    session = FakeSession(*[requests.exceptions.ConnectionError("отказано")] * 4)
    with patch("time.sleep"):
        with pytest.raises(LLMUnavailable, match="11434"):
            LLMClient(Settings(llm_retries=1), session).complete("s", "u")
    assert len(session.calls) == 2, "одна попытка и один повтор"


def test_timeout_is_retried():
    session = FakeSession(requests.exceptions.Timeout(), answer("Со второго раза."))
    reply = LLMClient(FAST, session).complete("s", "u")
    assert reply.text == "Со второго раза."
    assert len(session.calls) == 2


def test_retries_run_out_with_a_clear_error():
    session = FakeSession(requests.exceptions.Timeout(), requests.exceptions.Timeout())
    with pytest.raises(LLMError, match="попыток"):
        LLMClient(FAST, session).complete("s", "u")


def test_server_error_is_retried_but_client_error_is_not():
    """5xx бывает временной, 4xx — нет.

    «Нет такой модели» повторится ровно так же, только время уйдёт.
    """
    retried = FakeSession(FakeResponse(status_code=503, text="перегрузка"), answer("Ок."))
    assert LLMClient(FAST, retried).complete("s", "u").text == "Ок."

    refused = FakeSession(FakeResponse(status_code=404, text="model 'mistral' not found"))
    with pytest.raises(LLMError, match="404"):
        LLMClient(FAST, refused).complete("s", "u")
    assert len(refused.calls) == 1


def test_unparsable_answer_shows_what_came_back():
    session = FakeSession(FakeResponse(None, text="<html>Not Found</html>"))
    with pytest.raises(LLMError, match="Not Found"):
        LLMClient(FAST, session).complete("s", "u")


def test_answer_without_choices_is_an_error_not_an_empty_string():
    """Пустой ответ молча превратился бы в протокол без поручений."""
    session = FakeSession(FakeResponse({"choices": []}))
    with pytest.raises(LLMError):
        LLMClient(FAST, session).complete("s", "u")


def test_health_check_before_a_long_run():
    assert LLMClient(FAST, FakeSession(answer("ок"))).health() is True
    with patch("time.sleep"):
        dead = FakeSession(*[requests.exceptions.ConnectionError()] * 4)
        assert LLMClient(FAST, dead).health() is False


def test_max_tokens_can_be_overridden_per_request():
    session = FakeSession(answer("ок"))
    LLMClient(FAST, session).complete("s", "u", max_tokens=8)
    assert session.calls[0]["json"]["max_tokens"] == 8


def test_diagnose_is_silent_when_everything_works():
    assert LLMClient(FAST, FakeSession(answer("ок"))).diagnose() == ""


def test_server_down_and_model_missing_are_different_diagnoses():
    """Сервер не поднят — его запускают. Сервер отвечает, но отказывает —
    почти всегда не скачана модель. Сказать «не отвечает» во втором случае
    значит отправить человека чинить исправное.
    """
    with patch("time.sleep"):
        down = LLMClient(
            FAST, FakeSession(*[requests.exceptions.ConnectionError()] * 4)
        ).diagnose()
    assert "11434" in down
    assert "pull" not in down, "чинить надо сервер, а не скачивать модель"

    missing = LLMClient(
        Settings(llm_model="mistral", llm_retries=0),
        FakeSession(FakeResponse(status_code=404, text="model 'mistral' not found")),
    )
    trouble = missing.diagnose()
    assert "отвечает" in trouble
    assert "ollama pull mistral" in trouble


def test_local_model_is_not_routed_through_a_corporate_proxy():
    """На рабочей машине заданы HTTP_PROXY и HTTPS_PROXY, и requests гонит
    через прокси даже localhost. Прокси про него не знает и отказывает — со
    стороны это выглядит как «Ollama не запущена», хотя она работает.
    """
    local = LLMClient(Settings(llm_base_url="http://localhost:11434/v1"))
    assert local._session.trust_env is False


def test_remote_model_still_respects_the_environment():
    """Если модель вынесена на сервер, прокси может быть единственным путём
    к ней — отключать его там нельзя."""
    remote = LLMClient(Settings(llm_base_url="http://gpu-server.local:11434/v1"))
    assert remote._session.trust_env is True


class ModelsSession(FakeSession):
    """Заглушка, умеющая отвечать и на запрос списка моделей."""

    def __init__(self, models_response, *responses):
        super().__init__(*responses)
        self.models_response = models_response
        self.asked = []

    def get(self, url, *, timeout, headers=None):
        self.asked.append(url)
        if isinstance(self.models_response, Exception):
            raise self.models_response
        return self.models_response


def models(*names):
    return FakeResponse({"data": [{"id": name} for name in names]})


def test_installed_models_are_listed():
    """Человек должен выбирать из установленного, а не вспоминать точное
    написание: ошибиться в «qwen2.5:14b» проще простого."""
    session = ModelsSession(models("qwen2.5:14b", "mistral:latest"))
    assert LLMClient(FAST, session).available_models() == ["mistral:latest", "qwen2.5:14b"]
    assert session.asked[0].endswith("/v1/models")


def test_unreachable_server_gives_an_empty_list_not_an_error():
    """Спросить не удалось — не беда: имя модели можно вписать руками."""
    session = ModelsSession(requests.exceptions.ConnectionError())
    assert LLMClient(FAST, session).available_models() == []


def test_unexpected_answer_shape_is_survived():
    """На этом адресе может отвечать вовсе не Ollama."""
    assert LLMClient(FAST, ModelsSession(FakeResponse({"что-то": "иное"}))).available_models() == []
    assert LLMClient(FAST, ModelsSession(FakeResponse(None, 404, "nope"))).available_models() == []


def test_json_mode_is_asked_of_the_server():
    session = FakeSession(answer('{"tasks": []}'))
    LLMClient(FAST, session).complete("s", "u", json_mode=True)
    assert session.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_server_that_cannot_do_json_is_asked_again_without_it():
    """Строгий JSON понимают не все серверы. Отказ — не беда: разбор умеет и
    строки, просто мелкая модель справится хуже."""
    session = FakeSession(
        FakeResponse(status_code=400, text="response_format is not supported"),
        answer("Поручение: Сделать"),
    )
    reply = LLMClient(FAST, session).complete("s", "u", json_mode=True)

    assert reply.text == "Поручение: Сделать"
    assert "response_format" in session.calls[0]["json"]
    assert "response_format" not in session.calls[1]["json"], "второй раз просить незачем"


def test_address_without_a_path_gets_v1_added():
    """В поле адреса естественно вписать «http://localhost:11434» — так
    Ollama и представляется. Без /v1 сервер отвечает 404, неотличимым от
    «модель не найдена».
    """
    client = LLMClient(Settings(llm_base_url="http://localhost:11434"))
    assert client.endpoint == "http://localhost:11434/v1/chat/completions"


def test_address_with_a_path_is_left_alone():
    """LM Studio и другие серверы могут жить по своему пути."""
    client = LLMClient(Settings(llm_base_url="http://localhost:1234/v1/"))
    assert client.endpoint == "http://localhost:1234/v1/chat/completions"

    custom = LLMClient(Settings(llm_base_url="http://gpu.local/llm/openai"))
    assert custom.endpoint == "http://gpu.local/llm/openai/chat/completions"


def test_schema_is_asked_of_the_server():
    """Просто «какой-нибудь JSON» мелкой модели мало: она выдаёт безупречно
    правильный JSON собственной выдумки — {"name": ..., "description": ...} —
    и наполовину по-английски. Схема отсекает такое на каждом шаге.
    """
    session = FakeSession(answer('{"tasks": []}'))
    schema = {"type": "object", "properties": {"tasks": {"type": "array"}}}
    LLMClient(FAST, session).complete("s", "u", json_mode=True, schema=schema)

    asked = session.calls[0]["json"]["response_format"]
    assert asked["type"] == "json_schema"
    assert asked["json_schema"]["schema"] == schema


def test_server_without_schema_support_falls_back_by_one_step():
    """Уступаем по ступени: схему понимают не все, «просто JSON» — почти
    все, строки — везде. Терять сразу всё из-за старой версии незачем."""
    session = FakeSession(
        FakeResponse(status_code=400, text="json_schema is not supported"),
        answer('{"tasks": []}'),
    )
    LLMClient(FAST, session).complete("s", "u", json_mode=True, schema={"type": "object"})

    assert session.calls[0]["json"]["response_format"]["type"] == "json_schema"
    assert session.calls[1]["json"]["response_format"]["type"] == "json_object"


def test_truncated_answer_is_marked():
    """«Ответ обрезан» и «модель ничего не нашла» — разные беды, и лечатся
    они разным."""
    cut = FakeResponse({
        "choices": [{"message": {"content": '{"tasks": [{"what": "Подгот'}, "finish_reason": "length"}]
    })
    assert LLMClient(FAST, FakeSession(cut)).complete("s", "u").truncated is True
    assert LLMClient(FAST, FakeSession(answer("готово"))).complete("s", "u").truncated is False


def test_server_context_limit_is_asked_of_ollama():
    """Ollama держит собственный предел num_ctx, и всё сверх него молча
    отрезает: модель просто не видит конца фрагмента."""
    show = FakeResponse({
        "model_info": {"llama.context_length": 32768},
        "parameters": "num_ctx                        2048\nstop \"[INST]\"",
    })
    session = FakeSession(show)
    window, limit = LLMClient(FAST, session).context_window()

    assert (window, limit) == (32768, 2048)
    assert session.calls[0]["url"].endswith("/api/show")


def test_missing_context_info_is_not_an_error():
    """На этом адресе может отвечать не Ollama — тогда проверить нечем."""
    assert LLMClient(FAST, FakeSession(FakeResponse({}))).context_window() == (None, None)
    assert LLMClient(
        FAST, FakeSession(FakeResponse(None, 404, "nope"))
    ).context_window() == (None, None)


def test_show_request_carries_both_field_names():
    """У Ollama менялось имя поля: старые версии на новое отвечают отказом."""
    session = FakeSession(FakeResponse({"model_info": {"llama.context_length": 8192}}))
    LLMClient(FAST, session).context_window()

    sent = session.calls[0]["json"]
    assert sent["model"] == sent["name"] == "mistral"


def test_undeclared_limit_is_reported_as_unknown():
    """num_ctx виден, только если задан явно. У обычной модели его в ответе
    нет, а предел действует — и путать «неизвестен» с «нет предела» нельзя."""
    show = FakeResponse({
        "model_info": {"llama.context_length": 32768},
        "parameters": 'stop "[INST]"',
    })
    window, limit = LLMClient(FAST, FakeSession(show)).context_window()
    assert window == 32768
    assert limit is None


def test_key_is_sent_only_when_it_exists(monkeypatch):
    """Локальные серверы ключа не спрашивают, внешние без него отвечают
    отказом — и это выглядит как «модель не найдена»."""
    from minuteforge.config import LLM_KEY_ENV

    monkeypatch.delenv(LLM_KEY_ENV, raising=False)
    assert LLMClient(Settings())._headers() == {}

    monkeypatch.setenv(LLM_KEY_ENV, "sk-секрет")
    assert LLMClient(Settings())._headers() == {"Authorization": "Bearer sk-секрет"}


def test_key_never_lands_in_saved_settings(monkeypatch):
    """Файл настроек показывают, пересылают и кладут в репозиторий."""
    from dataclasses import asdict

    from minuteforge.config import LLM_KEY_ENV

    monkeypatch.setenv(LLM_KEY_ENV, "sk-секрет")
    assert "sk-секрет" not in str(asdict(Settings()))


def test_external_address_is_recognised_as_external():
    """Пока адрес локальный, стенограмма никуда не уходит. Внешний адрес это
    свойство отменяет, и знать об этом надо явно."""
    assert LLMClient(Settings(llm_base_url="http://localhost:11434")).is_local
    assert not LLMClient(Settings(llm_base_url="https://api.openai.com/v1")).is_local


def test_rejected_key_is_not_retried():
    """Ключ сам не появится: повторы только съедят время таймаута."""
    calls = []

    class Refusing:
        trust_env = True

        def post(self, url, **kwargs):
            calls.append(url)
            return FakeResponse({}, status_code=401)

    client = LLMClient(Settings(llm_base_url="https://api.example.com/v1"), Refusing())
    with pytest.raises(LLMError, match="ключ"):
        client.complete("система", "вопрос")
    assert len(calls) == 1


def test_model_matches_despite_the_tag():
    """Ollama показывает «mistral:latest», в настройках пишут «mistral».
    Без этого сравнения настройка не находилась в списке, и приложение
    молча брало первую по алфавиту — обычно bge-m3."""
    from minuteforge.llm import same_model

    assert same_model("mistral:latest", "mistral")
    assert same_model("qwen2.5:14b", "qwen2.5:7b")
    assert not same_model("mistral:latest", "llama3.1")


def test_embedding_models_are_recognised():
    """Разговаривать они не умеют: на просьбу найти поручения сервер вернёт
    пустоту, и выглядит это как поломка приложения."""
    from minuteforge.llm import is_embedder

    assert is_embedder("bge-m3:latest")
    assert is_embedder("nomic-embed-text")
    assert not is_embedder("mistral:latest")


def test_refused_connection_is_retried_before_giving_up():
    """Ollama перезапускает рантайм, когда модель не влезла в видеопамять —
    после распознавания её занимает torch. На эти секунды сервер отказывает
    в соединении, и раньше мы сдавались сразу."""
    tries = []

    class Flaky:
        trust_env = True

        def post(self, url, **kwargs):
            tries.append(url)
            if len(tries) < 3:
                raise requests.exceptions.ConnectionError("refused")
            return FakeResponse({"choices": [{"message": {"content": "Готово"}}]})

    client = LLMClient(Settings(llm_retries=3), Flaky())
    with patch("time.sleep"):
        assert client.complete("система", "вопрос").text == "Готово"
    assert len(tries) == 3


def test_server_that_never_answers_is_reported_once():
    """Повторы не бесконечны: если сервер не поднят, сказать надо прямо."""
    class Dead:
        trust_env = True

        def post(self, url, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

    client = LLMClient(Settings(llm_retries=1), Dead())
    with patch("time.sleep"):
        with pytest.raises(LLMUnavailable, match="не отвечает по адресу"):
            client.complete("система", "вопрос")
