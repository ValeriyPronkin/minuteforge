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

    def post(self, url, *, json, timeout):
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

    Повторять запрос бессмысленно: сам он не запустится, поэтому это
    отдельная ошибка с адресом в тексте.
    """
    session = FakeSession(requests.exceptions.ConnectionError("отказано"))
    with pytest.raises(LLMUnavailable, match="11434"):
        LLMClient(Settings(), session).complete("s", "u")
    assert len(session.calls) == 1, "повторов быть не должно"


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
    assert LLMClient(FAST, FakeSession(requests.exceptions.ConnectionError())).health() is False


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
    down = LLMClient(FAST, FakeSession(requests.exceptions.ConnectionError())).diagnose()
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

    def get(self, url, *, timeout):
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
