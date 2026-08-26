from minuteforge.config import HF_TOKEN_ENV, Settings


def test_chunk_budget_comes_from_the_model_window():
    """Нарезка существует потому, что стенограмма не влезает в окно."""
    assert Settings(llm_context_tokens=8192).chunk_budget == int(8192 * 0.65)


def test_huge_window_means_one_piece():
    """Часовое совещание — тысяч двадцать токенов. При окне в миллион делить
    нечего, и все хлопоты с нахлёстом и повторами отпадают сами."""
    from minuteforge.blocks import Block
    from minuteforge.chunking import split_into_chunks

    settings = Settings(llm_context_tokens=1_000_000)
    blocks = [Block("Иванов", "слово " * 200) for _ in range(200)]
    chunks = split_into_chunks(blocks, max_tokens=settings.chunk_budget)
    assert len(chunks) == 1


def test_explicit_budget_wins_over_the_window():
    """Задают его, чтобы сделать куски меньше расчётных: малая модель на
    длинном фрагменте теряет поручения из середины."""
    assert Settings(llm_context_tokens=8192, chunk_max_tokens=1500).chunk_budget == 1500


def test_budget_never_collapses_to_nothing():
    assert Settings(llm_context_tokens=10).chunk_budget == 500


def test_token_is_read_only_from_the_environment(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    assert Settings().hf_token is None
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_test")
    assert Settings().hf_token == "hf_test"


def test_secrets_never_reach_the_settings_dump(monkeypatch):
    """to_dict уходит в журнал: токену там не место."""
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_test")
    assert "hf_test" not in str(Settings().to_dict())


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("MINUTEFORGE_ASR_MODEL", "large-v3")
    monkeypatch.setenv("MINUTEFORGE_LLM_CONTEXT_TOKENS", "32768")
    settings = Settings.load("нет-такого-файла.yaml")
    assert settings.asr_model == "large-v3"
    assert settings.llm_context_tokens == 32768
