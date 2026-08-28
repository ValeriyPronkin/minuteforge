"""Интерфейс minuteforge.

    streamlit run app/streamlit_app.py

Приложение намеренно разделено на три шага, а не на одну кнопку: между
распознаванием и протоколом человек обязан вмешаться и сказать, кто такой
SPEAKER_00. Автоматически это не определить, а протокол с безымянными
участниками никому не нужен.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from minuteforge.blocks import (  # noqa: E402
    UNKNOWN,
    Transcript,
    blocks_from_segments,
    consolidate,
)
from minuteforge.audio import AudioError  # noqa: E402
from minuteforge.config import (  # noqa: E402
    CONFIG_EXAMPLE,
    CONFIG_FILE,
    HF_TOKEN_ENV,
    LLM_KEY_ENV,
    Settings,
)  # noqa: E402
from minuteforge.llm import LLMClient, is_embedder, same_model  # noqa: E402
from minuteforge.people import (  # noqa: E402
    merge_suggestions,
    mentioned_people,
    read_people,
    suggest_speakers,
)
from minuteforge.pipeline import (  # noqa: E402
    Meeting,
    cache_dir_for,
    cache_size,
    check_writable,
    clear_cache,
    protocol_from_transcript,
    _free_name,
    run_dir,
    save,
    save_transcript,
    transcribe_meeting,
)
from minuteforge.transcribe import (  # noqa: E402
    MissingToken,
    RecognitionError,
    check_model_access,
)

# Свой файл, а нет своего — образец из репозитория. Свой в репозитории не
# лежит: его правят под своё место работы, и обновление затирало бы правки.
_own = ROOT / CONFIG_FILE
BASE = Settings.load(_own if _own.exists() else ROOT / CONFIG_EXAMPLE)
WORK_DIR = (
    Path(BASE.work_dir) if Path(BASE.work_dir).is_absolute() else ROOT / BASE.work_dir
)
INPUT_DIR = Path(BASE.input_dir) if Path(BASE.input_dir).is_absolute() else ROOT / BASE.input_dir
OUTPUT_DIR = Path(BASE.output_dir) if Path(BASE.output_dir).is_absolute() else ROOT / BASE.output_dir

#: Что считаем записью совещания при выборе файла с диска.
MEDIA_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wav", ".m4a", ".mp3"}

st.set_page_config(page_title="minuteforge", page_icon="📝", layout="wide")


def full_width() -> dict:
    """Как попросить кнопку растянуться на всю ширину.

    В streamlit до версии 1.49 это ``use_container_width=True``, начиная с
    неё — ``width="stretch"``. Смотрим на саму сигнатуру, а не на номер
    версии: инструмент ставят в уже существующие окружения, где streamlit
    какой есть, и падать из-за оформления кнопки он не должен.
    """
    parameters = inspect.signature(st.button).parameters
    if "width" in parameters:
        return {"width": "stretch"}
    if "use_container_width" in parameters:
        return {"use_container_width": True}
    return {}


def pick_file_dialog() -> str | None:
    """Открывает обычный системный диалог выбора файла.

    Браузер путь к файлу не отдаёт и отдать не может — это его устройство,
    страница не должна знать, что лежит на диске. Поэтому загрузка через
    браузер всегда означает копирование гигабайтов, и обойти это в нём
    нельзя.

    Но приложение работает на той же машине, где лежит запись, и диалог
    можно открыть напрямую — так же, как это делало прежнее приложение.
    Возвращается путь, копировать нечего.

    Диалог запускается отдельным процессом: tkinter не любит, когда его
    зовут из потока streamlit, и на macOS от этого просто виснет.
    """
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "print(filedialog.askopenfilename(\n"
        "    title='Выберите запись совещания',\n"
        "    filetypes=[('Видео и аудио', '*.mp4 *.avi *.mov *.mkv *.webm *.wav *.m4a *.mp3'),\n"
        "               ('Все файлы', '*.*')]))\n"
    )
    try:
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
    except Exception as exc:
        st.error(f"Не удалось открыть диалог выбора файла: {exc}")
        return None
    if done.returncode != 0:
        st.error(
            "Не удалось открыть диалог выбора файла — в этой сборке Python нет "
            "tkinter. Впишите путь вручную в блоке ниже."
        )
        return None
    return done.stdout.strip() or None


class Live:
    """Показ хода работы: что идёт сейчас и что уже сделано.

    Полоска отмечает границы шагов, а внутри шага двигать её нечем: ни
    Whisper, ни модель не сообщают, сколько они прошли. Поэтому рядом с
    полоской — крутящийся значок и время: они видно живут, пока шаг длится,
    и человек не гадает, работает оно или повисло.
    """

    def __init__(self, title: str):
        self.started = time.monotonic()
        self.done: list[str] = []
        self.bar = st.progress(0.0, text=title)
        # st.status появился в streamlit 1.28; на более старых остаётся
        # полоска, и это не повод отказываться от неё вовсе.
        self.status = st.status(title, expanded=True) if hasattr(st, "status") else None

    def __call__(self, step) -> None:
        minutes, seconds = divmod(int(time.monotonic() - self.started), 60)
        clock = f"{minutes}:{seconds:02d}"
        if step.done:
            mark = (
                "не считался: взят из сохранённого расчёта этой же записи"
                if step.reused
                else f"{step.elapsed_s} с"
            )
            self.done.append(f"{step.title} — {mark}")
            if self.status is not None:
                self.status.write(f"{step.title} — {mark}")
            self.bar.progress(step.share, text=f"Прошло {clock}")
        else:
            label = f"{step.title}… (идёт {clock})"
            if self.status is not None:
                self.status.update(label=label)
            self.bar.progress(step.share, text=label)

    def finish(self, label: str) -> None:
        self.bar.progress(1.0, text=label)
        if self.status is not None:
            self.status.update(label=label, state="complete", expanded=False)


def transcript_from_state() -> Transcript | None:
    segments = st.session_state.get("segments")
    return Transcript(consolidate(blocks_from_segments(segments))) if segments else None


# ---------------------------------------------------------------- сайдбар
with st.sidebar:
    st.header("Запись")

    # Выбор файла системным диалогом: запись остаётся на месте, расчёт
    # читает её оттуда. Загрузка через браузер оставлена запасным путём —
    # она копирует гигабайты и нужна, только если запись на другой машине.
    if st.button("Выбрать файл…", **full_width()):
        chosen = pick_file_dialog()
        if chosen:
            st.session_state["source_path"] = chosen

    found = sorted(
        path for path in INPUT_DIR.glob("*")
        if path.suffix.lower() in MEDIA_SUFFIXES
    ) if INPUT_DIR.exists() else []
    if found:
        picked = st.selectbox(
            f"…либо из {INPUT_DIR.name}",
            ["", *[path.name for path in found]],
            format_func=lambda name: name or "— не выбрано —",
        )
        if picked:
            st.session_state["source_path"] = str(INPUT_DIR / picked)

    source_path: Path | None = None
    stored = st.session_state.get("source_path")
    if stored:
        candidate = Path(stored)
        if candidate.exists():
            source_path = candidate
        else:
            st.error(f"Файл не найден: {candidate}")
            st.session_state.pop("source_path", None)

    uploaded = None
    with st.expander("Запись на другой машине"):
        st.caption(
            "Загрузка через браузер копирует файл целиком: на часовом видео "
            "это минуты ожидания и вторая копия на диске. Нужна, только если "
            "записи нет на этой машине."
        )
        uploaded = st.file_uploader(
            "Загрузить", type=["mp4", "avi", "mov", "mkv", "wav", "m4a", "mp3"]
        )

    ready_segments = st.file_uploader("Готовая стенограмма (json)", type=["json"])

    # Путь показывается полным и заранее. Относительный «data/output» ничего
    # не говорит человеку, который запустил приложение ярлыком: искать файлы
    # он будет наугад.
    out_dir = Path(st.text_input(
        "Куда сохранять результаты",
        str(OUTPUT_DIR),
        help="Стенограмма и протокол ложатся сюда сразу, без кнопок «скачать». "
        "Можно указать сетевую папку, из которой их заберёт другое "
        "подразделение.",
    ).strip() or OUTPUT_DIR).expanduser().resolve()
    st.caption(f"Файлы появятся здесь: `{out_dir}`")
    # Проверяется сразу, а не при сохранении: папка часто сетевая, а узнавать
    # о недоступном диске после сорока минут распознавания — терять сорок
    # минут.
    trouble = check_writable(out_dir)
    if trouble:
        st.error(trouble)

    # Промежуточное копится незаметно: звук остаётся после каждой записи, а
    # нужен он только чтобы не считать заново. Пока размер не видно, его
    # никто и не чистит.
    busy = cache_size(WORK_DIR)
    if busy:
        # Показывается при любом размере: спрашивать «а можно ли почистить
        # сейчас» человеку приходится ровно потому, что кнопки не видно.
        size = f"{busy / 1024 ** 3:.1f} ГБ" if busy > 1024 ** 3 else f"{busy / 1024 ** 2:.0f} МБ"
        st.caption(f"Промежуточные файлы занимают {size}")
        if st.button("Очистить промежуточное", **full_width()):
            freed = clear_cache(WORK_DIR)
            gone = (
                f"{freed / 1024 ** 3:.1f} ГБ" if freed > 1024 ** 3
                else f"{freed / 1024 ** 2:.0f} МБ"
            )
            st.success(f"Освобождено {gone}. Готовые файлы на месте — они в другой папке.")

    st.header("Документ")
    roster_file = st.file_uploader(
        "Список участников (csv или txt)",
        type=["csv", "txt"],
        help="ФИО и должности. Не заменяет число говорящих: в списке все "
        "приглашённые, а голосов в записи обычно втрое меньше.",
    )
    template_file = st.file_uploader(
        "Форма протокола (md или txt)",
        type=["md", "txt"],
        help="Ваша форма документа с местами вида {{date}}, {{tasks}}. "
        "Без неё протокол собирается по встроенной разметке. "
        "Образец — data/sample/protocol_template.md.",
    )

    if st.button("Проверить доступ к моделям", **full_width()):
        # Секунда проверки против сорока минут распознавания, которое
        # упрётся в отказ на последнем шаге.
        access = check_model_access(BASE.hf_token)
        # Обычным if, а не тернарным выражением: streamlit печатает любое
        # «висящее» выражение, и на экран вываливается объект вместе со
        # своей документацией вместо короткого статуса.
        for model, why in access.items():
            name = model.split("/")[-1]
            if why:
                st.error(f"{name}: {why}")
            else:
                st.success(f"{name}: доступ есть")
        if any(access.values()):
            st.caption(
                "Условия принимаются на странице модели, на вкладке Model card: "
                "прокрутите вниз до «Agree and access repository». Кнопка "
                "«Use this model» — это подсказка с кодом, согласия она не даёт."
            )

    st.header("Распознавание")
    asr_model = st.selectbox(
        "Модель Whisper",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=3,
        help="Крупнее — точнее и заметно дольше. На записи с плохим микрофоном "
        "разница между small и medium решающая.",
    )
    language = st.text_input("Язык записи", BASE.language)
    speakers = st.number_input(
        "Сколько человек выступает, если известно", 0, 60, 0,
        help="Ноль — определить самой, и на большом совещании так и надо: "
        "когда из восьмидесяти подключений слово возьмут неизвестно сколько, "
        "любое число будет враньём. Указывайте, только когда знаете точно, — "
        "тогда это заметно помогает: иначе диаризация дробит одного человека "
        "на нескольких.",
    )

    st.header("Модель для поручений")

    # Адрес спрятан намеренно: когда сервер отвечает, список моделей внизу и
    # есть доказательство связи, а поле с localhost только занимает место.
    # Значение читается из состояния до отрисовки поля — иначе спросить
    # сервер по новому адресу вышло бы только со второго раза.
    address = st.session_state.get("llm_url", BASE.llm_base_url)
    installed = LLMClient(Settings(llm_base_url=address)).available_models()

    if installed:
        # По имени без тега: в настройках пишут «mistral», а Ollama
        # показывает «mistral:latest». Без этого настройка не находилась, и
        # бралась первая по алфавиту — обычно модель эмбеддингов.
        default = next(
            (i for i, name in enumerate(installed) if same_model(name, BASE.llm_model)),
            0,
        )
        llm_model = st.selectbox("Модель", installed, index=default)
        if is_embedder(llm_model):
            st.error(
                f"«{llm_model}» — модель эмбеддингов, разговаривать она не "
                "умеет и поручений не найдёт. Выберите разговорную: mistral, "
                "qwen2.5, llama3.1."
            )
        st.caption(
            "Для извлечения поручений семимиллиардные модели слабоваты — "
            "недобирают половину. Заметно лучше qwen2.5:14b, если хватает "
            "видеопамяти (около 10 ГБ), либо llama3.1:8b."
        )
    else:
        # Причин ровно три, и они сильно разной вероятности. Поле адреса —
        # самая редкая из них, поэтому сперва идёт то, что помогает почти
        # всегда: поднять сервер.
        st.error(f"Сервер модели не отвечает по адресу {address}")
        st.markdown(
            """
**Что делать, по порядку:**

1. **Запустите Ollama.** В Windows это значок в трее; в терминале —
   `ollama serve`. Страница обновится сама, как только он ответит.
2. **Проверьте, что модель скачана:** `ollama list`. Пусто — скачайте:
   `ollama pull mistral`.
3. **Если модель поднята в LM Studio** — у него другой адрес,
   `http://localhost:1234/v1`. Впишите его ниже.
            """
        )
        llm_model = st.text_input("Модель", BASE.llm_model)

    # Предел сервера важнее окна модели: Ollama по умолчанию держит num_ctx
    # небольшим и молча отрезает всё сверх него — модель не видит конца
    # фрагмента и половины поручений в нём.
    window, limit = LLMClient(Settings(llm_base_url=address)).context_window()
    # Внешняя модель отменяет главное свойство инструмента, и человек должен
    # это увидеть, а не вычитать в журнале. Запрета нет: у кого-то внешняя
    # модель разрешена, и тогда качество разбора несравнимо выше.
    probe = LLMClient(Settings(llm_base_url=address))
    if not probe.is_local:
        st.warning(
            "Модель внешняя — стенограмма уйдёт на этот сервер. Записи с "
            "ограничением доступа так обрабатывать нельзя.\n\n"
            f"Ключ берётся из переменной окружения `{LLM_KEY_ENV}`"
            + (" — она задана." if Settings().llm_key else ", а она не задана.")
        )

    ROOMY = 8192

    if limit and limit >= ROOMY:
        st.caption(f"Сервер принимает не больше {limit} токенов за раз.")
    elif limit:
        # Предел объявлен, но тесный. Это не ошибка настройки, а умолчание
        # Ollama, и без подсказки человек оставит его как есть: фрагменты
        # режутся молча, а выглядит это как будто модель плохо ищет.
        st.warning(
            f"Сервер принимает не больше {limit} токенов за раз — этого мало. "
            "Фрагменты придётся дробить мелко, и на границах теряются "
            f"поручения. Поднимите предел до {ROOMY}:\n\n"
            "```\nFROM <ваша модель>\nPARAMETER num_ctx 8192\n```\n"
            "Сохраните это в файл `Modelfile` и выполните "
            "`ollama create моя-модель -f Modelfile`, а её имя впишите выше."
        )
    elif window:
        # Предел не объявлен — значит действует умолчание сервера, а оно у
        # Ollama невелико: две-четыре тысячи. Всё сверх него молча
        # отрезается, и заметить это со стороны нельзя.
        st.warning(
            f"У модели окно {window} токенов, но сервер не сообщает свой "
            "предел (num_ctx). По умолчанию Ollama берёт 2048–4096 и молча "
            "отрезает остальное — конец фрагмента модель не увидит.\n\n"
            "Задайте предел явно, и тогда он станет виден:\n"
            "```\nFROM <ваша модель>\nPARAMETER num_ctx 8192\n```\n"
            "`ollama create моя-модель -f Modelfile`"
        )

    # Когда сервер молчит о своём пределе, берётся самое тесное значение, а
    # не то, что в настройках. Послать меньше, чем сервер примет, — потерять
    # связность фрагмента. Послать больше — потерять его конец молча, и это
    # хуже: со стороны выглядит как «модель ничего не находит».
    safe_default = limit if limit in (2048, 4096, 8192, 16384, 32768) else 2048
    context = st.select_slider(
        "Окно модели, токенов",
        options=[2048, 4096, 8192, 16384, 32768, 131072, 1_000_000],
        value=safe_default,
        help="Отсюда считается размер фрагмента. Нарезка нужна ровно потому, "
        "что стенограмма не влезает в окно целиком: при большом окне фрагмент "
        "будет один, и повторов на границах не возникнет вовсе.",
    )
    with st.expander("Свои указания модели"):
        st.caption(
            "Дописывается в конец указаний. Правила формата ответа этим не "
            "трогаются: их изменение сломает разбор. Сюда — предметное: что "
            "считать поручением у вас и как это называть."
        )
        prompt_extra = st.text_area(
            "Дополнительно",
            BASE.prompt_extra,
            placeholder="Поручения у нас называются заданиями.\n"
            "Не считай поручением просьбу подключиться или выступить.",
            label_visibility="collapsed",
        )

    if limit and int(context) > limit:
        st.warning(
            f"Окно больше того, что примет сервер ({limit}). Конец каждого "
            "фрагмента он отрежет, и поручений в нём модель не увидит. "
            "Поставьте окно по пределу сервера или поднимите num_ctx у модели."
        )

    fine = st.checkbox(
        "Дробить мельче",
        value=False,
        help="Фрагменты по 1200 токенов вместо расчётных. Мелкая модель на "
        "длинном куске теряет поручения из середины: коротких фрагментов "
        "больше, считать дольше, но находит она заметно больше.",
    )

    # Размер фрагмента — не настройка, а следствие двух других, и считать
    # его в уме человеку незачем. Заодно видно, когда «дробить мельче»
    # ничего не меняет: на тесном окне расчётный фрагмент и так мелкий.
    FINE_TOKENS = 1200
    planned = FINE_TOKENS if fine else max(500, int(int(context) * BASE.chunk_context_share))
    st.caption(f"Фрагмент выйдет ~{planned} токенов — примерно {planned // 150} мин разговора.")
    if fine and planned >= max(500, int(int(context) * BASE.chunk_context_share)):
        st.caption("На таком окне дробление мельче почти ничего не меняет.")

    # Раскрыт, когда связи нет: третий пункт из списка выше делается тут.
    with st.expander("Адрес сервера", expanded=not installed):
        st.text_input(
            "Адрес", address, key="llm_url",
            help="Ollama — 11434, LM Studio — 1234. Это корень API, а не "
            "страница: в браузере по нему будет 404, и это нормально. Живость "
            "сервера проверяется по http://localhost:11434 — ответит "
            "«Ollama is running».",
        )
        if st.button("Проверить связь", **full_width()):
            probe = LLMClient(
                Settings(llm_base_url=address, llm_model=llm_model, llm_retries=0)
            )
            trouble = probe.diagnose()
            if not trouble:
                st.success(f"Модель «{llm_model}» отвечает.")
            else:
                st.error(trouble)

    llm_url = st.session_state.get("llm_url", address)

settings = Settings(
    asr_model=asr_model,
    language=language,
    speakers=int(speakers) or None,
    llm_base_url=llm_url,
    llm_model=llm_model,
    llm_context_tokens=int(context),
    chunk_max_tokens=1200 if fine else BASE.chunk_max_tokens,
    prompt_extra=prompt_extra,
    prompt_file=BASE.prompt_file,
    chunk_overlap_blocks=BASE.chunk_overlap_blocks,
    offline_models=BASE.offline_models,
)

# ---------------------------------------------------------------- шапка
st.title("minuteforge")
st.caption("Протокол видеосовещания с поручениями. Ничего не уходит с этой машины.")


def running_version() -> str:
    """Какая версия сейчас в памяти.

    Streamlit перезапускает сценарий, но уже загруженные модули не
    перечитывает: после обновления приложение может работать старым кодом, и
    со стороны это выглядит как «изменения не появились». Строка внизу
    отвечает на этот вопрос сразу.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h %s"],
            capture_output=True, timeout=3,
        )
        # Байты, а не text=True: git отдаёт UTF-8, а Python на Windows
        # декодирует по локали — русский заголовок коммита превращается в
        # кракозябры.
        return out.stdout.decode("utf-8", "replace").strip() or "версия неизвестна"
    except Exception:
        return "версия неизвестна"


st.caption(f"Версия в памяти: {running_version()}")

with st.expander("Как это работает"):
    st.markdown(
        """
Три шага, и средний из них — ваш.

1. **Распознавание.** Из записи извлекается звук, речь превращается в текст,
   а диаризация делит его по говорящим: `SPEAKER_00`, `SPEAKER_01`. Самый
   долгий шаг: час записи на видеокарте — минуты, на процессоре — часы.
   Промежуточные результаты сохраняются, повторный запуск их не пересчитывает.
2. **Кто есть кто.** Модель слышит, что говорят разные люди, но не знает их
   имён. Сопоставить метки с участниками может только человек — по голосу,
   по содержанию, по порядку выступлений.
3. **Поручения и протокол.** Стенограмма уходит в локальную модель по частям,
   из каждой извлекаются поручения, они сводятся в один документ.

Поручения без названного исполнителя не выбрасываются и не получают
выдуманного: они попадают в раздел «Требуют уточнения».
        """
    )

# ---------------------------------------------------------------- шаг 1
st.subheader("Шаг 1. Распознавание")

if ready_segments is not None:
    st.session_state["segments"] = json.load(ready_segments)
    st.success(f"Загружена готовая стенограмма: {len(st.session_state['segments'])} сегментов.")

if source_path is not None or uploaded is not None:
    if source_path is not None:
        size = source_path.stat().st_size / 1024 / 1024
        st.write(f"Файл: `{source_path}` — {size:.0f} МБ, читается с диска.")
    else:
        st.write(f"Загружено: `{uploaded.name}`")

    # Плеер по требованию, а не всегда: работать он не помогает, а занимает
    # пол-экрана и тянет файл в браузер. Ставится галочкой, чтобы код с
    # видео вообще не выполнялся, пока его не попросят, — в раскрывающемся
    # блоке содержимое строится всё равно.
    if st.checkbox("Посмотреть запись", key="preview"):
        if source_path is not None and size > 300:
            st.caption(
                f"Файл на {size:.0f} МБ — проигрывать его в браузере незачем: "
                "откройте обычным плеером, если нужно свериться."
            )
        elif source_path is not None:
            st.video(str(source_path))
        elif uploaded.type.startswith("audio"):
            st.audio(uploaded)
        else:
            st.video(uploaded)

    span = st.columns(2)
    from_time = span[0].text_input(
        "Распознать с", placeholder="1:00:00",
        help="Пусто — с начала. На совещании с десятками подключений первый "
        "час уходит на перекличку: распознавать его — час работы видеокарты "
        "впустую. Время в стенограмме останется от начала записи.",
    )
    to_time = span[1].text_input("по", placeholder="2:17:00")

    fresh = st.checkbox(
        "Считать заново",
        help="Обычно готовые шаги берутся из сохранённого — так сбой на "
        "последнем шаге не стоит сорока минут работы. Отметьте, если хотите "
        "пересчитать эту запись с нуля.",
    )

    if st.button("Распознать", type="primary"):
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        if source_path is not None:
            # Ничего не копируем: расчёт читает файл там, где он лежит.
            source = source_path
        else:
            source = WORK_DIR / uploaded.name
            with st.spinner("Сохраняю загруженный файл…"):
                source.write_bytes(uploaded.getbuffer())
            # Копия видео — самое тяжёлое в рабочей папке. Помним, что она
            # наша, чтобы убрать вместе с остальным промежуточным.
            st.session_state["uploaded_copy"] = source
        st.session_state["cache_dir"] = cache_dir_for(
            source, WORK_DIR, start=from_time or None, end=to_time or None
        )
        live = Live("Готовлюсь…")
        try:
            transcript = transcribe_meeting(
                source, settings, work_dir=WORK_DIR, progress=live, fresh=fresh,
                start=from_time or None, end=to_time or None,
            )
            live.finish("Распознавание закончено")
            st.session_state["segments"] = [
                {"speaker": b.speaker, "text": b.text, "start": b.start, "end": b.end}
                for b in transcript.blocks
            ]
            # Кладём сразу в папку: «Загрузки» на этой машине — не то место,
            # откуда стенограмму заберёт другое подразделение.
            st.session_state["stem"] = source.stem
            # Своя папка на каждый разбор: иначе второй прогон той же записи
            # молча затирает первый, и сравнивать настройки не с чем.
            folder = run_dir(out_dir, source.stem)
            st.session_state["run_dir"] = folder
            saved = save_transcript(transcript, folder, stem="стенограмма")
            st.session_state["saved_transcript"] = saved
        except MissingToken as exc:
            st.error(str(exc))
            st.caption(
                "Токен нужен свой: он привязан к вашему аккаунту HuggingFace. "
                f"Кладётся в переменную окружения `{HF_TOKEN_ENV}` — "
                "подробнее в README, раздел «Токен HuggingFace»."
            )
            st.stop()
        except (RecognitionError, AudioError) as exc:
            st.error(str(exc))
            st.stop()

transcript = transcript_from_state()
if transcript is None:
    st.info("Загрузите запись слева или готовую стенограмму в json.")
    st.stop()

st.write(
    f"Реплик: **{len(transcript.blocks)}**, говорящих: **{len(transcript.speakers)}**, "
    f"длительность: **{transcript.duration_min} мин**"
)
for note in getattr(transcript, "notes", []):
    st.warning(f"Не хватило видеопамяти: {note}. Качество расшифровки будет ниже.")
with st.expander("Стенограмма"):
    st.text(transcript.as_text(with_time=True))
saved = st.session_state.get("saved_transcript")
if saved:
    st.success(f"Стенограмма сохранена в `{Path(saved['text']).parent}`")
    st.markdown(
        f"- `{Path(saved['text']).name}` — текстом, для передачи\n"
        f"- `{Path(saved['json']).name}` — для повторной сборки протокола"
    )

# Файлы уже сохранены в папку, и кнопка «скачать» — не главный путь, а
# запасной: она нужна тому, кто открыл приложение по сети со своей машины,
# потому что сохраняется всё на той, где приложение запущено.
with st.expander("Скачать себе"):
    st.caption(
        "Нужно, если вы открыли приложение по сети: файлы сохраняются на той "
        "машине, где оно запущено, а не у вас."
    )
    st.download_button(
        "Стенограмма json",
        json.dumps(st.session_state["segments"], ensure_ascii=False, indent=2).encode("utf-8"),
        "стенограмма.json",
        "application/json",
    )

# ---------------------------------------------------------------- шаг 2
st.subheader("Шаг 2. Кто есть кто")
st.caption(
    "Модель различает голоса, но не знает имён. Метку без имени оставьте "
    "пустой — она попадёт в протокол как есть, и это лучше, чем чужая "
    "фамилия под чужими словами."
)

roster = read_people(roster_file) if roster_file is not None else []
heard = mentioned_people(transcript.as_text())
people = merge_suggestions(roster, heard)
guesses = suggest_speakers(transcript.blocks)

if people:
    source = []
    if roster:
        source.append(f"из списка — {len(roster)}")
    if len(people) > len(roster):
        source.append(f"названы в записи — {len(people) - len(roster)}")
    st.caption("Кого удалось собрать: " + ", ".join(source) + ".")

OTHER = "другое…"
options = ["", *[person.full for person in people], OTHER]

# По убыванию времени речи: на двухчасовой записи диаризация нарезает
# десятки меток, но по существу говорят пятеро. Опознавать нужно сперва их,
# а не того, чей микрофон поймал два слова.
ordered = [s for s in transcript.speakers_by_time() if s != UNKNOWN]
spoken = transcript.speaking_time()
total_spoken = sum(spoken.values()) or 1.0

MAIN = 8
main_labels, rest_labels = ordered[:MAIN], ordered[MAIN:]

if rest_labels:
    st.caption(
        f"Голосов найдено {len(ordered)}. Первые {len(main_labels)} говорили "
        "дольше всех — обычно это и есть участники. Остальные чаще всего "
        "обрывки: эхо, кашель, реплика с чужого микрофона."
    )
    if len(ordered) > 12 and settings.speakers is None:
        st.warning(
            "Столько голосов на одном совещании — обычно признак того, что "
            "диаризация раздробила участников. Укажите слева, сколько человек "
            "говорит, и распознайте заново: точное число заметно помогает."
        )

names: dict[str, str] = {}


def name_field(label: str) -> None:
    """Одно поле сопоставления: метка, её доля речи и первая реплика."""
    seconds = spoken.get(label, 0.0)
    sample = next((b.text for b in transcript.blocks if b.speaker == label), "")
    # Догадка ставится по умолчанию, но остаётся догадкой: человек видит её
    # рядом с первой репликой и либо соглашается, либо правит.
    guess = guesses.get(label)
    default = options.index(guess.full) if guess and guess.full in options else 0
    title = f"{label} — {seconds / 60:.0f} мин ({seconds / total_spoken:.0%})"
    choice = st.selectbox(title, options, index=default, key=f"pick_{label}")
    if choice == OTHER:
        choice = st.text_input(
            "Кто это", key=f"name_{label}", placeholder="Фамилия И.О.",
            label_visibility="collapsed",
        )
    if guess and default:
        st.caption("Подсказка: назван перед этой репликой")
    st.caption(f"«{sample[:70]}…»" if len(sample) > 70 else f"«{sample}»")
    if choice and choice.strip():
        # В протоколе под репликами нужна фамилия, а должность — в шапке.
        names[label] = choice.split(",")[0].strip()


columns = st.columns(min(3, max(1, len(main_labels))))
for index, label in enumerate(main_labels):
    with columns[index % len(columns)]:
        name_field(label)

if rest_labels:
    with st.expander(f"Остальные голоса ({len(rest_labels)})"):
        tail = st.columns(3)
        for index, label in enumerate(rest_labels):
            with tail[index % 3]:
                name_field(label)

# ---------------------------------------------------------------- шаг 3
st.subheader("Шаг 3. Протокол")

left, right = st.columns(2)
with left:
    title = st.text_input("Заголовок", "Протокол совещания")
    date = st.text_input("Дата и время", placeholder="05.06.2025, 11:00")
    number = st.text_input("Номер протокола", placeholder="17")
with right:
    place = st.text_input("Место", placeholder="Переговорная 3")
    chair = st.text_input("Председатель", placeholder="Фамилия И.О.")
    secretary = st.text_input("Секретарь", placeholder="Фамилия И.О.")

if st.button("Собрать протокол", type="primary"):
    meeting = Meeting(
        title=title, date=date, place=place, chair=chair,
        secretary=secretary, number=number,
        names=names or None, people=people or None,
    )
    client = LLMClient(settings)

    # Здесь так же: модель отвечает по десятку секунд на фрагмент, и на
    # длинном совещании это минуты тишины.
    live = Live("Читаю стенограмму…")
    protocol = protocol_from_transcript(
        transcript, settings, meeting=meeting, client=client, progress=live
    )
    live.finish("Протокол собран")

    template_text = (
        template_file.getvalue().decode("utf-8-sig") if template_file is not None else None
    )
    # Всё по одному разбору лежит в своей папке. Стенограмма уже сохранена
    # на шаге распознавания — второй раз не пишем.
    stem = "протокол"
    folder = st.session_state.get("run_dir") or run_dir(
        out_dir, st.session_state.get("stem", "совещание")
    )
    st.session_state["run_dir"] = folder
    if template_text is None:
        written = save(protocol, folder, stem=stem, with_transcript=False)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        document = _free_name(folder / f"{stem}.md")
        document.write_text(protocol.render(template_text), encoding="utf-8")
        tasks = _free_name(folder / f"{stem}_поручения.csv")
        tasks.write_text(protocol.tasks_csv(), encoding="utf-8-sig")
        written = {"protocol": document, "tasks": tasks}
    st.session_state["saved_protocol"] = written
    st.session_state["protocol"] = protocol

    # Промежуточное убирается здесь, а не сразу после распознавания: до
    # готового протокола кеш ещё может пригодиться, если голоса захотят
    # переразметить. Чистится только эта запись — чужие разборы ни при чём.
    if not BASE.keep_work_files:
        freed = 0
        cache = st.session_state.get("cache_dir")
        if cache:
            freed += clear_cache(cache)
        copy = st.session_state.get("uploaded_copy")
        if copy and Path(copy).exists():
            freed += Path(copy).stat().st_size
            Path(copy).unlink(missing_ok=True)
        if freed:
            st.session_state["freed"] = freed
    st.session_state["template"] = (
        template_file.getvalue().decode("utf-8-sig") if template_file is not None else None
    )

protocol = st.session_state.get("protocol")
if protocol is not None:
    found, unclear = len(protocol.actionable), len(protocol.needs_clarification)
    metrics = st.columns(3)
    metrics[0].metric("Поручений", found + unclear)
    metrics[1].metric("С исполнителем", found)
    metrics[2].metric("Требуют уточнения", unclear)
    if unclear:
        st.warning(
            f"{unclear} поручений прозвучали без исполнителя. Они не выброшены "
            "и не получили выдуманного адресата — смотрите отдельный раздел."
        )

    if not protocol.tasks and protocol.answers:
        # Пустой результат выглядит одинаково при трёх разных бедах: модель
        # ничего не нашла, ответила не по формату или заговорила о своём.
        # Различить их можно только по самому ответу.
        st.error(
            "Модель не вернула ни одного поручения. Посмотрите, что она "
            "ответила: если там проза или пересказ задания, дело в модели — "
            "мелкие держат формат хуже. Помогает окно поменьше (фрагменты "
            "станут короче) или другая модель."
        )
        with st.expander("Что ответила модель"):
            for number, answer in enumerate(protocol.answers, 1):
                st.markdown(f"**Фрагмент {number}**")
                st.text(answer or "(пустой ответ)")

    written = st.session_state.get("saved_protocol")
    if written:
        folder = Path(next(iter(written.values()))).parent
        st.success(f"Протокол сохранён в `{folder}`")
        st.markdown("\n".join(f"- `{Path(path).name}`" for path in written.values()))
        freed = st.session_state.get("freed")
        if freed:
            st.caption(
                f"Промежуточное убрано, освободилось {freed / 1024 ** 2:.0f} МБ. "
                "Чтобы оставлять — keep_work_files: true в config.yaml."
            )

    template = st.session_state.get("template")
    document = protocol.render(template) if template else protocol.as_markdown()
    if template:
        st.caption("Документ собран по вашей форме.")
    st.markdown(document if not template else f"```\n{document}\n```")

    # Цитаты — то, ради чего таблица и открывается: по ним видно сразу,
    # поручение это или изложение доклада, и обращаться к записи приходится
    # только в спорных случаях.
    with st.expander(f"Проверить по стенограмме ({len(protocol.tasks)})"):
        st.caption(
            "Рядом с каждым поручением — время и реплика, из которой оно "
            "выписано. Если реплика на поручение не похожа, пункт лишний: "
            "вычеркните его при вычитке."
        )
        for number, task in enumerate(protocol.tasks, 1):
            stamp = task.at
            clock = (
                f"{int(stamp)//3600:02d}:{int(stamp)%3600//60:02d}:{int(stamp)%60:02d}"
                if stamp is not None else "—"
            )
            st.markdown(f"**{number}. {task.what}**  \n`{clock}` {task.said_by}: {task.quote}")

    with st.expander("Скачать себе"):
        st.caption(
            "Файлы уже сохранены в папку выше. Эти кнопки нужны, если вы "
            "открыли приложение по сети: сохраняется всё на той машине, где "
            "оно запущено, а не у вас."
        )
        files = st.columns(3)
        files[0].download_button(
            "Протокол", document.encode("utf-8"), "протокол.md", "text/markdown",
            **full_width(),
        )
        files[1].download_button(
            "Стенограмма", transcript.as_text(with_time=True).encode("utf-8"),
            "стенограмма.txt", "text/plain", **full_width(),
        )
        files[2].download_button(
            "Поручения csv", protocol.tasks_csv().encode("utf-8-sig"),
            "поручения.csv", "text/csv", **full_width(),
        )
