"""Интерфейс protocall.

    streamlit run app/streamlit_app.py

Приложение намеренно разделено на три шага, а не на одну кнопку: между
распознаванием и протоколом человек обязан вмешаться и сказать, кто такой
SPEAKER_00. Автоматически это не определить, а протокол с безымянными
участниками никому не нужен.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from protocall.blocks import (  # noqa: E402
    UNKNOWN,
    Transcript,
    blocks_from_segments,
    consolidate,
)
from protocall.config import HF_TOKEN_ENV, Settings  # noqa: E402
from protocall.llm import LLMClient  # noqa: E402
from protocall.pipeline import Meeting, protocol_from_transcript, transcribe_meeting  # noqa: E402
from protocall.transcribe import (  # noqa: E402
    MissingToken,
    RecognitionError,
    check_model_access,
)

BASE = Settings.load(ROOT / "config.yaml")
WORK_DIR = ROOT / "data" / "work"

st.set_page_config(page_title="protocall", page_icon="📝", layout="wide")


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


def transcript_from_state() -> Transcript | None:
    segments = st.session_state.get("segments")
    return Transcript(consolidate(blocks_from_segments(segments))) if segments else None


# ---------------------------------------------------------------- сайдбар
with st.sidebar:
    st.header("Запись")
    uploaded = st.file_uploader(
        "Видео или аудио", type=["mp4", "avi", "mov", "mkv", "wav", "m4a", "mp3"]
    )
    ready_segments = st.file_uploader("…либо готовая стенограмма (json)", type=["json"])

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
        "Сколько человек говорит", 0, 30, 0,
        help="Ноль — определить самой. Точное число заметно помогает: иначе "
        "диаризация склонна дробить одного человека на нескольких.",
    )

    st.header("Модель для поручений")
    llm_url = st.text_input("Адрес", BASE.llm_base_url)
    llm_model = st.text_input("Модель", BASE.llm_model)
    context = st.select_slider(
        "Окно модели, токенов",
        options=[4096, 8192, 16384, 32768, 131072, 1_000_000],
        value=BASE.llm_context_tokens,
        help="Отсюда считается размер фрагмента. Нарезка нужна ровно потому, "
        "что стенограмма не влезает в окно целиком: при большом окне фрагмент "
        "будет один, и повторов на границах не возникнет вовсе.",
    )
    if st.button("Проверить связь", **full_width()):
        probe = LLMClient(Settings(llm_base_url=llm_url, llm_model=llm_model, llm_retries=0))
        trouble = probe.diagnose()
        if not trouble:
            st.success(f"Модель «{llm_model}» отвечает.")
        else:
            st.error(trouble)

settings = Settings(
    asr_model=asr_model,
    language=language,
    speakers=int(speakers) or None,
    llm_base_url=llm_url,
    llm_model=llm_model,
    llm_context_tokens=int(context),
    chunk_max_tokens=BASE.chunk_max_tokens,
    chunk_overlap_blocks=BASE.chunk_overlap_blocks,
    offline_models=BASE.offline_models,
)

# ---------------------------------------------------------------- шапка
st.title("protocall")
st.caption("Протокол видеосовещания с поручениями. Ничего не уходит с этой машины.")

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

if uploaded is not None:
    if uploaded.type.startswith("audio"):
        st.audio(uploaded)
    else:
        st.video(uploaded)
    if st.button("Распознать", type="primary"):
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        source = WORK_DIR / uploaded.name
        source.write_bytes(uploaded.getbuffer())
        # Полоска по шагам, а не крутилка: на часовой записи человек иначе
        # сидит и гадает, идёт работа или всё повисло. Точной шкалы тут быть
        # не может — WhisperX не сообщает, сколько минут записи он прошёл, —
        # но видеть, какой шаг идёт и сколько занял предыдущий, уже достаточно.
        bar = st.progress(0.0, text="Готовлюсь…")
        done_steps: list[str] = []

        def show(step) -> None:
            if step.done:
                mark = "взят готовым" if step.reused else f"{step.elapsed_s} с"
                done_steps.append(f"{step.title} — {mark}")
            text = " · ".join([*done_steps, f"**{step.title}…**"]) if not step.done else " · ".join(done_steps)
            bar.progress(step.share, text=text or step.title)

        try:
            transcript = transcribe_meeting(
                source, settings, work_dir=WORK_DIR, progress=show
            )
            bar.progress(1.0, text=" · ".join(done_steps) or "Готово")
            st.session_state["segments"] = [
                {"speaker": b.speaker, "text": b.text, "start": b.start, "end": b.end}
                for b in transcript.blocks
            ]
        except MissingToken as exc:
            st.error(str(exc))
            st.caption(
                "Токен нужен свой: он привязан к вашему аккаунту HuggingFace. "
                f"Кладётся в переменную окружения `{HF_TOKEN_ENV}` — "
                "подробнее в README, раздел «Токен HuggingFace»."
            )
            st.stop()
        except RecognitionError as exc:
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
with st.expander("Стенограмма"):
    st.text(transcript.as_text(with_time=True))
st.download_button(
    "Скачать стенограмму json",
    json.dumps(st.session_state["segments"], ensure_ascii=False, indent=2).encode("utf-8"),
    "segments.json",
    "application/json",
    help="По этому файлу протокол пересобирается без видеокарты, на любой машине.",
)

# ---------------------------------------------------------------- шаг 2
st.subheader("Шаг 2. Кто есть кто")
st.caption(
    "Модель различает голоса, но не знает имён. Метку без имени оставьте "
    "пустой — она попадёт в протокол как есть, и это лучше, чем чужая "
    "фамилия под чужими словами."
)

labels = [s for s in transcript.speakers if s != UNKNOWN]
names: dict[str, str] = {}
columns = st.columns(min(3, max(1, len(labels))))
for index, label in enumerate(labels):
    with columns[index % len(columns)]:
        sample = next((b.text for b in transcript.blocks if b.speaker == label), "")
        value = st.text_input(label, key=f"name_{label}", placeholder="Фамилия И.О.")
        st.caption(f"«{sample[:70]}…»" if len(sample) > 70 else f"«{sample}»")
        if value.strip():
            names[label] = value.strip()

# ---------------------------------------------------------------- шаг 3
st.subheader("Шаг 3. Протокол")

left, right = st.columns(2)
with left:
    title = st.text_input("Заголовок", "Протокол совещания")
    date = st.text_input("Дата и время", placeholder="05.06.2025, 11:00")
with right:
    place = st.text_input("Место", placeholder="Переговорная 3")
    chair = st.text_input("Председатель", placeholder="Фамилия И.О.")

if st.button("Собрать протокол", type="primary"):
    meeting = Meeting(title=title, date=date, place=place, chair=chair, names=names or None)
    client = LLMClient(settings)
    with st.spinner("Ищу поручения. Местная модель отвечает небыстро…"):
        protocol = protocol_from_transcript(transcript, settings, meeting=meeting, client=client)
    st.session_state["protocol"] = protocol

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

    st.markdown(protocol.as_markdown())

    files = st.columns(2)
    files[0].download_button(
        "Скачать протокол md",
        protocol.as_markdown(with_transcript=True).encode("utf-8"),
        "protocol.md",
        "text/markdown",
        **full_width(),
    )
    files[1].download_button(
        "Скачать поручения csv",
        protocol.tasks_csv().encode("utf-8-sig"),
        "tasks.csv",
        "text/csv",
        **full_width(),
    )
