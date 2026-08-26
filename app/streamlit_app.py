"""Интерфейс protocall.

    streamlit run app/streamlit_app.py

Приложение намеренно разделено на три шага, а не на одну кнопку: между
распознаванием и протоколом человек обязан вмешаться и сказать, кто такой
SPEAKER_00. Автоматически это не определить, а протокол с безымянными
участниками никому не нужен.
"""

from __future__ import annotations

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
from protocall.transcribe import MissingToken, RecognitionError  # noqa: E402

BASE = Settings.load(ROOT / "config.yaml")
WORK_DIR = ROOT / "data" / "work"

st.set_page_config(page_title="protocall", page_icon="📝", layout="wide")


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
    if st.button("Проверить связь", width="stretch"):
        probe = LLMClient(Settings(llm_base_url=llm_url, llm_model=llm_model, llm_retries=0))
        st.success("Модель отвечает.") if probe.health() else st.error(
            "Модель не отвечает. Запущен ли Ollama или LM Studio?"
        )

settings = Settings(
    asr_model=asr_model,
    language=language,
    speakers=int(speakers) or None,
    llm_base_url=llm_url,
    llm_model=llm_model,
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
    st.audio(uploaded) if uploaded.type.startswith("audio") else st.video(uploaded)
    if st.button("Распознать", type="primary"):
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        source = WORK_DIR / uploaded.name
        source.write_bytes(uploaded.getbuffer())
        try:
            with st.spinner("Распознаю. На часовой записи это надолго…"):
                transcript = transcribe_meeting(source, settings, work_dir=WORK_DIR)
            st.session_state["segments"] = [
                {"speaker": b.speaker, "text": b.text, "start": b.start, "end": b.end}
                for b in transcript.blocks
            ]
        except MissingToken as exc:
            st.error(str(exc))
            st.caption(f"Токен кладётся в переменную окружения `{HF_TOKEN_ENV}`.")
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
        width="stretch",
    )
    files[1].download_button(
        "Скачать поручения csv",
        protocol.tasks_csv().encode("utf-8-sig"),
        "tasks.csv",
        "text/csv",
        width="stretch",
    )
