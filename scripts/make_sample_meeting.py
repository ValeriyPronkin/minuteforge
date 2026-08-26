"""Синтетическое совещание для отладки и демонстрации.

    python scripts/make_sample_meeting.py

Записи настоящих совещаний в репозиторий не попадают никогда: там живые люди,
внутренние дела и персональные данные. Поэтому пример придуман целиком — имена
вымышлены, обсуждение тоже, но структура ровно такая, какую отдаёт
распознавание с диаризацией: сегменты со спикером, текстом и временем.

Совещание нарочно неидеальное: одна реплика без метки говорящего, одно
поручение без исполнителя, одно — разложенное на две реплики, и один участник,
которого диаризация разделила надвое. Именно на таких местах инструмент и
проверяется.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: (спикер, текст, длительность в секундах)
SCRIPT: list[tuple[str | None, str, float]] = [
    ("SPEAKER_00", "Коллеги, начинаем. Повестка короткая: сроки по этапу интеграции и подготовка к приёмке.", 12),
    ("SPEAKER_01", "Доброе утро.", 2),
    ("SPEAKER_02", "Здравствуйте.", 2),
    ("SPEAKER_00", "Сначала интеграция. Что по срокам?", 5),
    ("SPEAKER_01", "Основной контур собран, осталась выгрузка справочников. Думаю, неделя.", 10),
    ("SPEAKER_00", "Неделя нас не устраивает, приёмка через десять дней.", 7),
    ("SPEAKER_01", "Тогда нужно подключить второго разработчика.", 6),
    ("SPEAKER_00", "Хорошо. Сергей, подготовьте план работ по выгрузке справочников", 8),
    ("SPEAKER_00", "и согласуйте его со мной до пятницы.", 5),
    ("SPEAKER_01", "Принято, сделаю.", 3),
    (None, "Извините, плохо слышно, повторите про сроки.", 4),
    ("SPEAKER_02", "У меня вопрос по приёмке. Кто готовит демонстрационный стенд?", 8),
    ("SPEAKER_00", "Стенд за вами. Разверните его на тестовом контуре к среде.", 8),
    ("SPEAKER_02", "Понял, разверну.", 3),
    ("SPEAKER_00", "Ещё нужно собрать справку по инцидентам за квартал, без неё приёмку не пройдём.", 11),
    ("SPEAKER_01", "А кто её собирает?", 3),
    ("SPEAKER_00", "Решим отдельно, вернёмся к этому завтра.", 5),
    ("SPEAKER_03", "Добавлю по документации: инструкции пользователя пока нет.", 8),
    ("SPEAKER_00", "Анна, направьте черновик инструкции участникам до конца недели.", 8),
    ("SPEAKER_03", "Хорошо.", 2),
    ("SPEAKER_00", "Тогда на этом всё. Спасибо, коллеги.", 5),
]

#: Кого диаризация «раздвоила»: последняя реплика председателя получила
#: отдельную метку. В жизни это обычное дело, когда человек отходит от
#: микрофона или меняется громкость.
SPLIT_SPEAKER = {20: "SPEAKER_04"}


def build_segments() -> list[dict]:
    segments = []
    clock = 0.0
    for index, (speaker, text, seconds) in enumerate(SCRIPT):
        label = SPLIT_SPEAKER.get(index, speaker)
        segments.append(
            {
                "speaker": label,
                "text": text,
                "start": round(clock, 2),
                "end": round(clock + seconds, 2),
            }
        )
        clock += seconds + 0.5
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    segments = build_segments()
    target = args.out / "meeting_segments.json"
    target.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    speakers = sorted({s["speaker"] for s in segments if s["speaker"]})
    minutes = segments[-1]["end"] / 60
    print(f"{len(segments)} сегментов -> {target}")
    print(f"говорящих: {len(speakers)} ({', '.join(speakers)}), длительность: {minutes:.1f} мин")


if __name__ == "__main__":
    main()
