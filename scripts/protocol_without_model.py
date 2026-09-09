"""Протокол по готовой стенограмме без модели — образец для сверки.

    python scripts/protocol_without_model.py стенограмма.txt 07.09.2026 куда/ справочник.csv Регион


Проходит ровно тем же путём, что и приложение: та же отбраковка переклички,
тот же отбор окон по фразам, те же правила исполнителя, направления и срока, та
же сборка документа. Подменена только сама модель: вместо неё поручением
объявляется та фраза, в которой правило его слышит.

Значит, сверять по этому образцу можно всё, кроме формулировок: время,
исполнителя, направление, срок и дату срока. Пунктов здесь будет больше, чем в
приложении: сведение повторов и проверочный проход требуют модели и здесь
отключены.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minuteforge.blocks import Block, Transcript
from minuteforge.config import Settings
from minuteforge.llm import Reply
from minuteforge.pipeline import Meeting, protocol_from_transcript
from minuteforge.tasks import is_directive, _sentences

LINE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+([^:]+):\s*(.*)$")


def read_transcript(path: Path) -> Transcript:
    """Стенограмма из сохранённого файла: json или текст.

    Json точнее: в нём настоящие границы реплик. В тексте конца реплики нет,
    и он восстанавливается по началу следующей — на отборе окон это не
    сказывается, а на длительности записи сказывается.
    """
    if path.suffix.lower() == ".json":
        from minuteforge.cli import load_transcript
        return load_transcript(path)
    blocks: list[Block] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        found = LINE.match(line.strip())
        if not found:
            if blocks and line.strip():
                blocks[-1].text += " " + line.strip()
            continue
        hours, minutes, seconds, speaker, text = found.groups()
        start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        blocks.append(Block(speaker.strip(), text.strip(), float(start), float(start)))
    # Конца реплики в тексте нет: берём начало следующей, последней даём
    # минуту. Время в протоколе идёт от начала, и на нём это не сказывается.
    for one, next_one in zip(blocks, blocks[1:]):
        one.end = next_one.start
    if blocks:
        blocks[-1].end = blocks[-1].start + 60
    return Transcript(blocks)


class RuleOnlyClient:
    """Заглушка вместо модели: поручением объявляет саму фразу.

    Так проверяется всё, что делается правилами, — а правилами теперь
    делается исполнитель, направление и срок.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunks = 0

    def complete(self, system, user, **kwargs):
        if "Стенограмма:" not in user:
            return Reply(text="")          # прогрев
        self.chunks += 1
        text = user.split("Стенограмма:\n", 1)[1]
        lines = []
        for sentence in _sentences(text):
            # Из фразы убирается метка говорящего: в поручении ей не место.
            clean = re.sub(r"^[A-ZА-Я_0-9 .]+:\s*", "", sentence).strip()
            if len(clean) > 6 and is_directive(clean):
                lines.append(f"Поручение: {clean}\nКому: \nСрок: ")
        return Reply(text="\n\n".join(lines) or "НЕТ ПОРУЧЕНИЙ")

    def diagnose(self):
        return ""


def main() -> int:
    source = Path(sys.argv[1])
    date = sys.argv[2] if len(sys.argv) > 2 else ""
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else source.parent / "образец"

    transcript = read_transcript(source)
    print(f"Реплик: {len(transcript.blocks)}, голосов: {len(transcript.speakers)}, "
          f"длительность: {transcript.duration_min} мин")

    settings = Settings(
        llm_json_mode=False,     # заглушке проще отвечать строками
        merge_similar=False,     # сведение повторов без модели невозможно
        verify_tasks=False,      # проверочный проход — тоже
        directory_file=Path(sys.argv[4]) if len(sys.argv) > 4 else None,
        directory_label=sys.argv[5] if len(sys.argv) > 5 else "Направление",
    )
    client = RuleOnlyClient(settings)
    protocol = protocol_from_transcript(
        transcript, settings, meeting=Meeting(date=date), client=client,
    )
    print(f"Окон отобрано: {client.chunks}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "протокол.md").write_text(protocol.as_markdown(), encoding="utf-8")
    (out / "протокол_поручения.csv").write_text(protocol.tasks_csv(), encoding="utf-8-sig")
    if protocol.journal is not None:
        (out / "протокол_разбор.md").write_text(
            protocol.journal.as_markdown(), encoding="utf-8"
        )

    tasks = protocol.tasks
    print(f"Поручений: {len(tasks)}")
    print(f"  с исполнителем: {sum(1 for t in tasks if t.who)}")
    print(f"  с направлением: {sum(1 for t in tasks if t.unit)}")
    print(f"  со сроком:      {sum(1 for t in tasks if t.due)}")
    print(f"  срок датой:     {sum(1 for t in tasks if t.due_date)}")
    print(f"  требуют уточнения: {len(protocol.needs_clarification)}")
    print(f"Записано в {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
