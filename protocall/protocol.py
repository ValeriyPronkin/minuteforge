"""Сборка протокола: из стенограммы и поручений — готовый документ.

Модуль намеренно не обращается к модели. Всё, что можно собрать правилами,
собирается правилами: заголовок, список участников, нумерация поручений,
разделение на исполнимые и требующие уточнения. Модель нужна там, где надо
понять смысл сказанного, а не там, где надо расставить пункты по порядку —
и каждый лишний вызов это ещё десятки секунд ожидания и ещё один шанс, что
она что-нибудь придумает.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import StringIO
from typing import Sequence

from .blocks import UNKNOWN, Transcript
from .tasks import Task


@dataclass
class Protocol:
    """Протокол совещания."""

    title: str = "Протокол совещания"
    #: Дата и время — строкой, как их принято писать в документе. Модуль их
    #: не выдумывает: если не переданы, строка останется пустой, и это лучше,
    #: чем дата обработки записи, выданная за дату совещания.
    date: str = ""
    place: str = ""
    chair: str = ""
    attendees: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    transcript: Transcript | None = None

    @property
    def actionable(self) -> list[Task]:
        """Поручения, у которых есть исполнитель."""
        return [t for t in self.tasks if t.who]

    @property
    def needs_clarification(self) -> list[Task]:
        """Поручения без исполнителя.

        Это не брак распознавания, а частая правда совещаний: прозвучало
        «надо подготовить справку», а кому — не сказали. Секретарь должен
        видеть такие пункты отдельно, чтобы уточнить их, а не разослать
        поручение в никуда.
        """
        return [t for t in self.tasks if not t.who]

    def as_markdown(self, *, with_transcript: bool = False) -> str:
        """Протокол как размеченный текст."""
        lines = [f"# {self.title}", ""]

        head = [
            ("Дата", self.date),
            ("Место", self.place),
            ("Председатель", self.chair),
        ]
        for label, value in head:
            if value:
                lines.append(f"**{label}:** {value}  ")
        if self.attendees:
            lines.append(f"**Участники:** {', '.join(self.attendees)}  ")
        if self.transcript is not None and self.transcript.duration_min:
            lines.append(f"**Длительность записи:** {self.transcript.duration_min} мин  ")
        lines.append("")

        lines.append("## Поручения")
        lines.append("")
        if not self.tasks:
            lines.append("Поручений не зафиксировано.")
        else:
            lines.extend(_task_table(self.actionable))
            if self.needs_clarification:
                lines.append("")
                lines.append("### Требуют уточнения")
                lines.append("")
                lines.append("Прозвучали, но исполнитель не назван:")
                lines.append("")
                for task in self.needs_clarification:
                    due = f" (срок: {task.due})" if task.due else ""
                    lines.append(f"- {task.what}{due}")

        if with_transcript and self.transcript is not None:
            lines.extend(["", "## Стенограмма", "", "```", self.transcript.as_text(with_time=True), "```"])

        return "\n".join(lines).rstrip() + "\n"

    def tasks_csv(self) -> str:
        """Поручения таблицей — то, что уходит в работу.

        Отдельно от протокола: документ читают, а таблицу разбирают по
        исполнителям и ставят на контроль.
        """
        buffer = StringIO()
        writer = csv.writer(buffer, delimiter=";", lineterminator="\n")
        writer.writerow(["№", "Поручение", "Исполнитель", "Срок", "Фрагмент"])
        for number, task in enumerate(self.tasks, 1):
            writer.writerow([number, task.what, task.who, task.due, task.chunk or ""])
        return buffer.getvalue()


def build_protocol(
    tasks: Sequence[Task],
    transcript: Transcript | None = None,
    *,
    title: str = "Протокол совещания",
    date: str = "",
    place: str = "",
    chair: str = "",
    attendees: Sequence[str] | None = None,
) -> Protocol:
    """Собирает протокол.

    Участники, если не переданы, берутся из стенограммы — это метки
    говорящих или уже сопоставленные с ними имена. Пустой список лучше
    выдуманного: приписать совещанию участника, которого не было, хуже, чем
    оставить строку незаполненной.
    """
    if attendees is None and transcript is not None:
        # Неопознанные реплики в стенограмме остаются, но участником
        # совещания UNKNOWN не является: в шапке протокола это выглядело бы
        # так, будто на совещании был человек с таким именем.
        attendees = [s for s in transcript.speakers if s != UNKNOWN]
    return Protocol(
        title=title,
        date=date,
        place=place,
        chair=chair,
        attendees=list(attendees or []),
        tasks=list(tasks),
        transcript=transcript,
    )


def _task_table(tasks: Sequence[Task]) -> list[str]:
    if not tasks:
        return ["Поручений с назначенным исполнителем нет."]
    rows = ["| № | Поручение | Исполнитель | Срок |", "|---|---|---|---|"]
    for number, task in enumerate(tasks, 1):
        rows.append(f"| {number} | {task.what} | {task.who} | {task.due or '—'} |")
    return rows
