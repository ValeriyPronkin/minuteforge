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
import re
from dataclasses import dataclass, field
from io import StringIO
from typing import Sequence

from .blocks import UNKNOWN, Transcript
from .people import Person, find
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
    #: Секретарь и номер — обязательные реквизиты протокола как документа.
    #: Распознавание их не даёт и дать не может: их знает делопроизводитель.
    secretary: str = ""
    number: str = ""
    attendees: list[str] = field(default_factory=list)
    #: Список участников с должностями, если он был. Отдельно от attendees:
    #: в шапке протокола нужны должности, а в поручениях — только фамилии.
    people: list[Person] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    transcript: Transcript | None = None
    #: Ответы модели как есть. В документ не идут, нужны для разбора: когда
    #: поручений не нашлось, только по ним и видно, в чём дело.
    answers: list[str] = field(default_factory=list)

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
            ("Номер", self.number),
            ("Дата", self.date),
            ("Место", self.place),
            ("Председатель", self.chair),
            ("Секретарь", self.secretary),
        ]
        for label, value in head:
            if value:
                lines.append(f"**{label}:** {value}  ")
        if self.attendees_full:
            # Списком, а не строкой через запятую: «Иванов И.И., начальник
            # отдела, Петров П.П., главный специалист» читается как перечень
            # из четырёх человек.
            lines.append("**Участники:**  ")
            lines.append("")
            lines.extend(f"* {item}" for item in self.attendees_full)
            # Пустая строка после списка обязательна: без неё разметка
            # приклеивает следующую строку к последнему участнику, и
            # длительность записи оказывается его должностью.
            lines.append("")
        if self.unnamed_count:
            lines.append(
                f"**Не опознано голосов:** {self.unnamed_count} — "
                "их реплики есть в стенограмме  "
            )
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

    def fields(self) -> dict[str, str]:
        """Значения для подстановки в шаблон.

        Пустое поле подставляется пустой строкой, а не пропускается: в
        готовой форме документа у каждой строки своё место, и молча съесть
        её нельзя — делопроизводитель должен увидеть, что реквизит не
        заполнен, и вписать его руками.
        """
        return {
            "title": self.title,
            "number": self.number,
            "date": self.date,
            "place": self.place,
            "chair": self.chair,
            "secretary": self.secretary,
            "attendees": "\n".join(f"{i}. {a}" for i, a in enumerate(self.attendees_full, 1)),
            "attendees_line": ", ".join(self.named_attendees),
            "duration": f"{self.transcript.duration_min} мин" if self.transcript else "",
            "tasks": "\n".join(_task_lines(self.actionable)),
            "unclear": "\n".join(f"- {t.what}" for t in self.needs_clarification),
            "tasks_table": "\n".join(_task_table(self.actionable)),
            "tasks_count": str(len(self.tasks)),
        }

    @property
    def named_attendees(self) -> list[str]:
        """Только те, кого опознали.

        Метка вида ``SPEAKER_13`` — не участник совещания, а обозначение
        голоса. В шапке протокола такому места нет: документ подписывают и
        рассылают, а список из двадцати девяти безымянных голосов делает его
        бумагой, которую нельзя показать.
        """
        return [name for name in self.attendees if not _is_label(name)]

    @property
    def unnamed_count(self) -> int:
        """Сколько голосов осталось неопознанными.

        Молча их не прячем: строка «столько-то голосов не опознано» честно
        говорит, что список неполон.
        """
        return sum(1 for name in self.attendees if _is_label(name))

    @property
    def attendees_full(self) -> list[str]:
        """Опознанные участники с должностями, если они известны."""
        result = []
        for name in self.named_attendees:
            person = find(self.people, name)
            result.append(person.full if person else name)
        return result

    def render(self, template: str) -> str:
        """Заполняет шаблон протокола.

        Форма протокола в каждой организации своя, и угадать её снаружи
        нельзя: где-то нужен номер и гриф, где-то «СЛУШАЛИ — ВЫСТУПИЛИ —
        РЕШИЛИ», где-то подписи двух человек. Поэтому форму приносит
        пользователь, а инструмент только подставляет в неё то, что знает.

        Неизвестное место в шаблоне остаётся как есть — это не ошибка, а
        подсказка тому, кто будет дописывать документ руками.
        """
        result = template
        for key, value in self.fields().items():
            for placeholder in (f"{{{{{key}}}}}", f"{{{{ {key} }}}}"):
                result = result.replace(placeholder, value)
        return result

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
    people: Sequence[Person] | None = None,
    secretary: str = "",
    number: str = "",
    answers: Sequence[str] | None = None,
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
        secretary=secretary,
        number=number,
        attendees=list(attendees or []),
        people=list(people or []),
        tasks=list(tasks),
        transcript=transcript,
        answers=list(answers or []),
    )


def _is_label(name: str) -> bool:
    """Похоже ли это на метку диаризации, а не на человека."""
    return bool(re.match(r"^(SPEAKER[_ -]*\d+|UNKNOWN)$", (name or "").strip(), re.IGNORECASE))


def _task_lines(tasks: Sequence[Task]) -> list[str]:
    """Поручения списком — то, что идёт в раздел «Решили»."""
    lines = []
    for number, task in enumerate(tasks, 1):
        due = f", срок — {task.due}" if task.due else ""
        lines.append(f"{number}. {task.who}: {task.what}{due}.")
    return lines


def _task_table(tasks: Sequence[Task]) -> list[str]:
    if not tasks:
        return ["Поручений с назначенным исполнителем нет."]
    rows = ["| № | Поручение | Исполнитель | Срок |", "|---|---|---|---|"]
    for number, task in enumerate(tasks, 1):
        rows.append(f"| {number} | {task.what} | {task.who} | {task.due or '—'} |")
    return rows
