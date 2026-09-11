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
from dataclasses import dataclass, field, replace
from io import StringIO
from typing import Sequence

from . import dates
from .blocks import UNKNOWN, Transcript
from .directory import DEFAULT_LABEL
from .people import Person, canonical, find
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
    #: Как называется графа «чей вопрос разбирали». У штаба это «Регион», у
    #: завода «Площадка», у холдинга «Общество» — форму разбора приносит
    #: организация, а не инструмент.
    unit_label: str = DEFAULT_LABEL
    #: Кому адресовать поручения по каждому направлению: «Ставропольский
    #: край» → «Правительству Ставропольского края». Берётся из справочника,
    #: третьей его колонки. Нет записи — в документ идёт название как есть:
    #: падежей инструмент не знает и склонять не берётся.
    addressees: dict[str, str] = field(default_factory=dict)
    #: Оборот, которым поручают: «Рекомендовать {кому}». Пусто — в документ
    #: идёт один адресат.
    decision_formula: str = ""
    #: Ответы модели как есть. В документ не идут, нужны для разбора: когда
    #: поручений не нашлось, только по ним и видно, в чём дело.
    answers: list[str] = field(default_factory=list)
    #: Ход разбора по стадиям. Тоже не для документа: по нему видно, где
    #: поручение потерялось и откуда взялось лишнее.
    journal: object | None = None

    @property
    def actionable(self) -> list[Task]:
        """Поручения, у которых есть адресат — человек или регион.

        Направление — не исполнитель, названный вслух, а тот, чей вопрос
        разбирали. Но для работы этого хватает: по нему ответственного
        находят в своём списке и рассылают поручение ему. Пункт с
        направлением и пустым исполнителем — это работа, которую можно
        начать, а не вопрос, который надо выяснять.
        """
        return [t for t in self.tasks if t.who or t.unit]

    @property
    def decisions(self) -> list["Decision"]:
        """Поручения, собранные по адресатам, — раздел «Решили».

        В готовом документе поручения не лежат плоским списком. Они собраны
        в пункты по тому, кому адресованы: один адресат — один пункт,
        несколько поручений — подпунктами, и общий срок под ними. Так его
        и исполняют: пункт целиком уходит в одну организацию.

        Порядок — по ходу совещания, а не по алфавиту: разбор шёл по кругу,
        и человек, слушавший его, ищет пункт там же, где он прозвучал.
        """
        return group_decisions(self.actionable, self.addressees)

    @property
    def decisions_text(self) -> str:
        return "\n".join(_decision_lines(self.decisions, self.decision_formula))

    @property
    def needs_clarification(self) -> list[Task]:
        """Поручения, у которых нет ни исполнителя, ни региона.

        Это не брак распознавания, а частая правда совещаний: прозвучало
        «надо подготовить справку», а кому — не сказали. Секретарь должен
        видеть такие пункты отдельно, чтобы уточнить их, а не разослать
        поручение в никуда.
        """
        return [t for t in self.tasks if not t.who and not t.unit]

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
        if self.transcript is not None and self.transcript.model:
            # Реквизит документа наравне с датой и местом: протокол — пересказ
            # расшифровки, а расшифровки разных моделей отличаются фамилиями и
            # цифрами. Тот, кто будет с документом спорить, должен знать, по
            # чему он составлен.
            hinted = (
                f", подсказка в {len(self.transcript.hints.split())} слов"
                if self.transcript.hints else ""
            )
            lines.append(f"**Распознано:** модель {self.transcript.model}{hinted}  ")
        lines.append("")

        lines.append("## Поручения")
        lines.append("")
        if not self.tasks:
            lines.append("Поручений не зафиксировано.")
        else:
            lines.extend(_task_table(self.actionable, self.unit_label))
            if self.needs_clarification:
                lines.append("")
                lines.append("### Требуют уточнения")
                lines.append("")
                lines.append(
                    "Прозвучали, но ни исполнитель, ни "
                    f"{self.unit_label.lower()} не названы:"
                )
                lines.append("")
                for task in self.needs_clarification:
                    due = f", срок: {_due(task)}" if (task.due or task.due_date) else ""
                    lines.append(f"- **{_clock(task.at)}** {task.what}{due}")

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
            "model": self.transcript.model if self.transcript else "",
            "tasks": "\n".join(_task_lines(self.actionable)),
            "decisions": self.decisions_text,
            "unclear": "\n".join(f"- {t.what}" for t in self.needs_clarification),
            "tasks_table": "\n".join(_task_table(self.actionable, self.unit_label)),
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
        # Цитата в таблице — то, ради чего она и открывается: по ней видно
        # сразу, поручение это или изложение доклада, и сверяться с записью
        # приходится только в спорных случаях. Выписывается куском, а не
        # одним предложением: «Просьба подтвердить» само по себе не говорит
        # ничего, а с соседней фразой — говорит всё.
        # Срок двумя колонками: сказанное и дата. По дате сортируют и
        # ставят напоминания, по сказанному спорят.
        writer.writerow(
            ["№", "Время", "Поручение", "Исполнитель", self.unit_label, "Срок",
             "Срок датой", "Кто сказал", "Цитата"]
        )
        for number, task in enumerate(self.tasks, 1):
            writer.writerow([
                number, _clock(task.at), task.what, task.who, task.unit,
                task.due, task.due_date, task.said_by, task.context or task.quote,
            ])
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
    unit_label: str = DEFAULT_LABEL,
    addressees: dict[str, str] | None = None,
    decision_formula: str = "",
) -> Protocol:
    """Собирает протокол.

    Участники, если не переданы, берутся из стенограммы — это метки
    говорящих или уже сопоставленные с ними имена. Пустой список лучше
    выдуманного: приписать совещанию участника, которого не было, хуже, чем
    оставить строку незаполненной.
    """
    # Сроки в даты: «через две недели» считаются от дня совещания, а не от
    # дня, когда делопроизводитель открыл протокол. Нет даты в шапке — нет и
    # дат в поручениях: выдумывать точку отсчёта нельзя.
    meeting_day = dates.parse_meeting_date(date)
    if meeting_day is not None:
        tasks = [
            replace(task, due_date=dates.as_text(dates.resolve(task.due, meeting_day)))
            for task in tasks
        ]

    if people:
        # Имена исполнителей приводятся к списку участников: в записи
        # порядок слов свободный, отчество распознаётся плохо, а в протоколе
        # один и тот же человек не должен встречаться в двух написаниях.
        tasks = [
            Task(
                what=task.what, who=canonical(task.who, people), due=task.due,
                chunk=task.chunk, at=task.at, quote=task.quote,
                context=task.context, said_by=task.said_by, unit=task.unit,
                due_date=task.due_date,
            )
            for task in tasks
        ]

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
        unit_label=unit_label,
        addressees=dict(addressees or {}),
        decision_formula=decision_formula,
    )


def _is_label(name: str) -> bool:
    """Похоже ли это на метку диаризации, а не на человека."""
    return bool(re.match(r"^(SPEAKER[_ -]*\d+|UNKNOWN)$", (name or "").strip(), re.IGNORECASE))


@dataclass
class Decision:
    """Пункт раздела «Решили»: кому и что поручено.

    Отдельная сущность, а не строка текста: тот же пункт нужен и в разметке
    протокола, и в шаблоне организации, и в выгрузке — а собирается он один
    раз здесь.
    """

    addressee: str
    tasks: list[Task] = field(default_factory=list)
    #: Взят ли адресат из справочника. Оборот приставляется только к таким:
    #: там дательный падеж выверен рукой, а к исполнителю, названному вслух,
    #: «Рекомендовать» приставить нельзя — «Рекомендовать все регионы».
    from_directory: bool = False

    @property
    def due(self) -> str:
        """Общий срок пункта. Пусто — сроки разные или их нет вовсе.

        Разные сроки в одну строку не сводятся: «Срок: 10.09.2026» под
        пунктом, где половина подпунктов к другому числу, — это не сокращение,
        а подлог. Тогда срок пишется у каждого подпункта отдельно.
        """
        said = [_on_date(task) for task in self.tasks if task.due or task.due_date]
        unique = list(dict.fromkeys(said))
        return unique[0] if len(unique) == 1 and len(said) == len(self.tasks) else ""


def group_decisions(
    tasks: Sequence[Task],
    addressees: dict[str, str] | None = None,
) -> list[Decision]:
    """Собирает поручения в пункты по адресатам.

    Адресат берётся сначала из направления, и только потом из исполнителя.
    Порядок именно такой: в документе поручают организации — «Правительству
    такой-то области», — а названный вслух человек в ней работает. Он не
    пропадает: в таблице поручений графа «Исполнитель» остаётся.
    """
    addressees = addressees or {}
    order: list[str] = []
    groups: dict[str, list[Task]] = {}
    known: set[str] = set()
    for task in tasks:
        who = addressees.get(task.unit, task.unit) if task.unit else task.who
        if not who:
            continue
        if task.unit and task.unit in addressees:
            known.add(who)
        if who not in groups:
            groups[who] = []
            order.append(who)
        groups[who].append(task)
    return [
        Decision(addressee=who, tasks=groups[who], from_directory=who in known)
        for who in order
    ]


def _decision_lines(decisions: Sequence[Decision], formula: str = "") -> list[str]:
    """Раздел «Решили» так, как он выглядит в документе."""
    lines: list[str] = []
    for number, point in enumerate(decisions, 1):
        common = point.due
        addressee = _upper_first(_addressed(point, formula))
        if len(point.tasks) == 1 and not common:
            # Одно поручение — в строку за двоеточием: заводить подпункт «а»
            # при единственном пункте документу незачем.
            lines.append(f"{number}. {addressee}: {_said(point.tasks[0])}.")
            lines.append("")
            continue
        lines.append(f"{number}. {addressee}:")
        for position, task in enumerate(point.tasks, 1):
            said = _said(task)
            if not common and (task.due or task.due_date):
                said = f"{said}, срок — {_on_date(task)}"
            end = "." if position == len(point.tasks) else ";"
            lines.append(f"- {said}{end}")
        if common:
            lines.append(f"Срок: {common}.")
        lines.append("")
    return lines[:-1] if lines else ["Поручений не зафиксировано."]


def _said(task: Task) -> str:
    """Поручение так, как оно читается подпунктом: со строчной буквы."""
    text = (task.what or "").strip().rstrip(".")
    return text[:1].lower() + text[1:] if text else text


def _addressed(point: Decision, formula: str) -> str:
    """Адресат с оборотом, которым поручают у вас.

    Оборот идёт только к адресатам из справочника. Исполнитель, названный
    вслух, приезжает в том падеже, в каком прозвучал, — «все регионы», «Ким
    С.А.», — и «Рекомендовать все регионы» вышло бы не по-русски.
    """
    if not formula or not point.from_directory:
        return point.addressee
    if "{кому}" in formula:
        return formula.replace("{кому}", point.addressee)
    return f"{formula} {point.addressee}"


def _upper_first(text: str) -> str:
    """Адресат открывает пункт документа, а пункт начинается с большой буквы.

    Именно первая буква, а не ``capitalize``: тот снёс бы заглавные внутри —
    «Правительству Иркутской Области» превратилось бы в «области».
    """
    text = (text or "").strip()
    return text[:1].upper() + text[1:] if text else text


def _on_date(task: Task) -> str:
    """Срок для документа: числом, если оно посчитано.

    В таблице для вычитки полезно и сказанное вслух — «через две недели»
    объясняет, откуда взялось число. В документе, по которому ставят на
    контроль, нужна дата.
    """
    return task.due_date or task.due


def _task_lines(tasks: Sequence[Task]) -> list[str]:
    """Поручения списком — то, что идёт в раздел «Решили»."""
    lines = []
    for number, task in enumerate(tasks, 1):
        due = f", срок — {_due(task)}" if (task.due or task.due_date) else ""
        place = f" [{_clock(task.at)}]" if task.at is not None else ""
        who = task.who or task.unit or "исполнитель не назван"
        lines.append(f"{number}. {who}: {task.what}{due}.{place}")
    return lines


def _task_table(tasks: Sequence[Task], label: str = DEFAULT_LABEL) -> list[str]:
    if not tasks:
        return ["Поручений с назначенным исполнителем нет."]
    # Время — колонка проверки. Без неё человек ищет место в двухчасовой
    # записи сам, и сверка восьмидесяти пунктов съедает больше, чем сэкономил
    # разбор.
    # Направление отдельной графой, а не вместе с исполнителем:
    # исполнителя назвали вслух, а направление выведено из хода совещания, и
    # смешивать их в одной клетке значит выдавать второе за первое.
    rows = [
        f"| № | Время | Поручение | Исполнитель | {label} | Срок |",
        "|---|---|---|---|---|---|",
    ]
    for number, task in enumerate(tasks, 1):
        rows.append(
            f"| {number} | {_clock(task.at)} | {task.what} | {task.who or '—'} "
            f"| {task.unit or '—'} | {_due(task)} |"
        )
    return rows


def _due(task: Task) -> str:
    """Срок так, как он прозвучал, и датой — если её удалось посчитать.

    Сказанное не подменяется вычисленным: спорить будут о сказанном, а
    работать по дате, и в документе должно стоять и то, и другое.
    """
    if task.due and task.due_date:
        return f"{task.due} ({task.due_date})"
    return task.due or task.due_date or "—"


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
