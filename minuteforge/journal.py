"""Журнал разбора: где поручение потерялось и откуда взялось лишнее.

Между стенограммой и протоколом список поручений проходит десяток стадий:
модель читает окна и выписывает, правила отсеивают не поручения, склейка
сводит сказанное одной фразой, проверка снимает изложение доклада. Видно при
этом только начало и конец — сорок пунктов на выходе, — и любой спор о
качестве упирается в догадки. «Модель не нашла» и «нашла, а мы отсеяли»
выглядят одинаково: пункта нет.

Здесь записываются все стадии целиком: сколько было, сколько стало, что
именно ушло и с какой формулировкой. И отдельно — окна, из которых модель не
вернула ничего: это единственное место, где видно, что она пропустила.

Файл ложится рядом с протоколом. Он не документ и не для рассылки: это
рабочая записка для того, кто настраивает разбор.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from loguru import logger

#: Сколько знаков окна показывать в записке. Окно — около сотни токенов,
#: целиком оно и есть примерно столько; обрезка нужна на случай длинной
#: реплики, которую распознавание не разбило на предложения.
QUOTE_LIMIT = 700


@dataclass
class Window:
    """Одно окно стенограммы и что модель по нему ответила."""

    index: int
    total: int
    at: float | None
    text: str
    answer: str
    found: int


@dataclass
class Step:
    """Одна стадия разбора: сколько было, сколько стало, что ушло."""

    title: str
    before: int
    after: int
    #: Что исчезло: отсеяно правилом, снято проверкой или поглощено склейкой.
    gone: list = field(default_factory=list)
    #: Что появилось: склейка выдаёт новую формулировку взамен нескольких.
    came: list = field(default_factory=list)
    why: str = ""


class Journal:
    """Что происходило с поручениями по дороге в протокол."""

    def __init__(self) -> None:
        self.windows: list[Window] = []
        self.steps: list[Step] = []

    # ------------------------------------------------------------ запись
    def window(self, chunk, answer: str, found: int) -> None:
        """Окно, ответ модели на него и сколько из ответа уцелело."""
        starts = [
            block.start for block in getattr(chunk, "blocks", [])
            if getattr(block, "start", None) is not None
        ]
        self.windows.append(Window(
            index=getattr(chunk, "index", len(self.windows) + 1),
            total=getattr(chunk, "total", 0),
            at=min(starts) if starts else None,
            text=getattr(chunk, "text", ""),
            answer=answer or "",
            found=found,
        ))

    def step(self, title: str, before: Sequence, after: Sequence, why: str = "") -> None:
        """Стадия разбора. Что ушло и что пришло, считается сравнением.

        Сравнение по тексту поручения, а не по объекту: склейка и сведение
        собирают новую запись из нескольких, и объекты после стадии — уже
        другие. Зато формулировка говорит человеку то, что ему нужно.
        """
        was = {task.key: task for task in before}
        now = {task.key: task for task in after}
        self.steps.append(Step(
            title=title,
            before=len(before),
            after=len(after),
            gone=[task for key, task in was.items() if key not in now],
            came=[task for key, task in now.items() if key not in was],
            why=why,
        ))

    # ------------------------------------------------------------ чтение
    @property
    def silent_windows(self) -> list[Window]:
        """Окна, из которых модель не вернула ни одного поручения.

        Самое ценное место записки. Правило показало модели фразу, в которой
        поручение слышно, — а пункта нет. Либо правило ошиблось, либо модель
        не справилась, и решается это чтением: другого способа отличить одно
        от другого не существует.
        """
        return [window for window in self.windows if not window.found]

    def as_markdown(self) -> str:
        lines = ["# Разбор поручений", ""]
        lines.append(
            "Рабочая записка, не документ. Показывает, что происходило со "
            "списком поручений между стенограммой и протоколом."
        )
        lines.append("")

        if self.windows:
            silent = self.silent_windows
            spoke = len(self.windows) - len(silent)
            lines.extend([
                "## Окна",
                "",
                f"Показано модели: **{len(self.windows)}**. "
                f"Вернула поручения из **{spoke}**, промолчала в **{len(silent)}**.",
                "",
            ])

        if self.steps:
            lines.extend([
                "## Воронка",
                "",
                "| Стадия | Было | Стало | Ушло |",
                "|---|---|---|---|",
            ])
            for step in self.steps:
                change = step.before - step.after
                lines.append(
                    f"| {step.title} | {step.before} | {step.after} | "
                    f"{change if change > 0 else '—'} |"
                )
            lines.append("")

        for step in self.steps:
            if not step.gone and not step.came:
                continue
            lines.append(f"### {step.title}")
            lines.append("")
            if step.why:
                lines.append(step.why)
                lines.append("")
            for task in step.gone:
                lines.append(f"- **снято** {_clock(task.at)} {task.what}")
                if task.quote:
                    lines.append(f"  - «{_short(task.quote)}»")
            for task in step.came:
                lines.append(f"- **стало** {_clock(task.at)} {task.what}")
            lines.append("")

        if self.silent_windows:
            lines.extend([
                "## Окна, из которых ничего не вышло",
                "",
                "Правило услышало здесь поручение, модель — нет. Прочитайте: "
                "если поручение в окне есть, промахнулась модель; если нет — "
                "правило, и окно показывать не стоило.",
                "",
            ])
            for window in self.silent_windows:
                lines.append(f"**Окно {window.index} — {_clock(window.at)}**")
                lines.append("")
                lines.append(f"> {_short(window.text)}")
                lines.append("")
                answer = " ".join((window.answer or "").split())
                lines.append(f"Ответ модели: {_short(answer, 300) or '(пусто)'}")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"


#: Уже заведённая запись в файл. Интерфейс перечитывает свой сценарий на
#: каждое нажатие, и без этого к журналу добавлялся бы новый обработчик по
#: десятку за сеанс: строки задваивались бы, а файлы копились открытыми.
_HANDLER: int | None = None


def setup_file_log(log_dir: str | Path, level: str = "INFO") -> Path | None:
    """Заводит запись журнала в файл рядом с настройками.

    До этого loguru писал только в консоль, а у интерфейса консоли, считай,
    нет: приложение запускают ярлыком, окно закрывают, и разбираться потом
    не с чем. Настройка ``log_dir`` при этом была — и не делала ничего.

    Файл на сутки, хранится две недели: журнал одного совещания это сотни
    строк, а разбираться приходится с прогоном, который был позавчера.
    """
    global _HANDLER
    if _HANDLER is not None:
        return Path(log_dir).expanduser()
    try:
        folder = Path(log_dir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "minuteforge_{time:YYYY-MM-DD}.log"
        _HANDLER = logger.add(
            str(path),
            level=level,
            rotation="00:00",
            retention="14 days",
            encoding="utf-8",
            # Без очереди: она пишет из отдельного потока, и последние
            # строки прогона оседают в ней до закрытия. Разбираются как раз
            # по последним, а на Windows фоновый поток ещё и держит файл
            # открытым, когда приложение уже закрыли.
            # Полный формат: разбирать приходится чужой прогон по присланному
            # файлу, и «в какой строке какого модуля» там единственная зацепка.
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        )
        logger.info("Журнал пишется в {}", folder.resolve())
        return folder
    except OSError as exc:
        # Недоступная папка — не повод не считать протокол. Журнал полезен,
        # но не обязателен, и падать из-за него нельзя.
        print(f"Журнал не ведётся: {exc}", file=sys.stderr)
        return None


def _short(text: str, limit: int = QUOTE_LIMIT) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
