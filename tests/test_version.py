"""Версия, с которой работает процесс."""

from pathlib import Path

import minuteforge
from minuteforge import LOADED_REVISION, disk_revision


def test_loaded_revision_is_read_once_at_import():
    """Значение должно быть снято в момент загрузки пакета, а не по запросу.

    Streamlit перезапускает сценарий, но загруженные модули не перечитывает.
    Версия, прочитанная «сейчас», покажет то, что лежит на диске, — а считает
    в это время старый код, оставшийся в памяти. Именно так однажды и вышло:
    интерфейс уже показывал новые пометки, а файл писал старый модуль.
    """
    assert isinstance(LOADED_REVISION, str)
    assert LOADED_REVISION == minuteforge.LOADED_REVISION


def test_disk_revision_is_asked_anew():
    """А эта — наоборот, каждый раз свежая: с ней и сравнивают."""
    assert disk_revision() == LOADED_REVISION, "в чистом репозитории они совпадают"


def test_no_repository_is_not_an_error(tmp_path: Path):
    """Инструмент ставят и без git — тогда версия просто неизвестна."""
    from minuteforge import _revision

    assert _revision(tmp_path) == ""
