"""Протокол видеосовещания с поручениями.

Здесь же запоминается версия, с которой запущен процесс. Момент важен:
значение считается один раз, при первой загрузке пакета в память. Streamlit
перезапускает сценарий приложения на каждое действие, но уже загруженные
модули не перечитывает — после ``git pull`` без перезапуска часть кода
остаётся старой. Версия, прочитанная из репозитория «сейчас», об этом не
скажет: на диске уже новый коммит, а работает старый. Прочитанная при
загрузке — скажет.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _revision(root: Path | None = None) -> str:
    """Короткий хеш и заголовок последнего коммита — или пусто."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root or _ROOT), "log", "-1", "--format=%h %s"],
            capture_output=True,
            timeout=3,
        )
    except Exception:  # pragma: no cover — git есть не везде
        return ""
    # Байты, а не text=True: git отдаёт UTF-8, а Python на Windows декодирует
    # по локали, и русский заголовок коммита превращается в кракозябры.
    return done.stdout.decode("utf-8", "replace").strip()


#: Версия кода, который сейчас в памяти. Считается при загрузке пакета.
LOADED_REVISION = _revision()


def disk_revision() -> str:
    """Версия, лежащая в репозитории сию минуту.

    Отличается от :data:`LOADED_REVISION` ровно тогда, когда обновление
    подтянули, а приложение не перезапустили.
    """
    return _revision()
