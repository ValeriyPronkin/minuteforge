#!/usr/bin/env bash
# Собирает то, что несут на машину без интернета.
#
# Собирать вручную нельзя: файл со старым содержимым и свежей датой выглядит
# как обновление, а им не является — обновление «не появляется», и причину
# ищут в приложении. Поэтому рядом с бандлом кладётся версия, которая в нём
# лежит, и её сверяют со строкой в шапке приложения.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/minuteforge-transfer}"
cd "$REPO"

if [ -n "$(git status --porcelain)" ]; then
    echo "Есть незакоммиченное — в перенос оно не попадёт:"
    git status --short
    echo
fi

mkdir -p "$OUT"
git bundle create "$OUT/minuteforge.bundle" --all
git log -1 --format="%h %ad %s" --date=format:"%d.%m.%Y %H:%M" > "$OUT/ВЕРСИЯ.txt"

echo
echo "Готово: $OUT/minuteforge.bundle"
echo "Внутри: $(cat "$OUT/ВЕРСИЯ.txt")"
echo
echo "На той машине, в папке с приложением:"
echo "    git pull <путь>\\minuteforge.bundle main"
echo "    (первый раз: git clone <путь>\\minuteforge.bundle minuteforge)"
echo
echo "Потом ОБЯЗАТЕЛЬНО перезапустить приложение целиком: Ctrl+C и заново"
echo "streamlit run. Streamlit перечитывает сценарий, но не модули в памяти."
echo
echo "Проверка: строка «Версия в памяти» в шапке должна совпасть с ВЕРСИЯ.txt."
