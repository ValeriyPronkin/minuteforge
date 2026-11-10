"""Распознать запись GigaAM и положить расшифровку в кэш minuteforge.

Испытание, а не часть конвейера. GigaAM живёт в своём окружении: его разбор
длинных записей тянет pyannote.audio 4 и torch 2.10, а основное окружение
(whisperx) держится на других версиях, и трогать его нельзя. Поэтому здесь —
только распознавание. Выравнивание, разметка голосов и протокол идут из
основного окружения, как обычно, с моделью ``gigaam-<вариант>``:

    minuteforge run запись.mp4 --model gigaam-v3_e2e_rnnt ...

minuteforge увидит готовый transcription.json в подпапке этой модели и
распознавать не станет — только выровняет и разметит.

Порядок для одной записи (команды — из папки проекта):
  1. основное окружение: minuteforge recognize <запись> — whisperX; он же
     извлекает звук в папку шагов записи (data/work/<запись>-<отпечаток>/);
  2. окружение GigaAM:  python scripts/asr_bench.py <запись>
  3. основное окружение: minuteforge recognize <запись> --model gigaam-v3_e2e_rnnt
     (и gigaam-v3_e2e_ctc).

<запись> — везде один и тот же файл, там, где он лежит (например, на I:).
Папку шагов скрипт находит по нему сам, тем же правилом, что minuteforge.
Можно передать и саму папку шагов.

Варианты — те, что пишут с пунктуацией и нормализацией: в пакете gigaam
они называются ``v3_e2e_ctc`` и ``v3_e2e_rnnt``. Веса пакет берёт со своего
сервера и сверяет по md5, а не с ревизий ai-sage/GigaAM-v3 на Hugging Face.

Окружение (Python 3.10+, ставится рядом с основным, не вместо):

    conda create -n gigaam python=3.12
    conda activate gigaam
    pip install torch==2.10.* torchaudio==2.10.* --index-url https://download.pytorch.org/whl/cu128
    pip install "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM@7447938d791c4f3e643386ee22c33777004293a5"
    python -c "import torch; print(torch.cuda.is_available())"

Номер CUDA в адресе — под драйвер видеокарты, сверить на pytorch.org. Без
него pip поставит torch без CUDA, и всё будет считать процессор.

Для разбиения записи GigaAM берёт pyannote/segmentation-3.0 — ту же
закрытую модель, что и диаризация, — с токеном из HF_TOKEN.

Видеопамяти на офисной машине 8 ГБ. minuteforge перед распознаванием
выгружает модели Ollama (llm.free_the_card), а после последней стадии
протокола Ollama держит модель ещё пять минут. Чтобы GigaAM считал в тех
же условиях, что whisperX, скрипт перед стартом выгружает всё, что Ollama
держит в памяти (--ollama-url; не ответила — значит её нет, и это не
ошибка). Сколько свободно после этого, скрипт пишет; пик расхода идёт в
реквизиты расшифровки.

Сюда ничего из записей не попадает: скрипт пишет только в папку шагов,
а она лежит в data/, закрытой .gitignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

#: Варианты с пунктуацией и нормализацией. Имена — как в пакете gigaam.
VARIANTS = ("v3_e2e_ctc", "v3_e2e_rnnt")

#: Должны совпадать с minuteforge.transcribe — держит тест.
ASR_META = "minuteforge_asr"
EXTERNAL_PREFIX = "gigaam-"

#: Расширения звука, который minuteforge извлекает в папку шагов.
_AUDIO = (".wav",)


def model_dir(model: str) -> str:
    """Подпапка кэша для модели — то же правило, что в minuteforge.transcribe."""
    return re.sub(r"[^\w.-]+", "_", model.strip()).strip("._") or "default"


def cache_for_record(source: Path, work_dir: Path) -> Path:
    """Папка шагов записи — то же правило, что minuteforge.pipeline.cache_dir_for
    для записи целиком (без отрезка). Совпадение держит тест."""
    stat = source.stat()
    mark = f"{source.name}|{stat.st_size}|{int(stat.st_mtime)}|0|0"
    short = hashlib.sha1(mark.encode("utf-8")).hexdigest()[:8]
    return work_dir / f"{source.stem}-{short}"


def minuteforge_name(variant: str) -> str:
    """Под каким именем вариант знает minuteforge: ``gigaam-v3_e2e_rnnt``."""
    return f"{EXTERNAL_PREFIX}{variant}"


def to_transcription(segments, *, variant: str, batch_size: int, extra: dict) -> dict:
    """Куски GigaAM в том виде, в каком minuteforge ждёт расшифровку WhisperX.

    Пустые куски — разбиение нашло голос, а модель слов не услышала —
    выбрасываются: WhisperX таких не выдаёт, а выравнивание на них падает.
    Сколько их было, остаётся в реквизитах.
    """
    out, empty = [], 0
    for seg in segments:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            empty += 1
            continue
        out.append({
            "text": text,
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
        })
    return {
        "segments": out,
        "language": "ru",
        ASR_META: {
            "model": minuteforge_name(variant),
            "batch_size": batch_size,
            # Подсказок GigaAM не принимает. Пусто — чтобы протокол не
            # приписал расшифровке того, чего в ней не было.
            "hints": "",
            "fallbacks": [],
            "engine": "gigaam",
            "empty_segments": empty,
            **extra,
        },
    }


def unload_ollama(base_url: str, *, opener=None) -> list[str]:
    """Выгружает все модели, которые Ollama держит в видеопамяти.

    То же, что делает minuteforge перед распознаванием, только без его
    настроек: имён моделей здесь не знают, поэтому спрашивают у Ollama, что
    загружено (/api/ps), и просят выгрузить каждую (keep_alive: 0).
    """
    import urllib.request

    opener = opener or urllib.request.urlopen
    root = base_url.rstrip("/")
    try:
        with opener(f"{root}/api/ps", timeout=5) as response:
            loaded = json.loads(response.read().decode("utf-8")).get("models") or []
    except Exception:
        return []
    freed = []
    for item in loaded:
        name = item.get("model") or item.get("name")
        if not name:
            continue
        request = urllib.request.Request(
            f"{root}/api/generate",
            data=json.dumps({"model": name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener(request, timeout=30):
                freed.append(name)
        except Exception:
            pass
    return freed


def find_audio(cache: Path) -> Path:
    found = sorted(p for p in cache.iterdir() if p.suffix.lower() in _AUDIO)
    if len(found) != 1:
        raise SystemExit(
            f"В {cache} ждали ровно один .wav, извлечённый minuteforge, а нашли "
            f"{len(found)}. Сначала прогоните запись whisperX в основном окружении "
            "или укажите звук явно: --audio."
        )
    return found[0]


def run(variant: str, audio: Path, cache: Path, *, device: str, batch_size: int,
        force: bool) -> Path:  # pragma: no cover — нужны gigaam и видеокарта
    import gigaam
    import torch

    target = cache / model_dir(minuteforge_name(variant)) / "transcription.json"
    if target.exists() and not force:
        print(f"{variant}: уже есть {target} — пропускаю (--force, чтобы заново)")
        return target

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = gigaam.load_model(variant, device=device)
    try:
        result = model.transcribe_longform(str(audio), fr_batch_size=batch_size)
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    extra = {
        "elapsed_s": round(time.perf_counter() - started, 1),
        "gigaam_version": getattr(gigaam, "__version__", ""),
        "torch_version": torch.__version__,
    }
    if device.startswith("cuda"):
        extra["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

    value = to_transcription(result.segments, variant=variant, batch_size=batch_size, extra=extra)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Внутри minuteforge следующие шаги этой модели лежат там же. Если они
    # остались от прошлой расшифровки, они ей и принадлежат — убираем.
    for stale in ("aligned.json", "segments.json"):
        (target.parent / stale).unlink(missing_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = value[ASR_META]
    print(
        f"{variant}: кусков {len(value['segments'])} (пустых {meta['empty_segments']}), "
        f"{meta['elapsed_s']} с, пик видеопамяти {meta.get('peak_vram_gb', '—')} ГБ -> {target}"
    )
    return target


def main(argv=None) -> int:  # pragma: no cover — нужны gigaam и видеокарта
    # Консоль Windows — cp866: без этого скрипт падает на первом же тире.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target", type=Path,
                        help="запись (видео или звук, там, где лежит) или её папка шагов")
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"),
                        help="work_dir из config.yaml, если он не data/work")
    parser.add_argument("--audio", type=Path, default=None,
                        help="звук, если это не .wav в папке шагов (например, исходный .wav)")
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None,
                        help="какой вариант; по умолчанию оба")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="сколько кусков за раз; уменьшить, если не хватит видеопамяти")
    parser.add_argument("--force", action="store_true", help="распознать заново")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="откуда выгрузить языковые модели перед стартом; пусто — не трогать")
    args = parser.parse_args(argv)

    if args.ollama_url:
        freed = unload_ollama(args.ollama_url)
        if freed:
            print("Выгружены из видеопамяти модели Ollama:", ", ".join(freed))

    import os

    import torch

    if args.target.is_dir():
        cache, record = args.target, None
    elif args.target.is_file():
        record = args.target
        cache = cache_for_record(record, args.work_dir.resolve())
    else:
        raise SystemExit(f"Нет ни записи, ни папки: {args.target}")
    if not cache.is_dir():
        raise SystemExit(
            f"Нет папки шагов {cache}. Сначала распознайте запись whisperX в "
            "основном окружении (minuteforge recognize, из папки проекта) — он "
            "извлечёт звук. Или --work-dir не тот."
        )
    print(f"Папка шагов: {cache}")
    if not os.getenv("HF_TOKEN"):
        print("Внимание: нет HF_TOKEN — разбиение записи не скачает "
              "pyannote/segmentation-3.0, если её нет в локальном кэше.")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("torch не видит CUDA: поставлена сборка без неё (см. начало файла)")
        free, total = torch.cuda.mem_get_info()
        print(f"Видеопамяти свободно {free / 1024 ** 3:.1f} ГБ из {total / 1024 ** 3:.1f}")
        if free < 4 * 1024 ** 3:
            print("Свободно меньше 4 ГБ даже после выгрузки Ollama — видеопамять "
                  "кем-то занята (nvidia-smi покажет кем). Замеры времени и памяти "
                  "с whisperX так не сравнить.")

    audio = args.audio
    if audio is None:
        extracted = [p for p in cache.iterdir() if p.suffix.lower() in _AUDIO]
        # Звук извлекается только из видео; запись-звук читается как есть.
        audio = record if not extracted and record is not None else find_audio(cache)
    for variant in args.variant or VARIANTS:
        run(variant, audio, cache, device=args.device,
            batch_size=args.batch_size, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
