#!/usr/bin/env python3
"""Подготовка материалов совещания до прихода видео.

Запуск:  python3 подготовка.py <папка-с-файлами-секретаря> [-o <папка-вывода>]

Делает три вещи, все — чистый парсинг, без интернета и без LLM:
  1. docx -> txt          (чтобы pipeline вообще увидел текст)
  2. xlsx формы -> реестр участников (регион / ФИО / должность) + список фамилий
  3. txt -> новые кандидаты в глоссарий (то, чего в постоянном глоссарии ещё нет)
  4. постоянный глоссарий -> initial_prompt.txt для whisper (с учётом лимита)

Только стандартная библиотека Python 3.7+. Работает на macOS, Windows, Linux.
"""
import argparse, csv, io, os, re, sys, zipfile
import xml.etree.ElementTree as ET

# Windows: cmd.exe в русской локали работает в cp866, и там нет ни длинного тире,
# ни кавычек-ёлочек — print на них падает с UnicodeEncodeError раньше, чем скрипт
# успеет что-либо сделать. Переводим вывод в UTF-8; если поток не переключаемый
# (перенаправлен в файл на старом Python), заменяем непредставимое.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def meetings_dir():
    """Папка заседаний — та, что задана настройкой ``meetings_dir``.

    Скрипт зовут и из приложения, и руками из консоли, и настройки при этом
    читаются те же самые: заседания переехали на общий диск — переехал и
    глоссарий, отдельно о нём помнить не надо. Настройки не прочлись —
    остаётся прежнее место, рядом с приложением.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    where = os.path.join('data', 'vks')
    try:
        sys.path.insert(0, repo)
        from minuteforge.config import Settings, CONFIG_FILE, CONFIG_EXAMPLE
        own = os.path.join(repo, CONFIG_FILE)
        where = str(Settings.load(
            own if os.path.exists(own) else os.path.join(repo, CONFIG_EXAMPLE)
        ).meetings_dir)
    except Exception:
        pass
    return where if os.path.isabs(where) else os.path.join(repo, where)


def docx_to_text(path):
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read('word/document.xml'))
    out = []
    for p in root.iter(W + 'p'):
        line = ''.join(t.text or '' for t in p.iter(W + 't'))
        out.append(line)
    return '\n'.join(out)


def _colnum(ref):
    s = re.match(r'[A-Z]+', ref).group()
    n = 0
    for ch in s:
        n = n * 26 + ord(ch) - 64
    return n


def xlsx_rows(path):
    """Читает первый лист. Поддерживает и sharedStrings, и inline-строки."""
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        shared = [''.join(t.text or '' for t in si.iter(S + 't'))
                  for si in ET.fromstring(z.read('xl/sharedStrings.xml')).iter(S + 'si')]
    sheet = next(n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter(S + 'row'):
        cells = {}
        for c in row.iter(S + 'c'):
            v, isx, val = c.find(S + 'v'), c.find(S + 'is'), ''
            if c.get('t') == 's' and v is not None and shared:
                val = shared[int(v.text)]
            elif isx is not None:
                val = ''.join(t.text or '' for t in isx.iter(S + 't'))
            elif v is not None:
                val = v.text or ''
            if val.strip():
                cells[_colnum(c.get('r'))] = val.strip()
        if cells:
            rows.append(cells)
    return rows


def roster(rows):
    """Экспорт формы регистрации -> [(организация, ФИО, должность)].

    Форма кладёт участников блоками по 4 колонки: ФИО / должность / контакт / телефон,
    поэтому должность ищется в колонке справа от ФИО.
    """
    hdr = rows[0]
    org_i = next((i for i, v in hdr.items() if v.startswith('ФОИВ/')), None)
    fio_i = sorted(i for i, v in hdr.items() if v.startswith('Введите ФИО участника'))
    if org_i is None or not fio_i:
        return []
    people = []
    for r in rows[1:]:
        org = r.get(org_i, '').strip()
        if not org:
            continue
        for fi in fio_i:
            fio = r.get(fi, '').strip()
            if fio:
                people.append((org, fio, r.get(fi + 1, '').strip()))
    return people


TERM_RE = re.compile(
    r'«[^»]{3,45}»'
    r'|\b(?:КПО|МСК|РДФ|RDF|ТКО|ТСОО|ППК|РЭО|ГГЭ|КЭР|СМР|МЭО)\b'
    r'|\b[А-ЯЁ][а-яё]+(?:ский|ская|ское)\s+(?:район|округ|кожуун)'
    r'|\bкожуун\w*|\bинсинератор\w*|\bрегоператор\w*|\bэкотехнопарк\w*'
)


def glossary_candidates(texts):
    hits = {}
    for t in texts:
        for m in TERM_RE.findall(t):
            k = m.strip('«»').strip()
            if not k:
                continue
            key = _norm(k)
            cnt, spelling = hits.get(key, (0, k))
            hits[key] = (cnt + 1, spelling)
    return sorted(((sp, c) for c, sp in hits.values()), key=lambda kv: (-kv[1], kv[0]))


def load_glossary(path):
    """Читает постоянный глоссарий: {раздел: [термины]} в порядке следования разделов.

    Возвращает две карты — термины (строки «- ») и текст подсказки (строки «> »).
    Если файла нет, обе пустые: без глоссария подготовка обязана продолжиться,
    а не упасть. Раньше здесь возвращалась одна пустая карта, и вызов
    `sections, prose = load_glossary(...)` ронял интерфейс на ValueError.
    """
    if not path or not os.path.exists(path):
        return {}, {}
    sections, prose, cur = {}, {}, None
    for line in io.open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('## '):
            cur = line[3:].strip()
            sections[cur] = []
            prose[cur] = []
        elif line.startswith('- ') and cur:
            sections[cur].append(line[2:].strip())
        elif line.startswith('>') and cur:
            prose[cur].append(line.lstrip('> ').strip())
    return sections, prose


def _norm(s):
    """Нормализация для сравнения: нижний регистр, единое тире, без кавычек и лишних пробелов."""
    s = s.lower().replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
    s = re.sub(r'[«»"\u2018\u2019]', '', s)
    s = re.sub(r'[\s-]+', ' ', s)
    return s.strip(' .,;:')


def aliases(term):
    """Из строки глоссария достаёт все написания, под которыми термин может встретиться.

    «РДФ / RDF» -> рдф, rdf | «ППК «РЭО»» -> ппк рэо, ппк, рэо
    «МЭО (Мордовский...)» -> обе части | «слышится → правильно» -> обе стороны
    """
    out = []
    for side in re.split(r'\s*\u2192\s*', term):
        # отрезаем пояснение после тире: «КПО — комплекс по переработке отходов»
        side = re.split(r'\s+[-\u2014\u2013]\s+', side)[0]
        for alt in re.split(r'\s*/\s*', side):
            inner = re.findall(r'\(([^)]*)\)', alt)
            outer = re.sub(r'\([^)]*\)', ' ', alt)
            for piece in [outer] + inner:
                n = _norm(piece)
                if n:
                    out.append(n)
                # аббревиатуры внутри строки — отдельными ключами
                for abbr in re.findall(r'\b[А-ЯЁA-Z]{2,}\b', piece):
                    out.append(_norm(abbr))
    return [a for a in out if a]


def known_terms(sections):
    known = set()
    for terms in sections.values():
        for t in terms:
            known.update(aliases(t))
    return known


def is_known(term, known):
    """Совпадение по нормализованной форме или по префиксу — чтобы падежи
    («инсинераторов», «кожууна», «регоператора») не считались новыми терминами."""
    n = _norm(term)
    if not n:
        return True
    if n in known:
        return True
    for k in known:
        if len(k) >= 6 and (n.startswith(k) or k.startswith(n)) and abs(len(n) - len(k)) <= 4:
            return True
    return False


GREET_RE = re.compile(r'Уважаем\w+\s+((?:[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)(?:\s*,\s*[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)*)')
NAME_IN_FILE_RE = re.compile(r'[А-ЯЁ]\.\s?[А-ЯЁ]\.\s*([А-ЯЁ][а-яё]+)')


def speaker_candidates(texts, filenames):
    """Кто на этой ВКС ведёт и докладывает — по обращению в тезисах и по именам
    в названиях файлов секретаря. Это подсказка человеку, не истина: имя в названии
    стоит в родительном падеже («А.Н. Головановой»), и выправить его должен он."""
    found = []
    for t in texts:
        for m in GREET_RE.findall(t):
            found.append(m.strip())
    for name in filenames:
        for m in NAME_IN_FILE_RE.findall(name):
            found.append(m)
    seen, out = set(), []
    for f in found:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def collect_people(vks_dir):
    """Накопительный справочник людей: объединение участники.csv всех заседаний.

    Нужен потому, что форма регистрации собирает представителей регионов, а ведущие
    и докладчики из Минприроды в неё не подают: на 17.09 ни Хатуова, ни Головановой
    в списке нет, а в списках 27.08 и 10.09 они есть. Чем больше заседаний прошло,
    тем меньше остаётся неразобранных фамилий."""
    people = {}
    if not os.path.isdir(vks_dir):
        return people
    for name in sorted(os.listdir(vks_dir)):
        path = os.path.join(vks_dir, name, 'участники.csv')
        if not os.path.exists(path):
            continue
        with io.open(path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, delimiter=';'):
                if len(row) >= 1 and row[0] and row[0] != 'ФИО':
                    people[row[0].strip()] = (row[1].strip() if len(row) > 1 else '',
                                              row[2].strip() if len(row) > 2 else '')
    return people


def resolve_speakers(candidates, people):
    """Кандидат -> фамилия в именительном падеже, по накопленному справочнику.

    Две формы: «Джамбулат Хизирович» (имя и отчество из обращения) и «Головановой»
    (родительный падеж из названия файла). Сверка идёт по отдельным словам ФИО,
    иначе «Александр Александрович» ложно совпадает с «Чакыров Алексей Александрович».
    """
    index = [(fio.split(), fio) for fio in people]
    resolved, guesses = [], []
    for cand in candidates:
        for part in re.split(r'\s*,\s*', cand):
            words = part.split()
            hits = []
            if len(words) == 2:
                # Имя-отчество из обращения — НЕ подставляется само. Председательствующий
                # в списки регионов не подаётся, и единственное совпадение запросто
                # окажется однофамильцем: «Александр Александрович» дал «Алексеева»,
                # тогда как речь была о Козлове. Такое уходит в подсказку человеку.
                hits = [p[0] for p, _ in index if len(p) >= 3
                        and p[1] == words[0] and p[2] == words[1]]
                guesses.append('%s -> %s' % (part, ', '.join(hits) if hits
                                             else 'в справочнике нет'))
                continue
            if len(words) == 1:
                # Фамилия в косвенном падеже — сверка по основе, это надёжно:
                # «Головановой» и «Голованова» совпадают на 9 буквах.
                w = words[0]
                hits = [p[0] for p, _ in index
                        if len(os.path.commonprefix([p[0].lower(), w.lower()])) >= 5
                        and abs(len(p[0]) - len(w)) <= 3]
                hits = sorted(set(hits))
            if len(hits) == 1 and hits[0] not in resolved:
                resolved.append(hits[0])
            elif len(hits) > 1:
                guesses.append('%s -> несколько: %s' % (part, ', '.join(hits)))
            elif not hits:
                guesses.append('%s -> в справочнике нет' % part)
    return resolved, guesses


def read_speakers(meeting_dir):
    """<заседание>/speakers.txt — по имени в строке, пустые и # игнорируются."""
    path = os.path.join(meeting_dir, 'speakers.txt')
    if not os.path.exists(path):
        return []
    out = []
    for line in io.open(path, encoding='utf-8'):
        line = line.strip()
        if line and not line.startswith('#'):
            out.append(line)
    return out


def build_asr_hints(sections, budget_chars, speakers=None):
    """Готовит файл для asr_hints_file.

    Берётся раздел «Подсказка распознаванию» как есть — связной фразой. Перечнем
    подсказку не собираем: на записи штаба список из 54 слов ухудшил расшифровку
    (см. asr_hints_file в config.example.yaml). Обрезка — по целым предложениям.
    """
    body = None
    for name, lines in sections.items():
        if name.startswith('Подсказка распознаванию'):
            body = ' '.join(l for l in lines if l)
            break
    if not body:
        return None
    # метка {докладчики} — переменная часть: кто ведёт и докладывает именно на этой ВКС
    if speakers:
        body = body.replace('{докладчики}', 'Докладывают ' + ', '.join(speakers) + '.')
    else:
        body = body.replace('{докладчики}', '').replace('  ', ' ')
    sents = re.split(r'(?<=\.)\s+', body)
    out, used = [], 0
    for sent in sents:
        if used + len(sent) + 1 > budget_chars:
            break
        out.append(sent)
        used += len(sent) + 1
    return ' '.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='папка с файлами секретаря')
    ap.add_argument('-o', '--out', default=None, help='папка вывода (по умолчанию <src>/prep)')
    ap.add_argument('-g', '--glossary', default=None,
                    help='постоянный глоссарий (по умолчанию glossary.md в папке заседаний)')
    ap.add_argument('--budget', type=int, default=550,
                    help='лимит initial_prompt в символах (whisper режет по ~224 токенам)')
    a = ap.parse_args()
    out = a.out or os.path.join(a.src, 'prep')
    os.makedirs(out, exist_ok=True)
    # Глоссарий держит настоящие ФИО, поэтому лежит он там же, где заседания:
    # внутри приложения эта папка целиком под запретом в .gitignore, а на общем
    # диске за доступом к ней следит тот, кто раздаёт права. Рядом со скриптом
    # (scripts/) ему не место: та папка отслеживается, и файл уехал бы в
    # публичный репозиторий.
    gpath = a.glossary or os.path.join(meetings_dir(), 'glossary.md')
    sections, prose = load_glossary(gpath)
    if sections:
        print('глоссарий %s — %d разделов, %d терминов'
              % (os.path.basename(gpath), len(sections), sum(len(v) for v in sections.values())))
    else:
        print('глоссарий не найден (%s) — кандидаты будут показаны целиком' % gpath)

    texts, people, names_seen = [], [], []
    for name in sorted(os.listdir(a.src)):
        path = os.path.join(a.src, name)
        if not os.path.isfile(path) or name.startswith('~$'):
            continue
        try:
            if name.lower().endswith('.docx'):
                txt = docx_to_text(path)
                texts.append(txt)
                names_seen.append(name)
                dst = os.path.join(out, os.path.splitext(name)[0] + '.txt')
                io.open(dst, 'w', encoding='utf-8').write(txt)
                print('txt   %s (%d симв.)' % (name, len(txt)))
            elif name.lower().endswith('.xlsx'):
                p = roster(xlsx_rows(path))
                if p:
                    people.extend(p)
                    print('реестр %s (%d участников)' % (name, len(p)))
                else:
                    print('пропуск %s — не похоже на форму регистрации' % name)
        except Exception as e:
            print('ОШИБКА %s: %s' % (name, e), file=sys.stderr)

    if people:
        orgs = {p[0] for p in people}
        with io.open(os.path.join(out, 'participants.md'), 'w', encoding='utf-8') as f:
            f.write('# Участники\n\nОрганизаций: %d · участников: %d\n' % (len(orgs), len(people)))
            cur = None
            for org, fio, post in sorted(people):
                if org != cur:
                    f.write('\n## %s\n' % org)
                    cur = org
                f.write('- **%s** — %s\n' % (fio, post or 'должность не указана'))
        # участники.csv по конвенции заседаний: рядом с протоколом и стенограммой,
        # колонки как у прошлых ВКС — ФИО;Должность;Организация. BOM, иначе Excel
        # в русской локали открывает кракозябрами.
        conv = os.path.join(os.path.dirname(os.path.abspath(out.rstrip(os.sep))), 'участники.csv')
        with io.open(conv, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(['ФИО', 'Должность', 'Организация'])
            w.writerows((fio, post or '', org) for org, fio, post in sorted(people))
        print('-> %s (%d строк)' % (os.path.basename(conv), len(people)))
        fams = sorted({p[1].split()[0] for p in people if p[1].split()})
        io.open(os.path.join(out, 'surnames.txt'), 'w', encoding='utf-8').write(', '.join(fams))
        print('-> participants.md, surnames.txt (%d фамилий)' % len(fams))

    if texts:
        sp = speaker_candidates(texts, names_seen)
        with io.open(os.path.join(out, 'speakers_candidates.txt'), 'w', encoding='utf-8') as f:
            f.write('# Кто ведёт и докладывает на этой ВКС — предположение по обращению\n')
            f.write('# в тезисах и по именам в названиях файлов. Проверь, выправи падеж\n')
            f.write('# и перенеси в <заседание>/speakers.txt — оттуда они попадут в подсказку.\n\n')
            for k in sp:
                f.write('%s\n' % k)
        print('-> speakers_candidates.txt (%d кандидатов)' % len(sp))

        # Заготовка speakers.txt — только если его ещё нет: перезаписать значило бы
        # затереть выправленные руками фамилии. Кандидаты идут закомментированными,
        # потому что падеж у них родительный («Головановой»), а нужен именительный.
        meeting = os.path.dirname(os.path.abspath(out.rstrip(os.sep)))
        draft = os.path.join(meeting, 'speakers.txt')
        if not os.path.exists(draft):
            with io.open(draft, 'w', encoding='utf-8') as f:
                f.write('# Кто ведёт и докладывает на этой ВКС. По фамилии в строке,\n')
                f.write('# в именительном падеже. Строки с # не читаются.\n')
                f.write('#\n')
                f.write('# Ниже — что нашлось в тезисах. Раскомментируй нужное, выправи\n')
                f.write('# падеж, лишнее удали. Потом запусти скрипт ещё раз — фамилии\n')
                f.write('# попадут в asr_hints.txt.\n')
                f.write('#\n')
                for k in sp:
                    f.write('# %s\n' % k)
            print('-> speakers.txt — заготовка создана, впиши фамилии и запусти ещё раз')

        cand = glossary_candidates(texts)
        known = known_terms(sections)
        fresh = [(k, v) for k, v in cand if not is_known(k, known)]
        with io.open(os.path.join(out, 'glossary_new.txt'), 'w', encoding='utf-8') as f:
            f.write('# Новое в материалах — чего ещё нет в постоянном глоссарии.\n')
            f.write('# Вычеркни мусор, остальное допиши в %s\n' % os.path.basename(gpath))
            f.write('# Формат: <частота>  <термин>\n\n')
            for k, v in fresh:
                f.write('%4d  %s\n' % (v, k))
        print('-> glossary_new.txt (%d новых из %d найденных)' % (len(fresh), len(cand)))

    if sections:
        meeting_dir = os.path.dirname(os.path.abspath(out.rstrip(os.sep)))
        speakers = read_speakers(meeting_dir)
        if speakers:
            print('докладчики из speakers.txt: %s' % ', '.join(speakers))
        else:
            есть = os.path.exists(os.path.join(meeting_dir, 'speakers.txt'))
            print('! speakers.txt %s — подсказка пойдёт без фамилий'
                  % ('пуст: все строки закомментированы' if есть else 'не найден'))
        hints = build_asr_hints(prose, a.budget, speakers)
        if hints:
            io.open(os.path.join(out, 'asr_hints.txt'), 'w', encoding='utf-8').write(hints)
            print('-> asr_hints.txt (%d симв. из лимита %d) -> asr_hints_file'
                  % (len(hints), a.budget))
        else:
            print('! в глоссарии нет раздела «Подсказка распознаванию» — файл для asr_hints_file не собран')


if __name__ == '__main__':
    main()
