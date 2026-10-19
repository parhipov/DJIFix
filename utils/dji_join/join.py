"""Склейка сегментов DJI O4 Pro (камера режет запись на куски ~4 минуты).

Запуск без аргументов обходит FOLDER и все её подпапки.
Или: python join.py "F:/clips" [--dry-run]

Сегменты одной записи идут подряд по номеру (…_0020_D, …_0021_D, …), и каждый
следующий начинается там, где кончился предыдущий (по времени из имени файла
и длительности). Склейка — mp4_merge из Gyroflow (github.com/gyroflow/mp4-merge):
без перекодирования, все дорожки (видео, телеметрия DJI djmd/dbgi) сохраняются,
это тот же код, которым Gyroflow сам склеивает разбитые записи.
Результат кладётся рядом: <имя первого файла>_joined.mp4; исходники после удачной
склейки переименовываются в _<имя>, чтобы их было легко проверить и удалить.
"""

FOLDER = r"f:\завод\паво20про"
MAX_GAP_SECONDS = 3.  # допустимый зазор между концом сегмента и началом следующего

MP4_MERGE_URL = 'https://github.com/gyroflow/mp4-merge/releases/download/v0.1.11/mp4_merge-windows64.exe'

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

import av

SEGMENT = re.compile(r'^DJI_(\d{14})_(\d{4})_D\.MP4$', re.IGNORECASE)
HERE = Path(__file__).resolve().parent


def mp4_merge_executable():
    executable = HERE / 'mp4_merge.exe'
    if not executable.exists():
        print(f'Скачиваю mp4_merge: {MP4_MERGE_URL}', flush=True)
        partial = executable.with_suffix('.part')
        urllib.request.urlretrieve(MP4_MERGE_URL, partial)
        os.replace(partial, executable)
    return executable


def duration_seconds(path):
    with av.open(str(path)) as container:
        if container.duration is None:
            raise ValueError(f'Не удалось прочитать длительность: {path}')
        return container.duration / av.time_base


def scan(folder):
    """Сегменты папки (без подпапок): [(номер, начало, конец, путь)] по номеру."""
    segments = []
    for name in os.listdir(folder):
        match = SEGMENT.match(name)
        if not match:
            continue
        path = Path(folder, name)
        start = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
        end = start + timedelta(seconds=duration_seconds(path))
        segments.append((int(match.group(2)), start, end, path))
    segments.sort()
    return segments


def groups(segments, max_gap=MAX_GAP_SECONDS):
    """Цепочки: следующий номер и начало в пределах max_gap от конца предыдущего."""
    chains = []
    for number, start, end, path in segments:
        if chains:
            last_number, _, last_end, _ = chains[-1][-1]
            gap = (start - last_end).total_seconds()
            if number == last_number + 1 and abs(gap) <= max_gap:
                chains[-1].append((number, start, end, path))
                continue
        chains.append([(number, start, end, path)])
    return [[item[3] for item in chain] for chain in chains if len(chain) > 1]


def join(paths, output, executable):
    partial = output.with_name(output.name + '.tmp')
    command = [str(executable), *(str(path) for path in paths), '--out', str(partial)]
    try:
        subprocess.run(command, check=True)
        expected = sum(path.stat().st_size for path in paths)
        if not partial.exists() or partial.stat().st_size < expected * 0.999:
            raise ValueError(f'Результат подозрительно мал: {partial}')
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def hide_sources(paths):
    for path in paths:
        hidden = path.with_name('_' + path.name)
        if hidden.exists():
            raise ValueError(f'Уже существует: {hidden}')
        path.rename(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folder', nargs='?', default=FOLDER)
    parser.add_argument('--dry-run', action='store_true', help='только показать, что склеилось бы')
    parser.add_argument('--max-gap', type=float, default=MAX_GAP_SECONDS, help='зазор между сегментами, с')
    parser.add_argument('--keep-names', action='store_true', help='не переименовывать исходники в _имя')
    args = parser.parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        raise ValueError(f'Папка не найдена: {root}')
    executable = None if args.dry_run else mp4_merge_executable()

    planned = done = 0
    for folder, _, _ in os.walk(root):
        for chain in groups(scan(folder), args.max_gap):
            output = chain[0].with_name(chain[0].stem + '_joined.mp4')
            print(f'{output}\n    <- ' + ' + '.join(path.name for path in chain), flush=True)
            if output.exists():
                print('    уже есть, пропускаю')
                continue
            planned += 1
            if args.dry_run:
                continue
            join(chain, output, executable)
            if not args.keep_names:
                hide_sources(chain)
            done += 1
    print(f'Склеено {done} из {planned}.' if not args.dry_run else f'К склейке: {planned}.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f'Ошибка: {error}')
