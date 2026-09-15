"""Обрезка DJI MP4 без перекодирования. Запуск: python src/trim_video.py."""

# Меняйте только эти три параметра. Время: секунды или "ЧЧ:ММ:СС.ссс".
FILE = r"F:\36\DJI_20260905181949_0005_D.MP4"
START = "00:10"  # None — с начала файла
END = "00:15"    # None — до конца файла

import math
import struct
import tempfile
from fractions import Fraction
from pathlib import Path

import pb
from dji_o4 import read_telemetry
from mp4parse import open_mp4, raw, read, u32a


def seconds(value):
    if value is None:
        return None
    parts = str(value).strip().replace(',', '.').split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError('Время: секунды, ММ:СС или ЧЧ:ММ:СС.')
    numbers = [Fraction(p) for p in parts]
    if any(n < 0 for n in numbers) or (len(parts) > 1 and
            any(n >= 60 for n in numbers[1:])):
        raise ValueError('Некорректное время.')
    result = Fraction(0)
    for number in numbers:
        result = result * 60 + number
    return result


def box(kind, data):
    return struct.pack('>I4s', len(data) + 8, kind) + data


def table(kind, rows, fmt):
    return box(kind, struct.pack('>II', 0, len(rows)) +
               b''.join(struct.pack('>' + fmt, *r) for r in rows))


def duration_box(f, node, duration):
    data = bytearray(read(f, node))
    # Offsets within the full-box payload, including version/flags.
    offset = (28 if data[0] == 1 else 20) if node.type == b'tkhd' else (
        24 if data[0] == 1 else 16)
    struct.pack_into('>Q' if data[0] == 1 else '>I', data, offset, duration)
    return box(node.type, data)


def packet(f, track, index):
    offset, size = track._trim_offsets[index]
    f.seek(offset)
    data = f.read(size)
    if len(data) != size:
        raise ValueError('Файл оборван: не удалось прочитать пакет.')
    return data


def with_headers(first, selected, fmt):
    """Carry clip/stream protobuf fields into the new first packet verbatim."""
    frame_field = {b'djmd': 3, b'dbgi': 2}.get(fmt)
    if frame_field is None:
        return selected
    present = {n for n, _, _, _ in pb.iter_fields(selected)}
    prefix = b''.join(first[a:b] for n, _, _, (a, b) in pb.iter_fields(first)
                      if n != frame_field and n not in present)
    return prefix + selected


def inspect(f, boxes, moov, tracks):
    """Fail before writing on layouts whose timing cannot be preserved here."""
    if any(b.type in (b'moof', b'sidx', b'mfra') for b in boxes) or moov.find(b'mvex'):
        raise ValueError('Фрагментированный MP4 пока не поддерживается.')
    videos = [t for t in tracks if t.formats in ([b'hvc1'], [b'hev1'], [b'avc1'])]
    if len(videos) != 1 or not any(t.formats == [b'djmd'] for t in tracks):
        raise ValueError('Ожидается DJI MP4 с одной видеодорожкой и djmd.')
    video = videos[0]
    if len(video.stts) != 1 or video.sample_count == 0:
        raise ValueError('Поддерживается постоянная частота кадров.')
    step = Fraction(video.stts[0][1], video.timescale)
    mvhd = read(f, moov.find(b'mvhd'))
    scale = struct.unpack_from('>I', mvhd, 20 if mvhd[0] == 1 else 12)[0]
    allowed = {b'stsd', b'stts', b'stss', b'stsc', b'stsz', b'stco', b'co64'}
    for t in tracks:
        if len(t.formats) != 1 or t.formats[0] not in (b'hvc1', b'hev1', b'avc1', b'djmd', b'dbgi'):
            raise ValueError(f'Дорожка {t.id}: неподдерживаемый формат {t.formats}; ничего не удалено.')
        if (t.sample_count != video.sample_count or len(t.stts) != 1 or
                t.stts[0][0] != t.sample_count or
                Fraction(t.stts[0][1], t.timescale) != step or
                t.duration != t.sample_count * t.stts[0][1]):
            raise ValueError('Дорожки имеют разные временные сетки; обрезка отменена.')
        unsupported = {c.type for c in t.stbl.children} - allowed
        if unsupported:
            raise ValueError(f'Неподдерживаемые таблицы {unsupported}; обрезка отменена.')
        if any(desc != 1 for _, _, desc in t.stsc):
            raise ValueError('Несколько описаний пакетов не поддерживаются.')
        edts = t.trak.find(b'edts')
        if edts:
            if [c.type for c in edts.children] != [b'elst']:
                raise ValueError('Неизвестная структура edts.')
            data = read(f, edts.children[0])
            version, count = data[0], struct.unpack_from('>I', data, 4)[0]
            if version not in (0, 1) or count != 1:
                raise ValueError('Сложный edit list не поддерживается.')
            duration, media_time, rate, fraction = struct.unpack_from(
                '>Qqhh' if version else '>Iihh', data, 8)
            if media_time != 0 or (rate, fraction) != (1, 0) or duration != t.duration * scale // t.timescale:
                raise ValueError('В файле уже есть сдвиг edit list; обрезка отменена.')
        t._trim_offsets = t.sample_offsets()
        if len(t._trim_offsets) != t.sample_count:
            raise ValueError('Некорректная таблица пакетов.')
    sync = video.stbl.find(b'stss')
    keys = [n - 1 for n in u32a(read(f, sync))[2:]] if sync else list(range(video.sample_count))
    if not keys or keys[0] != 0 or keys != sorted(set(keys)) or keys[-1] >= video.sample_count:
        raise ValueError('Некорректная таблица ключевых кадров.')
    return video, step, scale, keys


def rebuild(f, node, replacements):
    if node.offset in replacements:
        return replacements[node.offset]
    # Traverse only the structural boxes we actually modify. All metadata,
    # including opaque/unknown fields and DJI thumbnails, stays byte-identical.
    if node.type in (b'moov', b'trak', b'mdia', b'minf', b'stbl'):
        return box(node.type, b''.join(rebuild(f, c, replacements) for c in node.children))
    return raw(f, node)


def verify(source, result, start, stop):
    """Check every retained packet, preserved boxes, and decoded telemetry timing."""
    f, bs, moov, tracks = open_mp4(source)
    g, out_bs, out_moov, out_tracks = open_mp4(result)
    try:
        _, step, _, _ = inspect(f, bs, moov, tracks)
        inspect(g, out_bs, out_moov, out_tracks)
        if [(t.id, t.formats) for t in tracks] != [(t.id, t.formats) for t in out_tracks]:
            raise ValueError('Проверка: изменился набор дорожек.')
        def preserved(a, b):
            changed = {b'mvhd', b'tkhd', b'mdhd', b'edts', b'stbl'}
            if [x.type for x in a.children] != [x.type for x in b.children]:
                raise ValueError('Проверка: изменился набор MP4-блоков.')
            for x, y in zip(a.children, b.children):
                if x.type in changed:
                    continue
                if x.type in (b'trak', b'mdia', b'minf'):
                    preserved(x, y)
                elif raw(f, x) != raw(g, y):
                    raise ValueError(f'Проверка: изменились метаданные {x.type}.')
        preserved(moov, out_moov)
        original_extras = [raw(f, b) for b in bs if b.type not in (b'moov', b'mdat')]
        result_extras = [raw(g, b) for b in out_bs if b.type not in (b'moov', b'mdat')]
        if original_extras != result_extras:
            raise ValueError('Проверка: изменились внешние MP4-блоки.')
        for t, out in zip(tracks, out_tracks):
            if out.sample_count != stop - start:
                raise ValueError('Проверка: неправильное число пакетов.')
            if raw(f, t.stbl.find(b'stsd')) != raw(g, out.stbl.find(b'stsd')):
                raise ValueError('Проверка: изменилось описание дорожки.')
            for j, i in enumerate(range(start, stop)):
                expected = packet(f, t, i)
                if j == 0 and start:
                    expected = with_headers(packet(f, t, 0), expected, t.formats[0])
                if expected != packet(g, out, j):
                    raise ValueError(f'Проверка: дорожка {t.id}, пакет {j} отличается.')
    finally:
        f.close()
        g.close()
    clip, samples, frames = read_telemetry(source)
    out_clip, actual, out_frames = read_telemetry(result)
    expected = [s for s in samples if start <= s.frame < stop]
    expected_frames = [r for r in frames if start <= r['frame'] < stop]
    if clip.as_dict() != out_clip.as_dict() or not actual or len(expected) != len(actual):
        raise ValueError('Проверка: параметры камеры или количество кватернионов изменились.')
    if len(expected_frames) != stop - start or len(out_frames) != len(expected_frames):
        raise ValueError('Проверка: отсутствует телеметрия кадров.')
    shift_us = (expected_frames[0]['frame_timestamp_us'] - frames[0]['frame_timestamp_us']) / clip.fps_ratio
    worst = 0.0
    for a, b in zip(expected, actual):
        if a.q_dji != b.q_dji or a.frame - start != b.frame or a.index_in_frame != b.index_in_frame:
            raise ValueError('Проверка: нарушено соответствие кадр/кватернион.')
        worst = max(worst, abs(a.t_us - shift_us - b.t_us))
    if worst > 1e-5:
        raise ValueError(f'Проверка: ошибка временных меток {worst} мкс.')
    for a, b in zip(expected_frames, out_frames):
        for key in a:
            wanted = a[key] - start if key == 'frame' else (
                a[key] - float(start * step) if key == 'video_time_s' else a[key])
            if key == 'video_time_s':
                if not math.isclose(wanted, b[key], abs_tol=1e-9):
                    raise ValueError('Проверка: нарушена временная сетка.')
            elif wanted != b[key]:
                raise ValueError(f'Проверка: изменилось поле телеметрии {key}.')
    return {'frames': len(out_frames), 'quaternions': len(actual), 'timing_error_us': worst}


def trim(source, start=None, end=None, output=None):
    source = Path(source).resolve(strict=True)
    start, end = seconds(start), seconds(end)
    f, boxes, moov, tracks = open_mp4(source)
    temporary = None
    try:
        video, step, scale, keys = inspect(f, boxes, moov, tracks)
        duration = video.sample_count * step
        start = Fraction(0) if start is None else start
        end = duration if end is None else end
        if not 0 <= start < end <= duration:
            raise ValueError(f'Нужно 0 <= начало < конец <= {float(duration):.3f} с.')
        first = max(k for k in keys if k * step <= start)
        # Whole GOPs: expand the requested interval on both sides.
        last = next((k for k in keys if k * step >= end), video.sample_count)
        begin_s, end_s = float(first * step), float(last * step)
        if output is None:
            output = source.with_name(f'{source.stem}_trim_{begin_s:.3f}-{end_s:.3f}{source.suffix}')
        output = Path(output).resolve()
        if output == source or output.exists():
            raise FileExistsError(f'Не перезаписываю существующий файл: {output}')
        print(f'Фактический интервал: {begin_s:.3f} .. {end_s:.3f} с; {last-first} кадров.', flush=True)
        replacements = {}
        movie_duration = round((last - first) * step * scale)
        replacements[moov.find(b'mvhd').offset] = duration_box(f, moov.find(b'mvhd'), movie_duration)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.stem + '.', suffix='.partial', delete=False) as out:
            temporary = Path(out.name)
            for b in boxes:
                if b.type not in (b'moov', b'mdat'):
                    out.write(raw(f, b))
            mdat_at = out.tell()
            out.write(struct.pack('>I4sQ', 1, b'mdat', 0))
            offsets = {t.id: [] for t in tracks}
            sizes = {t.id: [] for t in tracks}
            for i in range(first, last):
                for t in tracks:
                    data = packet(f, t, i)
                    if i == first and first:
                        data = with_headers(packet(f, t, 0), data, t.formats[0])
                    offsets[t.id].append(out.tell())
                    sizes[t.id].append(len(data))
                    out.write(data)
            tail = out.tell()
            out.seek(mdat_at + 8)
            out.write(struct.pack('>Q', tail - mdat_at))
            out.seek(tail)
            for t in tracks:
                for node, ticks in ((t.trak.find(b'tkhd'), movie_duration),
                                    (t.trak.find(b'mdia', b'mdhd'), (last-first)*t.stts[0][1])):
                    replacements[node.offset] = duration_box(f, node, ticks)
                edts = t.trak.find(b'edts')
                if edts:
                    replacements[edts.offset] = box(b'edts', box(b'elst',
                        struct.pack('>IIQqhh', 1 << 24, 1, movie_duration, 0, 1, 0)))
                stbl = t.stbl
                for node in stbl.children:
                    kind = node.type
                    if kind == b'stts':
                        value = table(kind, [(last-first, t.stts[0][1])], 'II')
                    elif kind == b'stsc':
                        value = table(kind, [(1, 1, 1)], 'III')
                    elif kind == b'stsz':
                        value = box(kind, struct.pack('>III', 0, 0, last-first) +
                                    struct.pack(f'>{last-first}I', *sizes[t.id]))
                    elif kind in (b'stco', b'co64'):
                        use64 = kind == b'co64' or max(offsets[t.id]) > 0xffffffff
                        value = table(b'co64' if use64 else b'stco',
                                      [(n,) for n in offsets[t.id]], 'Q' if use64 else 'I')
                    elif kind == b'stss':
                        original_keys = u32a(read(f, node))[2:]
                        value = table(kind, [(n-first,) for n in original_keys if first < n <= last], 'I')
                    else:
                        continue
                    replacements[node.offset] = value
            out.write(rebuild(f, moov, replacements))
        print('Проверяю все пакеты и синхронизацию телеметрии...', flush=True)
        report = verify(source, temporary, first, last)
        # On Windows rename refuses to overwrite a destination created meanwhile.
        temporary.rename(output)
        temporary = None
        report.update(path=str(output), start_s=begin_s, end_s=end_s)
        print(f'Готово: {output}\nПроверено: {report["frames"]} кадров, '
              f'{report["quaternions"]} кватернионов; все дорожки и метаданные сохранены.', flush=True)
        return report
    finally:
        f.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        trim(FILE, START, END)
    except (OSError, ValueError, StopIteration, struct.error) as exc:
        raise SystemExit(f'Обрезка отменена: {exc}')
