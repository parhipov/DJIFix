"""Покадровая абсолютная разница двух видео, без звука.

Запуск без аргументов использует FILE_1 и FILE_2 ниже.
Или: python src/video_diff.py "before.mp4" "after.mp4" --gain 4
"""

FILE_1 = r"F:\36\54-56-DJI_20260517170526_0020_D_stabilized.mp4"
FILE_2 = r"F:\36\54-56-DJI_20260517170526_0020_D_stabilized_1.mp4"

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2


def ffmpeg_executable():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    local_deps = Path(__file__).resolve().parent.parent / '.cache' / 'video_diff'
    if local_deps.is_dir():
        sys.path.insert(0, str(local_deps))
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ValueError('Нужен FFmpeg: установите pip install imageio-ffmpeg.') from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def difference(frame_a, frame_b, gain=1., threshold=0):
    """abs(A-B); совпадения чёрные, gain усиливает различия без переполнения."""
    diff = cv2.absdiff(frame_a, frame_b)
    if threshold:
        # Threshold applies to original differences, before amplification.
        _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)
    if gain != 1.:
        diff = cv2.convertScaleAbs(diff, alpha=gain)
    return diff


def compare(file_1, file_2, output=None, gain=1., threshold=0, downscale=4, crf=30):
    if not isinstance(downscale, int) or downscale < 1:
        raise ValueError('Масштаб должен быть целым числом >= 1.')
    if not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError('CRF должен быть целым числом от 0 до 51.')
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('Усиление должно быть положительным конечным числом.')
    if not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError('Порог должен быть целым числом от 0 до 255.')
    sources = [Path(file_1).resolve(), Path(file_2).resolve()]
    for source in sources:
        if not source.is_file():
            raise ValueError(f'Файл не найден: {source}')
    output = (Path(output) if output else Path(__file__).resolve().parent.parent /
              'artifacts' / 'diff' / (sources[0].stem + f'_diff_small{downscale}.mp4')).resolve()
    if output in sources:
        raise ValueError('Выходной файл совпадает с одним из исходников.')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Для выходного файла укажите расширение .mp4.')
    if output.exists():
        raise ValueError(f'Файл уже существует: {output}. Укажите другое имя через -o.')

    caps = [cv2.VideoCapture(str(p)) for p in sources]
    writer, temporary, encoder_log = None, None, None
    try:
        for p, cap in zip(sources, caps):
            if not cap.isOpened():
                raise ValueError(f'Не удалось открыть видео: {p}')
        dimensions = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps]
        rates = [c.get(cv2.CAP_PROP_FPS) for c in caps]
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        if dimensions[0] != dimensions[1] or min(dimensions[0]) <= 0:
            raise ValueError(f'Разные или некорректные размеры кадров: {dimensions}.')
        if (any(not math.isfinite(fps) or fps <= 0 for fps in rates)
                or not math.isclose(*rates, rel_tol=1e-5, abs_tol=1e-4)):
            raise ValueError(f'Разная или некорректная частота кадров: {rates}.')
        if all(n > 0 for n in counts) and counts[0] != counts[1]:
            raise ValueError(f'Разное количество кадров: {counts}. Нужны одинаковые отрезки.')

        # H.264 yuv420p needs even dimensions; round down if necessary.
        output_size = tuple((n // downscale) // 2 * 2 for n in dimensions[0])
        if min(output_size) < 2:
            raise ValueError('Слишком сильное уменьшение для этих размеров.')
        executable = ffmpeg_executable()
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix='_diff_', suffix='.mp4', dir=output.parent)
        os.close(fd)
        temporary = Path(path)
        encoder_log = tempfile.TemporaryFile()
        writer = subprocess.Popen([
            executable, '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-video_size', f'{output_size[0]}x{output_size[1]}',
            '-framerate', str(rates[0]), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        print(f'{output_size[0]}x{output_size[1]}, {rates[0]:g} fps; H.264 CRF {crf}; '
              f'усиление {gain:g}, порог {threshold}', flush=True)
        done = 0
        while True:
            ok_a, a = caps[0].read()
            ok_b, b = caps[1].read()
            if not ok_a or not ok_b:
                if ok_a != ok_b:
                    raise ValueError(f'Один файл закончился или не декодируется на кадре {done}.')
                break
            if a.shape != b.shape or (a.shape[1], a.shape[0]) != dimensions[0]:
                raise ValueError(f'Изменились размеры кадра {done}.')
            diff = difference(a, b, gain, threshold)
            if output_size != dimensions[0]:
                diff = cv2.resize(diff, output_size, interpolation=cv2.INTER_AREA)
            writer.stdin.write(diff.tobytes())
            done += 1
            if done % 50 == 0:
                print(f'Кадров: {done}' + (f' / {counts[0]}' if counts[0] > 0 else ''), flush=True)
        if not done or any(n > 0 and done != n for n in counts):
            raise ValueError(f'Прочитано {done} кадров, заявлено {counts}; проверьте файлы.')
        writer.stdin.close()
        if writer.wait() != 0:
            encoder_log.seek(0)
            raise ValueError('Ошибка FFmpeg: ' + encoder_log.read().decode('utf-8', errors='replace'))
        writer = None
        # Detect output truncation before publishing the finished file.
        check = cv2.VideoCapture(str(temporary))
        try:
            if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != done:
                raise ValueError('Проверка выходного видео не прошла.')
        finally:
            check.release()
        temporary.rename(output)
        temporary = None
        print(f'Готово: {output}\nСравнено {done} кадров ({done / rates[0]:.2f} с).', flush=True)
        return output
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            if writer.poll() is None:
                writer.kill()
            writer.wait()
            if writer.stdin is not None and not writer.stdin.closed:
                writer.stdin.close()
        if encoder_log is not None:
            encoder_log.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', metavar='VIDEO')
    parser.add_argument('-o', '--output', help='выходной MP4 (по умолчанию artifacts/diff/<имя>_diff_small4.mp4)')
    parser.add_argument('--downscale', type=int, default=4,
                        help='уменьшить ширину и высоту в N раз (по умолчанию 4)')
    parser.add_argument('--crf', type=int, default=30,
                        help='сжатие H.264, 0–51: выше — меньше файл (по умолчанию 30)')
    parser.add_argument('--gain', type=float, default=1., help='усиление разницы, например 4 (по умолчанию 1)')
    parser.add_argument('--threshold', type=int, default=0,
                        help='обнулить разницу каналов <= порога, до усиления (по умолчанию 0)')
    args = parser.parse_args()
    if len(args.files) not in (0, 2):
        parser.error('Укажите ровно два видео или задайте FILE_1/FILE_2 в начале скрипта.')
    try:
        compare(*(args.files or [FILE_1, FILE_2]), args.output, args.gain, args.threshold,
                args.downscale, args.crf)
    except (ValueError, OSError, cv2.error) as exc:
        raise SystemExit(f'Сравнение отменено: {exc}')


if __name__ == '__main__':
    main()
