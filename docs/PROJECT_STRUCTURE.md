# Структура проекта

```
fix_telemetry.bat            запуск: перетащить DJI MP4 → NAME_telemetry_fixed.mp4 рядом с видео
requirements.txt             numpy, scipy, opencv-python, matplotlib
src/                         рабочий код
  main.py                    точка входа: измерение → исправление → проверка → очистка
  measure_rotation.py        один проход по видео: вращение камеры по картинке (кэш NAME_image.npz)
  fix_pipeline.py            сборка исправленной телеметрии и контроля, отчёт
  verify_fix.py              суд любого sidecar по картинке без рендера
  plot_report.py             PNG‑графики телеметрия/картинка/исправлено (main.py --plots)
  lenscal.py                 профиль линзы Gyroflow для измерения (--lens) и диагностический подбор фокуса
  timing.py                  покадровый тайминг (константа измеряется, экспозиция покадрово)
  imagerot.py                измерения по картинке: LK‑трекинг, Кабш, аффинный поток, гомография
  rotmath.py                 кватернионная математика (slerp, logv, поворотные векторы, окна)
  dji_o4.py                  разбор дорожки djmd, CLI extract/dump/sidecar/gcsv/patch/verify
  sidecar.py                 сборка MP4 только с телеметрией (данные движения для Gyroflow)
  mp4parse.py, pb.py         ISO‑BMFF и protobuf без схем
  quat.py                    скалярные кватернионы, преобразование DJI → Gyroflow
  telemetry.py               загрузчик .npz для работы с дампом в Python
  denoise.py, rollfix.py     винеровские кривые и исправление крена (используются конвейером)
  align.py                   старое выравнивание по опоре dbgi (режим --timing dbgi для A/B)
  trim_video.py              обрезка видео с сохранением всех дорожек телеметрии
  video_diff.py              покадровая разница двух видео
research/                    исследовательские скрипты (исторические, не нужны для запуска)
  session/                   сессионный конвейер: сравнение рендера с исходником
    compare_stabilization_telemetry.py, analyze_telemetry_outliers.py,
    create_telemetry_bugfix.py, evaluate_telemetry_fix.py
  selective.py, fullrate_trial.py, variants.py        локальные пробы и A/B‑наборы
  measure_clip.py, validate_events.py                 прежнее измерение крена и его проверка
  track_telemetry.py, fit_tracking_correction.py      трекинг с essential matrix
  trace_wobble.py, bridge_wobble.py                   разбор события 55 с на Lite
  gyroflow_output_check.py, analyze_gyroflow_output.py экспорт траектории Gyroflow
  verify_selective.py, mp4_dump.py, pbdump.py         проверки и диагностика
lens_profiles/               профили линз Gyroflow для --lens (Flywoo O4 Wide для Lite)
examples/                    два куска с телеметрией, run.bat, результат, графики, README с числами
tests/                       python -m unittest discover -s tests
docs/                        HANDOFF.md (история и текущее состояние), заметки по утилитам
external/                    исходники Gyroflow и telemetry-parser (не в репозитории, см. external/README.md)
artifacts/                   результаты (не в репозитории)
```

Запуск:

```bat
fix_telemetry.bat "F:\36\video.MP4"
python src\main.py "F:\36\video.MP4" --keep
python -m unittest discover -s tests
```

Исследовательские скрипты запускаются из корня, например
`python research/selective.py <video> -o <папка> --window 53:58`; они сами
добавляют `src/` в путь импорта.
