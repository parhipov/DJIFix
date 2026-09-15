# GyroGate fix: repairing DJI O4 / O4 Pro telemetry for Gyroflow

**The bug ("GyroGate").** Since February 2026 DJI ships the O4 Pro air unit with a
different gyro, the ICM‑40609‑D (marked I469D), the same part the O4 Lite has had
since launch. Footage from many of these units is fine straight from the camera
but judders once stabilized, in RockSteady and in Gyroflow alike. DJI confirmed the
sensor change and has shipped no fix. What the camera writes into the video is not
raw gyro but a *fused attitude* (quaternions in the `djmd` track), and that attitude
disagrees with what the camera actually did:

- **one‑frame spikes:** single frames report a yaw or pitch jump of up to 1.8°
  (88 °/s) that never happened, so Gyroflow yanks the frame for one frame;
- **roll jitter:** the reported roll trembles by ~0.1° from frame to frame while
  the image is smooth (≈0.01° measured from the same video), so the stabilized
  frame shivers clockwise‑counterclockwise even on a calm flight.

No smoothing setting removes a one‑frame error without smearing it onto the
neighbours, and Gyroflow itself is not at fault: it reads exactly what DJI wrote
(checked against its own CLI export, 1e‑3°).

**The fix.** This tool measures the camera's real rotation from the video itself
(OpenCV feature tracking through DJI's own lens model), repairs the telemetry
against that measurement frame by frame, and writes a small MP4 that Gyroflow
loads as external **motion data**. The video is not touched. Run
`python src/main.py clip.MP4` or drop the clip onto `fix_telemetry.bat`; the
fixed telemetry appears next to the clip.

![O4 Pro roll: DJI telemetry (red) jitters, the image (grey) is smooth, the fixed telemetry (green) follows the image](examples/o4pro_13-17s/o4pro_13-17s_roll.png)

*O4 Pro, calm flight, roll increment per frame: red is DJI's telemetry, grey is
the rotation measured from the video, green is the repaired telemetry.
More in [`examples/`](examples/). Docs below are in Russian;
[`docs/HANDOFF.md`](docs/HANDOFF.md) §0 has the technical state in English.*

**Related tools.** [DJI_04_Air_Unit_Gyro_Patcher](https://github.com/gmatocha/DJI_04_Air_Unit_Gyro_Patcher)
finds burst glitches by their magnitude and bridges them with SLERP;
[DJIGyroFix](https://github.com/kim2160/DJIGyroFix) smooths user‑selected time
ranges. Both rewrite the telemetry inside a copy of the MP4 from the telemetry
alone. This project differs in that the reference is the video: every correction
is measured against the image, single‑frame spikes are removed without touching
their neighbours, roll jitter is repaired as well, and the result is a separate
motion‑data file.

---

**Баг («GyroGate»).** С февраля 2026 DJI ставит в O4 Pro другой гироскоп,
ICM‑40609‑D (маркировка I469D), тот же, что стоит в O4 Lite с самого начала.
Видео с многих таких блоков прямо с камеры выглядит нормально, а после
стабилизации дёргается, что в RockSteady, что в Gyroflow. DJI смену сенсора
подтвердила, исправления не выпустила. В файл камера пишет не сырой гироскоп,
а *слитую ориентацию* (кватернионы в дорожке `djmd`), и эта ориентация
расходится с тем, что камера делала на самом деле:

- **одиночные выбросы:** отдельные кадры сообщают скачок рыскания или тангажа
  до 1.8° (88 °/с), которого не было, и Gyroflow дёргает кадр на один кадр;
- **дрожание крена:** телеметрия качает крен примерно на 0.1° от кадра к кадру,
  хотя картинка гладкая (≈0.01° по измерению с того же видео), и кадр мелко
  дрожит по часовой и против даже в спокойном полёте.

Никакое сглаживание не убирает ошибку в одном кадре, не размазав её на соседние,
а Gyroflow ни при чём: он читает ровно то, что записала DJI (сверено с его
собственным экспортом, 1e‑3°).

**Исправление.** Инструмент измеряет реальное вращение камеры по самому видео,
исправляет по нему телеметрию покадрово и отдаёт Gyroflow отдельным файлом
**данных движения** (Motion data). Видео не меняется.

## Быстрый старт

```bat
fix_telemetry.bat "F:\36\video.MP4"
```

или

```bash
python src/main.py "F:\36\video.MP4"
```

Результат — один файл рядом с видео: `<имя>_telemetry_fixed.mp4` (~5 МБ),
загрузить в Gyroflow: Motion data → открыть файл. Всё промежуточное удаляется;
`--keep` оставляет его рядом с видео:

| файл | что |
|---|---|
| `<имя>_00_control.mp4` | тот же тайминг, содержимое не тронуто — контроль для A/B |
| `<имя>_image.npz` | измерение вращения по картинке, кэш (~25 мин на 90 с 4K, переиспользуется) |
| `<имя>_fix_report.json` | что и на сколько исправлено, список событий с проверкой |
| `<имя>_verify.json` | суд по картинке: тайминг, дрожание крена по полосам скорости, тангаж/рыскание |

Если на камере не стоковая линза, укажите профиль Gyroflow, например для O4 Lite
с Flywoo O4 Wide: `fix_telemetry.bat "F:\clip.MP4" --lens flywoo`
(см. `lens_profiles/README.md`).

Ключи `main.py`: `--keep`, `--plots` (PNG‑графики, нужен `matplotlib`), `--lens`,
`--artifacts` (писать в `artifacts/main/<имя>/`, для разработки), `-o ПАПКА`,
`--no-image` (без прохода по видео: тайминг + винер), `--remeasure`, `--gain 0.7`,
`--r0`, `--no-events`, `--timing dbgi` (старое выравнивание для A/B), `--extract`,
`--no-verify`, `--gyroflow ПУТЬ` (Gyroflow.exe для финальной сверки; по умолчанию
переменная окружения `GYROFLOW`, затем стандартные папки установки; без Gyroflow
сверка пропускается). Тонкие параметры событий доступны в
`src/fix_pipeline.py` (см. `--help`). Зависимости: `numpy`, `scipy`, `opencv-python`
(`matplotlib` только для `--plots`).

## Что исправляется и почему

1. **Тайминг.** Gyroflow читает ориентацию кадра в момент `pts` (середина строк
   при включённой коррекции rolling shutter), одинаково для встроенной и внешней
   телеметрии. Измерено по картинке на двух камерах: `pts` верен с точностью ~1 мс.
   Постоянный сдвиг поэтому измеряется на клипе (выходит ≈0), а покадрово
   добавляется половина изменения экспозиции относительно медианы. Старое
   выравнивание по опоре из `dbgi` (`align.py`, +10.5 мс на Pro) опаздывало на
   ~10 мс: на Lite та же опора даёт сдвиг другого знака.
2. **Крен.** Слитая ориентация DJI дрожит по крену на ~0.12°/кадр независимо от
   реального движения. Крен, измеренный по картинке (чистое вращение лучей между
   соседними кадрами, `measure_rotation.py`), заменяет его выше 4 Гц с весом по
   скорости (R0 = 40 °/с), как в `rollfix.py`. Ось крена — z системы телеметрии
   (совпадение осей с камерой подтверждено оконной гомографией на быстрых участках).
3. **Шум тангажа/рыскания.** Винеровская кривая усиления выше 4 Гц, при
   возможности калибруется на клипе, ниже 4 Гц ничего не трогается.
4. **Выбросы и события тангажа/рыскания.** Главный дефект Lite оказался
   одиночными выбросами слитой ориентации: скачок рыскания на 1.77° за один кадр
   при 0.3° у соседей (55.72 с), Gyroflow честно дёргает кадр. Такие кадры ищутся
   по самой телеметрии (отклонение от локальной медианы ≥0.35°/кадр),
   подтверждаются картинкой (она скачка не показывает) и возвращаются на
   локальную медиану ещё до всех фильтров, чтобы фильтры выброс не «видели»
   (иначе винер размазывает его на соседние кадры).
   Более длинные события (расхождение картинки и телеметрии ≥0.1°/кадр и не
   менее половины самого движения, ≥0.1 с) правятся по покадровой форме
   картинки с нулевым итогом за окно. Каждое событие независимо проверяется по
   каналам сдвига аффинного потока; применяются только подтверждённые (и
   выбросы), неясные и противоречащие отбрасываются (`fix_pipeline.py --unclear-weight`).
   Потолка на величину нет (`fix_pipeline.py --event-max-deg`). Визуально проверено
   пользователем на обоих клипах 2026‑09‑15.

Все шаги измеряются, а не предполагаются: `verify_fix.py` судит любой sidecar
по картинке без рендера, `main.py` в конце проверяет через Gyroflow CLI, что
Gyroflow читает файл ровно так, как он записан (расхождение 1e‑3°).

## Результат

Короткие куски в `examples/`, по одному дефекту на камеру:

| | до | после | по картинке |
|---|---|---|---|
| O4 Pro, дрожание крена на спокойных 3 с, °/кадр | 0.11 | 0.019 | 0.0075 |
| O4 Lite, скачок рыскания в кадре 55.72 с, ° | 1.76 | 0.03 | 0.24 |
| O4 Lite, скачок рыскания в кадре 55.82 с, ° | 0.68 | 0.00 | 0.17 |

Полные клипы, 92 с каждый. Дрожание крена здесь — среднеквадратичное изменение
скорости крена от кадра к кадру, °/кадр. «Картинка» — то же самое, измеренное
по видео; ниже этого телеметрия опуститься не может.

| клип | участок | до | после | картинка |
|---|---|---|---|---|
| O4 Pro | спокойный <20 °/с | 0.075 | 0.044 | 0.037 |
| O4 Pro | 20–60 °/с | 0.090 | 0.074 | 0.059 |
| O4 Pro | быстрый >60 °/с | 0.85 | 0.85 | 0.72 |
| O4 Lite | спокойный <20 °/с | 0.051 | 0.046 | 0.041 |
| O4 Lite | 20–60 °/с | 0.089 | 0.088 | 0.080 |
| O4 Lite | быстрый >60 °/с | 0.45 | 0.46 | 0.30 |

На полном клипе Lite убрано 11 выбросов: 19.76, 25.78, 25.98, 26.10,
37.24–37.34, 41.14, 55.72, 55.82 и 91.54 с. Расхождение тангажа и рыскания
с картинкой по 99‑му процентилю 0.59 → 0.50 °/кадр. Крен у Lite и так почти
чистый. Выше 60 °/с ничего не правится: картинка смазана, телеметрия там
точнее неё. Правки тангажа и рыскания на Pro проверены только глазами,
картинка занижает их величину (см. «Ограничения»). Отчёты `<имя>_verify.json`
и `<имя>_fix_report.json` остаются рядом с видео при `--keep`.

## Примеры

В `examples/` два коротких куска с телеметрией, по одному на камеру, с
`run.bat`, результатом и графиками. Сами клипы (~70 МБ каждый) в репозиторий
не входят: скачайте `o4pro_13-17s.MP4` и `o4lite_53-58s.MP4` со страницы
Releases и положите в соответствующие папки; `run.bat` без клипа подскажет то же.

- `examples/o4pro_13-17s/` — O4 Pro, штатная линза: покадровое дрожание крена
  (0.1105 → 0.0187 °/кадр при пороге картинки 0.0075).
- `examples/o4lite_53-58s/` — O4 Lite с линзой Flywoo O4 Wide (запуск с
  `--lens flywoo`): одиночные выбросы рыскания телеметрии на 55.72 и 55.82 с
  (1.76 и 0.68° за кадр), из‑за которых Gyroflow дёргал кадр.

В каждой папке `README.md` с числами до/после и с тем, что осталось неидеальным.

## Ограничения, честно

- Тангаж и рыскание картинка измеряет с заниженным масштабом (0.5–0.75 на
  Lite, 0.7–0.95 на Pro): перемещение дрона над землёй неотделимо от поворота
  при малой базе. Поэтому ремонт событий опирается на *изменение* расхождения
  за доли секунды, а не на его величину, итог за окно обнуляется, и применяются
  только события, подтверждённые вторым каналом. Крен этой проблемы не имеет
  (усиление ~1.0 на обеих камерах).
- Модель объектива по умолчанию берётся из телеметрии клипа. Если на камере
  стоит другая линза (на тестовом Lite стояла Flywoo O4 Wide), передайте тот же
  профиль, что выбран в Gyroflow: `--lens flywoo` (часть имени файла в
  `lens_profiles/`, где лежит профиль Flywoo O4 Wide) или `--lens путь/к/профилю.json`
  из базы github.com/gyroflow/lens_profiles. Крен и вырезание выбросов от линзы не
  зависят; согласие тангажа/рыскания с телеметрией с правильным профилем на Lite
  улучшилось (остаток 0.73° → 0.56°/кадр), но заниженный масштаб остался и
  на Lite, и на Pro со стоковой линзой: это параллакс, а не линза. Поэтому
  `--fit-focal` (подбор масштаба фокуса по телеметрии на быстрых кадрах) только
  диагностический и применяется, лишь если усиления доходят до 1; на обоих
  тестовых клипах он корректно отказывается.
- Быстрее ~60 °/с картинка размыта; там ничего не исправляется.
- Крен ниже 4 Гц оставлен телеметрии: интеграл измерения по картинке дрейфует.

## Проверка и сравнение файлов

```bash
python src/verify_fix.py <video> --image <clip_image.npz> <sidecar1.mp4> [<sidecar2.mp4> ...]
```

Для каждого файла: сдвиг тайминга относительно `pts`, при котором крен телеметрии
лучше всего совпадает с картинкой (у исправленного файла должен быть 0 ± 2 мс),
дрожание крена по полосам скорости против порога картинки, и расхождение
тангажа/рыскания по проверенным окнам.

Экспорт Gyroflow без рендера (пути абсолютные, в цикле не запускать):

```bash
Gyroflow.exe video.MP4 -g sidecar.mp4 --export-metadata "3:C:\abs\camera.json" -f
```

Примечание из исходников Gyroflow 1.6.3 (`external/gyroflow`): с `-g` CLI не
разбирает собственную телеметрию видео, поэтому `frame_readout_time` в проекте
остаётся 0 и экспорт type 3 отмечен ровно `pts`; в GUI значение readout из
видео сохраняется, если видео загружено первым. Экспорт type 3 при readout > 0
ставит метку `pts + readout/2`, а рендер берёт среднюю строку в `pts`.

## Файлы

| файл | что |
|---|---|
| `src/main.py`, `fix_telemetry.bat` | точка входа |
| `src/measure_rotation.py` | один проход по видео: крен и сдвиги из аффинного потока (сырые и в нормализованных координатах), чистое вращение по парам кадров, гомография и чистое вращение по окнам в 5 кадров |
| `src/fix_pipeline.py` | сборка исправленной телеметрии и контрольного файла, отчёт |
| `src/timing.py` | покадровый тайминг |
| `src/verify_fix.py` | суд по картинке |
| `src/dji_o4.py` | CLI: `extract`, `dump`, `sidecar`, `gcsv`, `patch`, `verify`; разбор `djmd` |
| `src/sidecar.py`, `src/mp4parse.py`, `src/pb.py`, `src/quat.py` | sidecar MP4, ISO‑BMFF, protobuf, кватернионы |
| `src/telemetry.py` | загрузка `.npz`, векторные операции, запись sidecar |
| `src/rotmath.py` | кватернионная математика, общая для всего |
| `src/align.py`, `src/rollfix.py`, `src/denoise.py` | прежние методы; `denoise.py` даёт винеровские кривые конвейеру, `align.py` — режим `--timing dbgi` |
| `src/imagerot.py` | измерения по картинке: LK‑трекинг, Кабш, аффинный поток, гомография |
| `research/` | исследовательские скрипты и сессионный конвейер (история), см. `docs/PROJECT_STRUCTURE.md` |
| `external/` | исходники Gyroflow и telemetry-parser для сверки (не в репозитории, см. `external/README.md`) |

## Сравнение телеметрии до и после стабилизации (сессионный конвейер)

Скрипты лежат в `research/session/`. `compare_stabilization_telemetry.py` сопоставляет DJI-телеметрию исходника с
движением картинки исходного и стабилизированного видео. Пути к двум видео и
параметры интервала находятся в начале файла. Каждый запуск создаёт отдельную
папку `artifacts/sessions/YYYYMMDD_HHMMSS` с тремя одинаково устроенными CSV,
двумя графиками, описанием сессии и JSON со списком подозрительных интервалов.

```bash
python research/session/compare_stabilization_telemetry.py
python research/session/analyze_telemetry_outliers.py artifacts/sessions/YYYYMMDD_HHMMSS
python research/session/create_telemetry_bugfix.py artifacts/sessions/YYYYMMDD_HHMMSS
python research/session/evaluate_telemetry_fix.py
```

Этот конвейер требует стабилизированного рендера и остаётся инструментом анализа.
Исправления 2026‑09‑14: сетка времени скоростей сдвинута на `readout/2`
(раньше вставляемая деталь попадала не в фазу), правки всех моделей ограничены
`max_correction_deg`, в оценку добавлена метрика абсолютного отклонения за окно.
Класс `large_impulse` (всплеск в выходе при согласии гиро и картинки) не является
доказательством ошибки телеметрии; обоснованный критерий — `gyro_fault`.

## Обрезка видео с сохранением телеметрии

В `src/trim_video.py` сверху задайте три параметра и запустите `python src/trim_video.py`:

```python
FILE = r"F:\36\DJI_20260905181949_0005_D.MP4"
START = "00:10"  # None — с начала
END = "00:15"    # None — до конца
```

Границы расширяются до ключевых кадров, все дорожки (`djmd`, `dbgi`) режутся по
одним номерам кадров, абсолютные метки DJI сохраняются, файл перечитывается и
проверяется. Обрезанный ролик можно прогнать через `main.py` как обычный.
Подробнее в `docs/`. Тесты: `python -m unittest discover -s tests`.

## Что в файле

`djmd` — protobuf `dvtm_O4P.proto`/`dvtm_O4.proto`: 40 кватернионов на кадр
(2000 Гц номинально, слияние обновляется на 1000 Гц, каждое значение дважды),
время считывания сенсора (Pro 13.58 мс, Lite 18.13 мс), фокус и дисторсия
(fisheye OpenCV), экспозиция и ISO по кадрам. Сырых данных гироскопа DJI не
пишет. Преобразование в систему Gyroflow: `q_cam = (0,0,1,0) ⊗ q ⊗ (0.5,-0.5,-0.5,0.5)`
плюс знак непрерывности; шкала времени воспроизведена по telemetry-parser
точно (все метки совпадают с собственным разбором Gyroflow). Подробная карта
полей и история исследований — в `docs/HANDOFF.md`.

## Клипы и обратная связь / Sample clips wanted

Инструмент отлажен на двух блоках 2025 года. Если у вас O4 Pro выпуска 2026
(серийный номер `9F2KP2…` и позже, гироскоп I469D) и стабилизация дёргается,
пришлите короткий кусок исходника с телеметрией: 5–10 с оригинального MP4 без
перекодирования, лучше с описанием, где именно дёргает. Так же интересны клипы,
на которых инструмент не помог.

*The tool was developed on two 2025 units. If you have a 2026 O4 Pro (serial
`9F2KP2…` or later, I469D gyro) with juddering stabilization, send a short piece
of the original MP4 with its telemetry: 5–10 s, not re‑encoded, ideally with a
note of where it judders. Clips where the tool does not help are just as useful.*

Куда: закреплённый issue «Clips wanted» в [Issues](https://github.com/parhipov/DJIFix/issues),
файл на любой файлообменник, в issue ссылку. Личный контакт есть на профиле GitHub.
*Where: the pinned «Clips wanted» issue; upload the file anywhere and post the link.*

## Лицензия

MIT, см. `LICENSE`. Исходники Gyroflow и telemetry-parser в `external/` не входят в репозиторий и распространяются под своими лицензиями.
