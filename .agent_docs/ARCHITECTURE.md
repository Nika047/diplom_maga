# Архитектура

Состояние на 18.10.2026: этапы 0-2 реализованы и выполнены. Этап 3 написан и
проверен смоуком на рабочем ПК, полное обучение - на домашнем. Модули этапов 4 и
дальше помечены «план».

## Поток данных

```text
flight_logs/*.csv --> data/flight_logs.py: очистка, куски, ENU, перепривязка времени
                           |
                     data/phases.py: скорость, |w|, участки, начала маневров
                           |
                     data/splits.py: train / val / test / holdout / sim / excluded (целыми логами)
                           |
         data/generator.py: калибровка по train -> синтетика (только train + набор анализа)
                           |
         cache/stage0/*.pkl, splits.json, calibration.json
                           |
   radar/model.py: observe(Track) -> RadarTrack (отметки, ковариации, участки, начала маневров)
                           |
   filters/: cv_extrapolate, IMM(("CV",)) = фильтр Калмана, IMM() -> FilterRun
             (оценки, вероятности моделей, невязки, прогноз на 1..H обзоров)
                           |
   eval/metrics.py: пары (k, h) -> MAE, RMSE, ADE, FDE, R2; кривая динамической ошибки
                           |
   data/windows.py: окно W обзоров в системе курса IMM -> x (N, W, F); цель - поправка (N, H, 2)
                           |
   training/dataset.py: прогон РЛС + IMM + окна, кэш cache/stage2/*.npz, нормализация
                           |
   models/ (GRU) + training/trainer.py -> веса artifacts/stage2/*.pt
                           |
   гибрид: прогноз IMM + поправка (apply_correction) -> eval/metrics.py
                           |
   этап 3: LSTM, GRU, Transformer, Mamba (гибриды) + GRU, Mamba без IMM (окна build_windows_e2e)
           -> eval/compare.py (одинаковые пары (k, h) для всех методов)
                           |
   этап 4: tuning/search.py
           шаг imm   -> cache/stage4/imm_best.json (параметры IMM)
           шаг tune  -> cache/stage4/best_params.json (окно, сеть, обучение - для каждой модели)
           шаг final -> несколько seed, оценка val / test / holdout, results/stage4/
   этап 4б: tuning/arch_search.py (NSGA-II, 2 цели) -> cache/stage4/arch_best.json,
           фронт Парето и итог на test -> results/stage4b/
   дополнение этапа 4: run_stage4_baselines.py - подбор KF и CV -> results/stage4_baselines/
   этап 5: data/windows.onset_labels (метка начала маневра) + models/onset.py (GRU, две головы)
           -> обнаружители (сеть, вероятность CT, NIS CV) и прогноз -> results/stage5/
```

## Модули

| Модуль | Ответственность | Статус |
|---|---|---|
| `config.py` | Все параметры: очистка, разметка, разбиение, генератор, допуски проверок, потоки случайных чисел | есть |
| `common/logging_setup.py` | Журнал в консоль и `log/<запуск>_<время>.log`, исключения и предупреждения - тоже в файл | есть |
| `common/rng.py` | `make_rng(поток)` - именованные потоки из `RANDOM_SEED` (`config.RNG_STREAMS`, новые - только в конец) | есть |
| `data/track.py` | `Track` - единый формат истины для логов и синтетики | есть |
| `data/geo.py` | WGS84 -> ENU через ECEF | есть |
| `data/flight_logs.py` | Чтение CSV, очистка, разрез на куски, ENU, перепривязка времени | есть |
| `data/phases.py` | Кинематика, разметка участков, доли, `turn_omegas` | есть |
| `data/splits.py` | Разбиение целыми логами, `save_splits` / `load_splits` | есть |
| `data/generator.py` | Кинематический генератор, `calibrate_limits`, сценарии | есть |
| `eval/checks.py` | `StageSummary`: проверки PASS / FAIL, разделы, `summary.txt` | есть |
| `viz/plots.py`, `viz/map_html.py` | PNG и HTML-карта (Leaflet) | есть |
| `run_stage0.py` | Этап 0 | есть |
| `radar/model.py` | Модель РЛС: положение РЛС на трассу, полярный шум, пропуски, пересчёт в ENU с ковариацией | есть |
| `filters/models.py` | Модели CV, CA, CT на общем векторе `[x, y, z, vx, vy, vz, ax, ay, w]` | есть |
| `filters/imm.py` | IMM и прогноз на 1..H; с одной моделью - фильтр Калмана | есть |
| `filters/extrapolation.py` | Линейная экстраполяция по последним отметкам | есть |
| `eval/metrics.py` | Единые метрики и динамическая ошибка; метрики обнаружения начала маневра (этап 5): средняя точность, порог по F1, события, опережение, ложные тревоги, отклик | есть |
| `viz/filter_plots.py` | Рисунки этапа 1 | есть |
| `run_stage1.py` | Этап 1 | есть |
| `data/windows.py` | Нарезка окон (единственное место), система курса, `apply_correction`, `onset_labels` (метка начала маневра, этап 5) | есть |
| `training/dataset.py` | Сборка окон с кэшем и подписью параметров, `Normalizer` | есть |
| `training/trainer.py` | Обучение, ранняя остановка, overfit one batch (dropout отключается), `predict`; `loss_fn` - своя функция потерь (этап 5), по умолчанию MSE | есть |
| `models/__init__.py` | Реестр `CORRECTORS`: GRU, LSTM, Transformer, Mamba | есть |
| `models/gru.py`, `models/lstm.py` | Рекуррентные корректоры | есть |
| `models/transformer.py` | Кодировщик Transformer, токен - обзор, позиционное кодирование обучаемое | есть |
| `models/mamba.py` | Mamba на чистом PyTorch (селективное сканирование циклом по времени), `sequence()` для проверки причинности | есть |
| `eval/compare.py` | Сравнение методов на одинаковых парах, разделы сводки, рисунки; `mask_forecast` | есть |
| `viz/train_plots.py` | Кривые loss | есть |
| `run_stage2.py` | Этап 2, ключи `--smoke` и `--rebuild` | есть |
| `run_stage3.py` | Этап 3: все корректоры и модели без IMM; `--smoke`, `--rebuild`, `--only GRU,Mamba` | есть |
| `training/experiment.py` | Один эксперимент: нормализация, overfit, обучение, прогноз на val, веса, кривая loss. Общий для `run_stage3*.py` | есть |
| `run_stage3_window.py` | Дополнение этапа 3: те же модели на окнах 20 / 60 / 100 обзоров, оценка на общих обзорах; `--windows`, `--only`, `--smoke` | есть |
| `tuning/search.py` | Генетический подбор (Optuna, NSGA-II): цель для IMM и для корректоров, бюджет попыток, `config_override`, `split_params` | есть |
| `run_stage4.py` | Этап 4: `--phase imm / tune / final / all`, `--smoke`, `--only`, `--trials`, `--imm-trials` | есть |
| `models/stack.py` | `StackCorrector`: стопка блоков GRU / LSTM / Mamba / Attention с остаточными связями (этап 4б) | есть |
| `tuning/arch_search.py` | Геном архитектуры, две цели (ADE, время прогноза), важность групп признаков и типов блоков | есть |
| `run_stage4b.py` | Этап 4б: `--phase search / final / all`, `--smoke`, `--trials`. Нужен `cache/stage4/imm_best.json` | есть |
| `run_stage4_baselines.py` | Дополнение этапа 4: генетический подбор KF (`TUNE_KF_SPACE`) и перебор числа отметок CV (`TUNE_CV_POINTS`) с той же целью и бюджетом, что у IMM; без GPU | есть |
| `models/onset.py` | `OnsetGRU` - GRU с головами поправки и начала маневра (выход 2H + 1), `multitask_loss` (MSE + BCE) | есть |
| `run_stage5.py` | Этап 5: прогноз начала маневра, сравнение с обнаружителями по IMM, GRU с головой и без; `--smoke`. Нужны `imm_best.json` и `best_params.json` этапа 4 | есть |
| `predict.py` | Итоговая программа | план |

**Кэш этапа 0** (`cache/stage0/`) - вход следующих этапов:
- `tracks_real.pkl` - все куски, включая sim и excluded: фильтровать по
  `splits.json`;
- `tracks_synth_train.pkl`, `tracks_synth_analysis.pkl`;
- `splits.json` - {лог: выборка};
- `calibration.json` - `Limits`.

## Соответствие имён

| Сущность | В тексте диплома | В коде | В логе CSV |
|---|---|---|---|
| Файл лога | полёт | `log_file` (единица разбиения) | имя файла |
| Кусок полёта после очистки | полёт (фрагмент) | `Track`, `flight_id` = `<имя файла без .csv>#<номер>` | - |
| Синтетическая траектория | синтетическая траектория | `Track`, `flight_id` = `synth_<сценарий>_<номер>` (набор анализа - `synthA_...`) | - |
| Истинная траектория | истинная траектория | `Track.t`, `Track.pos` (N, 3) ENU | `Date`, `Time`, `GPS`, `Alt(m)` |
| Тип участка | прямая / вход / маневр / выход / медленно | `Track.phase` - индекс в `config.PHASES` = `straight`, `entry`, `turn`, `exit`, `slow` | - |
| Начало маневра | начало маневра | `Track.turn_onsets` - индексы точек | - |
| Скорость разворота | скорость разворота | `Track.omega`, град/с, + вправо, `nan` при скорости < `PHASE_V_MIN` | - |
| Трасса РЛС | трасса | `RadarTrack` (тот же `flight_id`, что у `Track`) | - |
| Отметка РЛС | отметка | `RadarTrack.polar` (`r`, `az`, `el`), `RadarTrack.meas` (ENU), `meas_cov` | - |
| Обзор | обзор | индекс `k`, время `RadarTrack.t` = k * `SCAN_PERIOD` | - |
| Горизонт | h обзоров | `h` = 1..`HORIZON_SCANS`, в массивах прогноза индекс `h - 1` | - |
| Прогноз | прогноз из обзора k на h | `forecast[k, h - 1]` - положение в момент k + h | - |
| Модель движения IMM | модель CV / CT | `IMM_MODELS`, столбцы `FilterRun.mu` в том же порядке | - |
| Скорость разворота в фильтре | w | состояние `W`, рад/с, **+ против часовой** (в разметке `Track.omega` - град/с, + вправо, то есть по часовой) | - |
| Окно корректора | окно наблюдения | `x[n]` - обзоры k-W+1..k; `WindowSet.track`, `.k`, `.psi` - чья трасса, какой обзор, курс | - |
| Система курса | система «поперёк / вдоль курса» | `to_local` / `to_global`; ось 0 - поперёк вправо, ось 1 - вдоль курса IMM в обзоре k, начало - оценка IMM в k | - |
| Поправка | поправка корректора | `y` = истина(k+h) - прогноз IMM(k, h) по горизонтали, в системе курса; в сети - делённая на `Normalizer.y_std` | - |
| Гибрид | IMM + корректор | `IMM+GRU` и т.п.; прогноз = `apply_correction(forecast_IMM, k, psi, поправка)`, высота - от IMM | - |
| Сквозная модель | модель без IMM | `GRU (без IMM)`, `Mamba (без IMM)`; окна `x_e2e` (4 признака), цель `y_e2e` - смещение истины от отметки k; прогноз - `apply_e2e`, курс `psi_e2e` - по МНК-прямой по последним отметкам | - |
| Тип аппарата | тип БПЛА | `Track.aircraft` | префикс имени файла |
| Режим полёта | режим | `Track.flight_mode` (без `*`) | `FM` |
| Путевая скорость | скорость | `Track.speed`, м/с, из координат | `GSpd(kmh)` (км/ч) - **не используется** |

## Системы координат и соглашения

- **ENU:** `x` - восток, `y` - север, `z` - вверх, метры. Начало - первая точка
  каждого куска (логи) или траектории (синтетика).
- **Курс** как в логе: от севера по часовой, `atan2(vx, vy)`.
- **Время** - секунды от начала куска. Для логов это время **после перепривязки**:
  исходные метки лога в кэш не попадают.
- **Разметка участков** использует центральное сглаживание курса: это эталон, а
  не вход модели. Входы моделей - только отметки РЛС (правило в `AGENTS.md`).
- **РЛС** стоит в ENU трассы. Азимут - от севера по часовой, угол места - от
  горизонта. Шум и положение РЛС зависят только от `flight_id`
  (`make_track_rng("radar", flight_id)`), поэтому одинаковы на любом компьютере
  и при любом составе выборки.

**Кэш этапов 2-3** (`cache/stage2/`, общий): `train.npz`, `val.npz`,
`train_smoke.npz` (поля для гибрида и для моделей без IMM, `FEATURES_VERSION = 2`)
+ `*.json` с подписью - параметры РЛС, IMM, длина окна, `FEATURES_VERSION`, хеш
списка трасс. Окна другой длины - `train_w60.npz`, `train_w100.npz` и т.п. (до
0,8 ГБ каждый). Длина окна передаётся параметром `window` в `dataset.build`,
`load_or_build`, `build_corrector`; по умолчанию `config.WINDOW_SCANS`. При несовпадении подписи кэш пересобирается сам. **Изменил признаки в
`data/windows.py` - увеличь `training/dataset.FEATURES_VERSION`**, иначе подпись
не заметит изменения.

**Кэш этапа 4** (`cache/stage4/`, смоук - `cache/stage4_smoke/`):
- `optuna.db` - все попытки подбора (SQLite). Прерванный подбор продолжается,
  бюджет считается по завершённым попыткам. Для подбора с нуля удалить базу;
- `imm_best.json`, `best_params.json` - лучшие параметры этапа 4;
  `arch_best.json` - кандидаты, важность признаков и блоков этапа 4б (тот же
  `optuna.db`, подбор `arch`). Этапы 4 и 4б одновременно не запускать: общая база
  SQLite;
- группы признаков для отбора - `data/windows.feature_groups(n_models)`, порядок
  тот же, что в `build_windows`;
- окна при подобранном IMM - `train_w<N>`, `val_w<N>`. Отдельно от
  `cache/stage2`, чтобы не перезаписывать окна этапов 2-3.

**Параметры IMM во время работы.** `IMM()` читает `IMM_P_STAY`, `IMM_MODELS`,
`IMM_MU0`, `HORIZON_SCANS` из config при создании, а модели движения читают
`FILTER_SIGMA_*` при каждом шаге. Поэтому подобранные параметры применяются через
`tuning.search.config_override(...)`, без правки `config.py`.

## Как расширять

- **Новый корректор:**
  1. Модуль в `models/`: вход (B, W, F), выход (B, H, 2), последний слой
     инициализирован нулями.
  2. Регистрация в `models.CORRECTORS`.
  3. Строка в `EXPERIMENTS` в `run_stage3.py`.
  4. Смоук: `run_stage3.py --smoke --only <имя>`.

- **Новый сценарий генератора:**
  1. Ветка в `make_plan`.
  2. Имя - в конец `config.SYNTH_SCENARIOS`.
  3. Число - в `SYNTH_COUNT_TRAIN` и `SYNTH_COUNT_ANALYSIS`.
  4. Запустить этап 0: проверки пределов и распределений должны остаться PASS.
- **Новый поток случайных чисел:** только в конец `config.RNG_STREAMS`.
- **Новая проверка этапа:** `summary.check(имя, условие, подробности)` в
  `run_stageN.py`.
