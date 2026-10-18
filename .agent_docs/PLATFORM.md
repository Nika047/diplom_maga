# Платформа

## Рабочий ПК (здесь работает агент)

Проверено 18.10.2026:
- Windows 10 Pro 10.0.19045, Windows PowerShell 5.1.19041. В PowerShell нет `&&`,
  `||`, `??`; цепочка пишется как `A; if ($?) { B }`. Есть также Git Bash.
- Системный Python 3.13.3, git 2.45.1.
- GPU нет: `nvidia-smi` отсутствует.
- Виртуальное окружение: `.venv` в корне проекта (Python 3.13.3), создано 18.10.2026.

## Домашний ПК (обучение)

Со слов пользователя: Windows 11, RTX 5070 Ti. Доступа у агента нет.

По выводу пользователя 18.10.2026: процессор AMD64 Family 26 (AuthenticAMD),
Windows 11 10.0.26200, numpy 2.5.4.

**Python на обоих ПК - 3.13.3** (решение пользователя 18.10.2026). Дома он
поставлен вместо 3.12.10, `.venv` пересоздан через `py -3.13 -m venv .venv`.
Побитового совпадения синтетики это не дало - процессоры разные. Сверка
синтетики - по отпечатку (`PROJECT_STATE.md`).
- **Подключение** - SSH через Radmin VPN: адрес `26.244.190.10`, пользователь
  `kirill`, вход паролем учётной записи (не PIN). Пароль в документы не
  записывать.
  - Проверка порта с рабочего ПК: `Test-NetConnection 26.244.190.10 -Port 22`
    (нужно `TcpTestSucceeded : True`).
  - Вход: `ssh kirill@26.244.190.10`.
  - Источник - документы старого проекта (`md\DAILY_ROUTINE.md`).
- **Код** на домашний ПК попадает через GitHub (`git pull`), репозиторий
  `https://github.com/Nika047/diplom_maga`.
  - Репозиторий **открытый** с 18.10.2026: `clone` и `pull` работают без входа.
  - Если его снова закроют: в ssh-сессии Git Credential Manager не может
    сохранить вход в хранилище Windows (`Unable to persist credentials with the
    'wincredman' credential store`). Тогда понадобятся:
    - хранилище DPAPI: `git config --global credential.credentialStore dpapi`;
    - токен (fine-grained, только `diplom_maga`, Contents: Read-only) вместо
      пароля. Токен в документы не записывать.
  - Клонирован на домашний ПК в `C:\Users\kirill\Desktop\diplom_maga`, этап 0 там
    выполнен 18.10.2026.
- Предположение, проверить при установке: RTX 5070 Ti требует сборку PyTorch с
  поддержкой CUDA 12.8 и новее.
- **Папка проекта:** `C:\Users\kirill\Desktop\diplom_maga`, клон с GitHub.
  Старый проект там же - `TATPN_diplom`, его не трогать.
- Подключение - по ssh из PowerShell рабочего ПК при запущенном Radmin VPN. После
  входа набрать `powershell`, иначе откроется `cmd`.
- **Кэш и результаты в git не хранятся.** Этап 0 выполняется на домашнем ПК один
  раз после клонирования. Он детерминирован: контрольные суммы в
  `results\stage0\summary.txt` должны совпасть с рабочим ПК (`PROJECT_STATE.md`).
- **Русский текст в ssh-консоли** может отображаться кракозябрами. Перед запуском:
  `[Console]::OutputEncoding = [Text.Encoding]::UTF8`. Сводку читать через
  `Get-Content ... -Encoding UTF8`.

## Библиотеки

Установлено (`requirements.txt`), 18.10.2026, с согласия пользователя:

| Библиотека | Версия | Для чего |
|---|---|---|
| numpy | 2.5.4 | Данные, генератор |
| matplotlib | 3.11.2 | PNG-рисунки |
| torch | 2.11.0 | Корректоры. Рабочий ПК - сборка `+cpu`, домашний - `+cu128` (проверено 18.10.2026: RTX 5070 Ti считает). Ставится отдельно, до `requirements.txt` (команды - в нём) |

Планируется, не установлено:

| Библиотека | Для чего |
|---|---|
| optuna | Генетический подбор, сэмплер NSGA-II (этап 4) |
| Mamba | Официальный `mamba-ssm` рассчитан на CUDA под Linux. Под Windows нужна реализация на чистом PyTorch (предположение, проверить) |

pandas не нужен: CSV читаются модулем `csv`.

HTML-карта - Leaflet с CDN внутри HTML, Python-зависимость не нужна.

## Каталоги результатов

| Каталог | Что |
|---|---|
| `log/` | Журналы запусков `<запуск>_<ГГГГММДД-ЧЧММСС>.log`, UTF-8 |
| `results/stageN/` | `summary.txt` (PASS / FAIL), рисунки, метрики |
| `cache/` | Промежуточные данные этапов |

Все три генерируются запуском и не коммитятся.

## Команды

PowerShell, из корня проекта:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_stage0.py
Get-Content results\stage0\summary.txt -Encoding UTF8
```

```powershell
.venv\Scripts\python.exe run_stage1.py
Get-Content results\stage1\summary.txt -Encoding UTF8
```

**Этап 2:**
- рабочий ПК - только смоук: `.venv\Scripts\python.exe run_stage2.py --smoke`,
  сводка в `results\stage2\summary_smoke.txt`. Без `--smoke` и без GPU скрипт
  завершается с кодом 2;
- домашний ПК - полное обучение: `run_stage2.py`. Первый запуск собирает окна
  train (прогон РЛС и IMM по всем трассам train, минуты на процессоре) и кладёт
  их в `cache\stage2\`. Дальше кэш переиспользуется;
- чтобы обрыв ssh не остановил обучение, запуск в фоне (`ReturnValue = 0` -
  запущено):

  ```powershell
  $P = (Get-Location).Path
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"cd /d $P && .venv\Scripts\python.exe -u run_stage2.py > stage2.out 2>&1`"" }
  ```

  Ход - `Get-Content stage2.out -Wait -Tail 20 -Encoding UTF8` или последний файл
  в `log\`.

`run_stageN.py` возвращает код 0, если все проверки PASS, иначе 1. Этап 1 читает
кэш этапа 0 и без него завершается с FAIL. На рабочем ПК идёт около 25 с, из них
IMM - около 16 с на 9684 обзора.

Длительность на рабочем ПК (журнал 18.10.2026): около 2 минут. Чтение логов -
2,4 с, калибровка с пробными генерациями и генерация синтетики - около 110 с
(цикл интегрирования на Python).
