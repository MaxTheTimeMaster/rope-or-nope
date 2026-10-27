# rope-or-nope

![CI](https://github.com/MaxTheTimeMaster/rope-or-nope/actions/workflows/ci.yml/badge.svg)

Лаборатория на FastAPI для экспериментов с позиционными кодированиями трансформеров, 
Интересный вопрос который можно проверить на cpu: какое кодирование (learned, sinusoidal, RoPE, ALiBi или NoPE)
позволяет крошечной модели, обученной на последовательностях длиной до 16, справляться с длиной 32–64.
Подойдет для студентов которые хотят посмотреть как происходит обобщение трансформеров на разные длины на cpu.  

**Статус:** работают реестр архитектур и генератор задач copy / reverse / sort / love57 
Обучение моделей и сравнение прогонов в разработке

## Маршруты

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | проверка, что сервис жив |
| GET | `/version` | версия приложения |
| GET | `/architectures` | список архитектур |
| GET | `/architectures/{name}` | конфиг архитектуры, 404 для неизвестной |
| GET | `/tasks/{name}/sample?length=8&seed=0` | пример задачи; 404 для неизвестной задачи, 422 при длине вне 1–64 |

## Установка

Нужен Python 3.13

```bash
git clone https://github.com/MaxTheTimeMaster/rope-or-nope.git
cd rope-or-nope
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
pip install -r requirements-dev.txt
```

## Запуск

Для разработки (перезапуск при изменении файлов):

```bash
uvicorn app_final:app --reload
```

Для обычной работы, без `--reload`:

```bash
uvicorn app_final:app --host 0.0.0.0 --port 8000
```

Интерактивная документация: http://127.0.0.1:8000/docs

## Переменные окружения

| Переменная | По умолчанию | Что задаёт |
|---|---|---|
| `LOG_LEVEL` | `INFO` | уровень логов: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DEFAULT_SAMPLE_LENGTH` | `8` | длина примера в `/tasks/{name}/sample`, если `length` не передан |

Задать можно в файле `.env` или прямо в окружении

## Тесты

```bash
python -m pytest -v
```

GitHub Actions запускает тесты на каждый push и pull request (`.github/workflows/ci.yml`)
После успешных тестов прогон сохраняет artifact `rope-or-nope-app` с файлами приложения

## Проверка и диагностика

- Жив ли сервис: http://127.0.0.1:8000/health → `{"status": "ok"}`
- Какая версия запущена: http://127.0.0.1:8000/version
- **Логи** пишутся в терминал, где запущен uvicorn - формат:
  `2026-10-27 21:00:00,000 INFO rope_or_nope: GET /health -> 200 (0.4 ms)`.
  Пишутся запуск и остановка, каждый запрос с кодом и временем ответа,
  сгенерированные примеры и запросы к несуществующим задачам или архитектурам (WARNING).
- Логи CI: вкладка Actions → прогон → шаг Run tests.
