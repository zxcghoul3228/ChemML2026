# Используем образ с Python 3.12 и установленным uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Рабочая папка внутри контейнера
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем библиотеки внутри образа
RUN uv sync --no-dev --no-install-project

# Копируем код
COPY generate_data.py train.py ./

# Создаем папки для данных и итогов
RUN mkdir -p data results

# Команда, которая запустится сама при старте контейнера
CMD ["uv", "run", "sh", "-c", "python generate_data.py && python train.py"]