# Credit Risk Model Stability MLPOps
Проект представляет собой демонстрационный MLOps-сервис для оценки кредитного риска заемщика.

Он реализует полный цикл работы с моделью машинного обучения: анализ данных, обучение модели, получение предсказаний и интерпретацию результатов через веб-интерфейс.

В проекте использованы предварительно предобработнные данные из соревания на Kaggle "Home Credit – Credit Risk Model Stability": https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data. 

**Tech Stack:** Python, FastAPI, Streamlit, seaborn, matplotlib, LightGBM, optuna

## Основные возможности:
* Исследование данных (EDA)
* Обучение модели кредитного скоринга LightGBM
* Предсказание кредитного риска

## Структура проекта:
```
backend/          # FastAPI сервис для обучения и инференса моделей
frontend/         # Streamlit веб-интерфейс
data/             # датасеты
notebooks/        # Jupyter-ноутбуки для анализа данных
config/           # конфигурационные файлы
```

## Запуск приложения через Docker
Приложение можно запустить с помощью контейнеризации (Docker или Podman).

1. Клонировать репозиторий:
```bash
git clone https://github.com/e-afanaseva/credit-risk-model-stability-mlops.git
cd credit-risk-model-stability-mlops
```
2. Собрать и запустить контейнеры:
```bash
docker compose up -d --build
```
При старте контейнеров:
* разворачивается backend-сервис с API для работы с моделью
* запускается веб-интерфейс для взаимодействия с системой
* загружается обученная модель и необходимые зависимости

3. После запуска веб-интерфейс будет доступен по следующему адресу:
  [http://localhost:8501](http://localhost:8601)

4. Чтобы остановить контейнеры:
```bash
docker compose down
```
