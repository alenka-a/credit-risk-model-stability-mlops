"""
Модуль: Модель для предсказания, какие клиенты с большей вероятностью не смогут выплатить свои кредиты.
Версия: 1.0
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File

from src.training.training import pipeline_training
from src.prediction.prediction import pipeline_prediction

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


@app.post("/train")
def train():
    """
    Обучение модели
    """
    metrics = pipeline_training(config_path=CONFIG_PATH)
    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_prediction(config_path=CONFIG_PATH, data_path=file.file)
    # выводим только часть предсказаний, чтобы не было зависаний
    return {"prediction": result[:10]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8500)
