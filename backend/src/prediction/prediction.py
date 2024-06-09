"""
Модуль: Предсказание на основе обученной модели
Версия: 1.0
"""
from typing import BinaryIO

import joblib
import pandas as pd
import yaml

from ..data.loading import get_dataset, save_dataset
from ..preprocess.preprocess import pipeline_preprocessing


def pipeline_prediction(
        config_path: str,
        dataset: pd.DataFrame = None,
        data_path: str | BinaryIO = None
) -> list:
    """
    Пайплайн с предобработкой входных данных и получением предсказаний
    :param dataset: датасет
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    :return: предсказания по входным данным
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]
    prediction_config = config["prediction"]

    if data_path:
        dataset = get_dataset(dataset_path=data_path)

    processed_dataset = pipeline_preprocessing(df_data=dataset, **preprocessing_config)

    model = joblib.load(train_config["model_path"])
    prediction = model.predict(processed_dataset)

    prediction_path = prediction_config["predicted_path"]
    if prediction_path:
        dataset["prediction"] = prediction
        save_dataset(dataset, prediction_path)
    return prediction.tolist()
