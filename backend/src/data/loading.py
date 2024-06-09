"""
Модуль: Загрузка и сохранение датасета
Версия: 1.0
"""
from typing import BinaryIO

import pandas as pd


def get_dataset(dataset_path: str | BinaryIO) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_parquet(dataset_path)


def save_dataset(dataset: pd.DataFrame, dataset_path: str, index: bool = False) -> None:
    """
    Сохранение данных в файл
    :param dataset: датасет
    :param dataset_path: путь до данных
    :param index: Если True, то индекс сохраняется в файл, если False - не сохраняется
    :return: датасет
    """
    return dataset.to_parquet(dataset_path, index=index)
