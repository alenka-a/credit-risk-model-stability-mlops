"""
Модуль: Загрузка датасета
Версия: 1.0
"""

import io
from io import BytesIO
from typing import Dict, Tuple

import pandas as pd


def get_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_parquet(dataset_path)


def dataset_to_bytes(
        dataset: pd.DataFrame, type_data: str
) -> Dict[str, Tuple[str, BytesIO, str]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param dataset: датасет
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset_bytes_obj = io.BytesIO()
    dataset.to_parquet(dataset_bytes_obj, index=False)
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.parquet", dataset_bytes_obj, "multipart/form-data")
    }
    return files
