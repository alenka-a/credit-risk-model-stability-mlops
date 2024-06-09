"""
Модуль: Получение и отрисовка предсказаний
Версия: 1.0
"""

from typing import Any

import pandas as pd
import requests
import streamlit as st


def predict(data: pd.DataFrame, endpoint: str, files: Any) -> None:
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    response = requests.post(endpoint, files=files, timeout=8000)
    result = response.json()["prediction"]
    data_ = data[:len(result)]
    data_["predict"] = result
    st.write(data_.head())
