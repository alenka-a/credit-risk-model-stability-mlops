"""
Модуль: Расчет и сохранение метрик
Версия: 1.0
"""
import json

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


def get_metrics(
        data_x: pd.DataFrame,
        data_y: pd.Series,
        groups: pd.Series,
        model: BaseEstimator,
        save_path: str = None
) -> dict:
    """
    Расчет и сохранение метрик
    :param data_x: объект-признаки
    :param data_y: целевая переменная
    :param groups: групповой признак
    :param model: модель
    :param save_path: путь для сохранения метрик
    :return: словарь с метриками
    """
    result_metrics = calculate_metrics(
        y_test=data_y,
        y_predicted=model.predict(data_x),
        y_score=model.predict_proba(data_x),
        groups=groups,
    )

    if save_path:
        with open(save_path, "w") as file:
            json.dump(result_metrics, file)

    return result_metrics


def calculate_metrics(y_test: pd.Series,
                      y_predicted: pd.Series,
                      y_score: ArrayLike,
                      groups: pd.Series) -> dict:
    """
    Расчет метрик по предсказанными значениям и генерация словаря с метриками
    :param y_test: фактические целевые значения
    :param y_predicted: предсказанные значения
    :param y_score: предсказанные вероятности
    :param groups: групповой признак
    :return: словарь с метриками
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel().tolist()

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_score[:, 1]),
        'precision': precision_score(y_test, y_predicted),
        'recall': recall_score(y_test, y_predicted),
        'f1': f1_score(y_test, y_predicted),
        'gini_stability': gini_stability(y_test, y_score[:, 1], groups),
        "confusion_matrix":
            {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp
            }
    }
    return metrics


def gini_stability(y_test: pd.Series,
                   y_score: ArrayLike,
                   weeks: pd.Series,
                   w_falling_rate: float = 88.0,
                   w_residuals_std: float = -0.5) -> float:
    """"
    Расчет показателя стабильности Джини.

    Показатель Джини рассчитывается для предсказаний, соответствующих каждой неделе
    Итоговая метрика рассчитывается по формуле:
        stability metric=mean(gini)+88.0⋅min(0,a)−0.5⋅std(residuals)
    :param y_test: фактические целевые значения
    :param y_score: предсказанные вероятности
    :param weeks: недели
    :param w_falling_rate: коэффициент
    :param w_residuals_std: коэффициент
    :return: метрика
    """
    df_score = pd.DataFrame(y_score, columns=["score"])
    df_score["week"] = weeks.to_numpy()
    df_score["target"] = y_test.to_numpy()

    gini_in_time = df_score.loc[:, ["week", "target", "score"]] \
        .sort_values("week") \
        .groupby("week")[["target", "score"]] \
        .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1) \
        .tolist()

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    mean_gini = np.mean(gini_in_time)
    metric = mean_gini + w_falling_rate * min(0, a) + w_residuals_std * res_std
    return metric
