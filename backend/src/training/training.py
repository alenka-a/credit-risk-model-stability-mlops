"""
Модуль: Обучение модели
Версия: 1.0
"""
import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier

from ..data.loading import get_dataset, save_dataset
from ..data.splitting import split_train_test_by_groups, get_train_test_data
from ..preprocess.preprocess import pipeline_preprocessing
from ..training.metrics import get_metrics
from ..training.parameters_selection import find_optimal_params, save_optimal_params


def pipeline_training(config_path: str) -> dict:
    """
    Полный цикл обучения модели с предобработкой данных и поиском лучших параметров
    :param config_path: путь до файла с конфигурациями
    :return: метрики
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    train_data = get_dataset(dataset_path=preprocessing_config["train_path"])
    train_data = pipeline_preprocessing(df_data=train_data, flg_prediction=False, **preprocessing_config)

    df_train, df_test = split_train_test_by_groups(dataset=train_data, **preprocessing_config)

    save_dataset(df_train, preprocessing_config["train_path_proc"])
    save_dataset(df_test, preprocessing_config["test_path_proc"])

    data_x_train, data_x_test, data_y_train, data_y_test, \
        groups_train, groups_test = get_train_test_data(df_train, df_test,
                                                        **preprocessing_config)

    if train_config["optuna_enable"]:
        study = find_optimal_params(data_x=data_x_train, data_y=data_y_train, groups=groups_train, **train_config)

        clf = train_model(
            data_x_train=data_x_train,
            data_y_train=data_y_train,
            model_params=study.best_params,
        )
        joblib.dump(study, train_config["study_path"])
        save_optimal_params(study, train_config["params_path"])
    else:
        clf = train_model(
            data_x_train=data_x_train,
            data_y_train=data_y_train
        )
    metrics = get_metrics(data_x=data_x_test, data_y=data_y_test, groups=groups_test, model=clf,
                          save_path=train_config["metrics_path"])
    joblib.dump(clf, train_config["model_path"])

    return metrics


def train_model(
        data_x_train: pd.DataFrame,
        data_y_train: pd.Series,
        model_params: dict = None,
) -> LGBMClassifier:
    """
    Обучение модели на переданных параметрах.
    Если параметры не переданы, то обучается на параметрах по умолчанию.
    :param data_x_train: тренировочный датасет
    :param data_y_train: данные с целевой переменной
    :param model_params: параметры модели
    :return: LGBMClassifier
    """
    ratio = float(np.sum(data_y_train == 0)) / np.sum(data_y_train == 1)

    if model_params:
        clf = LGBMClassifier(**model_params, scale_pos_weight=ratio)
    else:
        clf = LGBMClassifier(scale_pos_weight=ratio)

    clf.fit(data_x_train, data_y_train)

    return clf
