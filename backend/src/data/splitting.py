"""
Модуль: Деление датасета на тренировочный и тестовый
Версия: 1.0
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from typing import Tuple


def split_train_test_by_groups(dataset: pd.DataFrame,
                               **kwargs
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделение данных на train/test с последующим сохранением
    :param dataset: датасет
    :return: train/test датасеты
    """
    df_train, df_test = _split_train_test_by_groups(
        dataset,
        dataset[kwargs["groups_column"]],
        test_size=kwargs["test_size"],
        random_state=kwargs["random_state"],
    )
    return df_train, df_test


def get_train_test_data(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Получение train/test данных разбитых по отдельности на объект-признаки и целевую переменную
    :param data_train: train датасет
    :param data_test: test датасет
    :return: набор данных train/test
    """
    target = kwargs["target_column"]
    groups_column = kwargs["groups_column"]
    x_train, x_test = (
        data_train.drop([target, groups_column], axis=1),
        data_test.drop([target, groups_column], axis=1),
    )
    y_train, y_test = (
        data_train.loc[:, target],
        data_test.loc[:, target],
    )

    g_train, g_test = (
        data_train.loc[:, groups_column],
        data_test.loc[:, groups_column],
    )

    return x_train, x_test, y_train, y_test, g_train, g_test


def _split_train_test_by_groups(dataset: pd.DataFrame,
                                groups: pd.Series,
                                test_size=None,
                                random_state=None):
    """
    Разделение данных на train/test в соответствие с группами
    :param dataset: датасет
    :return: train/test датасеты
    """
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(gss.split(dataset, groups=groups))
    return dataset.iloc[train_idx], dataset.iloc[test_idx]
