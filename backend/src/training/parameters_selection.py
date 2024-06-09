"""
Модуль: Подбор параметров модели
Версия: 1.0
"""
import json
import optuna
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from optuna import Study

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


def find_optimal_params(
        data_x: pd.DataFrame,
        data_y: pd.Series,
        groups: pd.Series,
        **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param groups: данные с информацией о группах, которые будут использоваться при делении данные на фолды
    :return: [LGBMClassifier tuning, Study]
    """

    lr_study = find_learning_rate(data_x, data_y, groups, **kwargs)
    learning_rate = lr_study.best_params["learning_rate"]

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                study_name="LGB")
    function = lambda trial: objective(
        trial, data_x, data_y, groups,
        n_estimators=kwargs["optuna_n_estimators"],
        n_folds=kwargs["n_folds"],
        learning_rate=learning_rate,
        random_state=kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["optuna_n_trials"], show_progress_bar=True)
    return study


def find_learning_rate(
        data_x: pd.DataFrame,
        data_y: pd.Series,
        groups: pd.Series,
        **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param groups: данные с информацией о группах, которые будут использоваться при делении данные на фолды
    :return: [LGBMClassifier tuning, Study]
    """

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                study_name="LGB_LR")
    function = lambda trial: objective_lr(
        trial, data_x, data_y, groups,
        n_estimators=kwargs["optuna_n_estimators"],
        n_folds=kwargs["n_folds"],
        random_state=kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["optuna_n_trials"], show_progress_bar=True)
    return study


def objective(trial: optuna.trial.Trial,
              data_x: pd.DataFrame,
              data_y: pd.Series,
              groups: pd.Series,
              n_estimators: int,
              n_folds: int,
              learning_rate: float,
              random_state=10):
    """
        Целевая функция для поиска параметров
        :param trial: объект Trial, предоставляющий интерфейс для получения предлагаемых параметров
        :param data_x: данные объект-признаки
        :param data_y: данные с целевой переменной
        :param groups: данные с информацией о группах, которые будут использоваться при делении данные на фолды
        :param n_estimators: кол-во базовых алгоритмов
        :param n_folds: кол-во фолдов
        :param learning_rate: параметр модели learning_rate (скорость обучения)
        :param random_state: random_state
        :return: среднее значение метрики по фолдам
        """
    # TODO Параметры вынести в конфиг
    lgbm_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [n_estimators]),
        "learning_rate": trial.suggest_categorical("learning_rate", [learning_rate]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 70000, step=100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
        "min_split_gain": trial.suggest_int("min_gain_to_split", 0, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 0.9),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
    }

    return fit_lgbm_with_cross_validation(trial, data_x, data_y, groups, n_folds, random_state, lgbm_params)


def objective_lr(trial: optuna.trial.Trial,
                 data_x: pd.DataFrame,
                 data_y: pd.Series,
                 groups: pd.Series,
                 n_estimators: int,
                 n_folds: int,
                 random_state=10):
    """
        Целевая функция для поиска learning_rate
        :param trial: объект Trial, предоставляющий интерфейс для получения предлагаемых параметров
        :param data_x: данные объект-признаки
        :param data_y: данные с целевой переменной
        :param groups: данные с информацией о группах, которые будут использоваться при делении данные на фолды
        :param n_estimators: кол-во базовых алгоритмов
        :param n_folds: кол-во фолдов
        :param random_state: random_state
        :return: среднее значение метрики по фолдам
        """
    lgbm_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [n_estimators]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
    }

    return fit_lgbm_with_cross_validation(trial, data_x, data_y, groups, n_folds, random_state, lgbm_params)


def fit_lgbm_with_cross_validation(trial: optuna.trial.Trial,
                                   data_x: pd.DataFrame,
                                   data_y: pd.Series,
                                   groups: pd.Series,
                                   n_folds: int,
                                   random_state=10,
                                   lgbm_params: dict = None):
    """
    Обучает классификатор LGBMClassifier с кросс-валидацией

    Для деления данных на фолды используется StratifiedGroupKFold,
    который делит данные так, чтобы группы не пересекались
    :param trial: объект Trial
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param groups: данные с информацией о группах, которые будут использоваться при делении данные на фолды
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :param lgbm_params: словарь с параметрами для обучения классификатора
    :return: среднее значение метрики ROC-AUC по фолдам
    """
    cv = StratifiedGroupKFold(n_splits=n_folds,
                              shuffle=True,
                              random_state=random_state)

    cv_predicts = np.empty(n_folds)
    for idx, (train_idx, val_idx) in enumerate(cv.split(data_x, data_y, groups=groups)):
        x_train, x_val = data_x.iloc[train_idx], data_x.iloc[val_idx]
        y_train, y_val = data_y.iloc[train_idx], data_y.iloc[val_idx]

        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc", report_interval=100)
        model = LGBMClassifier(scale_pos_weight=ratio, silent=True, **lgbm_params)
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_val, y_val)],
                  eval_metric="auc",
                  callbacks=[early_stopping(100, verbose=False),
                             pruning_callback
                             ])

        predicts_proba = model.predict_proba(x_val)[:, 1]
        cv_predicts[idx] = roc_auc_score(y_val, predicts_proba)

    return np.mean(cv_predicts)


def save_optimal_params(study: Study, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(study.best_params, f)
