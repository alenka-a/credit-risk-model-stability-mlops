"""
Модуль: Предобработка данных
Версия: 1.0
"""
import pandas as pd
import numpy as np
import json

from typing import Any


def pipeline_preprocessing(df_data: pd.DataFrame,
                           flg_prediction: bool = True,
                           debug=False,
                           **kwargs):
    """
    Пайплайн по предобработке данных
    :param df_data: датасет
    :param flg_prediction: флаг для prediction
    :param debug: флаг для логирования дополнительной информации
    :return: датасет
    """
    df_data = df_data.drop(kwargs['drop_columns'], axis=1, errors="ignore")

    except_columns = [kwargs["target_column"], kwargs["groups_column"]]

    # Merge columns
    map_merge_columns = kwargs["map_merge_columns"]
    if map_merge_columns:
        for key in map_merge_columns.keys():
            df_data = merge_columns(df_data,
                                    main_column=map_merge_columns[key][0],
                                    add_column=map_merge_columns[key][1],
                                    new_column=key)

    # Bins
    map_bins_columns = kwargs["map_bins_columns"]
    if map_bins_columns:
        for key in map_bins_columns.keys():
            df_data[f"{key}_bins"] = df_data[key].apply(lambda x: get_bins(
                x,
                first_val=map_bins_columns[key][0],
                second_val=map_bins_columns[key][1],
            ))

    map_replace_rare_values = kwargs["map_replace_rare_values"]
    unique_values_path = kwargs["unique_values_path"]

    if flg_prediction:
        if map_replace_rare_values:
            df_data = replace_values_evaluate(
                df_data,
                map_replace_rare_values.keys(),
                unique_values_path=unique_values_path)

        df_data = check_columns_evaluate(df_data=df_data,
                                         unique_values_path=unique_values_path)

    else:
        if map_replace_rare_values:
            # Replace rare values
            df_data = replace_rare_values(df_data,
                                          map_replace_rare_values,
                                          debug=debug)

    # Fill nans
    cat_columns = df_data.select_dtypes("object").columns.to_list()
    numeric_columns = df_data.select_dtypes(
        exclude=["object", "bool", "category"]).columns.to_list()
    df_data = fill_nans(df_data, value="Unknown", columns=cat_columns)
    df_data = fill_nans(df_data, value=0, columns=numeric_columns)

    # To category
    category_maps = {
        key: "category"
        for key in df_data.select_dtypes(["object"]).columns
    }
    df_data = transform_types(df_data=df_data,
                              change_type_columns=category_maps)

    if not flg_prediction:
        corr_matrix_settings = kwargs["corr_matrix_settings"]
        if corr_matrix_settings and corr_matrix_settings["enable"]:
            df_data = drop_columns_by_corr_matrix(
                df_data,
                except_columns=except_columns,
                threshold=corr_matrix_settings["threshold"],
                debug=debug)

        save_unique_train_data(
            df_data=df_data,
            drop_columns=except_columns,
            unique_values_path=unique_values_path,
        )

    return df_data


def get_bins(data: (int, float), first_val: (int, float),
             second_val: (int, float)) -> str:
    """
    Генерация бинов для разных признаков
    :param data: числовое значение
    :param first_val: первый порог значения для разбиения на бины
    :param second_val: второй порог значения для разбиения на бины
    :return: наименование бина
    """
    assert isinstance(data, (int, float)), "Проблема с типом данных в признаке"
    result = ("small" if data <= first_val else
              "medium" if first_val < data <= second_val else "large")
    return result


def fill_nans(df_data: pd.DataFrame, value: Any,
              columns: list[str]) -> pd.DataFrame:
    """
    Заполняет пропуски переданным значением
    :param df_data: датасет
    :param value: значение, которым необходимо заполнить пропуску
    :param columns: список признаков
    :return: датасет
    """
    for column in columns:
        if np.mean(df_data[column].isnull()) > 0:
            df_data[column] = df_data[column].fillna(value)
    return df_data


def merge_columns(df_data: pd.DataFrame, main_column: str, add_column: str,
                  new_column: str) -> pd.DataFrame:
    """
    Объединение двух дублирующихся признаков в один.
    Если значение в main_column заполнено то берется это значение,
    иначе берется значение из add_column
    :param df_data: датасет
    :param main_column: основная признак
    :param add_column: дополнительный признак
    :param new_column: название нового признака
    :return: датасет
    """
    df_data[new_column] = df_data.apply(
        lambda x: x[add_column]
        if pd.isnull(x[main_column]) else x[main_column],
        axis=1).astype(df_data[main_column].dtype)

    df_data = df_data.drop([main_column, add_column], axis=1)
    return df_data


def replace_rare_values(df_data: pd.DataFrame,
                        columns: dict,
                        value: Any = "Other",
                        debug: bool = False):
    """
    Замена редко встречающихся значений
    :param df_data: датасет
    :param columns: словарь, где key - имя столбца,
                    value - порог, при котором заменяется значение столбца,
                    если его частота ниже
    :param value: новое значение
    :param debug: признак логирования дополнительной информации
    :return: датасет
    """
    for column in columns.keys():
        nunique = df_data[column].nunique()
        rare_values = df_data[column].value_counts(
            dropna=True, normalize=True)[df_data[column].value_counts(
            dropna=True, normalize=True) < columns[column]].index

        df_data[column] = df_data[column].apply(lambda x: "Other"
        if x in rare_values else x)

        if debug:
            print(
                f"{column} reduced nunique from {nunique} to {df_data[column].nunique()}"
            )
    return df_data


def replace_values_evaluate(df_data: pd.DataFrame,
                            columns: str,
                            unique_values_path: str,
                            value: Any = "Other"):
    """
    Замена значений, которые были заменены в train
    :param df_data: датасет
    :param columns: список столбцов
    :param unique_values_path: путь до списока с признаками train
    :param value: новое значение
    :return: датасет
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    for column in columns:
        train_values = unique_values[column]
        df_data[column] = df_data[column].apply(
            lambda x: x if x in train_values or pd.isnull(x) else "Other")
    return df_data


def drop_columns_by_corr_matrix(df_data: pd.DataFrame,
                                except_columns: list[str],
                                threshold: float = 0.9,
                                debug=False) -> pd.DataFrame:
    """
    Удаление сильно скоррелированных признаков
    :param df_data: датасет
    :param except_columns: исключаемые признаки
    :param threshold: порог, при котором признак удаляется
    :param debug: признак логирования дополнительной информации
    :return: датасет
    """
    cor_matrix = df_data.drop(columns=except_columns,
                              axis=1).corr(method='spearman',
                                           numeric_only=True).abs()
    cor_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

    to_drop = [
        column for column in cor_matrix.columns
        if any(cor_matrix[column] > threshold)
    ]

    if debug:
        for column in to_drop:
            print(column)
            print(cor_matrix[column][cor_matrix[column] > threshold])
            print()
    return df_data.drop(to_drop, axis=1)


def transform_types(df_data: pd.DataFrame,
                    change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param df_data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return df_data.astype(change_type_columns, errors="raise")


def save_unique_train_data(df_data: pd.DataFrame, drop_columns: list,
                           unique_values_path: str) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями для категориальных переменных
    и статистическими значениями для числовых переменных
    :param df_data: датасет
    :param drop_columns: список с признаками для удаления
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = df_data.drop(columns=drop_columns, axis=1, errors="ignore")

    category_columns = unique_df.select_dtypes(["object", "bool",
                                                "category"]).columns.to_list()
    dict_unique = {
        key:
            unique_df[key].unique().tolist()
            if key in category_columns else unique_df[key].describe().tolist()
        for key in unique_df.columns
    }

    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def check_columns_evaluate(df_data: pd.DataFrame,
                           unique_values_path: str) -> pd.DataFrame:
    """
    Удаление признаков, которых нет в train,
    проверка на наличие признаков из train
    и упорядочивание признаков согласно train
    :param df_data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    missing_features = set(column_sequence).difference(set(df_data.columns))

    assert not any(missing_features), f"Пропущены признаки: {missing_features}"
    return df_data[column_sequence]
