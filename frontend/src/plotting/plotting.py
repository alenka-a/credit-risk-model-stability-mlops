"""
Модуль: Отрисовка графиков
Версия: 1.0
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def barplot(df_data: pd.DataFrame, col_main: str, title: str) -> Figure:
    """
    Построение barplot с нормированными данными с выводом значений на графике
    :param df_data: датасет
    :param col_main: признак анализа, по нему происходит нормализация
    :param title: название графика
    :return: поле графика
    """
    norm_target = (df_data[col_main].value_counts(
        normalize=True).mul(100).rename("percentage").reset_index())

    fig = plt.figure(figsize=(15, 7))
    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_main,
                     palette='rocket',
                     data=norm_target)

    for container in ax.containers:
        ax.bar_label(container, fontsize=12, fmt='%.2f')

    plt.title(title, fontsize=16)
    plt.xlabel(col_main, fontsize=14)
    plt.ylabel("Percentage", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    sns.move_legend(ax, loc='best', fontsize=14, title_fontsize=14)
    return fig


def barplot_group(df_data: pd.DataFrame,
                  col_main: str,
                  col_group: str,
                  title: str,
                  is_horizontal: bool = True,
                  use_bar_labels: bool = True) -> Figure:
    """
    Построение barplot с группированными и нормированными данными,
    с выводом значений на графике
    :param df_data: датасет
    :param col_main: признак для анализа и нормалиации
    :param col_group: признак для группировки
    :param title: название графика
    :param is_horizontal: признак горизонтальной ориентации графика
    :param use_bar_labels: признак использования меток на столбцах графика
    :return: поле графика
    """
    data = (df_data.groupby(col_group)[col_main].value_counts(
        normalize=True).rename('percentage').mul(
        100).reset_index().sort_values(by=col_group))

    fig = plt.figure(figsize=(15, 7))
    if is_horizontal:
        ax = sns.barplot(x="percentage",
                         y=col_main,
                         hue=col_group,
                         data=data,
                         palette='rocket',
                         orient="h")
        xlabel = 'Percentage'
        ylabel = col_main
    else:
        ax = sns.barplot(x=col_main,
                         y="percentage",
                         hue=col_group,
                         data=data,
                         palette='rocket',
                         orient="v")
        xlabel = col_main
        ylabel = 'Percentage'

    if use_bar_labels:
        for container in ax.containers:
            ax.bar_label(container, fontsize=12, fmt='%.2f')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    sns.move_legend(ax, loc='best', fontsize=14, title_fontsize=14)
    return fig


def kdeplot(df_data: pd.DataFrame, x: str, hue: str, title: str,
            xlabel: str) -> Figure:
    """
    Построение кривой плотности распределения вероятностей

    :param df_data: датасет
    :param x: признак, для которого строится распределение
    :param hue: признак, по которому происходит разбивка на подмножества
    :param title: название графика
    :param xlabel: метка для оси X
    :return: поле графика
    """
    fig = plt.figure(figsize=(15, 7))
    ax = sns.kdeplot(data=df_data,
                     x=x,
                     hue=hue,
                     common_norm=False,
                     palette='rocket',
                     fill=True)

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    sns.move_legend(ax, loc='center right', fontsize=14, title_fontsize=14)
    return fig


def boxplot(df_data: pd.DataFrame, x: str, y: str, title: str, xlabel: str,
            ylabel: str) -> Figure:
    """
    Построение графика boxplot
    :param df_data: датасет
    :param x: признак для отрисовки по оси Х, по нему также будет выполняться группировка
    :param y: признак для отрисовки по оси Y
    :param title: название графика
    :param xlabel: метка для оси X
    :param ylabel: метка для оси Y
    :return: поле графика
    """
    fig = plt.figure(figsize=(15, 7))
    ax = sns.boxplot(x=x, y=y, hue=x, data=df_data, palette='rocket')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    sns.move_legend(ax, loc='center right', fontsize=14, title_fontsize=14)
    return fig
