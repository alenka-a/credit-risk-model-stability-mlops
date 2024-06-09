"""
Модуль: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st
from src.data.loading import get_dataset, dataset_to_bytes
from src.plotting.plotting import barplot_group, barplot, kdeplot, boxplot
from src.training.training import train
from src.prediction.prediction import predict

CONFIG_PATH = "../config/params.yml"


def overview():
    """
    Страница с описанием проекта
    """
    st.title("MLOps project:  Home Credit - Credit Risk Model Stability")
    st.markdown(
        """
        Данные взяты из соревания на Kaggle. Основная информация по данным представлена здесь https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data

        Цель - предсказать, какие клиенты с большей вероятностью не смогут выплатить свои кредиты. Оценка должна быть стабильна с течением времени
        
        Данные для соревнований изначально были преобразованы, есть следующие обозначения для сходных групп преобразований:
        - P - Преобразование DPD (просроченных дней)
        - M - Маскировка категорий
        - A - Преобразование сумм
        - D - Преобразование дат
        - T, L - Прочие преобразования
        Преобразования внутри группы обозначаются заглавной буквой в конце имени предиктора.
        
        Будем рассматривать только выборку данных из 100 000 объектов, которые уже предварительно объединены, частично отфильтрованы и сконвертированы, согласно вышеприведенному описанию.
        Основные колонки:
        - case_id -  уникальный идентификатор для каждого кредитного кейса. Этот идентификатор понадобится вам для объединения соответствующих таблиц с базовой таблицей.
        - moth_decision, weekday_decision -  месяц и день недели, когда было принято решение об одобрении кредита.
        - moth_decision -  дата, когда было принято решение об одобрении кредита.
        - WEEK_NUM - номер недели, используемый для агрегирования. Будет использоваться в оценке стабильности по Джини
        - **target - целевое значение, определяемое по истечении определенного периода времени в зависимости от того, допустил ли клиент дефолт по конкретному кредитному делу (займу).**
        
        Определения всех остальных столбцов можно найти в файле data/raw/feature_definitions.csv, но т.к. исторические данные были сгруппированы, то часть столбцов имеет префиксы:
         - "max_" - маскимальное значение
         - "mean_" - среднее значение
         - "mode_" - мода
         - "last_" - последнее значение
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    df_data = get_dataset(dataset_path=config["preprocessing"]["train_path"])
    st.write(df_data.head())

    # plotting with checkbox
    show_target_barplot = st.sidebar.checkbox("Соотношение клиентов, выплативших и не выплативших кредит")
    show_target_weeknum_barplot = st.sidebar.checkbox("Соотношение клиентов в разрезе номера недели недели")
    show_age_distribution = st.sidebar.checkbox("Распределение возраста клиентов")
    show_annuity_amount_boxplot = st.sidebar.checkbox("Соотношение размера ежемесячного платежа")
    show_tax_deductions_amount_boxplot = st.sidebar.checkbox("Соотношение суммы налоговых вычетов")
    show_riskassesment_barplot = st.sidebar.checkbox(
        "Соотношение клиентов в разрезе оценки рисков и целевой переменной")

    if show_target_barplot:
        st.pyplot(
            barplot(df_data=df_data,
                    col_main="target",
                    title="Соотношение клиентов, выплативших и не выплативших кредит")
        )

    if show_target_weeknum_barplot:
        st.pyplot(
            barplot_group(
                df_data=df_data,
                col_main="target",
                col_group="WEEK_NUM",
                title=
                "Соотношение клиентов, выплативших и не выплативших кредит в разрезе номера недели недели ",
                is_horizontal=False,
                use_bar_labels=False)
        )

    if show_age_distribution:
        st.pyplot(
            kdeplot(df_data=df_data,
                    x="birth_259D",
                    hue="target",
                    title='Распределение возраста клиентов',
                    xlabel='Age')
        )

    if show_annuity_amount_boxplot:
        st.pyplot(
            boxplot(
                df_data=df_data,
                x="target",
                y="annuity_780A",
                title="Соотношение размера ежемесячного платежа",
                xlabel="Target",
                ylabel="Monthly annuity amount"
            )
        )

    if show_tax_deductions_amount_boxplot:
        st.pyplot(
            boxplot(
                df_data=df_data,
                x="target",
                y="mean_amount_4527230A",
                title="Соотношение суммы налоговых вычетов",
                xlabel="Target",
                ylabel="Tax deductions amount"
            )
        )

    if show_riskassesment_barplot:
        st.pyplot(
            barplot_group(
                df_data=df_data,
                col_main="riskassesment_302T",
                col_group="target",
                title="Соотношение клиентов в разрезе оценки рисков и целевой переменной."
            )
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if st.button("Start training"):
        train(config=config, endpoint=config["endpoints"]["train"])


def prediction():
    """
    Получение предсказаний по данным из файла
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    upload_file = st.file_uploader(
        "Загрузите файл", type=["parquet"], accept_multiple_files=False,
        label_visibility="hidden"
    )
    if upload_file:
        dataset = get_dataset(upload_file)
        st.write("Dataset load")
        st.write(dataset.head())

        files = dataset_to_bytes(dataset=dataset, type_data="Test")
        if os.path.exists(config["train"]["model_path"]):
            if st.button("Predict"):
                predict(data=dataset, endpoint=config["endpoints"]["predict"], files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Главная страница
    """
    page_names_to_funcs = {
        "Overview": overview,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction from file": prediction,
    }
    selected_page = st.sidebar.radio(label="Раздел", options=page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
