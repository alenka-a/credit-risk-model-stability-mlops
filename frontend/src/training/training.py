"""
Модуль: Тренировка модели и отображение результатов
Версия: 1.0
"""
import os
import json
import requests
import joblib
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def train(config: dict, endpoint: str) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    train_config = config["train"]
    if os.path.exists(train_config["metrics_path"]):
        with open(train_config["metrics_path"]) as json_file:
            prev_metrics = json.load(json_file)
    else:
        prev_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0, "gini_stability": 0,
                        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}}

    with st.spinner("Выполняется обучение модели..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]
    plot_metrics(prev_metrics, new_metrics)
    plot_confusion_matrix(prev_metrics["confusion_matrix"], new_metrics["confusion_matrix"])

    if train_config["optuna_enable"]:
        study = joblib.load(train_config["study_path"])
        fig_imp = plot_param_importances(study)
        fig_history = plot_optimization_history(study)

        st.subheader("Hyperparameter importances")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.subheader("Optimization history of all trials in a study")
        st.plotly_chart(fig_history, use_container_width=True)


def plot_metrics(prev_metrics: dict, new_metrics: dict):
    st.subheader("Metrics")
    metrics_names = {
        "ROC-AUC": "roc_auc",
        "Precision": "precision",
        "Recall": "recall",
        "F1 score": "recall",
        "Gini Stability": "gini_stability",
    }
    columns = st.columns(5)
    for column, (title, metric) in zip(columns, metrics_names.items()):
        column.metric(
            title,
            f"{new_metrics[metric]:.4f}",
            f"{new_metrics[metric] - prev_metrics[metric]:.4f}",
        )


def plot_confusion_matrix(prev_confusion_matrix: dict, new_confusion_matrix: dict):
    metrics_names = {
        "TN": "tn",
        "FP": "fp",
        "FN": "fn",
        "TP": "tp",
    }

    st.subheader("Confusion Matrix")
    row1 = st.columns(2)
    row2 = st.columns(2)
    for column, (title, metric) in zip(row1 + row2, metrics_names.items()):
        container = column.container(border=True)
        container.metric(
            title,
            new_confusion_matrix[metric],
            new_confusion_matrix[metric] - prev_confusion_matrix[metric],
        )
