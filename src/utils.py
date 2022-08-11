from collections import defaultdict
from typing import Dict

import numpy as np
import scipy.fft as fft
from sklearn.metrics import classification_report


def arr_to_dict(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Массив словарей (данные по каждому человеку) приводит к виду:
    {
        key (e.g. label, time, ...): np.ndarray[n_users, dim_of_key]
    }
    """
    df_dict = defaultdict(list)
    for sample in arr:
        df_dict["signal_1"].append(sample["value"][:, 0])
        df_dict["signal_2"].append(sample["value"][:, 1])
        df_dict["time"].append(int(sample["time"]))
        df_dict["label"].append(sample["label"])
    result = {}
    for key, value in df_dict.items():
        result[key] = np.stack(value)
    del df_dict
    return result


def metric_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
) -> None:
    """
    Приводит метрики (precision И recall) для модели
    """
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    print("Train")
    print(classification_report(y_train, pred_train, target_names=classes), "\n")

    print("Val")
    print(classification_report(y_val, pred_val, target_names=classes), "\n")

    print("Test")
    print(classification_report(y_test, pred_test, target_names=classes), "\n")


def log_arr(arr: np.ndarray) -> np.ndarray:
    """
    Логарифмирование массива
    """
    mask = arr == 0
    arr = np.log(arr)
    arr[mask] = 0
    return arr


def fourier_transform(arr):
    """
    К сигналу применяется преобразование Фурье и вычисялется модуль полученных значений
    """
    transformed = fft.rfft(arr)
    return (transformed.imag**2 + transformed.real**2) ** 0.5
