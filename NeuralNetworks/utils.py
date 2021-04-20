import math
import os
from typing import Tuple

import matplotlib.pylab as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns


def get_data(test_data, test_events, channels_num):
    sampling_freq = test_data.info['sfreq']
    start_stop_seconds = np.array([0, 130])
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    channel_index = [x for x in range(channels_num)]
    temp_array = test_data[channel_index, start_sample:stop_sample]
    raw_selection = pd.DataFrame(columns=["time"] + [f"ch_{x}" for x in range(1, channels_num + 1)])
    raw_selection["time"] = temp_array[1]
    for i in range(len(temp_array[0])):
        raw_selection[f"ch_{i+1}"] = temp_array[0][i]
    raw_selection["class"] = None
    for event_idx in range(1, len(test_events[0])):
        raw_selection.loc[test_events[0][event_idx - 1][0]:test_events[0][event_idx][0],"class"] = test_events[0][event_idx][2]
    raw_selection.loc[test_events[0][-1][0]:,"class"] = test_events[0][-1][2]
    return raw_selection

## ReWrite!
def pretty_spectrogram(data: np.ndarray, K: int, N: int, n_samples: int, cutoff=0, norm: bool=True) -> np.ndarray:
    """Функция для рассчета спектра по отрезкам сигнала
    Arguments:
        data: массив с отсчетами сигнала
        K: количество отсчетов БПФ
        N: количество отсчетов сигнала
        n_samples: количество отрезков сигнала
        cutoff: количество вырезанных отрезков из начала сигнала, по умолчанию 0

    return:
        Спектограмму сигнала по всем отрезкам
    """

    spec_data = []
    for x in range(n_samples):
        spec_data.append(np.append(data[x * N:x * N + N], np.zeros(K - N), axis=0))
    spec_data = np.asarray(spec_data)

    spec = np.fft.fft(spec_data, axis=1, norm="ortho")
    spec = np.abs(spec) ** 2
    if norm:
        return spec[cutoff:, 12:53] / np.max(spec[cutoff:, 12:53])
    return spec[cutoff:, 12:53]

def create_cnn_data_from_raw_electrods(raw_data: np.ndarray, test_events: tuple, electrods: list, people_id: int, run_id: int, K: int, N: int, n_samples: int, cutoff:int=0, norm:bool=True):
    """Функция для построения датасетов для СНС
    Arguments:
        raw_data: набор входных данных
        test_events: разметка событий
        electrods: набор интересующих электродов
        people_id: id человека
        run_id: id эксперимента
        K: количество отсчетов БПФ
        N: количество отсчетов сигнала
        n_samples: количество отрезков сигнала
        cutoff: количество вырезанных отрезков из начала сигнала, по умолчанию 0

    return:
        Спектограмму сигнала по всем отрезкам
    """

    idx = []
    labels = []
    data = []
    for label in range(len(test_events[0]) - 1):
        if test_events[0][label, 2] != 1:
            idx.append((test_events[0][label, 0], test_events[0][label + 1, 0]))
            labels.append(test_events[0][label, 2] - 2)
    if test_events[0][-1, 2] != 1:
        idx.append((test_events[0][-1, 0], list(raw_data.index)[-1]))
        labels.append(test_events[0][-1, 2] - 2)
    
    true_labels = []
    
    if len(raw_data.iloc[idx[0][0]:idx[0][1], 1]) >= 640:
        dt = raw_data.iloc[idx[0][0]:idx[0][1], 1].astype("float64")[:640]
    else:
        dt = np.append(raw_data.iloc[idx[0][0]:idx[0][1], 1].astype("float64"), np.zeros(640 - len(raw_data.iloc[idx[0][0]:idx[0][1], 1])))

    one_vector = pretty_spectrogram(
        raw_data.iloc[idx[0][0]:idx[0][1], 1].astype("float64")[:640],
        K, N, n_samples, cutoff
    ).reshape((n_samples - cutoff, 41, 1))
    
    for ch in range(len(electrods) - 1):
        if len(raw_data.iloc[idx[0][0]:idx[0][1], ch + 1]) >= 640:
            dt = raw_data.iloc[idx[0][0]:idx[0][1], ch + 1].astype("float64")[:640]
        else:
            dt = np.append(raw_data.iloc[idx[0][0]:idx[0][1], ch + 1].astype("float64"), np.zeros(640 - len(raw_data.iloc[idx[0][0]:idx[0][1], ch + 1])))

        temp_vec = pretty_spectrogram(
            dt,
            K, N, n_samples, cutoff
        ).reshape((n_samples - cutoff, 41, 1))
        one_vector = np.dstack([one_vector, temp_vec])
    true_labels.append(labels[0])
    data.append(one_vector)
    
    for event in range(1, len(idx)):
        if len(raw_data.iloc[idx[event][0]:idx[event][1], 1]) >= 640:
            dt = raw_data.iloc[idx[event][0]:idx[event][1], 1].astype("float64")[:640]
        else:
            dt = np.append(raw_data.iloc[idx[event][0]:idx[event][1], 1].astype("float64"), np.zeros(640 - len(raw_data.iloc[idx[event][0]:idx[event][1], 1])))

        temp = pretty_spectrogram(
            dt,
            K, N, n_samples, cutoff
        ).reshape((n_samples - cutoff, 41, 1))

        for ch in range(len(electrods) - 1):
            if len(raw_data.iloc[idx[event][0]:idx[event][1], ch + 1]) >= 640:
                dt = raw_data.iloc[idx[event][0]:idx[event][1], ch + 1].astype("float64")[:640]
            else:
                dt = np.append(raw_data.iloc[idx[event][0]:idx[event][1], ch + 1].astype("float64"), np.zeros(640 - len(raw_data.iloc[idx[event][0]:idx[event][1], ch + 1])))
            temp_vec = pretty_spectrogram(
                dt,
                K, N, n_samples, cutoff
            ).reshape((n_samples - cutoff, 41, 1))
            temp = np.dstack([temp, temp_vec])
        
        true_labels.append(labels[event])
        data.append(temp)
    return data, true_labels
    
def create_data_from_raw_electrods(raw_data, test_events, electrods, people_id, run_id):
    idx = []
    labels = []
    for label in range(len(test_events[0]) - 1):
        if test_events[0][label, 2] != 1:
            idx.append((test_events[0][label, 0], test_events[0][label + 1, 0]))
            labels.append(test_events[0][label, 2] - 2)
    if test_events[0][-1, 2] != 1:
        idx.append((test_events[0][-1, 0], list(raw_data.index)[-1]))
        labels.append(test_events[0][-1, 2] - 2)
    
    one_vector = pretty_spectrogram(
        raw_data.iloc[idx[0][0]:idx[0][1], 1].astype("float64")[:640],
    ).reshape((-1))

    one_vector = one_vector / max(one_vector)
    for ch in range(len(electrods) - 1):
        if len(raw_data.iloc[idx[0][0]:idx[0][1], ch + 1]) >= 640:
            dt = raw_data.iloc[idx[0][0]:idx[0][1], ch + 1].astype("float64")[:640]
        else:
            dt = np.append(raw_data.iloc[idx[0][0]:idx[0][1], ch + 1].astype("float64"), np.zeros(640 - len(raw_data.iloc[idx[0][0]:idx[0][1], ch + 1])))
        temp_vec = pretty_spectrogram(
            dt,
        ).reshape((-1))
        one_vector = np.hstack([one_vector, temp_vec])
    one_vector = np.hstack([one_vector, labels[0], people_id, run_id])
    
    for event in range(1, len(idx)):
        temp = pretty_spectrogram(
            raw_data.iloc[idx[event][0]:idx[event][1], 1].astype("float64")[:640],
        ).reshape((-1))

        for ch in range(len(electrods) - 1):
            temp_vec = pretty_spectrogram(
                raw_data.iloc[idx[event][0]:idx[event][1], ch + 1].astype("float64")[:640],
            ).reshape((-1))
            temp = np.hstack([temp, temp_vec])
        temp = np.hstack([temp, labels[event], people_id, run_id])
        one_vector = np.vstack([one_vector, temp])
    
    return one_vector

def conf_matrix(data: Tuple[np.ndarray, np.ndarray], treshold: float = 0.5) -> dict:
    
    classification_metrics = {"TP" : 0, "FP": 0, "FN": 0, "TN": 0}
    diff = data[0] - (data[1] > treshold)
    classification_metrics["TP"] += float(len(np.where(data[0]==1)[0]) - np.sum(diff[np.where(data[0]==1)[0]]))
    classification_metrics["FP"] += float(np.sum(diff[np.where(data[0]==1)[0]]))
    classification_metrics["FN"] += float(len(np.where(diff<0)[0]))
    classification_metrics["TN"] += float(len(np.where(diff[np.where(data[0]==0)[0]] == 0)[0]))
    return classification_metrics

def confusion_data(data: tuple, score_beta: list = [1], treshold: float = .5) -> dict:
    
    classification_metrics = conf_matrix(data, treshold)
                
    condition_negative = float(classification_metrics["FP"] + classification_metrics["TN"]) + 1e-21
    condition_positive = float(classification_metrics["TP"] + classification_metrics["FN"]) + 1e-21
    predicted_condition_positive = float(classification_metrics["TP"] + classification_metrics["FP"]) + 1e-21
    predicted_condition_negative = float(classification_metrics["TN"] + classification_metrics["FN"]) + 1e-21
    total_population = float(len(data[0]))
    
    classification_metrics["Recall"] = float(classification_metrics["TP"]) / condition_positive
    classification_metrics["FPR"] = float(classification_metrics["FP"]) / condition_negative
    classification_metrics["FNR"] = float(classification_metrics["FN"]) / condition_positive
    classification_metrics["TNR"] = float(classification_metrics["TN"]) / condition_negative
    
    classification_metrics["Prevalence"] = condition_positive / total_population
    classification_metrics["Accuracy"] = (float(classification_metrics["TP"]) + float(classification_metrics["TN"])) / total_population
    classification_metrics["Precision"] = float(classification_metrics["TP"]) / predicted_condition_positive
    classification_metrics["FDR"] = float(classification_metrics["FP"]) / predicted_condition_positive
    classification_metrics["FOR"] = float(classification_metrics["FN"]) / predicted_condition_negative
    classification_metrics["NPV"] = float(classification_metrics["TN"]) / predicted_condition_negative
    
    try:
        classification_metrics["LR+"] = classification_metrics["Recall"] / classification_metrics["FPR"]
    except ZeroDivisionError:
        classification_metrics["LR+"] = 1
    
    try:
        classification_metrics["LR-"] = classification_metrics["FNR"] / classification_metrics["TNR"]
    except ZeroDivisionError:
        classification_metrics["LR-"] = 1        
    try:
        classification_metrics["DOR"] = classification_metrics["LR+"] / classification_metrics["LR-"]
    except ZeroDivisionError:
        classification_metrics["DOR"] = 1
    try:
        classification_metrics["PT"] = (math.sqrt(classification_metrics["Recall"]*(1 - classification_metrics["TNR"])) + classification_metrics["TNR"] - 1) / (classification_metrics["Recall"] + classification_metrics["TNR"] - 1)
    except ZeroDivisionError:
        classification_metrics["PT"] = 1
    classification_metrics["TS"] = classification_metrics["TP"] / (classification_metrics["TP"] + classification_metrics["FN"] + classification_metrics["FP"])
    
    classification_metrics["BA"] = (classification_metrics["Recall"] + classification_metrics["TNR"]) / 2
    try:
        classification_metrics["MCC"] = (classification_metrics["TP"] * classification_metrics["TN"] - classification_metrics["FP"] * classification_metrics["FN"]) / (math.sqrt((classification_metrics["TP"] + classification_metrics["FP"])*(classification_metrics["TP"] + classification_metrics["FN"])*(classification_metrics["TN"] + classification_metrics["FP"])*(classification_metrics["TN"] + classification_metrics["FN"])))
    except ZeroDivisionError:
        classification_metrics["MCC"] = 1
    classification_metrics["FM"] = math.sqrt(classification_metrics["Precision"] * classification_metrics["Recall"])
    
    classification_metrics["BM"] = classification_metrics["Recall"] + classification_metrics["TNR"] - 1
    classification_metrics["MK"] = classification_metrics["Precision"] + classification_metrics["NPV"] - 1
    
    
    for beta in score_beta:
        classification_metrics["F" + str(beta)] = (1.0 * beta ** 2 * classification_metrics["TP"]) / ((1+beta**2)*classification_metrics["TP"] + beta **2 * classification_metrics["FN"] + classification_metrics["FP"])    
    
    ones = np.sum(data[0])
    sorted_labels = np.hstack([data[1], data[0]])
    sorted_labels = sorted_labels[np.argsort(sorted_labels[:,0])][::-1]
    
    roc = [0]
    prc = np.zeros(shape=(2, 100))
    temp_roc_value = 0
    auc = 0
    
    for pos in range(len(sorted_labels)):
        if sorted_labels[pos][1] == 1:
            temp_roc_value += 1.0 / ones
        else:
            roc.append(temp_roc_value)
            auc += temp_roc_value
    
    idx = 0
    for temp_treshold in np.linspace(0, 1, num=prc.shape[1]):
        temp_matrix = conf_matrix(data, temp_treshold)
        
        try:
            prc[0, idx] = float(temp_matrix["TP"]) / float(temp_matrix["TP"] + temp_matrix["FP"])
        except ZeroDivisionError:
            prc[0, idx] = 1.0
        try:
            prc[1, idx] = float(temp_matrix["TP"]) / float(temp_matrix["TP"] + temp_matrix["FN"])
        except ZeroDivisionError:
            prc[1, idx] = 1.0
        
        idx += 1
    
    classification_metrics["ROC_AUC_cycle"] = auc / (len(sorted_labels) - ones)
    classification_metrics["ROC_AUC_analytic"] = (1 + classification_metrics["Recall"] - classification_metrics["FPR"]) / 2
    classification_metrics["ROC"] = roc
    classification_metrics["PRC"] = prc
    classification_metrics["PR_AUC"] = np.sum(prc[0, :]) / prc.shape[1]
    
    return classification_metrics
