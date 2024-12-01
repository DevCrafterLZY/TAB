# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
from ts_benchmark.evaluation.metrics.vus_metrics import metricor, generate_curve
from ts_benchmark.evaluation.metrics.utils import get_list_anomaly, find_length
from sklearn.metrics import precision_recall_curve

__all__ = ["auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]
# __all__ = ["best_ratio", "best_accuracy", "best_f_score", "best_precision", "best_recall", "auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]


metricor_grader = metricor()

def best_ratio(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return anomaly_rate
def best_f_score(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return best_f1_score


def best_accuracy(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return accuracy


def best_recall(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return recall


def best_precision(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    # Calculate F1 scores
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)

    # Find the index of the best F1 score
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # Get the best F1 score and the corresponding threshold
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[best_f1_score_index]

    # Create binary predicted labels based on the best threshold
    predicted_labels = [1 if p >= best_threshold else 0 for p in predicted]

    # Calculate confusion matrix components
    true_positives = sum((a == 1 and p == 1) for a, p in zip(actual, predicted_labels))
    false_positives = sum((a == 0 and p == 1) for a, p in zip(actual, predicted_labels))
    false_negatives = sum((a == 1 and p == 0) for a, p in zip(actual, predicted_labels))
    true_negatives = sum((a == 0 and p == 0) for a, p in zip(actual, predicted_labels))

    # Calculate precision, recall, and accuracy
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / len(actual)
    anomaly_rate = sum(predicted_labels) / len(actual)

    return precision

def auc_roc(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.roc_auc_score(actual, predicted)


def auc_pr(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.average_precision_score(actual, predicted)


def R_AUC_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_ROC


def R_AUC_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_PR


def VUS_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_ROC


def VUS_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    # slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_PR


# y_test = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
# pred_labels = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
# a = VUS_ROC(y_test, pred_labels)
# b = VUS_PR(y_test, pred_labels)
# # vus_results = get_range_vus_roc(y_test, pred_labels, 100)  # default slidingWindow = 100
# print("VUS_ROC", a)
# print("VUS_PR", b)

#
# import numpy as np
#
#
# score = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# label = np.array([0, 1, 1, 0, 1, 1, 1, 1])
# print("auc_roc:", auc_roc(label, score, label))
# print("auc_pr:", auc_pr(label, score, label))
# print("R_AUC_ROC:", R_AUC_ROC(label, score, label))
# print("R_AUC_PR:", R_AUC_PR(label, score, label))
#
# print("VUS_ROC:", VUS_ROC(label, score, label))
# print("VUS_PR:", VUS_PR(label, score, label))
