from __future__ import annotations

from sklearn.metrics import average_precision_score, precision_recall_curve as sk_precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve as sk_roc_curve

from .validation import check_binary_array, check_same_length, check_score_array


def precision_recall_curve(y_true, y_score):
    y_true = check_binary_array(y_true, name="y_true")
    y_score = check_score_array(y_score, name="y_score")
    check_same_length(y_true, y_score)
    return sk_precision_recall_curve(y_true, y_score)


def roc_curve(y_true, y_score):
    y_true = check_binary_array(y_true, name="y_true")
    y_score = check_score_array(y_score, name="y_score")
    check_same_length(y_true, y_score)
    return sk_roc_curve(y_true, y_score)


def auc_pr(y_true, y_score) -> float:
    y_true = check_binary_array(y_true, name="y_true")
    y_score = check_score_array(y_score, name="y_score")
    check_same_length(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def auc_roc(y_true, y_score) -> float:
    y_true = check_binary_array(y_true, name="y_true")
    y_score = check_score_array(y_score, name="y_score")
    check_same_length(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))
