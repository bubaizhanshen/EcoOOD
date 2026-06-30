from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearmanr(y_true, y_pred, nan_policy="omit").statistic),
        "bias": float(np.mean(y_pred - y_true)),
    }


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    uncertainty: np.ndarray,
    novelty: np.ndarray,
) -> dict[str, float]:
    width = upper - lower
    covered = (y_true >= lower) & (y_true <= upper)
    error = np.abs(y_true - (lower + upper) / 2)
    return {
        "coverage": float(covered.mean()),
        "mean_interval_width": float(width.mean()),
        "uncertainty_error_corr": _safe_corr(uncertainty, error),
        "uncertainty_novelty_corr": _safe_corr(uncertainty, novelty),
    }


def risk_coverage(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(score)
    abs_error = np.abs(y_true[order] - y_pred[order])
    coverages = np.linspace(1 / len(order), 1.0, len(order))
    cumulative_risk = np.cumsum(abs_error) / np.arange(1, len(abs_error) + 1)
    aurc = float(np.trapezoid(cumulative_risk, coverages))
    return coverages, cumulative_risk, aurc


def selective_risk_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    abstain_fractions: list[float] | tuple[float, ...],
) -> pd.DataFrame:
    order = np.argsort(score)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    abs_error_sorted = np.abs(y_true_sorted - y_pred_sorted)
    catastrophic_threshold = float(np.quantile(np.abs(y_true - y_pred), 0.9))
    baseline_catastrophic_rate = float(np.mean(np.abs(y_true - y_pred) >= catastrophic_threshold))

    rows: list[dict[str, float]] = []
    for abstain_fraction in abstain_fractions:
        keep_fraction = 1.0 - float(abstain_fraction)
        keep_count = max(1, int(np.floor(len(order) * keep_fraction)))
        retained_error = abs_error_sorted[:keep_count]
        retained_true = y_true_sorted[:keep_count]
        retained_pred = y_pred_sorted[:keep_count]
        catastrophic_rate = float(np.mean(retained_error >= catastrophic_threshold))
        rows.append(
            {
                "abstain_fraction": float(abstain_fraction),
                "retained_fraction": float(keep_count / len(order)),
                "rmse": float(np.sqrt(mean_squared_error(retained_true, retained_pred))),
                "mae": float(mean_absolute_error(retained_true, retained_pred)),
                "mean_abs_error": float(np.mean(retained_error)),
                "catastrophic_error_rate": catastrophic_rate,
                "catastrophic_error_reduction": float(baseline_catastrophic_rate - catastrophic_rate),
                "error_reduction": float(np.mean(np.abs(y_true - y_pred)) - np.mean(retained_error)),
            }
        )
    return pd.DataFrame(rows)


def ood_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    ood_label: np.ndarray | None,
) -> dict[str, float]:
    error = np.abs(y_true - y_pred)
    catastrophic = error >= np.quantile(error, 0.9)
    _, _, aurc = risk_coverage(y_true, y_pred, score)
    metrics = {
        "aurc": aurc,
        "catastrophic_error_capture_rate": float(np.mean(score[catastrophic] >= np.quantile(score, 0.8))) if catastrophic.any() else float("nan"),
    }
    if ood_label is not None and len(np.unique(ood_label)) > 1:
        metrics["auroc_id_vs_ood"] = float(roc_auc_score(ood_label, score))
        metrics["aupr_id_vs_ood"] = float(average_precision_score(ood_label, score))
    else:
        metrics["auroc_id_vs_ood"] = float("nan")
        metrics["aupr_id_vs_ood"] = float("nan")
    return metrics


def reference_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return {
            "auroc_id_vs_ood": float("nan"),
            "aupr_id_vs_ood": float("nan"),
            "fpr95": float("nan"),
        }
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    above = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[above[0]]) if len(above) else float("nan")
    return {
        "auroc_id_vs_ood": auroc,
        "aupr_id_vs_ood": aupr,
        "fpr95": fpr95,
    }


def score_method_metrics(
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    known_ood: np.ndarray | None,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> dict[str, float | str]:
    direct = ood_metrics(y_true, y_pred, score, known_ood)
    reference = reference_ood_metrics(id_scores=id_scores, ood_scores=ood_scores)
    return {
        "method": method,
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
        "aurc": direct["aurc"],
        "catastrophic_error_capture_rate": direct["catastrophic_error_capture_rate"],
        "known_ood_auroc": direct["auroc_id_vs_ood"],
        "known_ood_aupr": direct["aupr_id_vs_ood"],
        "reference_auroc": reference["auroc_id_vs_ood"],
        "reference_aupr": reference["aupr_id_vs_ood"],
        "reference_fpr95": reference["fpr95"],
    }


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_predictions(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
