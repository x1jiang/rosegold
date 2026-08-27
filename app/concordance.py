import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_concordance_metrics(df_merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates calibration and inter-annotator concordance metrics between Human Gold Standard and LLM Rose Gold labels.
    Expects DataFrame with columns 'human_positive' (bool) and 'llm_positive' (bool).
    """
    if df_merged.empty or len(df_merged) == 0:
        return {
            "total_evaluated": 0,
            "overall_agreement_pct": 0.0,
            "cohens_kappa": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "ppv": 0.0,
            "npv": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0
        }

    y_true = df_merged['human_positive'].astype(bool).values
    y_pred = df_merged['llm_positive'].astype(bool).values

    tp = int(np.sum((y_true == True) & (y_pred == True)))
    fp = int(np.sum((y_true == False) & (y_pred == True)))
    fn = int(np.sum((y_true == True) & (y_pred == False)))
    tn = int(np.sum((y_true == False) & (y_pred == False)))
    n = len(y_true)

    # Observed agreement
    po = (tp + tn) / n if n > 0 else 0.0

    # Expected chance agreement (for Cohen's Kappa)
    p_true_pos = (tp + fn) / n if n > 0 else 0.0
    p_true_neg = (tn + fp) / n if n > 0 else 0.0
    p_pred_pos = (tp + fp) / n if n > 0 else 0.0
    p_pred_neg = (tn + fn) / n if n > 0 else 0.0
    pe = (p_true_pos * p_pred_pos) + (p_true_neg * p_pred_neg)

    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 1.0

    # Diagnostic performance metrics
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 1.0

    return {
        "total_evaluated": n,
        "overall_agreement_pct": round(po * 100, 1),
        "cohens_kappa": round(kappa, 3),
        "sensitivity": round(sens * 100, 1),
        "specificity": round(spec * 100, 1),
        "ppv": round(ppv * 100, 1),
        "npv": round(npv * 100, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }
