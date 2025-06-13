# src/stats_tests.py
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def chi_square_table(contingency: pd.DataFrame) -> dict:
    """
    χ² 통계량 + Cramer's V 계산
    contingency : index = group (e.g., country), columns = category
    """
    chi2, p, dof, expected = chi2_contingency(contingency)

    # Cramer's V 수식: V = sqrt(χ² / (n * (min(k - 1, r - 1))))
    n = contingency.sum().sum()
    r, k = contingency.shape
    v = np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

    return {"chi2": chi2, "p": p, "cramers_v": v}
