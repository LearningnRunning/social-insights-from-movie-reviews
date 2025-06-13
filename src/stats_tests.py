# src/stats_tests.py
from __future__ import annotations

import pandas as pd
from pingouin import crm
from scipy.stats import chi2_contingency


def chi_square_table(contingency: pd.DataFrame) -> dict:
    """
    contingency : index=country, columns=category, values=count
    """
    chi2, p, dof, _ = chi2_contingency(contingency)
    v = crm(contingency, method="cramer")
    return {"chi2": chi2, "p": p, "cramers_v": v}
