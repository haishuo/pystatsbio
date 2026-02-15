"""
Diagnostic accuracy analysis for biomarker evaluation.

ROC analysis, sensitivity/specificity, predictive values, likelihood ratios,
and high-throughput batch AUC computation for biomarker panel screening.

Validates against: R packages pROC, OptimalCutpoints, epiR.
"""

from pystatsbio.diagnostic._common import ROCResult, DiagnosticResult
from pystatsbio.diagnostic._roc import roc, roc_test
from pystatsbio.diagnostic._accuracy import diagnostic_accuracy
from pystatsbio.diagnostic._cutoff import optimal_cutoff
from pystatsbio.diagnostic._batch import batch_auc

__all__ = [
    "ROCResult",
    "DiagnosticResult",
    "roc",
    "roc_test",
    "diagnostic_accuracy",
    "optimal_cutoff",
    "batch_auc",
]
