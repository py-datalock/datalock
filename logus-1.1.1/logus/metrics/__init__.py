from __future__ import annotations

from datalock.metrics.kanonymity import (
    KAnonymityAnalyzer,
    calculate_k_anonymity,
    KAnonymityReport,
    LDiversityReport,
    PrivacyMetricsReport,
)
from datalock.metrics.utility import (
    UtilityEvaluator,
    UtilityReport,
    ColumnUtilityReport,
)
from datalock.metrics.differential_privacy import (
    DifferentialPrivacy,
    DPBudget,
    DPResult,
    PrivacyAttackContext,
)
from datalock.metrics.risk_score import (
    ReidentificationRiskScorer,
    ReidentificationRiskReport,
    ColumnRiskProfile,
)

from datalock.metrics.fidelity import (
    FidelityEvaluator,
    FidelityReport,
    ColumnFidelityReport,
    CorrelationFidelityReport,
    TSTRReport,
    PrivacyScoreReport,
)

__all__ = [
    # k-Anonimato e l-Diversidade
    "KAnonymityAnalyzer",
    "calculate_k_anonymity",
    "KAnonymityReport",
    "LDiversityReport",
    "PrivacyMetricsReport",
    # Utilidade analítica
    "UtilityEvaluator",
    "UtilityReport",
    "ColumnUtilityReport",
    # Privacidade Diferencial
    "DifferentialPrivacy",
    "DPBudget",
    "DPResult",
    "PrivacyAttackContext",
    # Risk Score de Re-identificação
    "ReidentificationRiskScorer",
    "ReidentificationRiskReport",
    "ColumnRiskProfile",
    # Fidelidade de dados sintéticos
    "FidelityEvaluator",
    "FidelityReport",
    "ColumnFidelityReport",
    "CorrelationFidelityReport",
    "TSTRReport",
    "PrivacyScoreReport",
]
