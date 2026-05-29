"""
logus.check — Suite de métricas de privacidade e qualidade.

    from datalock import check

    report = check.fidelity(df_real, df_synth)
    report = check.kanon(df, quasi_identifiers=["uf", "idade"])
    report = check.risk(df, quasi_identifiers=["uf", "idade"])
    report = check.utility(df_original, df_masked)
    dp     = check.dp(epsilon=1.0)
"""

from __future__ import annotations


from datalock.metrics.fidelity    import FidelityEvaluator, FidelityReport
from datalock.metrics.kanonymity  import KAnonymityAnalyzer, calculate_k_anonymity, PrivacyMetricsReport
from datalock.metrics.risk_score  import ReidentificationRiskScorer, ReidentificationRiskReport
from datalock.metrics.utility     import UtilityEvaluator, UtilityReport
from datalock.metrics.differential_privacy import DifferentialPrivacy, DPBudget, PrivacyAttackContext
from datalock.metrics.tcloseness  import TClosenessAnalyzer, calculate_t_closeness, TClosenessReport

import pandas as pd
from datalock.utils.frames import to_pandas as _to_pd

__all__ = [
    "fidelity", "kanon", "risk", "utility", "dp", "tcloseness",
    "FidelityEvaluator", "FidelityReport",
    "KAnonymityAnalyzer", "PrivacyMetricsReport",
    "ReidentificationRiskScorer", "ReidentificationRiskReport",
    "UtilityEvaluator", "UtilityReport",
    "DifferentialPrivacy", "DPBudget",
    "TClosenessAnalyzer", "TClosenessReport",
]


def fidelity(df_real, df_synthetic, *, columns=None, tstr_target=None,
             n_sample=5000, psi_bins=10):
    """
    Avalia a fidelidade estatística de dados sintéticos em relação ao real.

    Métricas: KS Test, PSI, JSD, correlações Pearson/Spearman,
    TSTR (Train-on-Synthetic/Test-on-Real) e Privacy Score (DCR, NNDR).

    Args:
        df_real:      DataFrame de referência.
        df_synthetic: DataFrame sintético gerado pelo lg.clone().
        columns:      Colunas a avaliar (None = todas em comum).
        tstr_target:  Coluna alvo para TSTR (classificação ou regressão).
        n_sample:     Máximo de linhas amostradas.
        psi_bins:     Bins para PSI.

    Returns:
        FidelityReport com score geral e métricas por coluna.

    Example:
        report = lg.check.fidelity(df_real, df_synth)
        report.print_report()
    """
    df_real = _to_pd(df_real)
    df_synthetic = _to_pd(df_synthetic)
    evaluator = FidelityEvaluator(n_sample=n_sample, psi_bins=psi_bins,
                                  tstr_target=tstr_target)
    return evaluator.evaluate(df_real, df_synthetic, columns=columns)


def kanon(df, *, quasi_identifiers, sensitive_attributes=None, target_k=5):
    """
    Calcula k-anonimato e l-diversidade (ANPD recomenda k >= 5).

    Args:
        df:                   DataFrame a avaliar.
        quasi_identifiers:    Colunas quasi-identificadoras.
        sensitive_attributes: Colunas de atributos sensíveis (para l-diversidade).
        target_k:             k mínimo alvo (padrão 5, per ANPD 2023).

    Returns:
        PrivacyMetricsReport com k_value, l_value, compliant_anpd, recommendations.

    Example:
        report = lg.check.kanon(df, quasi_identifiers=["uf", "idade", "escolaridade"])
        print(f"k={report.k_anonymity.k_value} | ANPD: {report.compliant_anpd}")
    """
    return calculate_k_anonymity(df, quasi_identifiers=quasi_identifiers,
                                 sensitive_attributes=sensitive_attributes or [],
                                 target_k=target_k)


def risk(df, *, quasi_identifiers=None, direct_identifiers=None,
         masked_columns=None, anpd_k_threshold=5):
    """
    Calcula Risk Score de re-identificação (0 a 1).

    Combina unicidade, inferência por atributo e exposição de identificadores
    diretos em um score composto com interpretação e recomendações.

    Args:
        df:                DataFrame a avaliar.
        quasi_identifiers: Colunas quasi-identificadoras (None = auto-detecta).
        direct_identifiers: Colunas de identificação direta (CPF, email, etc.).
        masked_columns:    Colunas que já foram mascaradas pelo pipeline.
        anpd_k_threshold:  k mínimo considerado seguro (padrão 5).

    Returns:
        ReidentificationRiskReport com risk_score, risk_level, recommendations.

    Example:
        report = lg.check.risk(df_safe, quasi_identifiers=["uf", "idade"])
        print(f"Risk: {report.risk_score:.2f} | {report.risk_level}")
    """
    scorer = ReidentificationRiskScorer(anpd_k_threshold=anpd_k_threshold)
    return scorer.evaluate(
        df,
        quasi_identifiers=quasi_identifiers or [],
        direct_identifiers=direct_identifiers or [],
        masked_columns=masked_columns or [],
    )


def utility(df_original, df_masked, *, columns=None):
    """
    Avalia a perda de utilidade introduzida pelo mascaramento.

    Métricas: PSI por coluna, correlação preservada, distribuição de valores,
    e score geral de utilidade (0=inútil, 1=idêntico ao original).

    Args:
        df_original: DataFrame antes do mascaramento.
        df_masked:   DataFrame após o mascaramento.
        columns:     Colunas a comparar (None = todas em comum).
    Returns:
        UtilityReport com overall_score e métricas por coluna.

    Example:
        report = lg.check.utility(df_original, df_masked)
        print(f"Utilidade: {report.overall_score:.0%}")
    """
    df_original = _to_pd(df_original)
    df_masked = _to_pd(df_masked)
    evaluator = UtilityEvaluator()
    return evaluator.evaluate(df_original, df_masked, columns=columns)


def dp(epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
    """
    Cria um mecanismo de Privacidade Diferencial configurado.

    Args:
        epsilon:     Parâmetro de privacidade (menor = mais privado).
                     ε ≤ 1.0 → proteção forte. ε ≤ 5.0 → moderada (analytics).
        delta:       Parâmetro de relaxamento (apenas para mecanismo Gaussiano).
        sensitivity: Sensibilidade global da função de consulta.

    Returns:
        DifferentialPrivacy pronto para .laplace(), .gaussian(), .randomized_response().

    Example:
        dp = lg.check.dp(epsilon=1.0)
        noisy_mean = dp.laplace(df["renda"].mean(), sensitivity=df["renda"].max())
        dp.budget.report()
    """
    return DifferentialPrivacy(epsilon=epsilon, delta=delta, sensitivity=sensitivity)


def tcloseness(df, *, quasi_identifiers, sensitive_attribute, target_t=0.2):
    """
    Calcula t-closeness (Li et al., 2007).

    Exige que a distribuição do atributo sensível em cada grupo esteja
    dentro de t da distribuição global.

    Args:
        df:                  DataFrame a avaliar.
        quasi_identifiers:   Colunas quasi-identificadoras.
        sensitive_attribute: Coluna de atributo sensível.
        target_t:            t máximo aceitável (padrão 0.2).

    Returns:
        TClosenessReport com t_value, compliant, groups.

    Example:
        report = lg.check.tcloseness(df, quasi_identifiers=["uf"], sensitive_attribute="diagnostico")
        print(f"t={report.t_value:.3f} | OK: {report.compliant}")
    """
    df = _to_pd(df)
    return calculate_t_closeness(df, quasi_identifiers=quasi_identifiers,
                                  sensitive_attribute=sensitive_attribute,
                                  target_t=target_t)
