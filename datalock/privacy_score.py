"""
datalock/privacy_score.py
======================
Privacy Score — pontuação composta 0–100 para avaliação LGPD.

Componentes (pesos calibrados com literatura de PETs):
  1. PII exposta             (30pts)  — identificadores diretos sem mascaramento
  2. k-anonimato             (25pts)  — unicidade em quasi-identifiers
  3. Risco de re-identificação (20pts) — score composto do check.risk
  4. Cardinalidade de PII    (15pts)  — proporção de colunas PII no total
  5. Nulos em PII            (10pts)  — nulos indicam gaps no mascaramento

Usado por dd.profile() como report.privacy_score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl


@dataclass
class PrivacyScore:
    """Score composto de privacidade."""
    total:          int          # 0 → 100
    grade:          str          # "A" → "F"
    label:          str          # "Excelente" → "Crítico"
    recommendation: str

    # Componentes
    pii_exposure:   int          # 0–30
    k_anonymity:    int          # 0–25
    reident_risk:   int          # 0–20
    pii_density:    int          # 0–15
    null_coverage:  int          # 0–10

    # Contexto
    n_pii_columns:  int
    n_total_columns: int
    k_value:        Optional[int] = None
    risk_score:     Optional[float] = None

    def __str__(self) -> str:
        bar_filled = "█" * (self.total // 10)
        bar_empty  = "░" * (10 - self.total // 10)
        return (
            f"Privacy Score: {self.total}/100 [{self.grade}] {self.label}\n"
            f"  [{bar_filled}{bar_empty}]\n"
            f"  PII exposta:        {self.pii_exposure:>3}/30\n"
            f"  k-anonimato:        {self.k_anonymity:>3}/25\n"
            f"  Risco re-ident:     {self.reident_risk:>3}/20\n"
            f"  Densidade PII:      {self.pii_density:>3}/15\n"
            f"  Cobertura de nulos: {self.null_coverage:>3}/10\n"
            f"\n  ➜ {self.recommendation}"
        )

    def to_dict(self) -> Dict:
        return {
            "total":       self.total,
            "grade":       self.grade,
            "label":       self.label,
            "components": {
                "pii_exposure":  self.pii_exposure,
                "k_anonymity":   self.k_anonymity,
                "reident_risk":  self.reident_risk,
                "pii_density":   self.pii_density,
                "null_coverage": self.null_coverage,
            },
            "context": {
                "n_pii_columns":   self.n_pii_columns,
                "n_total_columns": self.n_total_columns,
                "k_value":         self.k_value,
                "risk_score":      self.risk_score,
            },
            "recommendation": self.recommendation,
        }


def calculate(
    df: Any,
    reports: Dict[str, Any],           # Dict[str, ColumnReport] do dd.scan()
    *,
    quasi_identifiers: Optional[List[str]] = None,
    sample_size: int = 2000,
) -> PrivacyScore:
    """
    Calcula o Privacy Score composto.

    Args:
        df:                DataFrame (original — não mascarado).
        reports:           Resultado do dd.scan().
        quasi_identifiers: Colunas quasi-identificadoras para k-anonimato.
                           None = auto-detecta por tipo QUASI_IDENTIFIER.
        sample_size:       Linhas amostradas para k-anonimato.

    Returns:
        PrivacyScore com total, grade e componentes detalhados.
    """
    df_pd = _to_pd(df)
    n_total_cols = len(df_pd.columns)
    n_pii_cols   = len(reports)

    from datalock.detectors.pii_detector import RiskLevel, PIIType, MaskStrategy

    # ── 1. PII exposta (0–30) ─────────────────────────────────────────────
    # Penaliza identificadores diretos não mascarados (hash/redact = mascarado)
    direct_ids = [
        r for r in reports.values()
        if r.is_direct_identifier and r.mask_strategy == MaskStrategy.PASSTHROUGH
    ]
    high_risk_unmasked = [
        r for r in reports.values()
        if r.risk_level == RiskLevel.HIGH and r.mask_strategy == MaskStrategy.PASSTHROUGH
    ]

    n_exposed = len(direct_ids) + len(high_risk_unmasked)
    if n_exposed == 0:
        pii_exposure = 30
    elif n_exposed == 1:
        pii_exposure = 20
    elif n_exposed == 2:
        pii_exposure = 10
    else:
        pii_exposure = max(0, 30 - n_exposed * 10)

    # ── 2. k-anonimato (0–25) ────────────────────────────────────────────
    k_value = None
    k_score = 0

    if quasi_identifiers is None:
        quasi_identifiers = [
            c for c, r in reports.items()
            if r.pii_type in (PIIType.QUASI_IDENTIFIER, PIIType.CATEGORICO)
            or r.risk_level == RiskLevel.LOW
        ][:4]  # Limita a 4 para eficiência

    if quasi_identifiers and all(q in df_pd.columns for q in quasi_identifiers):
        try:
            sample = df_pd[quasi_identifiers].dropna().head(sample_size)
            if not sample.empty:
                k_value = int(
                    sample.groupby(quasi_identifiers).size().min()
                )
                # k ≥ 5 (ANPD) = full score; k=1 = 0
                if k_value >= 10:
                    k_score = 25
                elif k_value >= 5:
                    k_score = 18
                elif k_value >= 3:
                    k_score = 10
                elif k_value >= 2:
                    k_score = 5
                else:
                    k_score = 0
        except Exception:
            k_score = 12  # inconclusivo → score médio

    else:
        k_score = 15  # sem quasi-ids → penalidade parcial (não conclusivo)

    # ── 3. Risco de re-identificação (0–20) ──────────────────────────────
    risk_val = None
    risk_score_pts = 10  # padrão moderado quando inconclusivo

    try:
        from datalock.metrics.risk_score import ReidentificationRiskScorer
        scorer = ReidentificationRiskScorer()
        risk_report = scorer.evaluate(
            df_pd,
            quasi_identifiers=quasi_identifiers or [],
            direct_identifiers=[c for c, r in reports.items() if r.is_direct_identifier],
            masked_columns=[c for c, r in reports.items()
                            if r.mask_strategy != MaskStrategy.PASSTHROUGH],
        )
        risk_val = risk_report.risk_score
        # risk_score ∈ [0,1] — menor é melhor
        risk_score_pts = round((1 - risk_val) * 20)
    except Exception:
        pass

    # ── 4. Densidade PII (0–15) ──────────────────────────────────────────
    # Proporção de colunas PII sobre total — mais PII = maior superfície de ataque
    if n_total_cols == 0:
        pii_density = 15
    else:
        pii_ratio = n_pii_cols / n_total_cols
        if pii_ratio == 0:
            pii_density = 15
        elif pii_ratio <= 0.2:
            pii_density = 12
        elif pii_ratio <= 0.4:
            pii_density = 8
        elif pii_ratio <= 0.6:
            pii_density = 4
        else:
            pii_density = 0

    # ── 5. Nulos em PII (0–10) ───────────────────────────────────────────
    # Nulos em colunas PII = dados faltantes que podem ser incompletos ou evasivos
    pii_cols_in_df = [c for c in reports if c in df_pd.columns]
    if not pii_cols_in_df:
        null_coverage = 10
    else:
        null_rates = [df_pd[c].isna().mean() for c in pii_cols_in_df]
        avg_null = sum(null_rates) / len(null_rates)
        if avg_null < 0.01:
            null_coverage = 10
        elif avg_null < 0.05:
            null_coverage = 8
        elif avg_null < 0.10:
            null_coverage = 5
        else:
            null_coverage = max(0, round(10 - avg_null * 20))

    # ── Total e Grade ─────────────────────────────────────────────────────
    total = pii_exposure + k_score + risk_score_pts + pii_density + null_coverage
    total = max(0, min(100, total))

    if total >= 90:
        grade, label = "A", "Excelente"
        recommendation = "Dados bem protegidos. Mantenha o processo e audite periodicamente."
    elif total >= 75:
        grade, label = "B", "Bom"
        recommendation = "Boa proteção. Revise quasi-identifiers e aplique k-anonimato ≥ 5."
    elif total >= 60:
        grade, label = "C", "Regular"
        recommendation = "Proteção básica. Mascare identificadores diretos e revise k-anonimato."
    elif total >= 40:
        grade, label = "D", "Insuficiente"
        recommendation = "Risco considerável. Execute dd.mask() com salt antes de qualquer compartilhamento."
    else:
        grade, label = "F", "Crítico"
        recommendation = "ATENÇÃO: dados expõem PII diretamente. Não compartilhe sem mascaramento completo."

    return PrivacyScore(
        total=total, grade=grade, label=label, recommendation=recommendation,
        pii_exposure=pii_exposure, k_anonymity=k_score,
        reident_risk=risk_score_pts, pii_density=pii_density,
        null_coverage=null_coverage,
        n_pii_columns=n_pii_cols, n_total_columns=n_total_cols,
        k_value=k_value, risk_score=risk_val,
    )


def _to_pd(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pl.LazyFrame):
        return df.collect().to_pandas()
    return pd.DataFrame(df)
