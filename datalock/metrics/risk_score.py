"""
metrics/risk_score.py
=====================
Risk Score Formal de Re-identificação.

Problema
--------
k-Anonimato responde "qual é o tamanho do menor grupo de equivalência?"
l-Diversidade responde "os grupos têm diversidade suficiente no atributo sensível?"

Mas nenhum responde: "qual é a probabilidade estimada de que um adversário
re-identifique um indivíduo específico neste dataset?"

Este módulo calcula um Risk Score composto que combina:
  1. Risco de singularização (k-anonimato → probabilidade de isolamento)
  2. Risco de inferência por atributo (l-diversidade → dominância de categoria)
  3. Risco de re-identificação por linkage (quasi-identifiers compartilháveis)
  4. Risco estrutural do dataset (unicidade, cardinalidade, completude)

O score resultante é calibrado de 0 (risco mínimo) a 1 (risco máximo)
e inclui interpretação ANPD-alinhada e recomendações de remediação.

Fundamentação
-------------
  Sweeney (2002): 87% dos norte-americanos são unicamente identificáveis
    por (ZIP code, sexo, data de nascimento).

  Golle (2006): Re-identificação via k-anonimato probabilístico.

  El Emam et al. (2011): Risk-based de-identification framework —
    propõe score de re-identificação baseado em unicidade.

  ANPD (2023): Guia de Anonimização — risco deve ser avaliado pelo
    "conjunto de meios razoavelmente utilizáveis" pelo adversário.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnRiskProfile:
    """Perfil de risco de re-identificação por coluna."""
    column:           str
    uniqueness_ratio: float   # nunique / n — quanto mais alto, mais identificável
    null_ratio:       float   # nulos reduzem a superfície de ataque
    is_quasi_id:      bool    # detectado como QI pelo PIIDetector
    is_direct_id:     bool    # identificador direto (CPF, email...)
    is_masked:        bool    # já foi mascarado pelo pipeline
    contribution:     float   # peso deste campo no score composto


@dataclass
class ReidentificationRiskReport:
    """
    Resultado do cálculo do Risk Score de Re-identificação.

    Interpretação do risk_score:
        0.00 – 0.20  → Risco muito baixo  (coluna verde — ANPD: baixo risco)
        0.20 – 0.40  → Risco baixo        (coluna verde-amarela)
        0.40 – 0.60  → Risco moderado     (coluna amarela — atenção)
        0.60 – 0.80  → Risco alto         (coluna laranja — ação necessária)
        0.80 – 1.00  → Risco crítico      (coluna vermelha — re-identificação provável)
    """
    risk_score:               float         # 0 a 1
    risk_level:               str           # "very_low" | "low" | "moderate" | "high" | "critical"
    risk_label:               str           # descrição humana
    n_records:                int
    n_columns_analyzed:       int
    n_quasi_identifiers:      int
    n_direct_identifiers:     int
    n_unmasked_direct_ids:    int
    uniqueness_score:         float         # componente de singularização
    inference_score:          float         # componente de inferência
    structural_score:         float         # componente estrutural do dataset
    column_profiles:          List[ColumnRiskProfile] = field(default_factory=list)
    recommendations:          List[str] = field(default_factory=list)
    anpd_compliant:           bool = False
    compliant_reason:         str = ""

    @property
    def risk_icon(self) -> str:
        icons = {
            "very_low":  "🟢",
            "low":       "🟡",
            "moderate":  "🟠",
            "high":      "🔴",
            "critical":  "🚨",
        }
        return icons.get(self.risk_level, "⚪")


class ReidentificationRiskScorer:
    """
    Calcula o Risk Score composto de re-identificação de um DataFrame.

    Pode ser usado antes (para diagnosticar o dado bruto) ou depois do
    mascaramento (para validar a eficácia do mascaramento aplicado).

    Parâmetros:
        quasi_identifiers:    Colunas conhecidas como quasi-identifiers.
                              Se None, detectadas automaticamente pelo PIIDetector.
        direct_identifiers:   Colunas com identificadores diretos (CPF, email...).
        masked_columns:       Colunas que já foram mascaradas pelo pipeline.
        anpd_k_threshold:     k mínimo para compliance ANPD (padrão: 5).

    Exemplo:
        scorer = ReidentificationRiskScorer(
            quasi_identifiers=["idade", "uf", "escolaridade"],
            direct_identifiers=["cpf", "email"],
            masked_columns=["cpf", "email", "nome"],
        )
        report = scorer.evaluate(df_mascarado)
        scorer.print_report(report)
    """

    def __init__(
        self,
        quasi_identifiers: Optional[List[str]] = None,
        direct_identifiers: Optional[List[str]] = None,
        masked_columns: Optional[List[str]] = None,
        anpd_k_threshold: int = 5,
    ) -> None:
        self.quasi_identifiers = quasi_identifiers or []
        self.direct_identifiers = direct_identifiers or []
        self.masked_columns = set(masked_columns or [])
        self.anpd_k_threshold = anpd_k_threshold

    def evaluate(self, df: pd.DataFrame) -> ReidentificationRiskReport:
        """
        Avalia o risco de re-identificação do DataFrame.

        O score é composto por três dimensões:
          1. Unicidade (0.5 de peso): mede singularização direta
          2. Inferência (0.3 de peso): mede risco de inferência por atributo
          3. Estrutural (0.2 de peso): mede exposição dos identificadores diretos
        """
        n = len(df)
        if n == 0:
            return self._empty_report()

        # --- Análise por coluna ---
        profiles = []
        for col in df.columns:
            series = df[col].dropna()
            uniqueness = series.nunique() / max(n, 1)
            null_ratio = df[col].isna().mean()
            is_qi   = col in self.quasi_identifiers
            is_did  = col in self.direct_identifiers
            is_mask = col in self.masked_columns

            # Contribuição ponderada ao risco
            contrib = 0.0
            if is_did and not is_mask:
                contrib = 1.0 * uniqueness          # identificador direto não mascarado
            elif is_did and is_mask:
                contrib = 0.1 * uniqueness          # identificador mascarado — risco residual baixo
            elif is_qi:
                contrib = 0.6 * uniqueness          # quasi-id contribui parcialmente
            else:
                contrib = 0.2 * uniqueness          # outras colunas: risco baixo

            contrib *= (1 - null_ratio * 0.5)       # nulos reduzem a superfície

            profiles.append(ColumnRiskProfile(
                column=col,
                uniqueness_ratio=round(uniqueness, 4),
                null_ratio=round(null_ratio, 4),
                is_quasi_id=is_qi,
                is_direct_id=is_did,
                is_masked=is_mask,
                contribution=round(contrib, 4),
            ))

        # --- Componente 1: Unicidade ---
        # Estima a proporção de registros unicamente identificáveis pelos QIs
        if self.quasi_identifiers:
            valid_qi = [c for c in self.quasi_identifiers if c in df.columns]
            if valid_qi:
                qi_df = df[valid_qi].dropna()
                qi_counts = qi_df.groupby(valid_qi).size()
                singularity_rate = (qi_counts == 1).sum() / max(len(qi_counts), 1)
                avg_qi_uniqueness = np.mean([
                    df[c].nunique() / max(n, 1) for c in valid_qi
                ])
                uniqueness_score = 0.6 * singularity_rate + 0.4 * avg_qi_uniqueness
            else:
                uniqueness_score = 0.0
        else:
            # Sem QIs conhecidos: usa unicidade média de todas as colunas
            uniqueness_score = np.mean([p.uniqueness_ratio for p in profiles])

        # --- Componente 2: Inferência por atributo ---
        # Mede dominância de categorias: se um grupo tem 95% da mesma categoria,
        # o adversário pode inferir com alta confiança
        inference_scores = []
        for col in self.quasi_identifiers:
            if col not in df.columns:
                continue
            vc = df[col].value_counts(normalize=True)
            if len(vc) > 0:
                dominance = float(vc.iloc[0])  # frequência da categoria mais comum
                inference_scores.append(dominance)
        inference_score = np.mean(inference_scores) if inference_scores else 0.2

        # --- Componente 3: Estrutural ---
        # Penaliza identificadores diretos não mascarados
        n_did = len(self.direct_identifiers)
        n_masked_did = len([c for c in self.direct_identifiers if c in self.masked_columns])
        n_unmasked_did = n_did - n_masked_did
        structural_score = n_unmasked_did / max(n_did, 1) if n_did > 0 else 0.0

        # --- Score composto ---
        risk_score = (
            0.5 * uniqueness_score +
            0.3 * inference_score +
            0.2 * structural_score
        )
        risk_score = min(max(risk_score, 0.0), 1.0)  # clamp [0, 1]

        risk_level, risk_label = _interpret_risk_score(risk_score)
        recommendations = self._build_recommendations(
            risk_score, risk_level, n_unmasked_did, profiles, df,
        )

        # ANPD compliance: risco baixo + QIs com k >= 5
        anpd_compliant, compliant_reason = self._check_anpd(risk_score, df)

        return ReidentificationRiskReport(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            risk_label=risk_label,
            n_records=n,
            n_columns_analyzed=len(df.columns),
            n_quasi_identifiers=len(self.quasi_identifiers),
            n_direct_identifiers=n_did,
            n_unmasked_direct_ids=n_unmasked_did,
            uniqueness_score=round(uniqueness_score, 4),
            inference_score=round(inference_score, 4),
            structural_score=round(structural_score, 4),
            column_profiles=profiles,
            recommendations=recommendations,
            anpd_compliant=anpd_compliant,
            compliant_reason=compliant_reason,
        )

    def print_report(self, report: ReidentificationRiskReport) -> None:
        """Imprime relatório formatado no console."""
        print()
        print("=" * 65)
        print("RISK SCORE DE RE-IDENTIFICAÇÃO")
        print("=" * 65)
        print(f"  {report.risk_icon} Score geral      : {report.risk_score:.4f} ({report.risk_level})")
        print(f"  Descrição         : {report.risk_label}")
        print(f"  Registros         : {report.n_records:,}")
        print(f"  IDs diretos       : {report.n_direct_identifiers} "
              f"({report.n_unmasked_direct_ids} não mascarados)")
        print(f"  Quasi-identifiers : {report.n_quasi_identifiers}")
        print()
        print(f"  Componentes:")
        print(f"    Unicidade (50%) : {report.uniqueness_score:.4f}")
        print(f"    Inferência (30%): {report.inference_score:.4f}")
        print(f"    Estrutural (20%): {report.structural_score:.4f}")
        print()
        anpd_icon = "✅" if report.anpd_compliant else "❌"
        print(f"  {anpd_icon} ANPD compliant    : {report.anpd_compliant}")
        print(f"     {report.compliant_reason}")
        print()
        if report.recommendations:
            print("  Recomendações:")
            for rec in report.recommendations:
                print(f"    • {rec}")
        print()
        # Top colunas por contribuição ao risco
        top = sorted(report.column_profiles, key=lambda p: p.contribution, reverse=True)[:5]
        print("  Top 5 colunas por contribuição ao risco:")
        for p in top:
            markers = []
            if p.is_direct_id: markers.append("ID")
            if p.is_quasi_id:  markers.append("QI")
            if p.is_masked:    markers.append("mascarado")
            tag = f"[{', '.join(markers)}]" if markers else ""
            print(f"    {p.column:<25} contrib={p.contribution:.4f}  uniq={p.uniqueness_ratio:.3f} {tag}")

    def _check_anpd(self, risk_score: float, df: pd.DataFrame) -> Tuple[bool, str]:
        """Verifica compliance simplificado com o Guia de Anonimização ANPD (2023)."""
        if self.quasi_identifiers:
            valid_qi = [c for c in self.quasi_identifiers if c in df.columns]
            if valid_qi:
                qi_counts = df[valid_qi].dropna().groupby(valid_qi).size()
                k = int(qi_counts.min()) if len(qi_counts) > 0 else 0
                if k >= self.anpd_k_threshold and risk_score < 0.4:
                    return True, f"k={k} ≥ {self.anpd_k_threshold} e risk_score={risk_score:.3f} < 0.4"
                else:
                    return False, f"k={k} (mín. {self.anpd_k_threshold}) ou risk_score={risk_score:.3f} ≥ 0.4"
        if risk_score < 0.2:
            return True, f"risk_score={risk_score:.3f} < 0.2 (sem QIs explícitos)"
        return False, f"risk_score={risk_score:.3f} ≥ 0.2 — requere análise de QIs"

    def _build_recommendations(
        self,
        score: float,
        level: str,
        n_unmasked: int,
        profiles: List[ColumnRiskProfile],
        df: pd.DataFrame,
    ) -> List[str]:
        recs = []
        if n_unmasked > 0:
            unmasked = [p.column for p in profiles if p.is_direct_id and not p.is_masked]
            recs.append(f"Mascarar identificadores diretos não protegidos: {unmasked}")
        if level in ("high", "critical"):
            recs.append("Aumentar generalização dos quasi-identifiers (ex: faixas etárias mais amplas).")
            recs.append("Considerar supressão de registros singulares (k=1).")
        if level in ("moderate", "high", "critical"):
            recs.append("Aplicar k-anonimato ≥ 5 sobre os quasi-identifiers (ANPD 2023).")
        high_uniq = [p.column for p in profiles if p.uniqueness_ratio > 0.8 and not p.is_masked]
        if high_uniq:
            recs.append(f"Colunas com alta unicidade não mascaradas: {high_uniq[:3]}")
        if score > 0.3:
            recs.append("Considerar privacidade diferencial (ε ≤ 1.0) para releases externos.")
        return recs

    def _empty_report(self) -> ReidentificationRiskReport:
        return ReidentificationRiskReport(
            risk_score=0.0, risk_level="very_low", risk_label="Dataset vazio",
            n_records=0, n_columns_analyzed=0, n_quasi_identifiers=0,
            n_direct_identifiers=0, n_unmasked_direct_ids=0,
            uniqueness_score=0.0, inference_score=0.0, structural_score=0.0,
        )


def _interpret_risk_score(score: float) -> Tuple[str, str]:
    if score < 0.20:
        return "very_low",  "Risco muito baixo — re-identificação improvável com meios razoáveis."
    elif score < 0.40:
        return "low",       "Risco baixo — dataset adequado para uso interno e analítico."
    elif score < 0.60:
        return "moderate",  "Risco moderado — ação preventiva recomendada antes de compartilhamento."
    elif score < 0.80:
        return "high",      "Risco alto — re-identificação viável; remediação necessária."
    else:
        return "critical",  "Risco crítico — re-identificação provável; dataset não deve ser compartilhado."
