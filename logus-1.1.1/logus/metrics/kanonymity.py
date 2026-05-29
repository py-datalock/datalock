"""
metrics/kanonymity.py
=====================
Métricas Formais de Privacidade: k-Anonimato e l-Diversidade.

Fundamentação Teórica
---------------------
k-Anonimato (Samarati & Sweeney, 1998; Sweeney, 2002):
    Uma tabela satisfaz k-anonimato se cada combinação de valores dos
    quasi-identifiers (QI) aparece em pelo menos k linhas. Nenhum indivíduo
    pode ser distinguido de pelo menos k-1 outros. k=1 significa que existem
    indivíduos únicos — re-identificáveis diretamente.

    Limitação conhecida: k-anonimato não protege contra ataques de
    homogeneidade (todos do mesmo grupo têm o mesmo atributo sensível)
    nem de background knowledge (conhecimento prévio do adversário).

l-Diversidade (Machanavajjhala et al., 2007):
    Estende k-anonimato exigindo que cada grupo de k registros tenha
    pelo menos l valores distintos do atributo sensível. Mitiga o ataque
    de homogeneidade do k-anonimato.

    Ref: Machanavajjhala, A., et al. "l-diversity: Privacy beyond
    k-anonymity." ACM TKDD 1.1 (2007).

t-Closeness (Li et al., 2007):
    Exige que a distribuição do atributo sensível em cada grupo esteja
    "próxima" (dentro de t) da distribuição global. Mitiga ataques de
    skewness e similarity do l-diversity.

    Ref: Li, N., et al. "t-closeness: Privacy beyond k-anonymity and
    l-diversity." ICDE 2007.

Interpretação prática para LGPD:
    A ANPD não exige k específico, mas recomenda que dados pseudonimizados
    satisfaçam k≥5 antes de compartilhamento com terceiros (Guia de
    Anonimização, ANPD 2023). k<3 é considerado alto risco de singularização.

Uso:
    from datalock.metrics.kanonymity import (
        KAnonymityAnalyzer, calculate_k_anonymity
    )
    analyzer = KAnonymityAnalyzer(quasi_identifiers=["idade_faixa", "uf", "genero"])
    report   = analyzer.evaluate(df_mascarado)
    print(f"k = {report.k_value} — {report.risk_level}")

    # Forma simplificada:
    report = calculate_k_anonymity(df, ["idade_faixa", "uf"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class KAnonymityReport:
    """
    Resultado do cálculo de k-anonimato empírico.

    Interpretação do k_value:
        k=1  → indivíduos únicos existem — risco CRÍTICO (re-identificação direta)
        k<3  → grupos muito pequenos — risco ALTO
        k<5  → risco MÉDIO — não recomendado para compartilhamento externo
        k≥5  → risco BAIXO — aceitável para maioria dos contextos (ANPD 2023)
        k≥10 → risco MUITO BAIXO — padrão conservador para dados de saúde
    """
    k_value:            int
    quasi_identifiers:  List[str]
    total_groups:       int
    singleton_groups:   int          # grupos com exatamente 1 registro
    singleton_fraction: float        # fração de grupos singularizados
    risk_level:         str          # "critical" | "high" | "medium" | "low"
    interpretation:     str
    group_size_dist:    Dict[str, int] = field(default_factory=dict)
    # percentis da distribuição de tamanho de grupos
    p25_group_size:     float = 0.0
    p50_group_size:     float = 0.0
    p75_group_size:     float = 0.0

    @property
    def satisfies_k5(self) -> bool:
        """k≥5 — padrão mínimo recomendado pela ANPD para compartilhamento."""
        return self.k_value >= 5

    @property
    def satisfies_k10(self) -> bool:
        """k≥10 — padrão conservador para dados sensíveis (saúde, biometria)."""
        return self.k_value >= 10


@dataclass
class LDiversityReport:
    """
    Resultado do cálculo de l-diversidade.

    l_value: mínimo de valores distintos do atributo sensível nos grupos.
    l=1 significa que algum grupo tem apenas 1 valor sensível — vulnerável
    ao ataque de homogeneidade (se você sabe que alguém está no grupo,
    sabe seu atributo sensível).
    """
    l_value:            int
    sensitive_attribute: str
    quasi_identifiers:  List[str]
    total_groups:       int
    homogeneous_groups: int          # grupos com l=1 (todos igual)
    homogeneous_fraction: float
    risk_level:         str
    interpretation:     str


@dataclass
class PrivacyMetricsReport:
    """Relatório consolidado com k-anonimato, l-diversidade e recomendações."""
    k_anonymity:     KAnonymityReport
    l_diversity:     Optional[LDiversityReport]
    overall_risk:    str
    recommendations: List[str]
    compliant_anpd:  bool    # satisfaz k≥5 conforme Guia ANPD 2023


# ---------------------------------------------------------------------------
# KAnonymityAnalyzer
# ---------------------------------------------------------------------------

class KAnonymityAnalyzer:
    """
    Avalia k-anonimato e l-diversidade de um DataFrame após mascaramento.

    Parâmetros:
        quasi_identifiers: Colunas que formam o perfil de quasi-identificador.
            Ex: ["idade_faixa", "uf", "genero", "escolaridade"]
            ATENÇÃO: inclua apenas as colunas que um adversário externo
            poderia cruzar com bases públicas (censo, redes sociais, etc.).
        target_k: k mínimo desejado. Usado para gerar recomendações.
        sensitive_attributes: Colunas de atributos sensíveis para l-diversidade.
            Ex: ["diagnostico", "salario_faixa"]

    Exemplo:
        analyzer = KAnonymityAnalyzer(
            quasi_identifiers=["idade_faixa", "uf", "genero"],
            target_k=5,
            sensitive_attributes=["diagnostico"],
        )
        report = analyzer.evaluate(df_mascarado)
        analyzer.print_report(report)
    """

    def __init__(
        self,
        quasi_identifiers: List[str],
        target_k: int = 5,
        sensitive_attributes: Optional[List[str]] = None,
    ):
        if not quasi_identifiers:
            raise ValueError(
                "KAnonymityAnalyzer: quasi_identifiers não pode ser vazio. "
                "Informe as colunas que formam o perfil de re-identificação."
            )
        self.quasi_identifiers   = quasi_identifiers
        self.target_k            = target_k
        self.sensitive_attributes = sensitive_attributes or []

    def evaluate(self, df: pd.DataFrame) -> PrivacyMetricsReport:
        """
        Calcula k-anonimato e l-diversidade do DataFrame.

        Args:
            df: DataFrame após aplicação do pipeline de mascaramento.

        Returns:
            PrivacyMetricsReport com todas as métricas e recomendações.
        """
        k_report = self._calculate_k(df)

        l_reports = []
        for attr in self.sensitive_attributes:
            if attr in df.columns:
                l_reports.append(self._calculate_l(df, k_report, attr))

        l_report = l_reports[0] if l_reports else None

        overall = self._overall_risk(k_report, l_report)
        recs = self._recommendations(k_report, l_report, df)

        return PrivacyMetricsReport(
            k_anonymity=k_report,
            l_diversity=l_report,
            overall_risk=overall,
            recommendations=recs,
            compliant_anpd=k_report.satisfies_k5,
        )

    def print_report(self, report: PrivacyMetricsReport) -> None:
        """Imprime relatório formatado no console."""
        k = report.k_anonymity
        _risk_icons = {
            "critical": "🔴 CRÍTICO",
            "high":     "🟠 ALTO",
            "medium":   "🟡 MÉDIO",
            "low":      "🟢 BAIXO",
        }

        print("\n" + "═" * 65)
        print("📊  MÉTRICAS DE PRIVACIDADE — k-ANONIMATO")
        print("═" * 65)
        print(f"  Quasi-identifiers : {', '.join(k.quasi_identifiers)}")
        print(f"  Total de grupos   : {k.total_groups:,}")
        print(f"  k-anonimato       : k = {k.k_value}  →  {_risk_icons.get(k.risk_level, k.risk_level)}")
        print(f"  Grupos únicos     : {k.singleton_groups:,} ({k.singleton_fraction:.1%} dos grupos)")
        print(f"  Distribuição      : p25={k.p25_group_size:.0f}  p50={k.p50_group_size:.0f}  p75={k.p75_group_size:.0f} registros/grupo")
        print(f"  Conformidade ANPD : {'✅ k≥5 (Guia ANPD 2023)' if k.satisfies_k5 else '❌ k<5 — abaixo do mínimo recomendado'}")

        if report.l_diversity:
            l = report.l_diversity
            print(f"\n  l-Diversidade ({l.sensitive_attribute})")
            print(f"    l = {l.l_value}  →  {_risk_icons.get(l.risk_level, l.risk_level)}")
            print(f"    Grupos homogêneos: {l.homogeneous_groups} ({l.homogeneous_fraction:.1%})")

        if report.recommendations:
            print("\n  ⚙️  Recomendações:")
            for rec in report.recommendations:
                print(f"    • {rec}")

        print("═" * 65 + "\n")

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _calculate_k(self, df: pd.DataFrame) -> KAnonymityReport:
        valid_qi = [c for c in self.quasi_identifiers if c in df.columns]
        missing = set(self.quasi_identifiers) - set(valid_qi)
        if missing:
            logger.warning(
                "KAnonymityAnalyzer: quasi-identifiers não encontrados no df: %s",
                missing,
            )

        if not valid_qi:
            return KAnonymityReport(
                k_value=0,
                quasi_identifiers=[],
                total_groups=0,
                singleton_groups=0,
                singleton_fraction=0.0,
                risk_level="unknown",
                interpretation="Nenhum quasi-identifier válido fornecido.",
            )

        counts = (
            df.groupby(valid_qi, dropna=False)
            .size()
            .reset_index(name="_n")
        )

        k_value         = int(counts["_n"].min())
        total_groups    = len(counts)
        singletons      = int((counts["_n"] == 1).sum())
        sing_frac       = round(singletons / max(total_groups, 1), 4)
        p25             = float(counts["_n"].quantile(0.25))
        p50             = float(counts["_n"].quantile(0.50))
        p75             = float(counts["_n"].quantile(0.75))
        dist            = counts["_n"].value_counts().sort_index().head(10).to_dict()
        dist_str        = {str(k): int(v) for k, v in dist.items()}

        risk, interp = _interpret_k(k_value, singletons, total_groups, self.target_k)

        logger.info(
            "k-anonimato | qi=%s | k=%d | singletons=%d/%d | risk=%s",
            valid_qi, k_value, singletons, total_groups, risk,
        )

        return KAnonymityReport(
            k_value=k_value,
            quasi_identifiers=valid_qi,
            total_groups=total_groups,
            singleton_groups=singletons,
            singleton_fraction=sing_frac,
            risk_level=risk,
            interpretation=interp,
            group_size_dist=dist_str,
            p25_group_size=p25,
            p50_group_size=p50,
            p75_group_size=p75,
        )

    def _calculate_l(
        self,
        df: pd.DataFrame,
        k_report: KAnonymityReport,
        sensitive_attr: str,
    ) -> LDiversityReport:
        qi = k_report.quasi_identifiers
        if not qi:
            return _empty_l(sensitive_attr, qi)

        l_values = (
            df.groupby(qi, dropna=False)[sensitive_attr]
            .nunique()
        )

        l_value       = int(l_values.min())
        total_groups  = len(l_values)
        homogeneous   = int((l_values == 1).sum())
        homo_frac     = round(homogeneous / max(total_groups, 1), 4)

        if l_value == 1:
            risk  = "critical"
            interp = (
                f"l=1: {homogeneous} grupos têm apenas 1 valor de '{sensitive_attr}'. "
                f"Qualquer um nesse grupo tem seu atributo sensível revelado "
                f"(ataque de homogeneidade — Machanavajjhala et al., 2007)."
            )
        elif l_value < 3:
            risk  = "high"
            interp = (
                f"l={l_value}: grupos com poucos valores distintos de '{sensitive_attr}'. "
                f"Vulnerável a ataques de similaridade. Recomendado l≥3."
            )
        else:
            risk  = "low"
            interp = (
                f"l={l_value}: cada grupo tem pelo menos {l_value} valores distintos "
                f"de '{sensitive_attr}'. Proteção adequada contra ataque de homogeneidade."
            )

        return LDiversityReport(
            l_value=l_value,
            sensitive_attribute=sensitive_attr,
            quasi_identifiers=qi,
            total_groups=total_groups,
            homogeneous_groups=homogeneous,
            homogeneous_fraction=homo_frac,
            risk_level=risk,
            interpretation=interp,
        )

    def _overall_risk(
        self,
        k: KAnonymityReport,
        l: Optional[LDiversityReport],
    ) -> str:
        _order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
        risks = [k.risk_level]
        if l:
            risks.append(l.risk_level)
        return min(risks, key=lambda r: _order.get(r, 99))

    def _recommendations(
        self,
        k: KAnonymityReport,
        l: Optional[LDiversityReport],
        df: pd.DataFrame,
    ) -> List[str]:
        recs = []

        if k.k_value < self.target_k:
            recs.append(
                f"k={k.k_value} está abaixo do alvo k={self.target_k}. "
                f"Aumente a generalização dos QIs ou aplique supressão de outliers."
            )

        if k.singleton_groups > 0:
            recs.append(
                f"{k.singleton_groups} registros únicos detectados. "
                f"Considere suprimir ou generalizar mais as colunas: "
                f"{', '.join(k.quasi_identifiers[:3])}."
            )

        if not k.satisfies_k5:
            recs.append(
                "Para compartilhamento com terceiros, k≥5 é o mínimo "
                "recomendado pelo Guia de Anonimização da ANPD (2023)."
            )

        if l and l.l_value < 2:
            recs.append(
                f"l-diversidade insuficiente em '{l.sensitive_attribute}' (l={l.l_value}). "
                f"Aplique generalização adicional ou embaralhamento dentro do grupo."
            )

        if not recs:
            recs.append(
                f"k={k.k_value} ≥ {self.target_k} (alvo). "
                f"Dataset satisfaz o padrão de privacidade configurado."
            )

        return recs


# ---------------------------------------------------------------------------
# Função de conveniência
# ---------------------------------------------------------------------------

def calculate_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: Optional[List[str]] = None,
    target_k: int = 5,
) -> PrivacyMetricsReport:
    """
    Calcula k-anonimato (e opcionalmente l-diversidade) em uma linha.

    Args:
        df: DataFrame a avaliar.
        quasi_identifiers: Colunas de quasi-identificadores.
        sensitive_attributes: Colunas de atributos sensíveis (para l-diversidade).
        target_k: k mínimo desejado para recomendações.

    Returns:
        PrivacyMetricsReport completo.

    Exemplo:
        report = calculate_k_anonymity(
            df_mascarado,
            quasi_identifiers=["idade_faixa", "uf", "genero"],
            sensitive_attributes=["renda_faixa"],
            target_k=5,
        )
        print(f"k = {report.k_anonymity.k_value}")
        print(f"Conforme ANPD: {report.compliant_anpd}")
    """
    analyzer = KAnonymityAnalyzer(
        quasi_identifiers=quasi_identifiers,
        target_k=target_k,
        sensitive_attributes=sensitive_attributes,
    )
    return analyzer.evaluate(df)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpret_k(
    k: int, singletons: int, total: int, target: int
) -> Tuple[str, str]:
    if k == 0:
        return "unknown", "Sem grupos — dados insuficientes."
    if k == 1:
        return "critical", (
            f"k=1: {singletons} grupos com registro único ({singletons/max(total,1):.1%} do total). "
            f"Esses registros são re-identificáveis diretamente por cruzamento com bases externas. "
            f"Aplique mais generalização ou supressão antes de compartilhar."
        )
    if k < 3:
        return "high", (
            f"k={k}: grupos muito pequenos — alto risco de re-identificação. "
            f"Qualquer adversário com {k-1} registros de contexto consegue singularizar. "
            f"Recomendado k≥5 para publicação (ANPD, 2023)."
        )
    if k < 5:
        return "medium", (
            f"k={k}: risco moderado. Dataset não atinge k≥5 recomendado pelo Guia ANPD. "
            f"Para uso interno pode ser aceitável; para compartilhamento externo, generalize mais."
        )
    return "low", (
        f"k={k}: cada combinação de quasi-identifiers aparece em ≥{k} registros. "
        f"{'Satisfaz k≥5 (ANPD 2023).' if k >= 5 else ''} "
        f"{'Satisfaz k≥10 (padrão conservador para dados sensíveis).' if k >= 10 else ''}"
    )


def _empty_l(attr: str, qi: List[str]) -> LDiversityReport:
    return LDiversityReport(
        l_value=0,
        sensitive_attribute=attr,
        quasi_identifiers=qi,
        total_groups=0,
        homogeneous_groups=0,
        homogeneous_fraction=0.0,
        risk_level="unknown",
        interpretation="Sem dados suficientes para calcular l-diversidade.",
    )
