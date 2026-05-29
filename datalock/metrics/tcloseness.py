"""
metrics/tcloseness.py
=====================
t-Closeness — Terceira Métrica Formal da Hierarquia de Privacidade.

Hierarquia formal:
    k-Anonimato < l-Diversidade < t-Closeness

Por que t-closeness é necessário?
----------------------------------
k-anonimato garante que ninguém é único pelos quasi-identifiers.
l-diversidade garante que o atributo sensível é diverso dentro do grupo.
Mas l-diversidade ainda falha quando:

  (1) Skewness attack: um grupo tem l valores distintos mas 99% concentrado
      num único valor (ex: 1 saudável, 99 com câncer de pâncreas).
  (2) Similarity attack: os l valores são semanticamente próximos
      (ex: "gastrite leve", "gastrite moderada", "gastrite severa" → todos
      revelam diagnóstico de gastrite ao adversário).

t-Closeness (Li et al., ICDE 2007) resolve isso exigindo que a distribuição
do atributo sensível em cada grupo de equivalência esteja a distância ≤ t
da distribuição global — usando Earth Mover Distance (Wasserstein Distance)
para atributos numéricos e Total Variation Distance para categóricos.

Interpretação prática de t:
    t ≤ 0.10  → muito rigoroso (dados de saúde, biometria)
    t ≤ 0.20  → rigoroso (financeiro, judicial)
    t ≤ 0.30  → moderado (analytics geral)
    t > 0.30  → fraco (pode revelar distribuição sensível)

Referências:
    Li, N., Li, T., Venkatasubramanian, S. (2007).
    t-Closeness: Privacy Beyond k-Anonymity and l-Diversity.
    IEEE International Conference on Data Engineering (ICDE) 2007.
    DOI: 10.1109/ICDE.2007.367856

    Machanavajjhala, A., et al. (2007). l-Diversity. ACM TKDD.
    (extensão do conceito para t-closeness)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class GroupTClosenessResult:
    """Resultado de t-closeness para um grupo de equivalência."""
    group_id:       int
    group_size:     int
    t_value:        float      # distância calculada para este grupo
    satisfies_t:    bool       # True se t_value <= target_t
    worst_case:     bool       # True se é o grupo mais distante da global


@dataclass
class TClosenessReport:
    """
    Resultado completo da análise de t-closeness.

    Interpretação:
        global_t_value:  distância máxima entre qualquer grupo e a distribuição
                         global — este é o t real do dataset.
        satisfies_t:     True se global_t_value <= target_t (dataset é t-close).
        n_violating:     grupos que excedem target_t (quanto maior, pior).
    """
    sensitive_attribute:  str
    quasi_identifiers:    List[str]
    target_t:             float
    global_t_value:       float        # t real do dataset (max entre grupos)
    mean_t_value:         float        # t médio (severidade geral)
    satisfies_t:          bool
    n_groups:             int
    n_violating:          int          # grupos que excedem target_t
    violating_fraction:   float        # n_violating / n_groups
    distance_type:        str          # "wasserstein" ou "total_variation"
    group_results:        List[GroupTClosenessResult] = field(default_factory=list)
    recommendations:      List[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if self.global_t_value <= 0.10:
            return "very_low"
        elif self.global_t_value <= 0.20:
            return "low"
        elif self.global_t_value <= 0.30:
            return "moderate"
        elif self.global_t_value <= 0.50:
            return "high"
        else:
            return "critical"

    @property
    def risk_icon(self) -> str:
        return {"very_low": "✅", "low": "🟡", "moderate": "🟠",
                "high": "🔴", "critical": "🚨"}.get(self.risk_level, "⚪")


# ---------------------------------------------------------------------------
# TClosenessAnalyzer
# ---------------------------------------------------------------------------

class TClosenessAnalyzer:
    """
    Avalia t-closeness sobre grupos de equivalência de quasi-identifiers.

    Para atributos sensíveis NUMÉRICOS: usa Earth Mover Distance (Wasserstein),
    como especificado em Li et al. (2007). A EMD mede o "trabalho" necessário
    para transformar a distribuição do grupo na distribuição global.

    Para atributos sensíveis CATEGÓRICOS: usa Total Variation Distance
    (variação total), que mede a diferença máxima entre probabilidades.

    Parâmetros:
        quasi_identifiers:  Colunas quasi-identificadoras (definem os grupos).
        sensitive_attribute: Coluna sensível a analisar (ex: "diagnostico", "renda").
        target_t:           Valor máximo de t aceitável (padrão: 0.20).

    Exemplo:
        from datalock.metrics.tcloseness import TClosenessAnalyzer

        analyzer = TClosenessAnalyzer(
            quasi_identifiers=["uf", "faixa_etaria"],
            sensitive_attribute="diagnostico",
            target_t=0.15,
        )
        report = analyzer.evaluate(df_mascarado)
        analyzer.print_report(report)
    """

    def __init__(
        self,
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        target_t: float = 0.20,
    ) -> None:
        if not quasi_identifiers:
            raise ValueError("quasi_identifiers não pode ser vazio.")
        if not 0 < target_t <= 1:
            raise ValueError(f"target_t deve estar em (0, 1]. Recebido: {target_t}")
        self.quasi_identifiers   = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.target_t            = target_t

    def evaluate(self, df: pd.DataFrame) -> TClosenessReport:
        """
        Calcula t-closeness para cada grupo de equivalência.

        Parâmetros:
            df: DataFrame (preferencialmente já mascarado) a avaliar.

        Retorna:
            TClosenessReport com t real do dataset, grupos violadores e recomendações.
        """
        missing = [c for c in self.quasi_identifiers + [self.sensitive_attribute]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas no DataFrame: {missing}")

        is_numeric = pd.api.types.is_numeric_dtype(df[self.sensitive_attribute])
        distance_type = "wasserstein" if is_numeric else "total_variation"

        # Distribuição global do atributo sensível
        global_series = df[self.sensitive_attribute].dropna()
        if len(global_series) == 0:
            raise ValueError(f"Coluna '{self.sensitive_attribute}' está completamente vazia.")

        # Agrupa pelos quasi-identifiers
        valid_qi = [c for c in self.quasi_identifiers if c in df.columns]
        groups = df.groupby(valid_qi, dropna=False)

        group_results: List[GroupTClosenessResult] = []
        t_values: List[float] = []

        for group_id, (group_key, group_df) in enumerate(groups):
            group_series = group_df[self.sensitive_attribute].dropna()
            if len(group_series) < 2:
                # Grupos com <2 registros têm distância máxima (não representativos)
                t_val = 1.0
            else:
                t_val = self._compute_distance(
                    group_series, global_series, is_numeric
                )

            t_values.append(t_val)
            group_results.append(GroupTClosenessResult(
                group_id=group_id,
                group_size=len(group_df),
                t_value=round(float(t_val), 6),
                satisfies_t=t_val <= self.target_t,
                worst_case=False,  # atualizado abaixo
            ))

        if not t_values:
            raise ValueError("Nenhum grupo de equivalência encontrado.")

        global_t  = float(max(t_values))
        mean_t    = float(np.mean(t_values))
        n_violate = sum(1 for t in t_values if t > self.target_t)

        # Marca o grupo mais distante
        worst_idx = int(np.argmax(t_values))
        group_results[worst_idx].worst_case = True

        satisfies = global_t <= self.target_t
        recommendations = self._build_recommendations(
            global_t, n_violate, len(group_results), is_numeric
        )

        logger.info(
            "TCloseness | attr=%s | t_real=%.4f | target=%.2f | satisfies=%s | "
            "grupos=%d | violadores=%d",
            self.sensitive_attribute, global_t, self.target_t,
            satisfies, len(group_results), n_violate,
        )

        return TClosenessReport(
            sensitive_attribute=self.sensitive_attribute,
            quasi_identifiers=self.quasi_identifiers,
            target_t=self.target_t,
            global_t_value=round(global_t, 6),
            mean_t_value=round(mean_t, 6),
            satisfies_t=satisfies,
            n_groups=len(group_results),
            n_violating=n_violate,
            violating_fraction=round(n_violate / max(len(group_results), 1), 4),
            distance_type=distance_type,
            group_results=group_results,
            recommendations=recommendations,
        )

    def print_report(self, report: TClosenessReport) -> None:
        """Imprime relatório de t-closeness formatado no console."""
        print()
        print("=" * 65)
        print("RELATÓRIO DE t-CLOSENESS")
        print("=" * 65)
        print(f"  {report.risk_icon} t real (global)  : {report.global_t_value:.4f}")
        print(f"  t alvo           : {report.target_t}")
        print(f"  Satisfaz t-close : {report.satisfies_t}")
        print(f"  Nível de risco   : {report.risk_level}")
        print(f"  Distância usada  : {report.distance_type}")
        print(f"  Atributo sensível: {report.sensitive_attribute}")
        print(f"  Quasi-ids        : {report.quasi_identifiers}")
        print()
        print(f"  Grupos totais    : {report.n_groups}")
        print(f"  Grupos violadores: {report.n_violating} ({report.violating_fraction:.1%})")
        print(f"  t médio          : {report.mean_t_value:.4f}")
        print()

        # Top 5 piores grupos
        sorted_groups = sorted(report.group_results, key=lambda g: -g.t_value)[:5]
        print(f"  Top 5 grupos mais distantes da distribuição global:")
        for g in sorted_groups:
            icon = "❌" if not g.satisfies_t else "✅"
            worst = " ← pior caso" if g.worst_case else ""
            print(f"    {icon} Grupo {g.group_id:03d} | n={g.group_size:4d} | "
                  f"t={g.t_value:.4f}{worst}")

        if report.recommendations:
            print()
            print("  Recomendações:")
            for rec in report.recommendations:
                print(f"    • {rec}")
        print()
        print("  Referência: Li et al. (2007). t-Closeness. ICDE 2007.")

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _compute_distance(
        self,
        group: pd.Series,
        global_: pd.Series,
        is_numeric: bool,
    ) -> float:
        """
        Computa a distância entre a distribuição do grupo e a global.

        Numérico: Earth Mover Distance / Wasserstein Distance (Li et al. 2007, §3.3).
        Categórico: Total Variation Distance = 0.5 × Σ|p_i - q_i| (Li et al. 2007, §3.2).
        """
        try:
            if is_numeric:
                return float(wasserstein_distance(
                    group.astype(float).values,
                    global_.astype(float).values,
                ))
            else:
                # Total Variation Distance para categóricos
                all_cats = set(group.astype(str).unique()) | set(global_.astype(str).unique())
                p = group.astype(str).value_counts(normalize=True)
                q = global_.astype(str).value_counts(normalize=True)
                tvd = 0.5 * sum(abs(p.get(c, 0.0) - q.get(c, 0.0)) for c in all_cats)
                return float(tvd)
        except Exception as e:
            logger.warning("TCloseness: erro ao calcular distância: %s", e)
            return 1.0

    def _build_recommendations(
        self,
        global_t: float,
        n_violate: int,
        n_groups: int,
        is_numeric: bool,
    ) -> List[str]:
        recs = []
        if global_t > self.target_t:
            recs.append(
                f"t real ({global_t:.4f}) excede o alvo ({self.target_t}). "
                f"Aplicar generalização adicional nos quasi-identifiers."
            )
        if n_violate > 0:
            pct = n_violate / max(n_groups, 1)
            recs.append(
                f"{n_violate} grupos ({pct:.0%}) violam t-closeness. "
                f"Considere suprimir ou mesclar grupos pequenos."
            )
        if is_numeric and global_t > 0.25:
            recs.append(
                "Para atributos numéricos sensíveis com alta dispersão, "
                "considere bucketização (faixas) antes da avaliação."
            )
        if not is_numeric and global_t > 0.30:
            recs.append(
                "Para atributos categóricos, t-closeness pode ser relaxado "
                "usando hierarquias de generalização (ex: CID→grupo diagnóstico)."
            )
        if not recs:
            recs.append(
                f"Dataset satisfaz t-closeness com t={global_t:.4f} ≤ {self.target_t}. "
                f"Distribuição do atributo sensível preservada dentro dos grupos."
            )
        return recs


# ---------------------------------------------------------------------------
# Função de conveniência
# ---------------------------------------------------------------------------

def calculate_t_closeness(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attribute: str,
    target_t: float = 0.20,
) -> TClosenessReport:
    """
    Calcula t-closeness — atalho para uso sem instanciar TClosenessAnalyzer.

    Parâmetros:
        df:                  DataFrame a avaliar.
        quasi_identifiers:   Colunas quasi-identificadoras.
        sensitive_attribute: Atributo sensível.
        target_t:            t máximo aceitável (padrão 0.20).

    Retorna:
        TClosenessReport com todos os resultados.

    Exemplo:
        report = dd.check.t_closeness(
            df_mascarado,
            quasi_identifiers=["uf", "faixa_etaria"],
            sensitive_attribute="diagnostico",
            target_t=0.15,
        )
        print(f"t real: {report.global_t_value} | satisfaz: {report.satisfies_t}")
    """
    analyzer = TClosenessAnalyzer(
        quasi_identifiers=quasi_identifiers,
        sensitive_attribute=sensitive_attribute,
        target_t=target_t,
    )
    return analyzer.evaluate(df)
