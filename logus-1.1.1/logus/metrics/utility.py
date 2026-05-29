"""
metrics/utility.py
==================
Métricas de Utilidade Analítica Pós-Mascaramento.

Fundamentação Teórica
---------------------
PSI — Population Stability Index (Siddiqi, 2006):
    Mede drift de distribuição entre original e mascarado.
    PSI < 0.10 → distribuição estável (mascaramento leve)
    PSI 0.10–0.25 → atenção — alguma distorção
    PSI > 0.25 → distribuição significativamente alterada

    Amplamente usado em modelos de crédito (Basel II) para detectar
    data drift. Ref: Siddiqi, N. "Credit Risk Scorecards", 2006.

KS Test — Kolmogorov-Smirnov (Kolmogorov, 1933):
    Teste não-paramétrico de igualdade de distribuições contínuas.
    KS statistic = max distância entre CDFs empíricas.
    p-value < 0.05 → rejeita H0 (distribuições são diferentes).
    Ref: Massey, F.J. "The Kolmogorov-Smirnov Test for Goodness of Fit."
    JASA, 1951.

Jensen-Shannon Divergence (Lin, 1991):
    Versão simétrica e limitada (0–1) da divergência KL.
    Usada para variáveis categóricas onde KS não se aplica.
    JSD = 0 → distribuições idênticas; JSD = 1 → completamente diferentes.

Utility Score (composto):
    Score 0–1 derivado de KS, PSI e diferenças de média/desvio.
    Score > 0.85 → alta utilidade preservada
    Score 0.70–0.85 → utilidade moderada
    Score < 0.70 → utilidade comprometida (considere técnicas menos agressivas)

Uso:
    from datalock.metrics.utility import UtilityEvaluator

    evaluator = UtilityEvaluator()
    report = evaluator.evaluate(df_original, df_mascarado)
    print(f"Utilidade geral: {report.overall_score:.1%}")
    print(f"PSI médio: {report.mean_psi:.3f}")
    evaluator.print_report(report)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class ColumnUtilityReport:
    """Métricas de utilidade para uma coluna individual."""
    column:          str
    dtype:           str
    # Numéricas
    orig_mean:       Optional[float] = None
    mask_mean:       Optional[float] = None
    orig_std:        Optional[float] = None
    mask_std:        Optional[float] = None
    mean_diff_pct:   Optional[float] = None   # |orig - mask| / |orig| × 100
    std_diff_pct:    Optional[float] = None
    ks_statistic:    Optional[float] = None
    ks_pvalue:       Optional[float] = None
    psi:             Optional[float] = None
    # Categóricas
    jsd:             Optional[float] = None   # Jensen-Shannon Divergence
    top_cat_overlap: Optional[float] = None   # overlap das top-10 categorias
    # Score composto 0–1
    utility_score:   float = 0.0
    assessment:      str = ""   # "✓ preservada" | "△ moderada" | "⚠ comprometida"


@dataclass
class CorrelationUtilityReport:
    """Preservação de correlações entre colunas."""
    pearson_mean_delta:   float   # média das diferenças absolutas de Pearson
    spearman_mean_delta:  float
    max_pearson_delta:    float
    most_changed_pairs:   List[Tuple[str, str, float]] = field(default_factory=list)
    correlation_score:    float = 0.0   # 1 - pearson_mean_delta


@dataclass
class UtilityReport:
    """Relatório consolidado de utilidade analítica."""
    column_reports:      List[ColumnUtilityReport]
    correlation_report:  Optional[CorrelationUtilityReport]
    overall_score:       float   # média ponderada de todos os scores
    mean_psi:            float   # PSI médio das colunas numéricas
    risk_reduction:      float   # estimativa de redução de identificabilidade
    assessment:          str     # avaliação textual

    @property
    def columns_compromised(self) -> List[str]:
        return [r.column for r in self.column_reports if r.utility_score < 0.70]

    @property
    def columns_preserved(self) -> List[str]:
        return [r.column for r in self.column_reports if r.utility_score >= 0.85]


# ---------------------------------------------------------------------------
# UtilityEvaluator
# ---------------------------------------------------------------------------

class UtilityEvaluator:
    """
    Avalia quanto de utilidade analítica foi preservada após mascaramento.

    Compara distribuições (PSI, KS), correlações e estatísticas descritivas
    entre o DataFrame original e o mascarado.

    Uso:
        evaluator = UtilityEvaluator()
        report = evaluator.evaluate(df_original, df_mascarado)
        evaluator.print_report(report)

        # Exporta como DataFrame para análise:
        df_util = evaluator.to_dataframe(report)
    """

    def __init__(
        self,
        psi_bins: int = 10,
        corr_threshold: float = 0.05,
    ):
        """
        Args:
            psi_bins: Número de bins para cálculo do PSI.
            corr_threshold: Delta mínimo de correlação para listar par alterado.
        """
        self.psi_bins       = psi_bins
        self.corr_threshold = corr_threshold

    def evaluate(
        self,
        df_orig: pd.DataFrame,
        df_mask: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> UtilityReport:
        """
        Avalia utilidade completa entre original e mascarado.

        Args:
            df_orig: DataFrame original (antes do mascaramento).
            df_mask: DataFrame mascarado (após pipeline).
            columns: Colunas a avaliar. None = todas as colunas em comum.

        Returns:
            UtilityReport com todas as métricas.
        """
        cols = columns or [c for c in df_orig.columns if c in df_mask.columns]
        col_reports = []

        for col in cols:
            try:
                rep = self._evaluate_column(df_orig[col], df_mask[col], col)
                col_reports.append(rep)
            except Exception as exc:
                logger.warning("Utilidade: erro ao avaliar '%s': %s", col, exc)

        corr_report = self._evaluate_correlations(df_orig[cols], df_mask[cols])
        risk_red    = self._estimate_risk_reduction(df_orig, df_mask)

        scores = [r.utility_score for r in col_reports]
        corr_s = corr_report.correlation_score if corr_report else 1.0
        overall = float(np.mean(scores + [corr_s])) if scores else 0.0

        psi_vals = [r.psi for r in col_reports if r.psi is not None]
        mean_psi = float(np.mean(psi_vals)) if psi_vals else 0.0

        return UtilityReport(
            column_reports=col_reports,
            correlation_report=corr_report,
            overall_score=round(overall, 4),
            mean_psi=round(mean_psi, 4),
            risk_reduction=round(risk_red, 4),
            assessment=_overall_assessment(overall, risk_red),
        )

    def print_report(self, report: UtilityReport) -> None:
        """Imprime relatório formatado."""
        print("\n" + "═" * 65)
        print("📈  MÉTRICAS DE UTILIDADE ANALÍTICA")
        print("═" * 65)
        print(f"  Utilidade geral   : {report.overall_score:.1%}  {_score_icon(report.overall_score)}")
        print(f"  PSI médio         : {report.mean_psi:.3f}  {_psi_label(report.mean_psi)}")
        print(f"  Redução de risco  : {report.risk_reduction:.1%}")
        print(f"  Avaliação         : {report.assessment}")

        if report.correlation_report:
            cr = report.correlation_report
            print(f"\n  Correlações")
            print(f"    Δ Pearson médio  : {cr.pearson_mean_delta:.4f}")
            print(f"    Δ Spearman médio : {cr.spearman_mean_delta:.4f}")
            if cr.most_changed_pairs:
                print(f"    Pares mais alterados:")
                for a, b, d in cr.most_changed_pairs[:3]:
                    print(f"      {a} × {b}: Δ={d:.4f}")

        print(f"\n  Por coluna ({len(report.column_reports)} avaliadas):")
        for r in sorted(report.column_reports, key=lambda x: x.utility_score):
            icon = _score_icon(r.utility_score)
            psi_str = f" PSI={r.psi:.3f}" if r.psi is not None else ""
            ks_str  = f" KS={r.ks_statistic:.3f}" if r.ks_statistic is not None else ""
            jsd_str = f" JSD={r.jsd:.3f}" if r.jsd is not None else ""
            print(
                f"  {icon} {r.column:<28} "
                f"score={r.utility_score:.2f}"
                f"{psi_str}{ks_str}{jsd_str}"
            )
        print("═" * 65 + "\n")

    def to_dataframe(self, report: UtilityReport) -> pd.DataFrame:
        """Retorna DataFrame com métricas por coluna."""
        rows = []
        for r in report.column_reports:
            rows.append({
                "coluna":         r.column,
                "dtype":          r.dtype,
                "utility_score":  round(r.utility_score, 4),
                "psi":            round(r.psi, 4) if r.psi is not None else None,
                "ks_stat":        round(r.ks_statistic, 4) if r.ks_statistic is not None else None,
                "ks_pvalue":      round(r.ks_pvalue, 4) if r.ks_pvalue is not None else None,
                "jsd":            round(r.jsd, 4) if r.jsd is not None else None,
                "mean_diff_pct":  round(r.mean_diff_pct, 2) if r.mean_diff_pct is not None else None,
                "std_diff_pct":   round(r.std_diff_pct, 2) if r.std_diff_pct is not None else None,
                "assessment":     r.assessment,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _evaluate_column(
        self, orig: pd.Series, mask: pd.Series, col: str
    ) -> ColumnUtilityReport:
        is_num = pd.api.types.is_numeric_dtype(orig) and pd.api.types.is_numeric_dtype(mask)

        if is_num:
            return self._numeric_metrics(orig, mask, col)
        else:
            return self._categorical_metrics(orig, mask, col)

    def _numeric_metrics(
        self, orig: pd.Series, mask: pd.Series, col: str
    ) -> ColumnUtilityReport:
        o = orig.dropna().astype(float)
        m = mask.dropna().astype(float)

        if len(o) == 0 or len(m) == 0:
            return ColumnUtilityReport(column=col, dtype="numeric", utility_score=0.0,
                                       assessment="⚠ vazia após limpeza")

        o_mean, m_mean = o.mean(), m.mean()
        o_std,  m_std  = o.std(),  m.std()

        mean_diff = abs(o_mean - m_mean) / max(abs(o_mean), 1e-9) * 100
        std_diff  = abs(o_std  - m_std)  / max(abs(o_std),  1e-9) * 100

        ks_stat, ks_pval = stats.ks_2samp(o.values, m.values)
        psi_val          = _psi(o.values, m.values, self.psi_bins)

        score = _numeric_score(ks_stat, psi_val, mean_diff, std_diff)

        return ColumnUtilityReport(
            column=col,
            dtype="numeric",
            orig_mean=round(o_mean, 4),
            mask_mean=round(m_mean, 4),
            orig_std=round(o_std, 4),
            mask_std=round(m_std, 4),
            mean_diff_pct=round(mean_diff, 2),
            std_diff_pct=round(std_diff, 2),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 4),
            psi=round(psi_val, 4),
            utility_score=round(score, 4),
            assessment=_col_assessment(score),
        )

    def _categorical_metrics(
        self, orig: pd.Series, mask: pd.Series, col: str
    ) -> ColumnUtilityReport:
        o_freq = orig.value_counts(normalize=True)
        m_freq = mask.value_counts(normalize=True)

        all_cats = set(o_freq.index) | set(m_freq.index)
        if not all_cats:
            return ColumnUtilityReport(column=col, dtype="categorical", utility_score=0.0,
                                       assessment="⚠ vazia")

        o_vec = np.array([o_freq.get(c, 0) for c in all_cats])
        m_vec = np.array([m_freq.get(c, 0) for c in all_cats])

        # Jensen-Shannon Divergence — scipy já normaliza para [0, 1]
        jsd = float(stats.entropy(o_vec + 1e-9, m_vec + 1e-9))
        # Clipa JSD para [0, 1] (scipy.stats.entropy retorna KL não limitada)
        jsd = min(jsd, 1.0)

        # Overlap das top-10 categorias
        top_o = set(o_freq.head(10).index)
        top_m = set(m_freq.head(10).index)
        overlap = len(top_o & top_m) / max(len(top_o | top_m), 1)

        score = max(0.0, 1.0 - jsd) * 0.7 + overlap * 0.3

        return ColumnUtilityReport(
            column=col,
            dtype="categorical",
            jsd=round(jsd, 4),
            top_cat_overlap=round(overlap, 4),
            utility_score=round(score, 4),
            assessment=_col_assessment(score),
        )

    def _evaluate_correlations(
        self, orig: pd.DataFrame, mask: pd.DataFrame
    ) -> Optional[CorrelationUtilityReport]:
        num_orig = orig.select_dtypes(include=[np.number])
        num_mask = mask.select_dtypes(include=[np.number])
        shared   = [c for c in num_orig.columns if c in num_mask.columns]

        if len(shared) < 2:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_orig = num_orig[shared].corr(method="pearson").fillna(0)
            p_mask = num_mask[shared].corr(method="pearson").fillna(0)
            s_orig = num_orig[shared].corr(method="spearman").fillna(0)
            s_mask = num_mask[shared].corr(method="spearman").fillna(0)

        p_delta = (p_orig - p_mask).abs()
        s_delta = (s_orig - s_mask).abs()

        triu = np.triu_indices_from(p_delta.values, k=1)
        mean_p = float(p_delta.values[triu].mean()) if len(triu[0]) else 0.0
        mean_s = float(s_delta.values[triu].mean()) if len(triu[0]) else 0.0

        changed = []
        for i, col_a in enumerate(shared):
            for col_b in shared[i + 1:]:
                d = float(p_delta.loc[col_a, col_b])
                if d >= self.corr_threshold:
                    changed.append((col_a, col_b, round(d, 4)))
        changed.sort(key=lambda x: -x[2])

        return CorrelationUtilityReport(
            pearson_mean_delta=round(mean_p, 4),
            spearman_mean_delta=round(mean_s, 4),
            max_pearson_delta=round(float(p_delta.values.max()), 4),
            most_changed_pairs=changed[:10],
            correlation_score=round(max(0.0, 1.0 - mean_p), 4),
        )

    def _estimate_risk_reduction(
        self, df_orig: pd.DataFrame, df_mask: pd.DataFrame
    ) -> float:
        """Estimativa de redução de unicidade (proxy de risco)."""
        scores = []
        for col in df_orig.columns:
            if col not in df_mask.columns:
                continue
            n = max(len(df_orig), 1)
            orig_u = df_orig[col].nunique() / n
            mask_u = df_mask[col].nunique() / n
            scores.append(max(0.0, min(orig_u - mask_u, 1.0)))
        return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index entre duas distribuições numéricas."""
    bins = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=n_bins)
    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual,   bins=bins)
    e_pct = np.where(e_counts == 0, 1e-6, e_counts / max(len(expected), 1))
    a_pct = np.where(a_counts == 0, 1e-6, a_counts / max(len(actual),   1))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _numeric_score(ks: float, psi: float, mean_d: float, std_d: float) -> float:
    return float(np.mean([
        max(0.0, 1.0 - ks),
        max(0.0, 1.0 - min(psi, 1.0)),
        max(0.0, 1.0 - min(mean_d / 100, 1.0)),
        max(0.0, 1.0 - min(std_d  / 100, 1.0)),
    ]))


def _col_assessment(score: float) -> str:
    if score >= 0.85:
        return "✓ preservada"
    if score >= 0.70:
        return "△ moderada"
    return "⚠ comprometida"


def _overall_assessment(overall: float, risk_red: float) -> str:
    if overall >= 0.85 and risk_red >= 0.3:
        return "✓ bom trade-off utilidade × privacidade"
    if overall < 0.70:
        return "⚠ utilidade comprometida — considere técnicas menos agressivas"
    if risk_red < 0.15:
        return "△ baixa redução de risco — pipeline pode não ser suficiente"
    return "△ trade-off aceitável"


def _score_icon(score: float) -> str:
    if score >= 0.85:
        return "✅"
    if score >= 0.70:
        return "🟡"
    return "🔴"


def _psi_label(psi: float) -> str:
    if psi < 0.10:
        return "(estável)"
    if psi < 0.25:
        return "(atenção)"
    return "(instável — distribuição muito alterada)"
