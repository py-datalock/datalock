"""
metrics/fidelity.py
====================
FidelityEvaluator — Avaliação de Fidelidade de Dados Sintéticos.

O que é fidelidade estatística?
---------------------------------
Fidelidade mede o quanto os dados sintéticos reproduzem as propriedades
estatísticas dos dados reais, sem copiar registros individuais.

Uma alta fidelidade significa que:
  - Um modelo de ML treinado no sintético terá performance similar ao real
  - As distribuições marginais e correlações são preservadas
  - O dataset sintético é um substituto válido para análises estatísticas

Métricas implementadas
-----------------------
1. MARGINAL FIDELITY (por coluna)
   - KS Test (Kolmogorov-Smirnov): diferença entre distribuições contínuas
   - JSD (Jensen-Shannon Divergence): diferença entre distribuições categóricas
   - PSI (Population Stability Index): estabilidade de distribuição (crédito)
   - Δ Média e Δ Desvio-padrão: diferença percentual de estatísticas básicas

2. CORRELATION FIDELITY (multivariada)
   - Matriz de correlação de Pearson (real vs sintético)
   - Matriz de correlação de Spearman (real vs sintético)
   - RMSE das matrizes de correlação

3. TRAIN-ON-SYNTHETIC / TEST-ON-REAL (TSTR)
   - Treina classificador/regressor no dado sintético
   - Avalia no dado real (porção de teste)
   - Compara com Train-on-Real / Test-on-Real (baseline)
   - Métrica: razão de AUC ou R² (sintético vs real)
   - Limiar aceitável: TSTR/TRTR ≥ 0.70

4. PRIVACY SCORE (anti-memorização)
   - DCR (Distance to Closest Record): distância mínima do sintético ao real
   - NNDR (Nearest Neighbor Distance Ratio): indica memorização se << 1
   - Ideal: DCR > 0 e NNDR > 0.5

Referências
-----------
  Xu, L., et al. (2019). CTGAN. NeurIPS 2019.

  Nikolenko, S. (2021). Synthetic Data for Deep Learning.
    Springer, Chapter 3: Evaluation metrics for synthetic data.

  Yale, A., et al. (2020).
    Assessing privacy and quality of synthetic tabular data.
    In Workshop on Privacy in ML, NeurIPS 2020.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class ColumnFidelityReport:
    """Fidelidade por coluna."""
    column:         str
    column_type:    str           # "numeric" ou "categorical"
    # Numéricas
    ks_statistic:   Optional[float] = None
    ks_pvalue:      Optional[float] = None
    psi:            Optional[float] = None
    mean_diff_pct:  Optional[float] = None
    std_diff_pct:   Optional[float] = None
    # Categóricas
    jsd:            Optional[float] = None
    top_overlap:    Optional[float] = None
    # Score composto por coluna
    fidelity_score: float = 0.0
    assessment:     str = ""


@dataclass
class CorrelationFidelityReport:
    """Fidelidade das correlações multivariadas."""
    pearson_rmse:  float   # RMSE entre matrizes de correlação Pearson
    spearman_rmse: float   # RMSE entre matrizes de correlação Spearman
    mean_abs_diff: float   # Diferença absoluta média por par de colunas
    assessment:    str


@dataclass
class TSTRReport:
    """Train-on-Synthetic / Test-on-Real."""
    task_type:        str    # "classification" ou "regression"
    target_column:    str
    tstr_score:       float  # AUC (classif.) ou R² (reg.) treinando no sintético
    trtr_score:       float  # Baseline: treinando no real
    ratio:            float  # tstr / trtr (ideal ≥ 0.70)
    assessment:       str
    model_used:       str


@dataclass
class PrivacyScoreReport:
    """Avaliação de risco de memorização."""
    mean_dcr:      float   # Distance to Closest Record (média)
    min_dcr:       float   # Caso mais extremo de memorização
    nndr:          float   # Nearest Neighbor Distance Ratio (ideal > 0.5)
    n_memorized:   int     # registros com DCR ≈ 0 (possível cópia)
    assessment:    str


@dataclass
class FidelityReport:
    """Relatório consolidado de fidelidade."""
    column_reports:       List[ColumnFidelityReport] = field(default_factory=list)
    correlation_report:   Optional[CorrelationFidelityReport] = None
    tstr_report:          Optional[TSTRReport] = None
    privacy_score:        Optional[PrivacyScoreReport] = None
    overall_fidelity:     float = 0.0   # 0 a 1
    overall_privacy:      float = 0.0   # 0 a 1 (maior = menos memorização)
    overall_assessment:   str = ""
    n_real:               int = 0
    n_synthetic:          int = 0

    @property
    def columns_high_fidelity(self) -> List[str]:
        return [r.column for r in self.column_reports if r.fidelity_score >= 0.80]

    @property
    def columns_low_fidelity(self) -> List[str]:
        return [r.column for r in self.column_reports if r.fidelity_score < 0.60]


# ---------------------------------------------------------------------------
# FidelityEvaluator
# ---------------------------------------------------------------------------

class FidelityEvaluator:
    """
    Avalia a fidelidade estatística de um dataset sintético em relação ao real.

    Parâmetros:
        psi_bins:       Bins para o cálculo do PSI (padrão 10).
        n_sample:       Máximo de registros amostrados para cálculos (eficiência).
        tstr_target:    Coluna alvo para TSTR. Se None, TSTR não é calculado.
        privacy_sample: Amostra para cálculo de DCR/NNDR (custoso para n grande).

    Exemplo:
        gen = TabularGenerative(epochs=100).fit(df_mascarado)
        df_sintetico = gen.sample(1000)

        evaluator = FidelityEvaluator(tstr_target="churn")
        report = evaluator.evaluate(df_mascarado, df_sintetico)
        evaluator.print_report(report)
    """

    def __init__(
        self,
        psi_bins: int = 10,
        n_sample: int = 5000,
        tstr_target: Optional[str] = None,
        privacy_sample: int = 500,
    ) -> None:
        self.psi_bins = psi_bins
        self.n_sample = n_sample
        self.tstr_target = tstr_target
        self.privacy_sample = privacy_sample

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df_real: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> FidelityReport:
        """
        Avalia a fidelidade estatística do dataset sintético.

        Parâmetros:
            df_real:     Dataset real (ou mascarado usado como referência).
            df_synthetic: Dataset sintético gerado pelo TabularGenerative.
            columns:     Colunas a avaliar (None = interseção das colunas).

        Retorna:
            FidelityReport com métricas por coluna e globais.
        """
        cols = columns or [c for c in df_real.columns if c in df_synthetic.columns]
        real = df_real[cols].copy()
        synth = df_synthetic[cols].copy()

        # Amostragem para eficiência
        if len(real) > self.n_sample:
            real = real.sample(self.n_sample, random_state=42)
        if len(synth) > self.n_sample:
            synth = synth.sample(self.n_sample, random_state=42)

        # --- Métricas por coluna ---
        col_reports = []
        for col in cols:
            try:
                r = self._evaluate_column(col, real[col], synth[col])
                col_reports.append(r)
            except Exception as e:
                logger.warning("FidelityEvaluator: erro na coluna '%s': %s", col, e)

        # --- Correlações ---
        corr_report = self._evaluate_correlations(real, synth, cols)

        # --- TSTR ---
        tstr = None
        if self.tstr_target and self.tstr_target in real.columns and self.tstr_target in synth.columns:
            tstr = self._evaluate_tstr(real, synth, self.tstr_target, cols)

        # --- Privacy Score ---
        privacy = self._evaluate_privacy(real, synth, cols)

        # --- Score global ---
        overall_fid = np.mean([r.fidelity_score for r in col_reports]) if col_reports else 0.0
        overall_priv = privacy.nndr if privacy else 0.5
        overall_assess = _assess_overall(overall_fid, tstr)

        return FidelityReport(
            column_reports=col_reports,
            correlation_report=corr_report,
            tstr_report=tstr,
            privacy_score=privacy,
            overall_fidelity=round(float(overall_fid), 4),
            overall_privacy=round(float(overall_priv), 4),
            overall_assessment=overall_assess,
            n_real=len(df_real),
            n_synthetic=len(df_synthetic),
        )

    def print_report(self, report: FidelityReport) -> None:
        """Imprime relatório de fidelidade no console."""
        print()
        print("=" * 65)
        print("RELATÓRIO DE FIDELIDADE — DADOS SINTÉTICOS")
        print("=" * 65)
        print(f"  Registros reais    : {report.n_real:,}")
        print(f"  Registros sintéticos: {report.n_synthetic:,}")
        print()
        print(f"  Fidelidade geral   : {report.overall_fidelity:.1%}  {_fidelity_icon(report.overall_fidelity)}")
        print(f"  Avaliação          : {report.overall_assessment}")
        print()

        if report.correlation_report:
            cr = report.correlation_report
            print(f"  Correlações (RMSE Pearson) : {cr.pearson_rmse:.4f}")
            print(f"  Correlações (RMSE Spearman): {cr.spearman_rmse:.4f}")
            print(f"  {cr.assessment}")
            print()

        if report.tstr_report:
            t = report.tstr_report
            ratio_icon = "✅" if t.ratio >= 0.70 else "⚠️"
            print(f"  TSTR ({t.task_type}, target={t.target_column!r}):")
            print(f"    Train-on-Synthetic score : {t.tstr_score:.4f}")
            print(f"    Train-on-Real (baseline) : {t.trtr_score:.4f}")
            print(f"    {ratio_icon} Ratio TSTR/TRTR       : {t.ratio:.4f} (mín. aceitável: 0.70)")
            print(f"    {t.assessment}")
            print()

        if report.privacy_score:
            ps = report.privacy_score
            mem_icon = "✅" if ps.nndr > 0.5 else "⚠️"
            print(f"  {mem_icon} Privacy Score (anti-memorização):")
            print(f"    DCR médio  : {ps.mean_dcr:.4f} (Distance to Closest Record)")
            print(f"    NNDR       : {ps.nndr:.4f} (ideal > 0.5)")
            print(f"    Memorização: {ps.n_memorized} registros suspeitos")
            print(f"    {ps.assessment}")
            print()

        print("  Fidelidade por coluna:")
        for r in sorted(report.column_reports, key=lambda x: x.fidelity_score):
            icon = _fidelity_icon(r.fidelity_score)
            print(f"    {icon} {r.column:<25} {r.fidelity_score:.1%}  ({r.assessment})")

    def to_dataframe(self, report: FidelityReport) -> pd.DataFrame:
        """Retorna métricas por coluna como DataFrame."""
        rows = []
        for r in report.column_reports:
            rows.append({
                "coluna":          r.column,
                "tipo":            r.column_type,
                "fidelity_score":  r.fidelity_score,
                "ks_statistic":    r.ks_statistic,
                "ks_pvalue":       r.ks_pvalue,
                "psi":             r.psi,
                "jsd":             r.jsd,
                "mean_diff_pct":   r.mean_diff_pct,
                "avaliacao":       r.assessment,
            })
        return pd.DataFrame(rows).sort_values("fidelity_score")

    # ------------------------------------------------------------------
    # Métricas individuais
    # ------------------------------------------------------------------

    def _evaluate_column(
        self, col: str, real: pd.Series, synth: pd.Series
    ) -> ColumnFidelityReport:
        """Avalia uma coluna individualmente."""
        is_numeric = pd.api.types.is_numeric_dtype(real) and pd.api.types.is_numeric_dtype(synth)

        if is_numeric:
            return self._evaluate_numeric(col, real.dropna(), synth.dropna())
        else:
            return self._evaluate_categorical(col, real.dropna(), synth.dropna())

    def _evaluate_numeric(
        self, col: str, real: pd.Series, synth: pd.Series
    ) -> ColumnFidelityReport:
        """KS Test + PSI + Δ estatísticas para colunas numéricas."""
        ks_stat, ks_pvalue = stats.ks_2samp(real.values, synth.values)
        psi = _compute_psi(real, synth, self.psi_bins)

        real_mean = float(real.mean()) if len(real) else 0
        synth_mean = float(synth.mean()) if len(synth) else 0
        real_std = float(real.std()) if len(real) else 1
        synth_std = float(synth.std()) if len(synth) else 1

        mean_diff_pct = abs(real_mean - synth_mean) / max(abs(real_mean), 1e-9) * 100
        std_diff_pct  = abs(real_std - synth_std) / max(abs(real_std), 1e-9) * 100

        score = np.mean([
            1 - ks_stat,
            1 - min(psi, 1.0),
            1 - min(mean_diff_pct / 100, 1.0),
            1 - min(std_diff_pct / 100, 1.0),
        ])
        score = max(0.0, min(1.0, float(score)))
        assessment = _assess_score(score)

        return ColumnFidelityReport(
            column=col, column_type="numeric",
            ks_statistic=round(ks_stat, 4), ks_pvalue=round(ks_pvalue, 4),
            psi=round(psi, 4), mean_diff_pct=round(mean_diff_pct, 2),
            std_diff_pct=round(std_diff_pct, 2),
            fidelity_score=round(score, 4), assessment=assessment,
        )

    def _evaluate_categorical(
        self, col: str, real: pd.Series, synth: pd.Series
    ) -> ColumnFidelityReport:
        """JSD + sobreposição de top categorias para colunas categóricas."""
        jsd = _compute_jsd(real.astype(str), synth.astype(str))

        real_top = set(real.value_counts().head(10).index)
        synth_top = set(synth.value_counts().head(10).index)
        top_overlap = len(real_top & synth_top) / max(len(real_top), 1)

        score = max(0.0, min(1.0, (1 - jsd) * 0.7 + top_overlap * 0.3))
        assessment = _assess_score(score)

        return ColumnFidelityReport(
            column=col, column_type="categorical",
            jsd=round(jsd, 4), top_overlap=round(top_overlap, 4),
            fidelity_score=round(score, 4), assessment=assessment,
        )

    def _evaluate_correlations(
        self, real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]
    ) -> Optional[CorrelationFidelityReport]:
        """Compara matrizes de correlação Pearson e Spearman."""
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
        if len(numeric_cols) < 2:
            return None

        try:
            real_p  = real[numeric_cols].corr(method="pearson").values
            synth_p = synth[numeric_cols].corr(method="pearson").values
            pearson_rmse = float(np.sqrt(np.mean((real_p - synth_p) ** 2)))

            real_s  = real[numeric_cols].corr(method="spearman").values
            synth_s = synth[numeric_cols].corr(method="spearman").values
            spearman_rmse = float(np.sqrt(np.mean((real_s - synth_s) ** 2)))

            mad = float(np.mean(np.abs(real_p - synth_p)))

            if pearson_rmse < 0.10:
                assessment = "✅ Correlações bem preservadas (RMSE < 0.10)"
            elif pearson_rmse < 0.20:
                assessment = "⚠️ Correlações parcialmente preservadas (RMSE < 0.20)"
            else:
                assessment = "❌ Correlações significativamente alteradas (RMSE ≥ 0.20)"

            return CorrelationFidelityReport(
                pearson_rmse=round(pearson_rmse, 4),
                spearman_rmse=round(spearman_rmse, 4),
                mean_abs_diff=round(mad, 4),
                assessment=assessment,
            )
        except Exception as e:
            logger.warning("FidelityEvaluator: erro no cálculo de correlações: %s", e)
            return None

    def _evaluate_tstr(
        self, real: pd.DataFrame, synth: pd.DataFrame,
        target: str, cols: List[str],
    ) -> Optional[TSTRReport]:
        """Train-on-Synthetic / Test-on-Real."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, r2_score
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            logger.warning("TSTR requer scikit-learn: pip install scikit-learn")
            return None

        feature_cols = [c for c in cols if c != target and pd.api.types.is_numeric_dtype(real[c])]
        if len(feature_cols) < 1:
            return None

        real_clean = real[feature_cols + [target]].dropna()
        synth_clean = synth[feature_cols + [target]].dropna()
        if len(real_clean) < 50 or len(synth_clean) < 50:
            return None

        is_classification = not pd.api.types.is_numeric_dtype(real_clean[target])

        X_synth = synth_clean[feature_cols].values
        y_synth = synth_clean[target].values
        X_real, X_test, y_real, y_test = train_test_split(
            real_clean[feature_cols].values,
            real_clean[target].values,
            test_size=0.3, random_state=42,
        )

        if is_classification:
            le = LabelEncoder()
            y_synth = le.fit_transform(y_synth)
            y_test  = le.transform([v for v in y_test if v in le.classes_])
            X_test  = X_test[:len(y_test)]
            y_real_tr = le.transform([v for v in y_real if v in le.classes_])
            X_real_tr = X_real[:len(y_real_tr)]

            model_synth = GradientBoostingClassifier(n_estimators=50, random_state=42)
            model_real  = GradientBoostingClassifier(n_estimators=50, random_state=42)
            model_synth.fit(X_synth, y_synth)
            model_real.fit(X_real_tr, y_real_tr)
            tstr_score = roc_auc_score(y_test, model_synth.predict_proba(X_test)[:, 1], multi_class="ovr") if len(le.classes_) == 2 else model_synth.score(X_test, y_test)
            trtr_score = roc_auc_score(y_test, model_real.predict_proba(X_test)[:, 1], multi_class="ovr") if len(le.classes_) == 2 else model_real.score(X_test, y_test)
            task_type = "classification"
            model_name = "GradientBoosting"
        else:
            model_synth = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model_real  = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model_synth.fit(X_synth, y_synth)
            model_real.fit(X_real, y_real)
            tstr_score = float(r2_score(y_test, model_synth.predict(X_test)))
            trtr_score = float(r2_score(y_test, model_real.predict(X_test)))
            task_type = "regression"
            model_name = "GradientBoosting"

        ratio = tstr_score / max(abs(trtr_score), 1e-9)
        if ratio >= 0.90:
            assessment = "✅ Excelente — modelo sintético comparável ao real (≥ 90%)"
        elif ratio >= 0.70:
            assessment = "⚠️  Aceitável — modelo sintético adequado para exploração (≥ 70%)"
        else:
            assessment = "❌ Insuficiente — modelo sintético muito inferior ao real (< 70%)"

        return TSTRReport(
            task_type=task_type, target_column=target,
            tstr_score=round(tstr_score, 4), trtr_score=round(trtr_score, 4),
            ratio=round(ratio, 4), assessment=assessment, model_used=model_name,
        )

    def _evaluate_privacy(
        self, real: pd.DataFrame, synth: pd.DataFrame, cols: List[str]
    ) -> Optional[PrivacyScoreReport]:
        """DCR e NNDR para detectar memorização."""
        try:
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(real[c])]
            if len(numeric_cols) < 1:
                return None

            real_n  = real[numeric_cols].dropna().sample(min(self.privacy_sample, len(real)), random_state=42).values
            synth_n = synth[numeric_cols].dropna().sample(min(self.privacy_sample, len(synth)), random_state=42).values

            # Normaliza [0,1]
            mins = real_n.min(axis=0)
            maxs = real_n.max(axis=0)
            denom = np.where(maxs - mins > 0, maxs - mins, 1.0)
            real_norm  = (real_n  - mins) / denom
            synth_norm = (synth_n - mins) / denom

            # DCR: distância ao registro real mais próximo
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(synth_norm, real_norm, metric="euclidean")

            dcr = dist_matrix.min(axis=1)
            mean_dcr = float(dcr.mean())
            min_dcr  = float(dcr.min())
            n_memorized = int((dcr < 1e-4).sum())

            # NNDR: ratio entre 1° e 2° vizinho mais próximo
            sorted_dists = np.sort(dist_matrix, axis=1)
            nndr_vals = sorted_dists[:, 0] / np.maximum(sorted_dists[:, 1], 1e-9)
            nndr = float(np.median(nndr_vals))

            if nndr > 0.5 and n_memorized == 0:
                assessment = "✅ Baixo risco de memorização (NNDR > 0.5)"
            elif nndr > 0.3:
                assessment = "⚠️  Risco moderado de memorização — verificar outliers"
            else:
                assessment = "❌ Alto risco — possível memorização de registros reais"

            return PrivacyScoreReport(
                mean_dcr=round(mean_dcr, 6), min_dcr=round(min_dcr, 6),
                nndr=round(nndr, 4), n_memorized=n_memorized,
                assessment=assessment,
            )
        except Exception as e:
            logger.warning("FidelityEvaluator: erro no privacy score: %s", e)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_psi(real: pd.Series, synth: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre duas séries numéricas."""
    try:
        real_clean  = real.dropna().values
        synth_clean = synth.dropna().values
        if len(real_clean) < bins or len(synth_clean) < bins:
            return 0.0
        breakpoints = np.percentile(real_clean, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        real_counts  = np.histogram(real_clean, bins=breakpoints)[0]
        synth_counts = np.histogram(synth_clean, bins=breakpoints)[0]
        real_pct  = real_counts  / max(real_counts.sum(), 1)
        synth_pct = synth_counts / max(synth_counts.sum(), 1)
        real_pct  = np.where(real_pct  == 0, 0.001, real_pct)
        synth_pct = np.where(synth_pct == 0, 0.001, synth_pct)
        psi = float(np.sum((synth_pct - real_pct) * np.log(synth_pct / real_pct)))
        return round(abs(psi), 6)
    except Exception:
        return 0.0


def _compute_jsd(real: pd.Series, synth: pd.Series) -> float:
    """Jensen-Shannon Divergence entre duas séries categóricas."""
    try:
        all_cats = set(real.unique()) | set(synth.unique())
        p = np.array([real.tolist().count(c) / max(len(real), 1) for c in all_cats])
        q = np.array([synth.tolist().count(c) / max(len(synth), 1) for c in all_cats])
        p = np.where(p == 0, 1e-9, p)
        q = np.where(q == 0, 1e-9, q)
        m = (p + q) / 2
        jsd = float((np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2)
        return round(min(abs(jsd), 1.0), 6)
    except Exception:
        return 0.0


def _assess_score(score: float) -> str:
    if score >= 0.85:  return "excelente"
    elif score >= 0.70: return "bom"
    elif score >= 0.55: return "moderado"
    else:               return "baixo"


def _assess_overall(fidelity: float, tstr: Optional[Any]) -> str:
    if fidelity >= 0.80:
        base = "Alta fidelidade estatística"
    elif fidelity >= 0.65:
        base = "Fidelidade moderada"
    else:
        base = "Baixa fidelidade — revisar treinamento"
    if tstr:
        tstr_part = f" | TSTR ratio={tstr.ratio:.2f}"
        return base + tstr_part
    return base


def _fidelity_icon(score: float) -> str:
    if score >= 0.85: return "✅"
    elif score >= 0.70: return "🟡"
    elif score >= 0.55: return "🟠"
    else:               return "❌"
