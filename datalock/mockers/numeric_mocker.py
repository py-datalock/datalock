"""
mockers/numeric_mocker.py
==========================
Simulação Paramétrica Numérica — gera dados de teste dentro dos limites reais.

Filosofia de Design
-------------------
O objetivo NÃO é replicar a distribuição estatística exata (isso seria síntese
de dados completa). O objetivo é gerar valores plausíveis e sintaticamente
corretos que:

  1. Não quebrem o código do desenvolvedor (tipo correto, limites respeitados)
  2. Não sejam os valores reais (privacidade)
  3. Sejam rápidos de gerar (sem overhead de modelagem)

Estratégia: lê min e max reais da coluna e gera valores uniformes ou com
perturbação leve nesse intervalo. Para colunas com distribuição assimétrica
(ex: renda, salário), usa perturbação log-normal que evita valores negativos.

Uso:
    mocker = NumericMocker()
    df["renda"] = mocker.mock(df["renda"])

    # Com semente para reprodutibilidade em testes:
    mocker = NumericMocker(random_state=42)
    df["idade"] = mocker.mock(df["idade"], as_int=True)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Limiar de assimetria (skewness) para detecção automática de distribuição log-normal.
# Distribuições com skew > 1.5 têm cauda direita longa (ex: renda, salário, patrimônio).
_LOGNORMAL_SKEW_THRESHOLD = 1.5


class NumericMocker:
    """
    Gerador de valores numéricos de teste baseado nos limites da coluna original.

    Parâmetros:
        strategy: "uniform" (padrão) ou "perturb".
            - "uniform": valores aleatórios uniformes em [min, max].
              Mantém os limites mas não a forma da distribuição.
            - "perturb": adiciona ruído leve preservando melhor a forma.
              Para distribuições simétricas usa ruído gaussiano.
              Para distribuições assimétricas (skew > 1.5) usa perturbação
              log-normal — evita valores negativos em colunas como renda.
        distribution: "auto" (padrão), "gaussian", "lognormal" ou "uniform".
            - "auto": detecta automaticamente com base no skewness da coluna.
            - "gaussian": força perturbação gaussiana (pode gerar negativos).
            - "lognormal": força perturbação em escala logarítmica.
            - "uniform": força distribuição uniforme (ignora forma original).
        noise_fraction: Usado só no modo "perturb". Fração do desvio-padrão
            usada como escala do ruído. 0.05 = 5% do std.
        random_state: Semente para reprodutibilidade.
        non_negative: Se True, faz floor em 0 para colunas sem valores negativos.
            Evita gerar renda negativa, idade negativa, etc.
    """

    def __init__(
        self,
        strategy: str = "perturb",
        distribution: str = "auto",
        noise_fraction: float = 0.05,
        random_state: int = 42,
        non_negative: bool = True,
    ):
        if strategy not in ("uniform", "perturb"):
            raise ValueError("strategy deve ser 'uniform' ou 'perturb'.")
        if distribution not in ("auto", "gaussian", "lognormal", "uniform"):
            raise ValueError("distribution deve ser 'auto', 'gaussian', 'lognormal' ou 'uniform'.")
        # strategy='uniform' gera valores aleatórios em [min, max] — destrói correlações.
        # 'perturb' (padrão) preserva melhor a forma da distribuição.
        self.strategy = strategy
        self.distribution = distribution
        self.noise_fraction = noise_fraction
        self.random_state = random_state
        self.non_negative = non_negative

    def _perturb_lognormal(
        self,
        rng: np.random.Generator,
        v_min: float,
        v_max: float,
        n: int,
        series: pd.Series,
    ) -> np.ndarray:
        """
        Perturbação log-normal para colunas assimétricas (renda, salário, etc.).

        Converte para escala log, adiciona ruído gaussiano nessa escala,
        converte de volta. Hard clamp em [v_min, v_max] garante que 100%
        dos valores gerados ficam no range original.
        """
        filled = series.fillna(series.median()).values.astype(float)
        # Shift para escala positiva antes do log (caso v_min <= 0)
        shift = max(0.0, 1.0 - v_min)
        shifted = filled + shift
        shifted = np.where(shifted <= 0, 1e-10, shifted)

        log_vals = np.log(shifted)
        log_std = float(np.std(log_vals))
        scale = (log_std * self.noise_fraction) if log_std > 0 else (v_max - v_min) * 0.01

        noise = rng.normal(loc=0, scale=scale, size=n)
        log_noisy = log_vals + noise
        values = np.exp(log_noisy) - shift

        # Hard clamp — garante que 100% dos valores ficam no range original
        values = np.clip(values, v_min, v_max)
        return values

    def mock(
        self,
        series: pd.Series,
        as_int: bool = False,
        col_min: Optional[float] = None,
        col_max: Optional[float] = None,
    ) -> pd.Series:
        """
        Gera versão mockada de uma Series numérica.

        Args:
            series: Série original.
            as_int: Se True, arredonda para inteiro (útil para idade, contagens).
            col_min: Override do mínimo. Se None, usa o mínimo real da série.
            col_max: Override do máximo. Se None, usa o máximo real da série.

        Returns:
            Series com valores mockados. NaN preservados nas mesmas posições.
        """
        rng = np.random.default_rng(self.random_state)
        clean = series.dropna()

        if len(clean) == 0:
            logger.warning("NumericMocker: '%s' vazia — retornando sem alteração.", series.name)
            return series.copy()

        v_min = col_min if col_min is not None else float(clean.min())
        v_max = col_max if col_max is not None else float(clean.max())

        if v_min == v_max:
            logger.warning(
                "NumericMocker: '%s' é constante (%.4f) — variação mínima aplicada.",
                series.name, v_min,
            )
            v_min -= 1.0
            v_max += 1.0

        n = len(series)

        if self.strategy == "uniform" or self.distribution == "uniform":
            values = rng.uniform(low=v_min, high=v_max, size=n)
        else:
            # "perturb" — detecta ou usa distribuição configurada
            effective_dist = self.distribution
            if effective_dist == "auto":
                try:
                    skewness = float(clean.astype(float).skew())
                except Exception:
                    skewness = 0.0
                effective_dist = (
                    "lognormal" if skewness > _LOGNORMAL_SKEW_THRESHOLD else "gaussian"
                )
                logger.debug(
                    "NumericMocker: '%s' skew=%.2f → distribuição=%s",
                    series.name, skewness, effective_dist,
                )

            if effective_dist == "lognormal":
                values = self._perturb_lognormal(rng, v_min, v_max, n, series)
            else:
                # Gaussian
                filled = series.fillna(series.median()).values.astype(float)
                std = float(clean.std())
                scale = std * self.noise_fraction if std > 0 else (v_max - v_min) * 0.01
                noise = rng.normal(loc=0, scale=scale, size=n)
                values = np.clip(filled + noise, v_min, v_max)

        # Restrição não-negativa
        if self.non_negative and v_min >= 0:
            values = np.clip(values, 0, None)

        result = pd.Series(values, index=series.index, name=series.name)
        result[series.isna()] = np.nan

        if as_int or pd.api.types.is_integer_dtype(series):
            result = result.round(0).astype("Int64")

        logger.debug(
            "NumericMocker | col=%s strategy=%s distribution=%s range=[%.2f, %.2f]",
            series.name, self.strategy, self.distribution, v_min, v_max,
        )
        return result

    def mock_from_report(self, series: pd.Series, col_min: float, col_max: float) -> pd.Series:
        """
        Versão conveniente que aceita os limites diretamente do ColumnReport.
        """
        as_int = pd.api.types.is_integer_dtype(series)
        return self.mock(series, as_int=as_int, col_min=col_min, col_max=col_max)

    def __repr__(self) -> str:
        return (
            f"NumericMocker(strategy='{self.strategy}', "
            f"distribution='{self.distribution}', "
            f"noise_fraction={self.noise_fraction}, "
            f"non_negative={self.non_negative})"
        )
