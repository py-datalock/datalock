"""
mockers/category_mocker.py
===========================
Sorteio Categórico por Amostragem Probabilística Empírica.

Filosofia de Design
-------------------
Colunas categóricas (cargo, departamento, estado civil, UF...) não devem ser
simplesmente embaralhadas — um shuffle simples desfaz correlações mas mantém
os valores reais visíveis no código. O objetivo aqui é gerar novos valores
sorteados com base nas frequências originais, garantindo:

  1. As proporções de categoria são mantidas (ex: 60% "SP", 20% "RJ")
  2. Os valores individuais são desassociados dos demais campos da linha
  3. O código do desenvolvedor não vê nenhum valor real de indivíduo

Permutação Condicionada por Blocos
------------------------------------
O CategoryMocker suporta também a permutação condicionada: embaralha valores
de uma coluna dentro de grupos definidos por outra. Exemplo: salários são
embaralhados apenas dentro de cada faixa de cargo, evitando combinações
absurdas como salário de diretor em linha de operador.

Uso:
    mocker = CategoryMocker()
    df["uf"] = mocker.mock(df["uf"])

    # Permutação condicionada por grupo:
    df["salario"] = mocker.mock_within_group(df, "salario", group_by="cargo")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CategoryMocker:
    """
    Gera valores categóricos fictícios preservando as proporções da coluna original.

    Parâmetros:
        random_state: Semente para reprodutibilidade.
        preserve_nulls: Se True, mantém NaN nas mesmas posições originais.
        min_category_freq: Frequência mínima para incluir uma categoria no sorteio.
            Categorias abaixo deste threshold são suprimidas para evitar que
            casos raros (ex: doenças raras, grupos minoritários) sejam re-identificáveis
            pelo padrão de frequência. Padrão: 0.01 (1%).
    """

    def __init__(
        self,
        random_state: int = 42,
        preserve_nulls: bool = True,
        min_category_freq: float = 0.01,
    ):
        self.random_state = random_state
        self.preserve_nulls = preserve_nulls
        self.min_category_freq = min_category_freq

    def mock(
        self,
        series: pd.Series,
        value_freq: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Sorteia novos valores para a coluna baseado nas frequências reais.

        Args:
            series: Série original.
            value_freq: Dicionário {valor: frequência_relativa} vindo do
                ColumnReport. Se None, calcula diretamente da série.

        Returns:
            Series com valores sorteados — sem associação com os originais.
        """
        rng = np.random.default_rng(self.random_state)
        null_mask = series.isna()
        n = len(series)

        # Calcula ou usa frequências fornecidas
        if value_freq is not None and len(value_freq) > 0:
            categories = list(value_freq.keys())
            weights = np.array(list(value_freq.values()), dtype=float)
        else:
            vc = series.dropna().astype(str).value_counts(normalize=True)
            if len(vc) == 0:
                logger.warning(
                    "CategoryMocker: '%s' vazia — retornando sem alteração.", series.name
                )
                return series.copy()
            categories = vc.index.tolist()
            weights = vc.values

        # Filtra categorias raras (abaixo de min_category_freq).
        # Categorias com frequência muito baixa revelam casos raros quando preservadas —
        # ex: uma doença com 0.001% de prevalência indica os mesmos indivíduos.
        if self.min_category_freq > 0:
            mask = weights >= self.min_category_freq
            if mask.any():
                suppressed = len(categories) - mask.sum()
                if suppressed > 0:
                    logger.debug(
                        "CategoryMocker | col=%s | %d categoria(s) rara(s) suprimida(s) "
                        "(freq < %.1f%%)",
                        series.name, suppressed, self.min_category_freq * 100,
                    )
                categories = [c for c, m in zip(categories, mask) if m]
                weights = weights[mask]

        # Normaliza pesos (sum=1) caso venham truncados (top_n) ou após supressão
        weights = weights / weights.sum()

        # Sorteia n valores com as proporções originais
        sampled = rng.choice(categories, size=n, replace=True, p=weights)
        result = pd.Series(sampled, index=series.index, name=series.name, dtype=object)

        if self.preserve_nulls:
            result[null_mask] = np.nan

        logger.debug(
            "CategoryMocker | col=%s | categorias=%d | n=%d",
            series.name, len(categories), n,
        )
        return result

    def mock_within_group(
        self,
        df: pd.DataFrame,
        column: str,
        group_by: Union[str, List[str]],
    ) -> pd.Series:
        """
        Permutação Condicionada por Blocos.

        Embaralha os valores de `column` dentro de cada grupo definido por
        `group_by`. Preserva distribuição dentro do grupo mas desfaz a
        associação linha-a-linha — um analista não consegue mais ligar um valor
        específico a um indivíduo específico.

        Exemplo:
            df["salario"] = mocker.mock_within_group(df, "salario", "cargo")
            # Salários são permutados apenas entre colegas do mesmo cargo.

        Args:
            df: DataFrame completo.
            column: Coluna a ser permutada.
            group_by: Coluna(s) que define os blocos.

        Returns:
            Series com valores permutados dentro dos grupos.
        """
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        missing = [c for c in [column] + group_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"CategoryMocker.mock_within_group: colunas não encontradas: {missing}. "
                f"Disponíveis: {list(df.columns)}"
            )

        rng = np.random.default_rng(self.random_state)
        result = df[column].copy()

        for _, group in df.groupby(group_by, dropna=False):
            idx = group.index
            permuted = rng.permutation(group[column].values)
            result.loc[idx] = permuted

        logger.debug(
            "CategoryMocker.mock_within_group | col=%s group_by=%s",
            column, group_by,
        )
        return result

    def __repr__(self) -> str:
        return f"CategoryMocker(random_state={self.random_state})"
