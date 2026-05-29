"""
utils/frames.py
===============
Conversores pd.DataFrame ↔ pl.DataFrame compartilhados.

Centraliza a lógica de conversão que antes estava duplicada em
analytics.py e polars_adapter.py, eliminando divergência futura.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pandas as pd

import polars as pl
_POLARS_AVAILABLE = True  # Polars obrigatório desde v1.0.4

if TYPE_CHECKING:
    import polars as pl

AnyFrame = Union[pd.DataFrame, "pl.DataFrame"]


def to_polars(df: AnyFrame) -> "pl.DataFrame":
    """Converte pd.DataFrame → pl.DataFrame. Passthrough se já for Polars."""
    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def to_pandas(df: AnyFrame) -> pd.DataFrame:
    """Converte pl.DataFrame → pd.DataFrame. Passthrough se já for pandas."""
    if isinstance(df, pd.DataFrame):
        return df
    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Esperado pd.DataFrame ou pl.DataFrame, recebido {type(df).__name__}")


def ensure_pandas(df: AnyFrame) -> pd.DataFrame:
    """Garante pd.DataFrame — alias semântico de to_pandas()."""
    return to_pandas(df)


def is_polars(df: object) -> bool:
    return _POLARS_AVAILABLE and isinstance(df, pl.DataFrame)


def as_list(x: Union[str, list]) -> list:
    """Normaliza str | list → list."""
    return [x] if isinstance(x, str) else list(x)
