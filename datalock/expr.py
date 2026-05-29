"""
datalock/expr.py
=============
dd.col() — expressões de coluna agnósticas ao engine.

Com Polars instalado (obrigatório desde v1.0.4), dd.col() É pl.col() —
acesso completo a todos os métodos nativos: rank, over, str.*, dt.*, etc.

Este módulo expõe apenas as classes auxiliares que o pipeline interno usa:
  - Col       → fachada para pl.col() com operadores Python naturais
  - ColExpr   → expressão aritmética (dd.col('x') * 0.27)
  - Expr      → expressão de filtro (dd.col('x') > 5)
  - _WhenExpr → CASE WHEN (movida para analytics.py, reexportada aqui por compat)

Nota: desde v1.0.4 dd.col = pl.col diretamente no __init__.py.
      Este módulo é mantido para o fallback de pandas e para when().
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import polars as pl


# ---------------------------------------------------------------------------
# Col — referência de coluna com operadores expressivos
# ---------------------------------------------------------------------------

class Col:
    """
    Referência a uma coluna com operadores expressivos.

    Criada via dd.col('nome_da_coluna') quando Polars não está disponível.
    Com Polars instalado, dd.col = pl.col diretamente (mais poderoso).

    Exemplos:
        dd.where(df, dd.col('uf') == 'SP')
        dd.where(df, dd.col('renda').between(5000, 15000))
        dd.where(df, dd.col('uf').is_in(['SP','RJ','MG']))
        dd.where(df, ~dd.col('inadimplente'))
        dd.where(df, (dd.col('uf') == 'SP') & (dd.col('renda') > 5000))
    """

    def __init__(self, name: str) -> None:
        self._name = name

    # ── Comparação ──────────────────────────────────────────────────────────

    def __eq__(self, other: Any) -> "Expr":
        return Expr(self._name, "is_null", None) if other is None else Expr(self._name, "eq", other)

    def __ne__(self, other: Any) -> "Expr":
        return Expr(self._name, "is_not_null", None) if other is None else Expr(self._name, "ne", other)

    def __gt__(self, other: Any) -> "Expr":  return Expr(self._name, "gt", other)
    def __ge__(self, other: Any) -> "Expr":  return Expr(self._name, "ge", other)
    def __lt__(self, other: Any) -> "Expr":  return Expr(self._name, "lt", other)
    def __le__(self, other: Any) -> "Expr":  return Expr(self._name, "le", other)

    # ── Membership ──────────────────────────────────────────────────────────

    def is_in(self, values: Iterable) -> "Expr":
        return Expr(self._name, "is_in", list(values))

    def is_null(self) -> "Expr":
        return Expr(self._name, "is_null", None)

    def is_not_null(self) -> "Expr":
        return Expr(self._name, "is_not_null", None)

    # ── Range ───────────────────────────────────────────────────────────────

    def between(self, low: Any, high: Any) -> "Expr":
        return Expr(self._name, "between", (low, high))

    is_between = between  # alias

    # ── String ──────────────────────────────────────────────────────────────

    @property
    def str(self) -> "_StrNamespace":
        return _StrNamespace(self._name)

    def contains(self, pattern: str) -> "Expr":
        return Expr(self._name, "str_contains", pattern)

    def starts_with(self, prefix: str) -> "Expr":
        return Expr(self._name, "str_starts_with", prefix)

    def ends_with(self, suffix: str) -> "Expr":
        return Expr(self._name, "str_ends_with", suffix)

    # ── Datetime ────────────────────────────────────────────────────────────

    @property
    def dt(self) -> "_DtNamespace":
        return _DtNamespace(self._name)

    # ── Lógico ──────────────────────────────────────────────────────────────

    def __invert__(self) -> "Expr":
        return Expr(None, "not", None, _left=Expr(self._name, "eq", True))

    # ── Aritmético (retorna ColExpr para uso em add_column) ─────────────────

    def __mul__(self, other) -> "ColExpr":   return ColExpr(self._name, "*",  other)
    def __rmul__(self, other) -> "ColExpr":  return ColExpr(self._name, "*",  other)
    def __add__(self, other) -> "ColExpr":   return ColExpr(self._name, "+",  other)
    def __radd__(self, other) -> "ColExpr":  return ColExpr(self._name, "+",  other)
    def __sub__(self, other) -> "ColExpr":   return ColExpr(self._name, "-",  other)
    def __rsub__(self, other) -> "ColExpr":  return ColExpr(self._name, "-",  other)
    def __truediv__(self, other) -> "ColExpr":  return ColExpr(self._name, "/",  other)
    def __floordiv__(self, other) -> "ColExpr": return ColExpr(self._name, "//", other)
    def __mod__(self, other) -> "ColExpr":   return ColExpr(self._name, "%",  other)
    def __pow__(self, other) -> "ColExpr":   return ColExpr(self._name, "**", other)
    def __neg__(self) -> "ColExpr":          return ColExpr(self._name, "neg", None)
    def __abs__(self) -> "ColExpr":          return ColExpr(self._name, "abs", None)

    def alias(self, name: str) -> "ColExpr":
        ce = ColExpr(self._name, "identity", None)
        ce._alias = name
        return ce

    def __repr__(self) -> str:
        return f"dd.col({self._name!r})"


# ---------------------------------------------------------------------------
# ColExpr — expressão aritmética sobre coluna
# ---------------------------------------------------------------------------

class ColExpr:
    """
    Expressão aritmética. Retornada por dd.col('x') * 0.27 etc.
    Usada em dd.add_column() e convertida para pl.Expr internamente.
    """

    def __init__(self, col_name: str, op: str, arg: Any) -> None:
        self._col   = col_name
        self._op    = op
        self._arg   = arg
        self._alias: Optional[str] = None

    def alias(self, name: str) -> "ColExpr":
        self._alias = name
        return self

    def to_polars(self) -> pl.Expr:
        c = pl.col(self._col)

        def _resolve(v: Any) -> Any:
            if isinstance(v, ColExpr): return v.to_polars()
            if isinstance(v, Col):     return pl.col(v._name)
            if isinstance(v, Expr):    return v.to_polars()
            return v

        _OP = {
            "*":        lambda: c * _resolve(self._arg),
            "+":        lambda: c + _resolve(self._arg),
            "-":        lambda: c - _resolve(self._arg),
            "/":        lambda: c / _resolve(self._arg),
            "//":       lambda: c // _resolve(self._arg),
            "%":        lambda: c % _resolve(self._arg),
            "**":       lambda: c ** _resolve(self._arg),
            "neg":      lambda: -c,
            "abs":      lambda: c.abs(),
            "identity": lambda: c,
        }
        result = _OP[self._op]()
        return result.alias(self._alias) if self._alias else result

    def to_pandas_series(self, df: Any) -> Any:
        s = df[self._col]

        def _r(v: Any) -> Any:
            if isinstance(v, ColExpr): return v.to_pandas_series(df)
            if isinstance(v, Col):     return df[v._name]
            return v

        _OP = {
            "*":   lambda: s * _r(self._arg),
            "+":   lambda: s + _r(self._arg),
            "-":   lambda: s - _r(self._arg),
            "/":   lambda: s / _r(self._arg),
            "//":  lambda: s // _r(self._arg),
            "%":   lambda: s % _r(self._arg),
            "**":  lambda: s ** _r(self._arg),
            "neg": lambda: -s,
            "abs": lambda: s.abs(),
            "identity": lambda: s,
        }
        result = _OP[self._op]()
        if self._alias:
            result.name = self._alias
        return result

    # Permite encadear: (dd.col('x') * 0.27) + 100
    def __add__(self, other):  return ColExpr(self._col, "+", other)
    def __sub__(self, other):  return ColExpr(self._col, "-", other)
    def __mul__(self, other):  return ColExpr(self._col, "*", other)
    def __truediv__(self, other): return ColExpr(self._col, "/", other)

    def __repr__(self) -> str:
        return f"ColExpr(dd.col({self._col!r}) {self._op} {self._arg!r})"


# ---------------------------------------------------------------------------
# Namespaces auxiliares
# ---------------------------------------------------------------------------

class _StrNamespace:
    """dd.col('x').str.contains(...)"""
    def __init__(self, col: str) -> None:
        self._col = col

    def contains(self, p: str) -> "Expr":
        return Expr(self._col, "str_contains", p)

    def starts_with(self, p: str) -> "Expr":
        return Expr(self._col, "str_starts_with", p)

    def ends_with(self, p: str) -> "Expr":
        return Expr(self._col, "str_ends_with", p)

    def to_lowercase(self) -> "TransformExpr":
        return TransformExpr(self._col, "str_lower")

    def to_uppercase(self) -> "TransformExpr":
        return TransformExpr(self._col, "str_upper")

    def strip(self) -> "TransformExpr":
        return TransformExpr(self._col, "str_strip")

    def slice(self, offset: int, length: int) -> "TransformExpr":
        return TransformExpr(self._col, "str_slice", (offset, length))

    def replace(self, pattern: str, value: str) -> "TransformExpr":
        return TransformExpr(self._col, "str_replace", (pattern, value))


class _DtNamespace:
    """dd.col('data').dt.year > 2000"""
    def __init__(self, col: str) -> None:
        self._col = col

    @property
    def year(self) -> "_ColExtracted":
        return _ColExtracted(self._col, "year")

    @property
    def month(self) -> "_ColExtracted":
        return _ColExtracted(self._col, "month")

    @property
    def day(self) -> "_ColExtracted":
        return _ColExtracted(self._col, "day")


class _ColExtracted(Col):
    """Coluna derivada de extração de atributo (ex: dt.year)."""
    def __init__(self, col: str, attr: str) -> None:
        super().__init__(col)
        self._attr = attr

    def _make_expr(self, op: str, other: Any) -> "Expr":
        return Expr(self._name, f"dt_{op}", (self._attr, other))

    def __eq__(self, other: Any) -> "Expr":  return self._make_expr("eq", other)
    def __gt__(self, other: Any) -> "Expr":  return self._make_expr("gt", other)
    def __ge__(self, other: Any) -> "Expr":  return self._make_expr("ge", other)
    def __lt__(self, other: Any) -> "Expr":  return self._make_expr("lt", other)
    def __le__(self, other: Any) -> "Expr":  return self._make_expr("le", other)


class TransformExpr:
    """Expressão de transformação (não filtro). Usada em dd.add_column()."""
    def __init__(self, col: str, op: str, arg: Any = None) -> None:
        self._col = col
        self._op  = op
        self._arg = arg
        self._alias: Optional[str] = None

    def to_polars(self) -> pl.Expr:
        c = pl.col(self._col)
        _ops = {
            "str_lower":   lambda: c.str.to_lowercase(),
            "str_upper":   lambda: c.str.to_uppercase(),
            "str_strip":   lambda: c.str.strip_chars(),
            "str_slice":   lambda: c.str.slice(*self._arg),
            "str_replace": lambda: c.str.replace(*self._arg),
        }
        result = _ops[self._op]()
        return result.alias(self._alias) if self._alias else result

    def alias(self, name: str) -> "TransformExpr":
        self._alias = name
        return self


# ---------------------------------------------------------------------------
# Expr — expressão de filtro lazy
# ---------------------------------------------------------------------------

class Expr:
    """
    Expressão de filtro lazy — avaliada quando passada para dd.where() etc.

    Suporta &, |, ~ para combinação lógica.

    Exemplos:
        e = dd.col('uf') == 'SP'
        e = dd.col('uf').is_in(['SP','RJ']) & (dd.col('renda') > 5000)
        e = ~dd.col('inadimplente')
    """

    def __init__(
        self,
        col: Optional[str],
        op: str,
        arg: Any,
        _left: Optional["Expr"] = None,
        _right: Optional["Expr"] = None,
    ) -> None:
        self._col   = col
        self._op    = op
        self._arg   = arg
        self._left  = _left
        self._right = _right

    # ── Combinação ──────────────────────────────────────────────────────────

    def __and__(self, other: "Expr") -> "Expr":
        return Expr(None, "and", None, _left=self, _right=other)

    def __or__(self, other: "Expr") -> "Expr":
        return Expr(None, "or", None, _left=self, _right=other)

    def __invert__(self) -> "Expr":
        return Expr(None, "not", None, _left=self)

    # ── Conversão para pl.Expr ──────────────────────────────────────────────

    def to_polars(self) -> pl.Expr:
        if self._op == "and":
            return self._left.to_polars() & self._right.to_polars()
        if self._op == "or":
            return self._left.to_polars() | self._right.to_polars()
        if self._op == "not":
            return ~self._left.to_polars()

        c = pl.col(self._col)
        _map = {
            "eq":              lambda: c == self._arg,
            "ne":              lambda: c != self._arg,
            "gt":              lambda: c > self._arg,
            "ge":              lambda: c >= self._arg,
            "lt":              lambda: c < self._arg,
            "le":              lambda: c <= self._arg,
            "is_in":           lambda: c.is_in(self._arg),
            "is_null":         lambda: c.is_null(),
            "is_not_null":     lambda: c.is_not_null(),
            "between":         lambda: c.is_between(self._arg[0], self._arg[1]),
            "str_contains":    lambda: c.cast(pl.String).str.contains(self._arg),
            "str_starts_with": lambda: c.cast(pl.String).str.starts_with(self._arg),
            "str_ends_with":   lambda: c.cast(pl.String).str.ends_with(self._arg),
            "dt_eq":           lambda: getattr(c.dt, self._arg[0]) == self._arg[1],
            "dt_gt":           lambda: getattr(c.dt, self._arg[0]) > self._arg[1],
            "dt_ge":           lambda: getattr(c.dt, self._arg[0]) >= self._arg[1],
            "dt_lt":           lambda: getattr(c.dt, self._arg[0]) < self._arg[1],
            "dt_le":           lambda: getattr(c.dt, self._arg[0]) <= self._arg[1],
        }
        fn = _map.get(self._op)
        if fn is None:
            raise ValueError(f"Operação desconhecida: {self._op!r}")
        return fn()

    # ── Conversão para máscara pandas ───────────────────────────────────────

    def to_pandas_mask(self, df: Any) -> Any:
        import pandas as _pd
        if self._op == "and":
            return self._left.to_pandas_mask(df) & self._right.to_pandas_mask(df)
        if self._op == "or":
            return self._left.to_pandas_mask(df) | self._right.to_pandas_mask(df)
        if self._op == "not":
            return ~self._left.to_pandas_mask(df)

        col_s = df[self._col]
        _map = {
            "eq":              lambda: col_s == self._arg,
            "ne":              lambda: col_s != self._arg,
            "gt":              lambda: col_s > self._arg,
            "ge":              lambda: col_s >= self._arg,
            "lt":              lambda: col_s < self._arg,
            "le":              lambda: col_s <= self._arg,
            "is_in":           lambda: col_s.isin(self._arg),
            "is_null":         lambda: col_s.isna(),
            "is_not_null":     lambda: col_s.notna(),
            "between":         lambda: col_s.between(self._arg[0], self._arg[1]),
            "str_contains":    lambda: col_s.astype(str).str.contains(self._arg, na=False),
            "str_starts_with": lambda: col_s.astype(str).str.startswith(self._arg),
            "str_ends_with":   lambda: col_s.astype(str).str.endswith(self._arg),
        }
        fn = _map.get(self._op)
        if fn is None:
            raise ValueError(f"Operação desconhecida: {self._op!r}")
        return fn()

    def __repr__(self) -> str:
        if self._op in ("and", "or"):
            return f"({self._left!r} {self._op.upper()} {self._right!r})"
        if self._op == "not":
            return f"NOT ({self._left!r})"
        return f"dd.col({self._col!r}).{self._op}({self._arg!r})"


# ---------------------------------------------------------------------------
# Função pública
# ---------------------------------------------------------------------------

def col(name: str) -> Col:
    """
    Cria referência a uma coluna com operadores expressivos.

    Nota: quando Polars está instalado (obrigatório desde v1.0.4),
    dd.col é diretamente pl.col — com acesso a todos os métodos nativos.
    Esta função é o fallback para uso programático via datalock.expr.col().
    """
    return Col(name)
