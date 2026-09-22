"""
frame_proxy.py
==============
dd.frame(df) — acesso a QUALQUER método do Polars (DataFrame ou LazyFrame)
sem precisar `import polars as pl` no código do usuário.

Motivação
---------
datalock já embrulha manualmente ~30 operações comuns (dd.where, dd.sort,
dd.groupby, dd.pivot, dd.melt, dd.explode, dd.shift/lead/lag, dd.describe,
dd.value_counts, dd.corr, dd.nunique, dd.count_nulls, dd.sql, ...). Isso
cobre o caso comum, mas o Polars tem 200+ métodos — reescrever um wrapper
para cada um não escala e sempre vai ficar um passo atrás da API do Polars.

FrameProxy resolve isso de forma genérica: qualquer método que exista em
pl.DataFrame/pl.LazyFrame passa a existir em dd.frame(df), com o mesmo
nome e a mesma assinatura do Polars — incluindo métodos que ainda não
existem hoje e vierem a existir em versões futuras do Polars. O usuário
nunca precisa saber se um método já foi "portado" pelo datalock ou não.

Uso básico
----------
    import datalock as dd

    resultado = (
        dd.frame(df)                              # aceita pd.DataFrame ou pl.DataFrame
          .filter(dd.col("uf") == "SP")            # qualquer método polars
          .with_columns((dd.col("renda") * 1.1).alias("renda_reajustada"))
          .join(outro_df, on="cpf", how="left")    # outro_df pode ser DataFrame comum
          .sort("renda_reajustada", descending=True)
          .head(10)
          .native()                                # devolve no MESMO tipo do input original
    )

Regras de conversão automática
-------------------------------
  - Se o método retorna pl.DataFrame ou pl.LazyFrame → embrulhado de novo em
    FrameProxy (permite encadear .filter().select().sort()... sem interrupção).
  - Qualquer outro tipo de retorno (int, dict, pl.Series, string, etc.) →
    devolvido como está, sem embrulhar.
  - Argumentos de métodos (ex.: `.join(outro, ...)`) que sejam pd.DataFrame
    ou outro FrameProxy são convertidos para pl.DataFrame/pl.LazyFrame
    automaticamente — o usuário nunca precisa converter manualmente.

Métodos terminais (converters explícitos)
-------------------------------------------
    .native()    → mesmo tipo do DataFrame original passado a dd.frame()
                   (pd.DataFrame → pd.DataFrame, pl.DataFrame → pl.DataFrame)
    .pandas()    → força pd.DataFrame (materializa se for lazy)
    .polars()    → força pl.DataFrame (materializa se for lazy)
    .lazy_pl()   → o pl.LazyFrame/pl.DataFrame interno, sem conversão
    .collect()   → alias de Polars — materializa LazyFrame, mantém FrameProxy

Sem terminal, um FrameProxy também tem __repr__ delegando ao objeto interno
— então basta fazer `print(dd.frame(df).filter(...))` para inspecionar
resultados intermediários sem precisar de `.native()`.

Isso NÃO substitui os wrappers dedicados de dd.* que já existem (dd.mask,
dd.where, dd.groupby, etc.) — continua sendo mais legível usar
dd.where(df, uf="SP") para o caso comum. dd.frame() é o escape hatch
genérico para os outros 90% de métodos do Polars que não têm um wrapper
dedicado, e para quem prefere a sintaxe de encadeamento nativa do Polars.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import polars as pl

_WRAPPABLE_RETURN = (pl.DataFrame, pl.LazyFrame)


def _is_polars_native(obj: Any) -> bool:
    """
    Detecta objetos "intermediários" do Polars que não são DataFrame/LazyFrame
    mas ainda fazem parte de uma cadeia de métodos — ex.: o retorno de
    df.group_by(...) é um objeto GroupBy, não um DataFrame; só vira
    DataFrame depois de .agg(...). O mesmo vale para rolling(), over(), etc.
    Sem tratar isso, a cadeia dd.frame(df).group_by(...).agg(...) quebraria
    no meio (group_by() devolveria o GroupBy "cru", sem o proxy).
    """
    module = type(obj).__module__
    return isinstance(module, str) and module.startswith("polars")


def _unwrap_arg(value: Any) -> Any:
    """Converte FrameProxy/pd.DataFrame em pl.DataFrame antes de repassar a um método Polars."""
    if isinstance(value, FrameProxy):
        return value._data
    if isinstance(value, pd.DataFrame):
        return pl.from_pandas(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap_arg(v) for v in value)
    return value


class FrameProxy:
    """
    Proxy fluente sobre pl.DataFrame/pl.LazyFrame — ver módulo docstring.

    Não instancie diretamente; use dd.frame(df).
    """

    __slots__ = ("_data", "_was_pandas", "_was_lazy")

    def __init__(self, data: Any):
        if isinstance(data, FrameProxy):
            self._data = data._data
            self._was_pandas = data._was_pandas
            self._was_lazy = data._was_lazy
            return

        self._was_pandas = isinstance(data, pd.DataFrame)
        self._was_lazy = isinstance(data, pl.LazyFrame)

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            self._data = data
        elif isinstance(data, pd.DataFrame):
            self._data = pl.from_pandas(data)
        else:
            raise TypeError(
                f"dd.frame() aceita pd.DataFrame, pl.DataFrame ou pl.LazyFrame "
                f"(recebido {type(data).__name__})."
            )

    # ------------------------------------------------------------------
    # Dispatch genérico — o coração do proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # __slots__ garante que atributos "reais" (_data, etc.) nunca caem
        # aqui — só chegamos em __getattr__ para nomes que NÃO existem
        # neste objeto, ou seja, candidatos a métodos do Polars.
        target = self._data
        if not hasattr(target, name):
            raise AttributeError(
                f"dd.frame(...): '{name}' não existe em "
                f"{type(target).__name__} (Polars). Verifique o nome do "
                f"método na documentação do Polars — dd.frame() expõe "
                f"exatamente a API do Polars, sem alterações."
            )
        attr = getattr(target, name)
        if not callable(attr):
            return attr  # propriedades (ex.: .columns, .shape, .dtypes)

        def _dispatched(*args, **kwargs):
            args = tuple(_unwrap_arg(a) for a in args)
            kwargs = {k: _unwrap_arg(v) for k, v in kwargs.items()}
            result = attr(*args, **kwargs)
            if isinstance(result, _WRAPPABLE_RETURN):
                wrapped = FrameProxy.__new__(FrameProxy)
                wrapped._data = result
                wrapped._was_pandas = self._was_pandas
                wrapped._was_lazy = self._was_lazy or isinstance(result, pl.LazyFrame)
                return wrapped
            if _is_polars_native(result):
                # Objeto intermediário (GroupBy, RollingGroupBy, Expr, ...) —
                # mantém o encadeamento embrulhando também, mas os terminais
                # (.native()/.pandas()/.polars()) vão recusar com uma
                # mensagem clara enquanto o objeto não virar DataFrame/LazyFrame.
                wrapped = FrameProxy.__new__(FrameProxy)
                wrapped._data = result
                wrapped._was_pandas = self._was_pandas
                wrapped._was_lazy = self._was_lazy
                return wrapped
            return result

        return _dispatched

    # ------------------------------------------------------------------
    # Terminais — saída explícita
    # ------------------------------------------------------------------

    def native(self) -> Any:
        """Devolve no MESMO tipo passado originalmente a dd.frame()."""
        self._require_frame("native")
        data = self._data.collect() if isinstance(self._data, pl.LazyFrame) else self._data
        return data.to_pandas() if self._was_pandas else data

    def pandas(self) -> pd.DataFrame:
        """Força conversão para pd.DataFrame (materializa se for LazyFrame)."""
        self._require_frame("pandas")
        data = self._data.collect() if isinstance(self._data, pl.LazyFrame) else self._data
        return data.to_pandas()

    def polars(self) -> pl.DataFrame:
        """Força conversão para pl.DataFrame (materializa se for LazyFrame)."""
        self._require_frame("polars")
        return self._data.collect() if isinstance(self._data, pl.LazyFrame) else self._data

    def lazy_pl(self) -> Any:
        """Devolve o objeto Polars interno sem qualquer conversão (DataFrame ou LazyFrame)."""
        return self._data

    def unwrap(self) -> Any:
        """Alias de native() — nome alternativo mais comum em outras libs."""
        return self.native()

    def _require_frame(self, method_name: str) -> None:
        if not isinstance(self._data, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(
                f"dd.frame(...).{method_name}(): o encadeamento ainda não "
                f"chegou a um DataFrame — o objeto atual é "
                f"{type(self._data).__name__} (Polars), um estado "
                f"intermediário (ex.: resultado de group_by()/rolling() "
                f"antes de .agg()). Complete a cadeia (ex.: .agg(...)) "
                f"antes de chamar .{method_name}()."
            )

    # ------------------------------------------------------------------
    # Convivência com Python
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "lazy" if isinstance(self._data, pl.LazyFrame) else "eager"
        origin = "pandas" if self._was_pandas else "polars"
        return f"FrameProxy({kind}, origem={origin})\n{self._data!r}"

    def __len__(self) -> int:
        data = self._data
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        return len(data)

    def __getitem__(self, key: Any) -> Any:
        data = self._data
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        result = data[key]
        if isinstance(result, _WRAPPABLE_RETURN):
            return FrameProxy(result)
        return result

    def __eq__(self, other: Any) -> bool:
        other_data = other._data if isinstance(other, FrameProxy) else other
        return self._data.equals(other_data) if hasattr(self._data, "equals") else NotImplemented

    def __iter__(self):
        data = self._data
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        return iter(data)


def frame(data: Any) -> FrameProxy:
    """
    Ponto de entrada público — dd.frame(df).

    Aceita pd.DataFrame, pl.DataFrame ou pl.LazyFrame. Retorna um FrameProxy
    fluente que expõe qualquer método do Polars, encadeável, convertendo
    de volta ao tipo original com .native() ao final.
    """
    return FrameProxy(data)
