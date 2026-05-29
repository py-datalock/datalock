"""
datalock/analytics.py  — v1.1.0
==============================
Um vocabulário, linguagem natural/SQL, performance de Polars.

Princípios:
  1. TYPE PRESERVATION  input pl.DataFrame → pl.DataFrame, pd.DataFrame → pd.DataFrame
  2. DUAL ENGINE        Polars para pl.DataFrame/LazyFrame; pandas para pd.DataFrame
                        Sem round-trip pl↔pd desnecessário
  3. SQL/NATURAL NAMES  where, sort, groupby, select, unique, fill_null, add_column
  4. SQL ALIASES        order_by, group_by, distinct, union_all, limit, coalesce

Aliases de nomes SQL/pandas:
  WHERE       → where()      filter_()   q()
  SELECT      → select()
  ORDER BY    → sort()       order_by()
  GROUP BY    → groupby()    group_by()
  DISTINCT    → unique()     distinct()  drop_duplicates()
  LIMIT       → head()       limit()
  UNION ALL   → concat()     union_all()
  COALESCE    → fill_null()  fillna()    coalesce()
  CASE WHEN   → when()
  ALTER ADD   → add_column() with_column() assign()
  UNPIVOT     → melt()       unpivot()
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd

import polars as pl
_PL = True  # Polars é dependência obrigatória desde v1.0.4

AnyFrame = Any   # pd.DataFrame | pl.DataFrame


# ─── type helpers (zero-overhead) ────────────────────────────────────────────

def _ipl(df) -> bool:
    return _PL and isinstance(df, pl.DataFrame)

def _ilazy(df) -> bool:
    return _PL and isinstance(df, pl.LazyFrame)

def _asl(x) -> list:
    if x is None: return []
    return [x] if isinstance(x, str) else list(x)


# ─── WHERE ────────────────────────────────────────────────────────────────────

def where(df, expr=None, **kw):
    """
    Filtra linhas.  SQL: WHERE

    Sintaxes (sem import polars):
      kwargs      dd.where(df, uf='SP')
                  dd.where(df, uf='SP', tipo_pessoa='PF')
      lista=isin  dd.where(df, uf=['SP','RJ','MG'])
      range       dd.where(df, renda_mensal=(5000, 15000))
      operadores  dd.where(df, renda_mensal=('>', 5000))
                  dd.where(df, uf=('!=', 'SP'))
                  dd.where(df, nome=('contains', 'Silva'))
                  dd.where(df, nome=('startswith', 'Ana'))
                  dd.where(df, nome=('endswith', '.br'))
                  dd.where(df, nome=('like', '%Silva%'))
                  dd.where(df, doc=('isnull',))
                  dd.where(df, doc=('notnull',))
      None=null   dd.where(df, doc=None)
      dd.col()    dd.where(df, dd.col('renda') > 5000)
      pl.Expr     dd.where(df, pl.col('uf') == 'SP')
      string      dd.where(df, 'uf == "SP" and renda > 5000')
      callable    dd.where(df, lambda d: d['uf'] == 'SP')
      LazyFrame   dd.where(lf, pl.col('uf') == 'SP')  → LazyFrame

    Exemplos:
        dd.where(df, uf='SP')
        dd.where(df, uf=['SP','RJ'])
        dd.where(df, renda_mensal=('>', 5000))
        dd.where(df, nome=('contains', 'Silva'))
        dd.where(df, dd.col('renda').is_between(5000, 15000))
        dd.where(df, ~dd.col('inadimplente'))
        dd.where(df, (dd.col('uf')=='SP') & (dd.col('renda')>5000))
    """
    try:
        from datalock.expr import Expr as _LE
    except ImportError:
        _LE = None

    if kw:
        expr = dict(kw) if expr is None else ({**expr, **kw} if isinstance(expr, dict) else expr)

    if expr is None:
        raise ValueError("where() precisa de pelo menos um critério")

    # ── LazyFrame ──────────────────────────────────────────────────────
    if _ilazy(df):
        if _LE and isinstance(expr, _LE): return df.filter(expr.to_polars())
        if _PL and isinstance(expr, pl.Expr): return df.filter(expr)
        if isinstance(expr, dict):
            m = _dp(list(expr.items())[0][0], list(expr.items())[0][1])
            for k,v in list(expr.items())[1:]: m = m & _dp(k,v)
            return df.filter(m)
        if isinstance(expr, str): return df.collect().filter(pl.Expr.from_json(expr)).lazy()
        raise TypeError(f"LazyFrame where: use pl.Expr ou dict, recebido {type(expr).__name__}")

    # ── pl.DataFrame ──────────────────────────────────────────────────
    if _ipl(df):
        if _LE and isinstance(expr, _LE): return df.filter(expr.to_polars())
        if isinstance(expr, pl.Expr): return df.filter(expr)
        if isinstance(expr, dict):
            m = _dp(list(expr.items())[0][0], list(expr.items())[0][1])
            for k,v in list(expr.items())[1:]: m = m & _dp(k,v)
            return df.filter(m)
        if isinstance(expr, str):
            try: return df.filter(pl.Expr.from_json(expr))
            except Exception: pass
        if callable(expr): return df[expr(df.to_pandas()).values]
        if isinstance(expr, str):
            mask = df.to_pandas().eval(expr)
            return df[mask.values]
        raise TypeError(f"pl.DataFrame where: tipo não suportado {type(expr).__name__}")

    # ── pd.DataFrame ──────────────────────────────────────────────────
    df_pd: pd.DataFrame = df
    if _LE and isinstance(expr, _LE): return df_pd[expr.to_pandas_mask(df_pd)].reset_index(drop=True)
    if _PL and isinstance(expr, pl.Expr):
        return df_pd[pl.from_pandas(df_pd).filter(expr).to_pandas().index].reset_index(drop=True)
    if isinstance(expr, dict):
        mask = pd.Series(True, index=df_pd.index)
        for k,v in expr.items(): mask &= _dpd(df_pd, k, v)
        return df_pd[mask].reset_index(drop=True)
    if isinstance(expr, str):
        try: return df_pd.query(expr).reset_index(drop=True)
        except Exception as e: raise ValueError(f"Filtro inválido: {expr!r} — {e}") from None
    if callable(expr): return df_pd[expr(df_pd)].reset_index(drop=True)
    raise TypeError(f"where(): tipo não suportado {type(expr).__name__}")


def _dp(col, val):
    """(col, val) → pl.Expr"""
    c = pl.col(col)
    if val is None: return c.is_null()
    if isinstance(val, list): return c.is_in(val)
    if isinstance(val, tuple):
        if len(val)==1:
            o=str(val[0]).lower()
            if o in ('isnull','is_null'): return c.is_null()
            if o in ('notnull','is_not_null'): return c.is_not_null()
        if len(val)==2:
            op,arg=val
            if isinstance(op,str):
                ol=op.lower()
                M={'>':lambda:c>arg,'>=':lambda:c>=arg,'<':lambda:c<arg,'<=':lambda:c<=arg,
                   '!=':lambda:c!=arg,'<>':lambda:c!=arg,'==':lambda:c==arg,
                   'eq':lambda:c==arg,'ne':lambda:c!=arg,'gt':lambda:c>arg,'ge':lambda:c>=arg,
                   'lt':lambda:c<arg,'le':lambda:c<=arg,
                   'contains':lambda:c.cast(pl.String).str.contains(str(arg)),
                   'startswith':lambda:c.cast(pl.String).str.starts_with(str(arg)),
                   'endswith':lambda:c.cast(pl.String).str.ends_with(str(arg)),
                   'like':lambda:c.cast(pl.String).str.contains(str(arg).replace('%','.*').replace('_','.')),
                   'isin':lambda:c.is_in(list(arg) if not isinstance(arg,list) else arg),
                   'not_in':lambda:~c.is_in(list(arg) if not isinstance(arg,list) else arg),
                   'matches':lambda:c.cast(pl.String).str.contains(str(arg)),
                }
                if ol in M: return M[ol]()
            if isinstance(op,(int,float)) and isinstance(arg,(int,float)): return c.is_between(op,arg)
    return c==val


def _dpd(df, col, val):
    """(col, val) → pd.Series bool mask"""
    s=df[col]
    if val is None: return s.isna()
    if isinstance(val,list): return s.isin(val)
    if isinstance(val,tuple):
        if len(val)==1:
            o=str(val[0]).lower()
            if o in ('isnull','is_null'): return s.isna()
            if o in ('notnull','is_not_null'): return s.notna()
        if len(val)==2:
            op,arg=val
            if isinstance(op,str):
                ol=op.lower()
                OPS={'>':lambda:s>arg,'>=':lambda:s>=arg,'<':lambda:s<arg,'<=':lambda:s<=arg,
                     '!=':lambda:s!=arg,'<>':lambda:s!=arg,'==':lambda:s==arg,
                     'eq':lambda:s==arg,'ne':lambda:s!=arg,'gt':lambda:s>arg,'ge':lambda:s>=arg,
                     'lt':lambda:s<arg,'le':lambda:s<=arg,
                     'contains':lambda:s.astype(str).str.contains(str(arg),na=False),
                     'startswith':lambda:s.astype(str).str.startswith(str(arg)),
                     'endswith':lambda:s.astype(str).str.endswith(str(arg)),
                     'like':lambda:s.astype(str).str.contains(str(arg).replace('%','.*').replace('_','.'),na=False),
                     'isin':lambda:s.isin(list(arg) if not isinstance(arg,list) else arg),
                     'not_in':lambda:~s.isin(list(arg) if not isinstance(arg,list) else arg),
                     'matches':lambda:s.astype(str).str.contains(str(arg),na=False),
                }
                if ol in OPS: return OPS[ol]()
            if isinstance(op,(int,float)) and isinstance(arg,(int,float)): return s.between(op,arg)
    return s==val

filter_ = where
q       = where


# ─── SELECT / DROP / RENAME ──────────────────────────────────────────────────

def select(df, cols) -> AnyFrame:
    """
    Seleciona colunas.  SQL: SELECT col1, col2

    Exemplos:
        dd.select(df, 'uf')
        dd.select(df, ['uf','renda_mensal','tipo_pessoa'])
        dd.select(df, dd.cols(df, 'renda'))
    """
    cl = _asl(cols) if isinstance(cols,(str,list)) else list(cols)
    if _ipl(df): return df.select(cl)
    if _ilazy(df): return df.select(cl)
    return df[cl]


def drop(df, cols) -> AnyFrame:
    """
    Remove colunas.  SQL: SELECT * EXCEPT(col)

    Exemplos:
        dd.drop(df, 'coluna_inutil')
        dd.drop(df, ['col1','col2'])
    """
    cl = _asl(cols) if isinstance(cols,(str,list)) else list(cols)
    if _ipl(df): return df.drop(cl)
    if _ilazy(df): return df.drop(cl)
    return df.drop(columns=cl)


def rename(df, mapping: Dict[str,str]) -> AnyFrame:
    """
    Renomeia colunas.  SQL: SELECT col AS novo_nome

    Exemplos:
        dd.rename(df, {'cpf': 'documento', 'renda': 'renda_mensal'})
    """
    if _ipl(df) or _ilazy(df): return df.rename(mapping)
    return df.rename(columns=mapping)


# ─── SORT ────────────────────────────────────────────────────────────────────

def sort(df, by, *, desc: bool=False,
         ascending=None, nulls_last: bool=True) -> AnyFrame:
    """
    Ordena.  SQL: ORDER BY ... ASC|DESC

    Args:
        by:          Coluna(s).
        desc:        True = decrescente (natural: 'maior primeiro').
        ascending:   Alternativa: True/False ou [True,False,...].
        nulls_last:  Nulos no final (padrão True).

    Exemplos:
        dd.sort(df, 'renda_mensal')                     # crescente
        dd.sort(df, 'renda_mensal', desc=True)          # decrescente
        dd.sort(df, ['uf','renda'], desc=True)          # multi-coluna
        dd.sort(df, ['uf','renda'], ascending=[True,False])
    """
    by_list = _asl(by) if isinstance(by,str) else list(by)
    if ascending is not None:
        descending = [not ascending]*len(by_list) if isinstance(ascending,bool) else [not a for a in ascending]
    else:
        descending = [desc]*len(by_list)

    if _ipl(df) or _ilazy(df):
        return df.sort(by_list, descending=descending, nulls_last=nulls_last)

    asc_pd = [not d for d in descending]
    na_pos = 'last' if nulls_last else 'first'
    return df.sort_values(by_list, ascending=asc_pd, na_position=na_pos).reset_index(drop=True)

order_by = sort


# ─── GROUP BY ────────────────────────────────────────────────────────────────

def groupby(df, by, agg: Dict[str,Any], *,
            having=None, sort=None, desc: bool=False, limit: int=None) -> AnyFrame:
    """
    Agrupa e agrega.  SQL: GROUP BY ... HAVING ... ORDER BY ... LIMIT

    Args:
        by:     Coluna(s) de agrupamento.
        agg:    {nome: funcao | (col,func) | [funcs]}
                Funções: mean sum min max count std var first last n_unique median
                Atalho: {'renda':'mean'} → agrega 'renda' com mean
        having: Filtro pós-grupo: {'n':('>', 100)}
        sort:   Ordena resultado por esta(s) coluna(s).
        desc:   Ordem decrescente ao usar sort=.
        limit:  TOP N grupos.

    Exemplos:
        dd.groupby(df, 'uf', {'renda_mensal':'mean'})
        dd.groupby(df, 'uf', {'media':('renda_mensal','mean'), 'n':('*','count')})
        dd.groupby(df, ['uf','tipo'], {'renda':['mean','sum']})
        dd.groupby(df, 'uf', {'media':('renda_mensal','mean'), 'n':('*','count')},
                   having={'n':('>',100)}, sort='media', desc=True, limit=10)
    """
    was_pd = isinstance(df, pd.DataFrame) and not _ipl(df)

    if not _PL or was_pd:
        # pandas path — stay in pandas
        df_pd = df if isinstance(df, pd.DataFrame) else df.to_pandas()
        by_list = _asl(by) if isinstance(by,str) else list(by)
        pd_agg = {}
        for col, spec in agg.items():
            if isinstance(spec, str): pd_agg[col] = spec
            elif isinstance(spec, tuple) and len(spec)==2:
                src, fn = spec
                if src != '*': pd_agg[src] = fn
        result = df_pd.groupby(by_list).agg(pd_agg).reset_index()
        return result

    # Polars path
    df_pl = df if _ipl(df) or _ilazy(df) else pl.from_pandas(df)
    by_list = _asl(by) if isinstance(by,str) else list(by)

    _F = {'mean':pl.mean,'sum':pl.sum,'min':pl.min,'max':pl.max,
          'count':pl.len,'std':pl.std,'var':pl.var,'first':pl.first,
          'last':pl.last,'n_unique':pl.n_unique,'median':pl.median,'len':pl.len}

    exprs = []
    for name, spec in agg.items():
        if isinstance(spec, str):
            fn = spec.lower()
            exprs.append(pl.len().alias(name) if fn in ('count','len') else _F.get(fn, pl.mean)(name).alias(name))
        elif isinstance(spec, list):
            for fn in spec:
                fn = fn.lower()
                exprs.append(pl.len().alias(f'{name}_{fn}') if fn in ('count','len') else _F.get(fn,pl.mean)(name).alias(f'{name}_{fn}'))
        elif isinstance(spec, tuple) and len(spec)==2:
            src, fn = spec[0], spec[1].lower()
            exprs.append(pl.len().alias(name) if fn in ('count','len') or src=='*' else _F.get(fn,pl.mean)(src).alias(name))

    if _ilazy(df_pl):
        result = df_pl.group_by(by_list).agg(exprs)
        if having:
            for h_col, h_val in having.items(): result = result.filter(_dp(h_col, h_val))
        if sort:
            sc = _asl(sort) if isinstance(sort,str) else list(sort)
            result = result.sort(sc, descending=[desc]*len(sc))
        if limit: result = result.head(limit)
        return result  # stays LazyFrame

    result = df_pl.group_by(by_list).agg(exprs)
    if having:
        for h_col, h_val in having.items(): result = result.filter(_dp(h_col, h_val))
    if sort:
        sc = _asl(sort) if isinstance(sort,str) else list(sort)
        result = result.sort(sc, descending=[desc]*len(sc))
    if limit: result = result.head(limit)

    return result if not was_pd else result.to_pandas()

group_by = groupby


# ─── HEAD / TAIL / SAMPLE ────────────────────────────────────────────────────

def head(df, n: int=5) -> AnyFrame:
    """Primeiras N linhas.  SQL: LIMIT N"""
    if _ipl(df) or _ilazy(df): return df.head(n)
    return df.head(n)

def tail(df, n: int=5) -> AnyFrame:
    """Últimas N linhas."""
    if _ipl(df): return df.tail(n)
    if _ilazy(df): return df.collect().tail(n)
    return df.tail(n)

limit = head

def sample(df, n=None, frac=None, *, seed: int=42) -> AnyFrame:
    """Amostra aleatória de N linhas ou fração."""
    if _ipl(df):
        if frac is not None: return df.sample(fraction=frac, seed=seed)
        return df.sample(n=n or 100, seed=seed)
    if _ilazy(df):
        df_c = df.collect()
        if frac is not None: return df_c.sample(fraction=frac, seed=seed)
        return df_c.sample(n=n or 100, seed=seed)
    return df.sample(n=n, frac=frac, random_state=seed).reset_index(drop=True)


# ─── UNIQUE / DISTINCT ───────────────────────────────────────────────────────

def unique(df, subset=None, *, keep: str='first') -> AnyFrame:
    """
    Remove duplicatas.  SQL: SELECT DISTINCT

    Args:
        subset: Coluna(s) para considerar (None = todas).
        keep:   'first' | 'last' | 'none' (remove todas as cópias).

    Exemplos:
        dd.unique(df)                  # linhas completamente únicas
        dd.unique(df, 'cpf')           # uma linha por CPF
        dd.unique(df, ['uf','tipo'])   # uma linha por combinação
    """
    sub = _asl(subset) if subset else None
    if _ipl(df): return df.unique(subset=sub, keep=keep, maintain_order=True)
    if _ilazy(df): return df.unique(subset=sub, keep=keep)
    keep_pd = keep if keep!='none' else False
    return df.drop_duplicates(subset=sub, keep=keep_pd).reset_index(drop=True)

distinct        = unique
drop_duplicates = unique


# ─── ADD / TRANSFORM COLUMNS ─────────────────────────────────────────────────

def add_column(df, **cols) -> AnyFrame:
    """
    Adiciona ou substitui colunas.  SQL: SELECT *, expr AS nome

    Aceita dd.col(), dd.when(), pl.Expr, scalar, callable, array.
    Engine Polars — executa todas as expressões em um único passo.

    Exemplos:
        dd.add_column(df,
            imposto       = dd.col('renda_mensal') * 0.27,
            renda_liquida = dd.col('renda_mensal') * 0.73,
            faixa         = dd.when(dd.col('renda_mensal') > 10000, 'alta')
                              .when(dd.col('renda_mensal') > 5000,  'media')
                              .otherwise('baixa'),
            media_uf      = dd.col('renda_mensal').mean().over('uf'),
            rank_renda    = dd.col('renda_mensal').rank('dense', descending=True),
            nome_lower    = dd.col('nome').str.to_lowercase(),
            ano_nasc      = dd.col('data_nascimento').str.to_date('%Y-%m-%d').dt.year(),
        )
    """
    if not cols: return df

    try:
        from datalock.expr import Expr as _LE
    except ImportError:
        _LE = None

    was_pd = isinstance(df, pd.DataFrame) and not _ipl(df)

    if _PL:
        df_pl = df if (_ipl(df) or _ilazy(df)) else pl.from_pandas(df)
        exprs = []
        deferred = {}

        for name, val in cols.items():
            if isinstance(val, pl.Expr):
                exprs.append(val.alias(name))
            elif isinstance(val, _WhenExpr):
                exprs.append(val.to_polars(alias=name))
            elif _LE and isinstance(val, _LE):
                try: exprs.append(val.to_polars().alias(name))
                except: deferred[name] = val
            elif hasattr(val, 'to_polars') and callable(val.to_polars):
                try:
                    e = val.to_polars()
                    exprs.append(e.alias(name) if not (hasattr(val,'_alias') and val._alias) else e)
                except: deferred[name] = val
            elif callable(val):
                deferred[name] = val
            elif hasattr(val,'__iter__') and not isinstance(val,(str,dict)):
                try:
                    import numpy as _np
                    arr = list(val) if not isinstance(val,_np.ndarray) else val.tolist()
                    exprs.append(pl.Series(name, arr).alias(name))
                except: deferred[name] = val
            else:
                exprs.append(pl.lit(val).alias(name))

        if exprs:
            df_pl = df_pl.with_columns(exprs) if not _ilazy(df_pl) else df_pl.with_columns(exprs)

        for name, val in deferred.items():
            if callable(val):
                df_tmp = df_pl.collect().to_pandas() if _ilazy(df_pl) else df_pl.to_pandas()
                series_result = val(df_tmp)
                vals = series_result.values.tolist() if hasattr(series_result,'values') else list(series_result)
                if _ilazy(df_pl): df_pl = df_pl.collect().with_columns(pl.Series(name, vals)).lazy()
                else: df_pl = df_pl.with_columns(pl.Series(name, vals))

        return df_pl.to_pandas() if was_pd else df_pl

    # pandas fallback
    df_pd = df.copy()
    for name, val in cols.items():
        if isinstance(val, _WhenExpr): df_pd[name] = _eval_when_pd(df_pd, val)
        elif callable(val): df_pd[name] = val(df_pd)
        elif hasattr(val,'__iter__') and not isinstance(val,str): df_pd[name] = list(val)
        else: df_pd[name] = val
    return df_pd

with_column = add_column
assign      = add_column


# ─── FILL NULL ────────────────────────────────────────────────────────────────

def fill_null(df, value) -> AnyFrame:
    """
    Preenche nulos.  SQL: COALESCE(col, value)

    Args:
        value: Escalar, dict {col:val}, 'forward' ou 'backward'.

    Exemplos:
        dd.fill_null(df, 0)
        dd.fill_null(df, {'renda_mensal': 0, 'uf': 'DESCONHECIDO'})
        dd.fill_null(df, 'forward')
    """
    if _ipl(df) or _ilazy(df):
        if isinstance(value, str) and value in ('forward','backward','forward_fill','backward_fill'):
            strategy = 'forward' if 'forward' in value else 'backward'
            return df.fill_null(strategy=strategy)
        if isinstance(value, dict):
            return df.with_columns([pl.col(c).fill_null(v).alias(c) for c,v in value.items()])
        return df.fill_null(value)
    # pandas
    if isinstance(value, str) and value in ('forward','ffill'): return df.ffill()
    if isinstance(value, str) and value in ('backward','bfill'): return df.bfill()
    return df.fillna(value)

fillna   = fill_null
coalesce = fill_null


# ─── CAST ────────────────────────────────────────────────────────────────────

def cast(df, schema: Dict[str,str]) -> AnyFrame:
    """
    Converte tipos.  SQL: CAST(col AS tipo)

    Tipos aceitos: int int32 int64 float float32 float64
                   str string bool boolean date datetime
                   categorical cat
    Também aceita tipos Polars diretamente: 'Int32', 'Float32', 'Utf8'

    Exemplos:
        dd.cast(df, {'renda_mensal': 'float32', 'idade': 'int32'})
        dd.cast(df, {'flag': 'bool', 'data': 'date'})
    """
    _TM = {'int':pl.Int64 if _PL else 'int64','int8':pl.Int8 if _PL else 'int8',
           'int16':pl.Int16 if _PL else 'int16','int32':pl.Int32 if _PL else 'int32',
           'int64':pl.Int64 if _PL else 'int64','uint8':pl.UInt8 if _PL else 'uint8',
           'uint16':pl.UInt16 if _PL else 'uint16','uint32':pl.UInt32 if _PL else 'uint32',
           'uint64':pl.UInt64 if _PL else 'uint64',
           'float':pl.Float64 if _PL else 'float64','float32':pl.Float32 if _PL else 'float32',
           'float64':pl.Float64 if _PL else 'float64',
           'str':pl.String if _PL else 'str','string':pl.String if _PL else 'str',
           'utf8':pl.String if _PL else 'str','varchar':pl.String if _PL else 'str',
           'bool':pl.Boolean if _PL else 'bool','boolean':pl.Boolean if _PL else 'bool',
           'date':pl.Date if _PL else 'datetime64[ns]',
           'datetime':pl.Datetime if _PL else 'datetime64[ns]',
           'categorical':pl.Categorical if _PL else 'category',
           'cat':pl.Categorical if _PL else 'category'} if _PL else {}

    if _ipl(df) or _ilazy(df):
        exprs = []
        for col, dtype_str in schema.items():
            pl_type = _TM.get(dtype_str.lower(), getattr(pl, dtype_str, pl.String))
            exprs.append(pl.col(col).cast(pl_type, strict=False).alias(col))
        return df.with_columns(exprs)
    pd_schema = {c: _TM.get(t.lower(), t) if _PL else t for c,t in schema.items()}
    return df.astype({c: str(t) if _PL else t for c,t in pd_schema.items()}, errors='ignore')


# ─── CONCAT ──────────────────────────────────────────────────────────────────

def concat(frames: List[AnyFrame], *, axis: int=0) -> AnyFrame:
    """
    Concatena DataFrames.  SQL: UNION ALL (axis=0)

    Engine Polars: zero-copy para frames do mesmo schema.

    Exemplos:
        dd.concat([df_jan, df_fev, df_mar])         # empilha linhas
        dd.concat([df_a, df_b], axis=1)             # combina colunas
    """
    if not frames: return pd.DataFrame()
    any_pl = _PL and any(_ipl(f) or _ilazy(f) for f in frames)
    any_pd = any(isinstance(f, pd.DataFrame) and not _ipl(f) for f in frames)

    if _PL and any_pl and not any_pd and axis==0:
        pl_frames = [f.collect() if _ilazy(f) else f for f in frames]
        return pl.concat(pl_frames, how='diagonal')
    if any_pd or not _PL:
        pd_frames = [f.to_pandas() if (_PL and _ipl(f)) else f for f in frames]
        return pd.concat(pd_frames, axis=axis, ignore_index=(axis==0))
    # mixed — convert all to polars
    if _PL:
        pl_frames = [f.collect() if _ilazy(f) else (_to_pl_safe(f) if not _ipl(f) else f) for f in frames]
        return pl.concat(pl_frames, how='diagonal')
    return pd.concat([f for f in frames], axis=axis, ignore_index=(axis==0))

def _to_pl_safe(df):
    if _ipl(df): return df
    return pl.from_pandas(df)

union_all = concat


# ─── PIVOT / MELT ────────────────────────────────────────────────────────────

def pivot(df, *, index, columns: str, values: str, aggfunc: str='first') -> AnyFrame:
    """
    Wide: transforma linhas em colunas.  SQL: PIVOT

    Exemplos:
        dd.pivot(df, index='uf', columns='tipo_pessoa',
                 values='renda_mensal', aggfunc='mean')
    """
    was_pd = isinstance(df, pd.DataFrame) and not _ipl(df)
    if _PL:
        agg_map = {'first':'first','sum':'sum','mean':'mean',
                   'count':'count','min':'min','max':'max'}
        df_pl = df if _ipl(df) else pl.from_pandas(df) if was_pd else df.collect()
        result = df_pl.pivot(on=columns, index=index, values=values,
                             aggregate_function=agg_map.get(aggfunc,'first'))
        return result.to_pandas() if was_pd else result
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)


def melt(df, *, id_cols=None, value_cols=None,
         name: str='variable', value: str='value') -> AnyFrame:
    """
    Long: transforma colunas em linhas.  SQL: UNPIVOT

    Exemplos:
        dd.melt(df, id_cols=['uf','tipo'], value_cols=['renda_mensal','idade'])
    """
    was_pd = isinstance(df, pd.DataFrame) and not _ipl(df)
    id_list  = _asl(id_cols)   if id_cols   else None
    val_list = _asl(value_cols) if value_cols else None
    if _PL:
        df_pl = df if _ipl(df) else pl.from_pandas(df) if was_pd else df.collect()
        result = df_pl.unpivot(on=val_list, index=id_list,
                               variable_name=name, value_name=value)
        return result.to_pandas() if was_pd else result
    return pd.melt(df, id_vars=id_list, value_vars=val_list,
                   var_name=name, value_name=value)

unpivot = melt


# ─── DESCRIBE / SCHEMA / INFO ────────────────────────────────────────────────

def describe(df, percentiles=None) -> pd.DataFrame:
    """Estatísticas descritivas. Sempre retorna pd.DataFrame (é um relatório)."""
    pct = percentiles or [0.25, 0.5, 0.75]
    if _ipl(df):
        try:
            r = df.describe(percentiles=pct)
            return r.to_pandas().set_index(r.columns[0])
        except Exception:
            pass
    df_pd = df.to_pandas() if _ipl(df) else (df.collect().to_pandas() if _ilazy(df) else df)
    return df_pd.describe(percentiles=pct)


def schema(df) -> Dict[str,str]:
    """
    Schema: {coluna: tipo}.  SQL: DESCRIBE TABLE / SHOW COLUMNS

    Exemplos:
        dd.schema(df)
        # {'nome':'String', 'renda_mensal':'Float64', 'inadimplente':'Boolean'}
    """
    if _ipl(df): return {k: str(v) for k,v in df.schema.items()}
    if _ilazy(df): return {k: str(v) for k,v in df.schema.items()}
    return {c: str(df[c].dtype) for c in df.columns}

dtypes = schema


def info(df) -> None:
    """
    Resumo: shape, tipos, nulos, memória.  Como df.info() no pandas.

    Exemplos:
        dd.info(df)
    """
    if _ipl(df):
        n_rows, n_cols = df.height, df.width
        null_c = {c: df[c].null_count() for c in df.columns}
        mem_mb = df.estimated_size('mb')
        type_label = 'polars.DataFrame'
    elif _ilazy(df):
        df = df.collect()
        n_rows, n_cols = df.height, df.width
        null_c = {c: df[c].null_count() for c in df.columns}
        mem_mb = df.estimated_size('mb')
        type_label = 'polars.LazyFrame (collected)'
    else:
        n_rows, n_cols = df.shape
        null_c = df.isnull().sum().to_dict()
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        type_label = 'pandas.DataFrame'

    sc = schema(df)
    print(f'<{type_label}>')
    print(f'Shape: {n_rows:,} linhas × {n_cols} colunas  |  ~{mem_mb:.1f} MB')
    print(f'{"Coluna":<25} {"Tipo":<22} {"Nulos":>8}  {"Nulos%":>7}')
    print('─' * 67)
    for col, dtype_str in sc.items():
        n_null = null_c.get(col, 0)
        pct = n_null / max(n_rows,1) * 100
        flag = ' ⚠️' if pct > 10 else ''
        print(f'  {col:<23} {dtype_str:<22} {n_null:>8,}  {pct:>6.1f}%{flag}')


def count_nulls(df) -> pd.Series:
    """Conta nulos por coluna.  Sempre retorna pd.Series (relatório)."""
    if _ipl(df): return df.null_count().to_pandas().iloc[0]
    if _ilazy(df): return df.collect().null_count().to_pandas().iloc[0]
    return df.isnull().sum()

null_counts = count_nulls
isnull      = count_nulls


def count(df, col=None):
    """Conta linhas ou não-nulos.  SQL: COUNT(*)"""
    if col is None:
        if _ipl(df): return df.height
        if _ilazy(df): return df.collect().height
        return len(df)
    if _ipl(df): return int(df[col].drop_nulls().len())
    if _ilazy(df): return int(df.collect()[col].drop_nulls().len())
    return int(df[col].count())


def nunique(df) -> pd.Series:
    """Valores únicos por coluna."""
    if _ipl(df):
        return df.select([pl.col(c).n_unique().alias(c) for c in df.columns]).to_pandas().iloc[0]
    if _ilazy(df):
        df_c = df.collect()
        return df_c.select([pl.col(c).n_unique().alias(c) for c in df_c.columns]).to_pandas().iloc[0]
    return df.nunique()


def value_counts(df, col: str, *, normalize: bool=False, n: int=20) -> pd.Series:
    """
    Frequência de valores.  SQL: SELECT col, COUNT(*) GROUP BY col ORDER BY 2 DESC

    Exemplos:
        dd.value_counts(df, 'uf')
        dd.value_counts(df, 'uf', normalize=True, n=5)
    """
    if _ipl(df) or _ilazy(df):
        df_pl = df if _ipl(df) else df.collect()
        vc = df_pl[col].value_counts(sort=True).head(n).to_pandas()
        cnt_col = 'count' if 'count' in vc.columns else vc.columns[-1]
        s = vc.set_index(col)[cnt_col]
        return (s/s.sum() if normalize else s).rename(col)
    return df[col].value_counts(normalize=normalize).head(n)


def corr(df, method: str='pearson') -> pd.DataFrame:
    """Correlação entre colunas numéricas."""
    df_pd = df.to_pandas() if (_PL and (_ipl(df) or _ilazy(df))) else df
    if _ilazy(df): df_pd = df.collect().to_pandas()
    num = df_pd.select_dtypes(include=['number'])
    return num.corr(method=method) if not num.empty and num.shape[1] >= 2 else pd.DataFrame()


def shape(df) -> Tuple[int,int]:
    """Dimensões: (linhas, colunas)."""
    if _ipl(df): return (df.height, df.width)
    if _ilazy(df): return df.collect().shape
    return df.shape


# ─── TOP N PER GROUP ─────────────────────────────────────────────────────────

def top_n(df, n: int, by: str, *, group_by=None, desc: bool=True) -> AnyFrame:
    """
    Top N linhas por grupo.  Operação muito comum em análise de dados.

    Sem group_by: top N global.
    Com group_by: top N dentro de cada grupo (window function internamente).

    Exemplos:
        dd.top_n(df, 3, 'renda_mensal')              # top 3 global
        dd.top_n(df, 3, 'renda_mensal', group_by='uf')  # top 3 por UF
    """
    was_pd = isinstance(df, pd.DataFrame) and not _ipl(df)

    if group_by is None:
        df_src = df if _ipl(df) else (pl.from_pandas(df) if was_pd else df)
        return (df_src.sort(by, descending=desc).head(n).to_pandas()
                if was_pd else df_src.sort(by, descending=desc).head(n))

    gb_list = _asl(group_by) if isinstance(group_by,str) else list(group_by)

    if _PL:
        df_pl = df if _ipl(df) else pl.from_pandas(df)
        result = (df_pl
            .with_columns(pl.col(by).rank('ordinal', descending=desc).over(gb_list).alias('__rank__'))
            .filter(pl.col('__rank__') <= n)
            .drop('__rank__')
            .sort(gb_list + [by], descending=[False]*len(gb_list) + [desc])
        )
        return result.to_pandas() if was_pd else result

    # pandas fallback
    return (df.sort_values(by, ascending=not desc)
              .groupby(gb_list).head(n)
              .reset_index(drop=True))


# ─── APPLY / CLIP ────────────────────────────────────────────────────────────

def apply(df, funcs: Dict[str,Any]) -> AnyFrame:
    """
    Aplica função a colunas.

    Exemplos:
        dd.apply(df, {'uf': str.upper, 'email': str.lower})
        dd.apply(df, {'renda': lambda v: round(v, 2)})
    """
    was_pd = isinstance(df, pd.DataFrame) and not _ipl(df)
    df_pd = df.to_pandas() if (_PL and _ipl(df)) else df
    df_pd = df_pd.copy()
    for col, fn in funcs.items():
        if col not in df_pd.columns: raise KeyError(f"Coluna '{col}' não encontrada")
        df_pd[col] = df_pd[col].map(fn)
    if _PL and not was_pd: return pl.from_pandas(df_pd)
    return df_pd


def clip(df, bounds: Dict[str,tuple]) -> AnyFrame:
    """
    Recorta valores.  SQL: GREATEST(min, LEAST(max, col))

    Exemplos:
        dd.clip(df, {'renda_mensal': (1320, 500_000), 'idade': (0, 120)})
    """
    if _ipl(df) or _ilazy(df):
        exprs = []
        for col, (lo, hi) in bounds.items():
            if lo is not None and hi is not None: exprs.append(pl.col(col).clip(lo,hi).alias(col))
            elif lo is not None: exprs.append(pl.when(pl.col(col)<lo).then(lo).otherwise(pl.col(col)).alias(col))
            elif hi is not None: exprs.append(pl.when(pl.col(col)>hi).then(hi).otherwise(pl.col(col)).alias(col))
        return df.with_columns(exprs)
    df_pd = df.copy()
    for col,(lo,hi) in bounds.items():
        df_pd[col] = df_pd[col].clip(lower=lo, upper=hi)
    return df_pd


# ─── WRITE ───────────────────────────────────────────────────────────────────

def write(df, path: str, **kw) -> None:
    """
    Escreve em arquivo detectando formato pela extensão.

    Formatos: .csv .tsv .parquet .xlsx .json .ndjson .feather .ipc .avro

    Exemplos:
        dd.write(df, 'resultado.csv')
        dd.write(df, 'resultado.parquet')
        dd.write(df, 'resultado.xlsx')
    """
    from pathlib import Path as _P
    ext = _P(path).suffix.lower()

    if _PL and (_ipl(df) or _ilazy(df)):
        df_pl = df if _ipl(df) else df.collect()
        W = {'.csv':lambda:df_pl.write_csv(path,**kw),
             '.tsv':lambda:df_pl.write_csv(path,separator='\t',**kw),
             '.parquet':lambda:df_pl.write_parquet(path,**kw),
             '.xlsx':lambda:df_pl.write_excel(path,**kw),
             '.json':lambda:df_pl.write_json(path,**kw),
             '.ndjson':lambda:df_pl.write_ndjson(path,**kw),
             '.feather':lambda:df_pl.write_ipc(path,**kw),
             '.ipc':lambda:df_pl.write_ipc(path,**kw),
             '.avro':lambda:df_pl.write_avro(path,**kw)}
        fn = W.get(ext)
        if fn: fn(); return

    df_pd = df.to_pandas() if (_PL and _ipl(df)) else df
    W2 = {'.csv':lambda:df_pd.to_csv(path,index=False,**kw),
          '.tsv':lambda:df_pd.to_csv(path,sep='\t',index=False,**kw),
          '.parquet':lambda:df_pd.to_parquet(path,index=False,**kw),
          '.xlsx':lambda:df_pd.to_excel(path,index=False,**kw),
          '.json':lambda:df_pd.to_json(path,orient='records',**kw),
          '.ndjson':lambda:df_pd.to_json(path,orient='records',lines=True,**kw),
          '.feather':lambda:df_pd.to_feather(path,**kw)}
    fn2 = W2.get(ext)
    if fn2: fn2()
    else: raise ValueError(f"Formato '{ext}' não suportado. Use: {list(W2)}")


# ─── COLS SELECTOR ────────────────────────────────────────────────────────────

def cols(df, pattern=None, *, dtype=None, exclude=None) -> List[str]:
    """
    Lista colunas por padrão de nome ou tipo.

    Exemplos:
        dd.cols(df, 'renda')                         # colunas com 'renda'
        dd.cols(df, ['renda','idade'])               # múltiplos padrões
        dd.cols(df, dtype='String')                  # todas string
        dd.cols(df, dtype='Float64')                 # todas float
        dd.cols(df, dtype='String', exclude=['nome']) # strings exceto nome
        dd.select(df, dd.cols(df, 'renda'))          # combina com select
    """
    sc = schema(df)
    result = list(sc.keys())
    if pattern is not None:
        pats = [pattern] if isinstance(pattern,str) else list(pattern)
        result = [c for c in result if any(p.lower() in c.lower() for p in pats)]
    if dtype is not None:
        dl = dtype.lower()
        result = [c for c in result if dl in sc.get(c,'').lower()]
    if exclude:
        excl = set(exclude)
        result = [c for c in result if c not in excl]
    return result


# ─── TYPE CONVERSION ─────────────────────────────────────────────────────────

def to_pandas(df) -> pd.DataFrame:
    """Converte para pd.DataFrame. Passthrough se já for pandas."""
    if isinstance(df, pd.DataFrame) and not (_PL and _ipl(df)): return df
    if _PL:
        if _ipl(df): return df.to_pandas()
        if _ilazy(df): return df.collect().to_pandas()
    return pd.DataFrame(df)


def to_polars(df) -> "pl.DataFrame":
    """Converte para pl.DataFrame. Passthrough se já for Polars."""
    if not _PL: raise ImportError("Polars não instalado")
    if _ipl(df): return df
    if _ilazy(df): return df.collect()
    return pl.from_pandas(df)


# ─── WHEN / CASE WHEN ────────────────────────────────────────────────────────

class _WhenExpr:
    """
    CASE WHEN sem import polars.  Usada em dd.add_column().

    Exemplos:
        dd.when(dd.col('renda') > 10000, 'alta')
          .when(dd.col('renda') > 5000, 'media')
          .otherwise('baixa')
    """
    def __init__(self, cond, val):
        self._cases = [(cond, val)]
        self._default = None
        self._alias_name = None

    def when(self, cond, val) -> '_WhenExpr':
        self._cases.append((cond, val)); return self

    def otherwise(self, val) -> '_WhenExpr':
        self._default = val; return self

    def alias(self, name: str) -> '_WhenExpr':
        self._alias_name = name; return self

    def to_polars(self, alias=None):
        try:
            from datalock.expr import Expr as LE
        except ImportError:
            LE = None
        def _e(v):
            if LE and isinstance(v, LE): return v.to_polars()
            if isinstance(v, pl.Expr): return v
            return pl.lit(v)
        c0, v0 = self._cases[0]
        chain = pl.when(_e(c0)).then(_e(v0))
        for c,v in self._cases[1:]: chain = chain.when(_e(c)).then(_e(v))
        default = pl.lit(None) if self._default is None else _e(self._default)
        expr = chain.otherwise(default)
        name = alias or self._alias_name
        return expr.alias(name) if name else expr


def when(condition, value) -> _WhenExpr:
    """
    CASE WHEN sem import polars.

    Exemplos:
        dd.add_column(df,
            faixa=dd.when(dd.col('renda_mensal') > 10000, 'alta')
                    .when(dd.col('renda_mensal') > 5000, 'media')
                    .otherwise('baixa')
        )
    """
    return _WhenExpr(condition, value)


def _eval_when_pd(df: pd.DataFrame, expr: _WhenExpr) -> pd.Series:
    import numpy as np
    try:
        from datalock.expr import Expr as LE
    except ImportError:
        LE = None
    conds, choices = [], []
    for cond, val in expr._cases:
        if LE and isinstance(cond, LE): conds.append(cond.to_pandas_mask(df))
        elif callable(cond): conds.append(cond(df))
        else: conds.append(pd.Series([bool(cond)]*len(df), index=df.index))
        choices.append(val)
    return pd.Series(np.select(conds, choices, default=expr._default), index=df.index)


# ─── BACKWARD COMPAT ─────────────────────────────────────────────────────────
query = where

# ─── SHIFT / LAG / LEAD ──────────────────────────────────────────────────────

def shift(
    df: AnyFrame,
    n: int = 1,
    *,
    columns: Optional[Union[str, List[str]]] = None,
    fill_value: Optional[Any] = None,
) -> AnyFrame:
    """
    Desloca valores N posições para baixo (lag) ou para cima (lead).

    Equivalente ao pandas df.shift() e SQL LAG() / LEAD().

    Args:
        df:          pd.DataFrame ou pl.DataFrame.
        n:           Posições a deslocar. Positivo=lag (passado), negativo=lead (futuro).
        columns:     Colunas a deslocar. None = todas as numéricas.
        fill_value:  Valor para novas posições. None → null.

    Returns:
        DataFrame com colunas deslocadas, mesmo tipo do input.

    Exemplos:
        dd.shift(df, 1)                          # valor do período anterior
        dd.shift(df, -1, columns="renda")        # próximo valor de renda
        dd.shift(df, 3, fill_value=0.0)          # lag 3 com 0 nos nulls
    """
    was_pd = isinstance(df, pd.DataFrame)
    df_pl  = pl.from_pandas(df) if was_pd else df
    if isinstance(df_pl, pl.LazyFrame):
        df_pl = df_pl.collect()

    cols = ([columns] if isinstance(columns, str) else list(columns)) if columns else [
        c for c in df_pl.columns
        if df_pl[c].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16,
                               pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    exprs = []
    for col in cols:
        expr = pl.col(col).shift(n)
        if fill_value is not None:
            expr = expr.fill_null(pl.lit(fill_value))
        exprs.append(expr.alias(col))

    if not exprs:
        return df_pl.to_pandas() if was_pd else df_pl

    result = df_pl.with_columns(exprs)
    return result.to_pandas() if was_pd else result


def lag(df: AnyFrame, n: int = 1, **kw) -> AnyFrame:
    """Alias para shift(n) — valor N períodos no passado."""
    return shift(df, n, **kw)


def lead(df: AnyFrame, n: int = 1, **kw) -> AnyFrame:
    """Alias para shift(-n) — valor N períodos no futuro."""
    return shift(df, -n, **kw)


# ─── EXPLODE ─────────────────────────────────────────────────────────────────

def explode(
    df: AnyFrame,
    columns: Union[str, List[str]],
) -> AnyFrame:
    """
    Expande colunas de listas em múltiplas linhas (one-to-many).

    Essencial para dados JSON/API onde um campo contém uma lista de valores.
    Equivalente a pandas df.explode() e SQL UNNEST().

    Args:
        df:      pd.DataFrame ou pl.DataFrame.
        columns: Coluna(s) do tipo lista a expandir.

    Returns:
        DataFrame com linhas expandidas, mesmo tipo do input.

    Exemplos:
        # df com coluna "tags" = [["a","b"],["c"]]
        dd.explode(df, "tags")
        # resultado: uma linha por tag

        # Múltiplas colunas (devem ter o mesmo comprimento de lista)
        dd.explode(df, ["tags", "scores"])
    """
    was_pd = isinstance(df, pd.DataFrame)
    df_pl  = pl.from_pandas(df) if was_pd else df
    if isinstance(df_pl, pl.LazyFrame):
        df_pl = df_pl.collect()
    cols   = [columns] if isinstance(columns, str) else list(columns)
    result = df_pl.explode(cols)
    return result.to_pandas() if was_pd else result
