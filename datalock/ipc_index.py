"""
ipc_index.py
============
Lógica de row group statistics, pruning e predicate pushdown para o formato .dlk v1.2.0.

Este módulo isola completamente a lógica de índice de Arrow IPC do secure_file.py,
permitindo:
  - Testes unitários independentes de criptografia
  - Reutilização em ferramentas de diagnóstico (dd.inspect)
  - Evolução independente do algoritmo de pruning

Funções públicas:
  compute_batch_stats(batch, content_type)   → dict com min/max/null_count por coluna
  prune_row_groups(row_groups_meta, filters)  → set[int] de índices relevantes
  apply_arrow_filters(batch, filters)         → pa.RecordBatch filtrado
  normalize_filters(filters)                 → list[tuple] canônica

Formato de filters aceito (inspirado em PyArrow/DuckDB):
  {"uf": "SP"}                              # igualdade
  {"uf": ["SP", "RJ"]}                      # in list
  {"renda_mensal": (">", 10_000)}           # comparação: >, >=, <, <=, !=
  {"renda_mensal": (5_000, 50_000)}         # range fechado [a, b]
  {"uf": "SP", "renda_mensal": (">", 5_000)} # múltiplas colunas (AND implícito)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ---------------------------------------------------------------------------
# normalize_filters
# ---------------------------------------------------------------------------

def normalize_filters(
    filters: Optional[Dict[str, Any]],
) -> List[Tuple[str, str, Any]]:
    """
    Normaliza o dict de filters para lista canônica de tuplas (col, op, value).

    Formatos de entrada → saída normalizada:
      "SP"              → [("col", "==", "SP")]
      ["SP", "RJ"]      → [("col", "in", ["SP", "RJ"])]
      (">", 10_000)     → [("col", ">", 10_000)]
      (5_000, 50_000)   → [("col", "range", (5_000, 50_000))]
      (5_000, None)     → [("col", ">=", 5_000)]
      (None, 50_000)    → [("col", "<=", 50_000)]

    Args:
        filters: Dict de filtros ou None.

    Returns:
        Lista de tuplas (col, op, value). Lista vazia se filters=None ou {}.

    Raises:
        ValueError: Se o formato do filtro for inválido.
    """
    if not filters:
        return []

    _VALID_OPS = frozenset({">", ">=", "<", "<=", "==", "!="})
    result: List[Tuple[str, str, Any]] = []

    for col, val in filters.items():
        if not isinstance(col, str) or not col:
            raise ValueError(f"Nome de coluna inválido em filters: {col!r}")

        if isinstance(val, list):
            # in list
            result.append((col, "in", val))

        elif isinstance(val, tuple):
            if len(val) != 2:
                raise ValueError(
                    f"Filtro para '{col}': tuple deve ter 2 elementos, "
                    f"recebeu {len(val)}: {val!r}"
                )
            a, b = val

            if isinstance(a, str) and a in _VALID_OPS:
                # (">", 10_000) — comparação explícita
                result.append((col, a, b))

            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                # (5_000, 50_000) — range fechado [a, b]
                result.append((col, "range", (a, b)))

            elif a is None and b is not None:
                # (None, 50_000) → col <= b
                result.append((col, "<=", b))

            elif a is not None and b is None:
                # (5_000, None) → col >= a
                result.append((col, ">=", a))

            else:
                raise ValueError(
                    f"Filtro para '{col}': formato de tuple inválido: {val!r}. "
                    "Use ('op', valor), (num, num), (None, num) ou (num, None)."
                )

        elif isinstance(val, (str, int, float, bool)) or val is None:
            # igualdade simples
            result.append((col, "==", val))

        else:
            raise ValueError(
                f"Filtro para '{col}': tipo não suportado {type(val).__name__!r}. "
                "Use str, número, lista ou tuple."
            )

    return result


# ---------------------------------------------------------------------------
# compute_batch_stats
# ---------------------------------------------------------------------------

def compute_batch_stats(batch: Any, content_type: str = "masked_dataframe") -> Dict[str, Any]:
    """
    Calcula estatísticas por coluna de um Arrow RecordBatch.

    Para raw_dataframe: retorna apenas dtype — min/max/null_count de dados não
    mascarados poderiam revelar distribuições sensíveis (faixa de renda, etc.),
    violando o princípio de exposição mínima mesmo dentro de um arquivo .dlk.

    Para masked_dataframe (e qualquer outro tipo): calcula min, max e null_count
    por coluna numérica ou categórica. Os valores são hashes HMAC — seguros
    para indexação.

    Args:
        batch:        pyarrow.RecordBatch.
        content_type: "raw_dataframe" | "masked_dataframe" | outros.

    Returns:
        Dict[col_name → {dtype, min?, max?, null_count?}]
    """
    import pyarrow as pa

    stats: Dict[str, Any] = {}
    expose_stats = (content_type != "raw_dataframe")

    for i, field in enumerate(batch.schema):
        col = batch.column(i)
        col_name = field.name
        dtype_str = str(field.type)

        stat: Dict[str, Any] = {"dtype": dtype_str}

        if not expose_stats:
            stats[col_name] = stat
            continue

        try:
            stat["null_count"] = col.null_count

            # Tipos numéricos — min/max direto
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                if col.null_count < len(col):  # pelo menos um não-nulo
                    import pyarrow.compute as pc
                    stat["min"] = pc.min(col).as_py()
                    stat["max"] = pc.max(col).as_py()

            # Strings/categorias — min/max lexicográfico
            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                if col.null_count < len(col):
                    import pyarrow.compute as pc
                    stat["min"] = pc.min(col).as_py()
                    stat["max"] = pc.max(col).as_py()

            # Datas e timestamps — min/max como ISO string
            elif pa.types.is_date(field.type) or pa.types.is_timestamp(field.type):
                if col.null_count < len(col):
                    import pyarrow.compute as pc
                    min_val = pc.min(col).as_py()
                    max_val = pc.max(col).as_py()
                    stat["min"] = str(min_val) if min_val is not None else None
                    stat["max"] = str(max_val) if max_val is not None else None

        except Exception:
            # Falha silenciosa por coluna — nunca impede o pack()
            pass

        stats[col_name] = stat

    return stats


# ---------------------------------------------------------------------------
# prune_row_groups
# ---------------------------------------------------------------------------

def prune_row_groups(
    row_groups_meta: Optional[List[Dict]],
    filters: Optional[Dict[str, Any]],
) -> Set[int]:
    """
    Retorna o conjunto de índices de batches que podem conter linhas
    satisfazendo os filtros fornecidos.

    A lógica é conservadora: na dúvida (sem stats, tipo incompatível, etc.)
    o batch É incluído — nunca exclui um batch que poderia ter resultados.
    Só exclui quando as estatísticas garantem que o batch não tem nenhuma
    linha relevante.

    Args:
        row_groups_meta: Lista de dicts com {batch_index, stats: {col: {min, max, ...}}}.
                         None ou [] → retorna todos os índices (sem pruning).
        filters:         Dict de filtros no formato normalize_filters().
                         None ou {} → retorna todos os índices.

    Returns:
        Set[int] com os batch_index a ler. Se row_groups_meta estiver vazio,
        retorna {-1} como sentinela de "leia tudo sem indexação".
    """
    if not row_groups_meta:
        return {-1}  # sentinela: sem índice → lê tudo

    if not filters:
        return {e["batch_index"] for e in row_groups_meta}

    normalized = normalize_filters(filters)
    if not normalized:
        return {e["batch_index"] for e in row_groups_meta}

    relevant: Set[int] = set()

    for entry in row_groups_meta:
        batch_idx = entry["batch_index"]
        batch_stats = entry.get("stats", {})

        if _batch_satisfies_filters(batch_stats, normalized):
            relevant.add(batch_idx)

    return relevant


def _batch_satisfies_filters(
    batch_stats: Dict[str, Any],
    normalized_filters: List[Tuple[str, str, Any]],
) -> bool:
    """
    Verifica se um batch PODE conter linhas que satisfazem todos os filtros.

    Retorna True (inclui o batch) se:
      - A coluna não tem stats (não podemos descartar)
      - As stats não têm min/max (não podemos descartar)
      - Os ranges se sobrepõem

    Retorna False (exclui o batch) apenas quando as stats garantem que
    nenhuma linha do batch pode satisfazer o filtro.
    """
    for col, op, value in normalized_filters:
        if col not in batch_stats:
            return True  # sem info → conservador: inclui

        col_stat = batch_stats[col]
        batch_min = col_stat.get("min")
        batch_max = col_stat.get("max")

        if batch_min is None or batch_max is None:
            return True  # sem min/max → conservador: inclui

        try:
            if op == "==" :
                # batch pode ter o valor se min <= value <= max
                if not (batch_min <= value <= batch_max):
                    return False

            elif op == "in":
                # batch pode ter algum valor da lista se há sobreposição
                if not any(batch_min <= v <= batch_max for v in value):
                    return False

            elif op == ">":
                # batch pode ter linhas > value se batch_max > value
                if not (batch_max > value):
                    return False

            elif op == ">=":
                if not (batch_max >= value):
                    return False

            elif op == "<":
                # batch pode ter linhas < value se batch_min < value
                if not (batch_min < value):
                    return False

            elif op == "<=":
                if not (batch_min <= value):
                    return False

            elif op == "!=":
                # batch pode ter linhas != value se não é um batch constante = value
                # (batch_min == batch_max == value → batch inteiro = value → descarta)
                if batch_min == batch_max == value and batch_stat.get("null_count", 1) == 0:
                    return False

            elif op == "range":
                range_min, range_max = value
                # Descarta se batch_max < range_min ou batch_min > range_max
                if batch_max < range_min or batch_min > range_max:
                    return False

        except (TypeError, ValueError):
            # Comparação inválida (ex: string vs número) → conservador: inclui
            return True

    return True  # todos os filtros passaram → batch relevante


# ---------------------------------------------------------------------------
# apply_arrow_filters
# ---------------------------------------------------------------------------

def apply_arrow_filters(batch: Any, filters: Optional[Dict[str, Any]]) -> Any:
    """
    Aplica filtros Arrow em nível de linha em um RecordBatch.

    Chamado após o row group pruning — aplica os filtros às linhas reais
    do batch (post-pruning exacto). A combinação pruning + apply é o que
    garante resultados corretos: pruning é otimista (pode incluir falsos
    positivos), apply_arrow_filters é exato.

    Args:
        batch:   pyarrow.RecordBatch.
        filters: Dict de filtros. None → retorna batch sem alteração.

    Returns:
        pyarrow.RecordBatch filtrado (pode ter 0 linhas).
    """
    if not filters:
        return batch

    import pyarrow as pa
    import pyarrow.compute as pc

    normalized = normalize_filters(filters)
    if not normalized:
        return batch

    # Constrói máscara booleana AND de todos os filtros
    mask = None

    for col, op, value in normalized:
        if col not in batch.schema.names:
            continue  # coluna não existe no batch → ignora filtro para ela

        col_array = batch.column(batch.schema.get_field_index(col))

        try:
            col_mask = _build_column_mask(col_array, op, value, pc)
        except Exception:
            continue  # falha silenciosa por coluna — não filtra essa coluna

        if col_mask is None:
            continue

        mask = col_mask if mask is None else pc.and_(mask, col_mask)

    if mask is None:
        return batch

    return batch.filter(mask)


def _build_column_mask(col_array: Any, op: str, value: Any, pc: Any) -> Any:
    """Constrói máscara booleana para uma única condição."""
    if op == "==":
        return pc.equal(col_array, value)

    elif op == "!=":
        return pc.not_equal(col_array, value)

    elif op == ">":
        return pc.greater(col_array, value)

    elif op == ">=":
        return pc.greater_equal(col_array, value)

    elif op == "<":
        return pc.less(col_array, value)

    elif op == "<=":
        return pc.less_equal(col_array, value)

    elif op == "in":
        if not value:
            import pyarrow as pa
            return pa.array([False] * len(col_array))
        # isin via OR de equals
        masks = [pc.equal(col_array, v) for v in value]
        result = masks[0]
        for m in masks[1:]:
            result = pc.or_(result, m)
        return result

    elif op == "range":
        range_min, range_max = value
        return pc.and_(
            pc.greater_equal(col_array, range_min),
            pc.less_equal(col_array, range_max),
        )

    return None


# ---------------------------------------------------------------------------
# Constante de sentinela
# ---------------------------------------------------------------------------

#: Sentinela retornado por prune_row_groups quando não há índice disponível.
#: Indica ao chamador que deve ler todos os batches sem verificação de índice.
ALL_BATCHES_SENTINEL = -1
