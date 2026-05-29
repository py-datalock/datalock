"""
datalock/sql_transpiler.py
=======================
SQL Transpiler — dd.mask_sql()

Transforma queries SQL para que colunas PII sejam automaticamente
mascaradas na própria query, sem mover dados para Python.

Uso:
    reports = dd.scan(df)
    masked_sql = dd.mask_sql(
        "SELECT cpf, nome, email, renda_mensal FROM clientes WHERE uf = 'SP'",
        reports=reports,
        dialect="postgresql",
    )
    # SQL retornado:
    # SELECT
    #   encode(hmac(cpf::text, 'salt', 'sha256'), 'hex') AS cpf,
    #   'REDACTED' AS nome,
    #   encode(hmac(email::text, 'salt', 'sha256'), 'hex') AS email,
    #   renda_mensal    ← não é PII, mantida
    # FROM clientes
    # WHERE uf = 'SP'

Dialetos suportados: postgresql, mysql, sqlite, sqlserver, bigquery, duckdb
"""
from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional, Set

from datalock.detectors.pii_detector import MaskStrategy, PIIType


# ---------------------------------------------------------------------------
# Templates SQL por dialeto e estratégia
# ---------------------------------------------------------------------------

_HASH_TEMPLATES = {
    "postgresql":  "encode(hmac({col}::text, {salt}, 'sha256'), 'hex')",
    "mysql":       "SHA2(CONCAT({col}, {salt}), 256)",
    "sqlite":      "hex({col} || {salt})",   # SQLite sem SHA256 nativo
    "sqlserver":   "CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', CAST({col} AS NVARCHAR) + {salt}), 2)",
    "bigquery":    "TO_HEX(SHA256(CONCAT(CAST({col} AS STRING), {salt})))",
    "duckdb":      "hex(sha256(CONCAT({col}::TEXT, {salt})))",
}

_REDACT_TEMPLATES = {
    "postgresql":  "'REDACTED'",
    "mysql":       "'REDACTED'",
    "sqlite":      "'REDACTED'",
    "sqlserver":   "N'REDACTED'",
    "bigquery":    "'REDACTED'",
    "duckdb":      "'REDACTED'",
}

_TRUNCATE_CEP = {
    "postgresql":  "CONCAT(LEFT({col}::text, 5), '-XXX')",
    "mysql":       "CONCAT(LEFT({col}, 5), '-XXX')",
    "sqlite":      "SUBSTR({col}, 1, 5) || '-XXX'",
    "sqlserver":   "CONCAT(LEFT(CAST({col} AS VARCHAR), 5), '-XXX')",
    "bigquery":    "CONCAT(SUBSTR(CAST({col} AS STRING), 1, 5), '-XXX')",
    "duckdb":      "CONCAT(SUBSTR({col}::TEXT, 1, 5), '-XXX')",
}

_GENERALIZE_DATE = {
    "postgresql":  "TO_CHAR({col}, 'YYYY-MM') || '-XX'",
    "mysql":       "DATE_FORMAT({col}, '%Y-%m-XX')",
    "sqlite":      "SUBSTR({col}, 1, 7) || '-XX'",
    "sqlserver":   "FORMAT({col}, 'yyyy-MM') + '-XX'",
    "bigquery":    "CONCAT(FORMAT_DATE('%Y-%m', {col}), '-XX')",
    "duckdb":      "STRFTIME({col}::DATE, '%Y-%m') || '-XX'",
}

_PHONE_MASK = {
    "postgresql":  "CONCAT('(', LEFT({col}::text, 2), ') XXXXX-XXXX')",
    "mysql":       "CONCAT('(', LEFT({col}, 2), ') XXXXX-XXXX')",
    "sqlite":      "'(' || SUBSTR({col}, 1, 2) || ') XXXXX-XXXX'",
    "sqlserver":   "CONCAT('(', LEFT(CAST({col} AS VARCHAR), 2), ') XXXXX-XXXX')",
    "bigquery":    "CONCAT('(', SUBSTR(CAST({col} AS STRING), 1, 2), ') XXXXX-XXXX')",
    "duckdb":      "CONCAT('(', SUBSTR({col}::TEXT, 1, 2), ') XXXXX-XXXX')",
}


# ---------------------------------------------------------------------------
# Engine de transpilação
# ---------------------------------------------------------------------------

def mask_sql(
    query: str,
    *,
    reports: Dict[str, Any],
    dialect: str = "postgresql",
    salt: Optional[str] = None,
    view_name: Optional[str] = None,
    table_alias: Optional[str] = None,
    annotate: bool = True,
) -> str:
    """
    Transpila um SELECT SQL adicionando mascaramento inline para colunas PII.

    As transformações são aplicadas diretamente no SQL — os dados nunca
    saem do banco. Útil para criar views mascaradas.

    Args:
        query:       SELECT SQL a transformar.
        reports:     Dict[str, ColumnReport] do dd.scan().
        dialect:     "postgresql" | "mysql" | "sqlite" | "sqlserver" | "bigquery" | "duckdb"
        salt:        Salt HMAC para hash. None → usa '${DATALOCK_SALT}' como placeholder.
        view_name:   Se fornecido, encapsula em CREATE OR REPLACE VIEW.
        table_alias: Prefixo de tabela usado nos SELECTs (ex: "c" em "c.cpf").
        annotate:    Se True, adiciona comentários SQL indicando estratégia.

    Returns:
        SQL transformado como string.

    Exemplo:
        reports = dd.scan(df)
        sql_safe = dd.mask_sql(
            "SELECT cpf, email, renda_mensal, uf FROM clientes",
            reports=reports,
            dialect="postgresql",
            salt=SALT,
        )
    """
    dialect = dialect.lower().strip()
    if dialect not in _HASH_TEMPLATES:
        raise ValueError(
            f"Dialeto '{dialect}' não suportado. "
            f"Use: {list(_HASH_TEMPLATES)}"
        )

    salt_literal = f"'{salt}'" if salt else "'${DATALOCK_SALT}'"

    # Extrai as colunas do SELECT
    cols_in_query = _extract_select_columns(query)
    pii_cols = set(reports.keys())
    alias = f"{table_alias}." if table_alias else ""

    # Gera expressões mascaradas para cada coluna PII no SELECT
    masked_exprs: Dict[str, str] = {}
    for col in cols_in_query:
        bare_col = col.split(".")[-1].strip().lower()
        if bare_col not in {c.lower() for c in pii_cols}:
            continue

        # Encontra o report correspondente (case-insensitive)
        report = next((r for c, r in reports.items() if c.lower() == bare_col), None)
        if report is None:
            continue

        expr = _build_mask_expr(
            col=col, report=report, dialect=dialect,
            salt_literal=salt_literal, annotate=annotate,
        )
        masked_exprs[col] = expr

    if not masked_exprs:
        if annotate:
            return f"-- datalock: nenhuma coluna PII detectada no SELECT\n{query}"
        return query

    # Reconstrói o SELECT substituindo as colunas PII
    result_sql = _replace_select_columns(query, masked_exprs, annotate=annotate)

    if view_name:
        if dialect in ("postgresql", "sqlite", "duckdb"):
            result_sql = f"CREATE OR REPLACE VIEW {view_name} AS\n{result_sql}"
        elif dialect == "mysql":
            result_sql = f"CREATE OR REPLACE VIEW {view_name} AS\n{result_sql}"
        elif dialect == "sqlserver":
            result_sql = (
                f"CREATE OR ALTER VIEW {view_name} AS\n{result_sql}"
            )
        elif dialect == "bigquery":
            result_sql = f"CREATE OR REPLACE VIEW {view_name} AS\n{result_sql}"

    if annotate:
        header = (
            f"-- Generated by logus {_logus_version()} — mask_sql(dialect='{dialect}')\n"
            f"-- Mascaramento: {', '.join(masked_exprs.keys())}\n"
            f"-- AVISO: nunca commite o salt diretamente. Use variável de ambiente.\n"
        )
        result_sql = header + result_sql

    return result_sql


def generate_view(
    df: Any,
    table: str,
    *,
    reports: Dict[str, Any],
    dialect: str = "postgresql",
    salt: Optional[str] = None,
    view_suffix: str = "_masked",
) -> str:
    """
    Gera CREATE VIEW completo para uma tabela, mascarando todas as colunas PII.

    Todas as colunas não-PII são incluídas como estão.
    As colunas PII recebem o mascaramento adequado por tipo.

    Args:
        df:       DataFrame de referência (para obter todas as colunas).
        table:    Nome da tabela no banco.
        reports:  Dict[str, ColumnReport] do dd.scan().
        dialect:  Dialeto SQL.
        salt:     Salt HMAC.
        view_suffix: Sufixo para o nome da view (padrão "_masked").

    Returns:
        SQL CREATE VIEW completo.
    """
    try:
        cols = list(df.columns)
    except Exception:
        cols = list(reports.keys())

    salt_literal = f"'{salt}'" if salt else "'${DATALOCK_SALT}'"
    dialect = dialect.lower()

    select_parts = []
    masked_cols = []

    for col in cols:
        bare = col.lower()
        report = next((r for c, r in reports.items() if c.lower() == bare), None)

        if report is None:
            select_parts.append(f"    {col}")
        else:
            expr = _build_mask_expr(
                col=col, report=report, dialect=dialect,
                salt_literal=salt_literal, annotate=False,
            )
            select_parts.append(f"    {expr} AS {col}")
            masked_cols.append(col)

    view_name = f"{table}{view_suffix}"
    cols_sql = ",\n".join(select_parts)

    if dialect in ("postgresql", "duckdb", "mysql"):
        create = f"CREATE OR REPLACE VIEW {view_name} AS"
    elif dialect == "sqlserver":
        create = f"CREATE OR ALTER VIEW {view_name} AS"
    else:
        create = f"CREATE VIEW IF NOT EXISTS {view_name} AS"

    header = (
        f"-- Generated by logus {_logus_version()} — generate_view()\n"
        f"-- Tabela: {table}  →  View: {view_name}\n"
        f"-- Colunas mascaradas: {', '.join(masked_cols)}\n"
        f"-- Dialeto: {dialect}\n"
    )

    return f"{header}{create}\nSELECT\n{cols_sql}\nFROM {table};\n"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _build_mask_expr(col: str, report: Any, dialect: str,
                     salt_literal: str, annotate: bool) -> str:
    """Gera a expressão SQL de mascaramento para uma coluna."""
    strategy = report.mask_strategy

    if strategy == MaskStrategy.HASH:
        tmpl = _HASH_TEMPLATES[dialect]
    elif strategy == MaskStrategy.REDACT:
        tmpl = _REDACT_TEMPLATES[dialect]
        if annotate:
            return f"{tmpl} /* REDACTED: {report.pii_type.value} */"
        return tmpl
    elif strategy == MaskStrategy.TRUNCATE:
        tmpl = _TRUNCATE_CEP[dialect]
    elif strategy == MaskStrategy.GENERALIZE_DATE:
        tmpl = _GENERALIZE_DATE[dialect]
    elif strategy == MaskStrategy.MASK_PHONE_DDD:
        tmpl = _PHONE_MASK[dialect]
    else:
        # Fallback: hash para qualquer estratégia não mapeada
        tmpl = _HASH_TEMPLATES[dialect]

    expr = tmpl.format(col=col, salt=salt_literal)
    if annotate:
        expr += f" /* {strategy.value}: {report.pii_type.value} */"
    return expr


def _extract_select_columns(query: str) -> List[str]:
    """Extrai as colunas listadas no SELECT (heurística simples)."""
    # Normaliza espaços
    q = re.sub(r"\s+", " ", query.strip())

    # Encontra a parte entre SELECT e FROM
    m = re.search(r"(?i)\bSELECT\b(.*?)\bFROM\b", q, re.DOTALL)
    if not m:
        return []

    select_clause = m.group(1).strip()

    # SELECT * — não há colunas explícitas
    if select_clause.strip() == "*":
        return []

    # Divide por vírgulas (ignora vírgulas dentro de funções com parênteses)
    parts = _split_columns(select_clause)
    result = []
    for part in parts:
        part = part.strip()
        # Remove alias: "col AS alias" → "col"
        alias_m = re.search(r"(?i)\bAS\b\s+\w+$", part)
        if alias_m:
            part = part[:alias_m.start()].strip()
        # Remove qualificadores de tabela: "t.col" → "t.col" (mantém)
        result.append(part)
    return result


def _split_columns(select_clause: str) -> List[str]:
    """Divide SELECT clause por vírgulas respeitando parênteses."""
    parts = []
    depth = 0
    current = ""
    for ch in select_clause:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def _replace_select_columns(query: str, masked_exprs: Dict[str, str],
                             annotate: bool) -> str:
    """Substitui colunas no SELECT pelas expressões mascaradas."""
    # Encontra o trecho SELECT...FROM
    m = re.search(r"(?i)(\bSELECT\b)(.*?)(\bFROM\b)", query, re.DOTALL)
    if not m:
        return query

    select_kw    = m.group(1)
    select_cols  = m.group(2)
    from_onwards = m.group(3) + query[m.end():]

    cols = _split_columns(select_cols)
    new_cols = []
    for part in cols:
        original = part.strip()
        # Detecta o nome base da coluna
        bare = original.split(".")[-1].strip().lower()
        alias_m = re.search(r"(?i)\bAS\b\s+(\w+)$", original)
        bare_no_alias = (original[:alias_m.start()].strip().split(".")[-1].lower()
                         if alias_m else bare)

        # Procura nos masked_exprs (case-insensitive)
        matched_expr = next(
            (expr for col_key, expr in masked_exprs.items()
             if col_key.lower() == bare_no_alias),
            None,
        )
        if matched_expr:
            # Mantém alias original se existia
            col_name = original[:alias_m.start()].strip() if alias_m else original
            alias_name = alias_m.group(1) if alias_m else col_name.split(".")[-1]
            new_cols.append(f"    {matched_expr} AS {alias_name}")
        else:
            new_cols.append(f"    {original}")

    new_select = ",\n".join(new_cols)
    return f"{select_kw}\n{new_select}\n{from_onwards}"


def _logus_version() -> str:
    try:
        import datalock as lg
        return dd.__version__
    except Exception:
        return "1.1.0"
