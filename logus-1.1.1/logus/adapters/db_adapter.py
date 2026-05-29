"""
adapters/db_adapter.py
======================
SecureDBAdapter — Adapter para bancos de dados com mascaramento automático.

Suporte completo a PostgreSQL, MySQL, SQLite, SQL Server, Oracle e qualquer
banco acessível via SQLAlchemy. Três modos de operação:

  1. PULL & MASK  — lê localmente, mascara em Python, devolve DataFrame
  2. PUSH         — escreve DataFrame mascarado de volta ao banco
  3. IN-DB MASK   — mascara sem puxar dados: executa UPDATE/CREATE VIEW no banco

IN-DB MASKING (modo mais seguro):
  Os dados brutos nunca saem do banco. O mascaramento é executado via:
  - PostgreSQL: UPDATE com pgcrypto HMAC ou SHA256
  - MySQL:      UPDATE com SHA2()
  - SQLite:     UPDATE com hex(randomblob()) como proxy de hash
  Os dados brutos saem para Python apenas uma amostra de detecção (500 linhas).

Conexão:
  - URL string:     "postgresql://user:pass@host/db"
  - SQLAlchemy:     create_engine(url)
  - psycopg2:       psycopg2.connect(...)
  - pyodbc:         pyodbc.connect(...)

Cache de schema:
  O PIIDetector é custoso. O adapter cacheia o plano de detecção por
  fingerprint de schema e reutiliza em queries subsequentes da mesma tabela.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from typing import Union, List, Any, Dict, List, Literal, Optional, Union

import pandas as pd

from datalock.adapters.pandas_adapter import secure_dataframe, _MaskingEngine
from datalock.detectors.pii_detector import PIIDetector, MaskStrategy, PIIType
from datalock.utils.secret_str import SecretStr

logger = logging.getLogger(__name__)

Dialect = Literal["postgresql", "mysql", "sqlite", "sqlserver", "oracle", "bigquery"]

# ---------------------------------------------------------------------------
# SecureDBAdapter
# ---------------------------------------------------------------------------

class SecureDBAdapter:
    """
    Executa queries SQL e retorna DataFrames mascarados.

    Parâmetros:
        engine:      SQLAlchemy Engine, URL string, ou conexão direta.
        salt:        Chave HMAC para pseudonimização.
        random_state: Semente para mockers.
        detector_kwargs: Parâmetros para PIIDetector.
        cache_schema:  Cacheia plano de detecção por schema (padrão True).
        dialect:       Dialeto SQL para operações in-DB (auto-detectado se SQLAlchemy).
    """

    def __init__(
        self,
        engine: Any,
        salt: str,
        random_state: int = 42,
        detector_kwargs: Optional[Dict] = None,
        cache_schema: bool = True,
        dialect: Optional[Dialect] = None,
    ) -> None:
        self._engine = _resolve_engine(engine)
        self._raw_engine = engine  # preserva para operações in-DB
        self._salt = SecretStr(salt)
        self._random_state = random_state
        self._detector = PIIDetector(**(detector_kwargs or {}))
        self._cache_schema = cache_schema
        self._reports_cache: Dict[str, Dict] = {}
        self._dialect: Dialect = dialect or _detect_dialect(self._engine)

    # ------------------------------------------------------------------
    # 1. PULL & MASK — lê e mascara localmente
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        params: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        **read_sql_kwargs,
    ) -> pd.DataFrame:
        """
        Executa SQL e retorna resultado mascarado.

        Args:
            sql:     Instrução SQL. Use parâmetros vinculados (%s / :param).
            params:  Parâmetros da query.
            columns: Mascara apenas estas colunas (None = todas PII detectadas).
            exclude: Exclui estas colunas do mascaramento.

        Returns:
            pd.DataFrame mascarado.

        Exemplo:
            df = adapter.query(
                "SELECT * FROM clientes WHERE uf = %s",
                params=("SP",),
            )
        """
        t0 = time.perf_counter()
        df_raw = self._execute(sql, params, **read_sql_kwargs)
        if df_raw.empty:
            return df_raw

        df_safe = self._mask(df_raw, columns=columns, exclude=exclude)
        logger.info(
            "db.query | linhas=%d | cols=%d | %.3fs",
            len(df_safe), len(df_safe.columns), time.perf_counter() - t0,
        )
        return df_safe

    def query_table(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        schema: Optional[str] = None,
        mask_columns: Optional[List[str]] = None,
        exclude_mask: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Lê tabela com filtros opcionais e retorna resultado mascarado.

        Args:
            table:        Nome da tabela.
            columns:      Colunas a selecionar (None = todas).
            where:        Cláusula WHERE sem a keyword.
            limit:        Máximo de linhas.
            schema:       Schema do banco (ex: "public").
            mask_columns: Mascara apenas estas colunas.
            exclude_mask: Não mascara estas colunas.
        """
        table_ref = f'"{schema}"."{table}"' if schema else f'"{table}"'
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table_ref}"
        if where:
            sql += f" WHERE {where}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.query(sql, columns=mask_columns, exclude=exclude_mask, **kwargs)

    def query_chunked(
        self,
        sql: str,
        chunksize: int = 10_000,
        params: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Executa query em chunks, mascarando cada bloco antes do próximo.

        Minimiza dados brutos em memória — ideal para tabelas grandes.
        """
        chunks_safe: List[pd.DataFrame] = []
        reports: Optional[Dict] = None

        for chunk in pd.read_sql(sql, self._engine, params=params, chunksize=chunksize):
            if chunk.empty:
                continue
            if reports is None:
                reports = self._detect_with_cache(chunk)
                if columns:
                    reports = {k: v for k, v in reports.items() if k in columns}
                if exclude:
                    reports = {k: v for k, v in reports.items() if k not in exclude}
            from datalock.adapters.pandas_adapter import _MaskingEngine
            engine = _MaskingEngine(salt=self._salt.get(), random_state=self._random_state)
            chunks_safe.append(engine.apply(chunk, reports) if reports else chunk.copy())

        return pd.concat(chunks_safe, ignore_index=True) if chunks_safe else pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. PUSH — escreve DataFrame no banco
    # ------------------------------------------------------------------

    def write(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: Literal["fail", "replace", "append"] = "append",
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        index: bool = False,
        **to_sql_kwargs,
    ) -> int:
        """
        Escreve um DataFrame no banco (dados já mascarados pelo chamador).

        Args:
            df:        DataFrame a gravar (deve estar mascarado antes).
            table:     Nome da tabela destino.
            if_exists: Comportamento se a tabela existir: fail/replace/append.
            schema:    Schema do banco.
            chunksize: Linhas por batch (None = tudo de uma vez).
            index:     Se True, grava o índice como coluna.

        Returns:
            Número de linhas gravadas.

        Exemplo:
            df_safe = lg.mask(df, salt="chave")
            n = adapter.write(df_safe, "clientes_masked")
        """
        _require_sqlalchemy(self._engine)
        t0 = time.perf_counter()
        df.to_sql(
            table,
            self._engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            **to_sql_kwargs,
        )
        n = len(df)
        logger.info(
            "db.write | table=%s | linhas=%d | %.3fs",
            table, n, time.perf_counter() - t0,
        )
        return n

    def create_table(
        self,
        df: pd.DataFrame,
        table: str,
        *,
        schema: Optional[str] = None,
        if_exists: str = "fail",
    ) -> None:
        """
        Cria tabela no banco com o schema inferido do DataFrame.

        Se a tabela já existir, comportamento definido por if_exists:
          - "fail"    (padrão): lança ValueError
          - "replace": recria a tabela
          - "ignore":  não faz nada se já existir

        Args:
            df:        DataFrame de referência para inferir o schema.
            table:     Nome da tabela.
            schema:    Schema do banco.
            if_exists: Comportamento se tabela existir.

        Exemplos:
            banco.create_table(df, "clientes")
            banco.create_table(df, "clientes", if_exists="replace")
        """
        _require_sqlalchemy(self._engine)
        from sqlalchemy import inspect as _inspect
        insp = _inspect(self._engine)
        table_exists = insp.has_table(table, schema=schema)

        if table_exists:
            if if_exists == "fail":
                raise ValueError(f"Tabela '{table}' já existe. Use if_exists='replace' ou 'ignore'.")
            elif if_exists == "ignore":
                return
            # replace: drop and recreate via to_sql

        # Use pandas to_sql to create (replace works as CREATE + INSERT)
        df.head(0).to_sql(table, self._engine, schema=schema,
                           if_exists="replace" if if_exists == "replace" else "fail",
                           index=False)
        logger.info("db.create_table | table=%s | columns=%d", table, len(df.columns))

    def upsert(
        self,
        df: pd.DataFrame,
        table: str,
        *,
        on: Union[str, List[str]],
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
    ) -> int:
        """
        INSERT ... ON CONFLICT UPDATE (upsert) para PostgreSQL e SQLite.

        Para outros dialetos, faz DELETE por chave seguido de INSERT (fallback).

        Args:
            df:        DataFrame com os dados a inserir/atualizar.
            table:     Nome da tabela.
            on:        Coluna(s) de chave para detectar conflito.
            schema:    Schema do banco.
            chunksize: Linhas por batch.

        Returns:
            Número de linhas afetadas.

        Exemplos:
            banco.upsert(df_new, "clientes", on="cpf")
            banco.upsert(df_new, "pedidos", on=["cpf", "data"])
        """
        _require_sqlalchemy(self._engine)
        from sqlalchemy import text as _text
        on_cols = [on] if isinstance(on, str) else list(on)
        dialect = str(self._engine.dialect.name).lower()

        chunks = [df] if chunksize is None else [
            df.iloc[i:i+chunksize] for i in range(0, len(df), chunksize)
        ]
        total = 0

        for chunk in chunks:
            if dialect == "postgresql":
                total += self._upsert_postgres(chunk, table, on_cols, schema)
            elif dialect == "sqlite":
                total += self._upsert_sqlite(chunk, table, on_cols, schema)
            else:
                # Fallback: delete + insert
                total += self._upsert_delete_insert(chunk, table, on_cols, schema)

        logger.info("db.upsert | table=%s | linhas=%d", table, total)
        return total

    def _upsert_postgres(self, df, table, on_cols, schema) -> int:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy import Table, MetaData
        meta = MetaData()
        tbl  = Table(table, meta, schema=schema, autoload_with=self._engine)
        stmt = pg_insert(tbl).values(df.to_dict(orient="records"))
        update_cols = {c: stmt.excluded[c] for c in df.columns if c not in on_cols}
        stmt = stmt.on_conflict_do_update(index_elements=on_cols, set_=update_cols)
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount if result.rowcount >= 0 else len(df)

    def _upsert_sqlite(self, df, table, on_cols, schema) -> int:
        col_list = ', '.join('"' + c + '"' for c in df.columns)
        placeholders = ', '.join(['?'] * len(df.columns))
        sql = 'INSERT OR REPLACE INTO "' + table + '" (' + col_list + ') VALUES (' + placeholders + ')'
        rows = [tuple(row) for row in df.itertuples(index=False)]
        raw = self._engine.raw_connection()
        try:
            raw.cursor().executemany(sql, rows)
            raw.commit()
        finally:
            raw.close()
        return len(df)

    def _upsert_delete_insert(self, df, table, on_cols, schema) -> int:
        from sqlalchemy import text as _text
        table_ref = f"{schema}.{table}" if schema else table
        for _, row in df.iterrows():
            conds = " AND ".join([f"{c} = :{c}" for c in on_cols])
            with self._engine.begin() as conn:
                conn.execute(_text(f"DELETE FROM {table_ref} WHERE {conds}"),
                             {c: row[c] for c in on_cols})
        self.write(df, table, if_exists="append", schema=schema)
        return len(df)

    def read_and_write_masked(
        self,
        source_table: str,
        dest_table: str,
        if_exists: Literal["fail", "replace", "append"] = "replace",
        where: Optional[str] = None,
        chunksize: int = 50_000,
        schema: Optional[str] = None,
    ) -> int:
        """
        Lê uma tabela, mascara em Python e grava na tabela destino.

        Mais seguro que in_db_mask() quando o banco não suporta HMAC nativo.
        Adequado para migrações one-shot (ex: popular tabela de dev a partir de prod).

        Returns:
            Total de linhas gravadas.
        """
        total = 0
        first = True

        table_ref = f'"{schema}"."{source_table}"' if schema else f'"{source_table}"'
        sql = f"SELECT * FROM {table_ref}"
        if where:
            sql += f" WHERE {where}"

        reports: Optional[Dict] = None
        for chunk in pd.read_sql(sql, self._engine, chunksize=chunksize):
            if chunk.empty:
                continue
            if reports is None:
                reports = self._detect_with_cache(chunk)
            engine = _MaskingEngine(salt=self._salt.get(), random_state=self._random_state)
            chunk_safe = engine.apply(chunk, reports) if reports else chunk.copy()
            ie = if_exists if first else "append"
            self.write(chunk_safe, dest_table, if_exists=ie, schema=schema)
            total += len(chunk_safe)
            first = False

        logger.info(
            "db.read_and_write_masked | %s → %s | %d linhas",
            source_table, dest_table, total,
        )
        return total

    # ------------------------------------------------------------------
    # 3. IN-DB MASK — mascara sem puxar dados para Python
    # ------------------------------------------------------------------

    def in_db_mask(
        self,
        table: str,
        schema: Optional[str] = None,
        sample_size: int = 500,
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Mascara dados diretamente no banco sem puxar para Python.

        Puxo apenas uma amostra (sample_size linhas) para detectar PII,
        depois executa UPDATE SQL no banco com funções de hash nativas.
        Os dados brutos nunca saem integralmente do banco.

        Suporte:
          - PostgreSQL: HMAC via pgcrypto (forte) ou encode(sha256()) (fallback)
          - MySQL/MariaDB: SHA2(CONCAT(salt, col), 256)
          - SQLite:       hex(randomblob(8)) como aproximação (sem HMAC nativo)
          - SQL Server:   HASHBYTES('SHA2_256', CONCAT(salt, col))

        Args:
            table:       Tabela a mascarar.
            schema:      Schema (ex: "public").
            sample_size: Linhas para detecção PII (não os dados reais).
            columns:     Mascara apenas estas colunas.
            exclude:     Não mascara estas colunas.
            dry_run:     Se True, retorna os SQLs sem executar.

        Returns:
            Dict com colunas mascaradas, SQLs executados e status.

        AVISO:
            Esta operação modifica os dados originais no banco de forma
            irreversível. Faça backup antes de executar em produção.
            Use dry_run=True para revisar os SQLs antes.

        Exemplo:
            # Revisar antes
            result = adapter.in_db_mask("clientes", dry_run=True)
            print(result["sql_statements"])

            # Executar
            result = adapter.in_db_mask("clientes")
            print(f"Mascaradas: {result['columns_masked']}")
        """
        warnings.warn(
            "in_db_mask() modifica dados no banco IRREVERSIVELMENTE. "
            "Certifique-se de ter um backup antes de prosseguir. "
            "Use dry_run=True para revisar os SQLs primeiro.",
            UserWarning,
            stacklevel=2,
        )

        table_ref = f'"{schema}"."{table}"' if schema else f'"{table}"'
        sample_sql = f"SELECT * FROM {table_ref} LIMIT {sample_size}"

        try:
            sample = pd.read_sql(sample_sql, self._engine)
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao ler amostra de '{table}': {exc}"
            ) from None

        if sample.empty:
            return {"columns_masked": [], "sql_statements": [], "status": "empty_table"}

        reports = self._detector.detect_dict(sample)
        if columns:
            reports = {k: v for k, v in reports.items() if k in columns}
        if exclude:
            reports = {k: v for k, v in reports.items() if k not in exclude}

        if not reports:
            return {"columns_masked": [], "sql_statements": [], "status": "no_pii_detected"}

        salt_val = self._salt.get()
        sqls = _generate_in_db_update_sqls(
            table_ref=table_ref,
            reports=reports,
            salt=salt_val,
            dialect=self._dialect,
        )

        if dry_run:
            return {
                "columns_masked":  list(reports.keys()),
                "sql_statements":  sqls,
                "status":          "dry_run",
                "dialect":         self._dialect,
            }

        executed = []
        errors = []
        _require_sqlalchemy(self._engine)
        with self._engine.begin() as conn:
            for sql_stmt in sqls:
                try:
                    conn.execute(_text(sql_stmt))
                    executed.append(sql_stmt)
                    logger.info("in_db_mask | executed: %s...", sql_stmt[:80])
                except Exception as exc:
                    errors.append({"sql": sql_stmt, "error": str(exc)})
                    logger.error("in_db_mask | ERRO: %s | sql=%s", exc, sql_stmt[:80])

        return {
            "columns_masked":   list(reports.keys()),
            "sql_statements":   executed,
            "errors":           errors,
            "status":           "success" if not errors else "partial_error",
            "dialect":          self._dialect,
        }

    def create_masked_view(
        self,
        table: str,
        view_name: Optional[str] = None,
        schema: Optional[str] = None,
        sample_size: int = 500,
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        replace: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Cria uma VIEW mascarada no banco sem modificar os dados originais.

        Mais seguro que in_db_mask() para ambientes de produção — os dados
        originais permanecem intactos. Desenvolvedores consultam apenas a view.

        Args:
            table:      Tabela de origem com dados reais.
            view_name:  Nome da view (padrão: {table}_masked).
            schema:     Schema.
            sample_size: Linhas para detecção PII.
            columns:    Mascara apenas estas colunas na view.
            exclude:    Não mascara estas colunas na view.
            replace:    Se True, usa CREATE OR REPLACE VIEW.
            dry_run:    Retorna SQL sem executar.

        Returns:
            Dict com o SQL da view e status.

        Exemplo:
            result = adapter.create_masked_view("clientes")
            # Agora: SELECT * FROM clientes_masked
        """
        view = view_name or f"{table}_masked"
        table_ref = f'"{schema}"."{table}"' if schema else f'"{table}"'
        view_ref  = f'"{schema}"."{view}"'  if schema else f'"{view}"'

        sample_sql = f"SELECT * FROM {table_ref} LIMIT {sample_size}"
        try:
            sample = pd.read_sql(sample_sql, self._engine)
        except Exception as exc:
            raise RuntimeError(f"Falha ao ler amostra de '{table}': {exc}") from None

        if sample.empty:
            return {"view": view, "sql": "", "status": "empty_table"}

        reports = self._detector.detect_dict(sample)
        if columns:
            reports = {k: v for k, v in reports.items() if k in columns}
        if exclude:
            reports = {k: v for k, v in reports.items() if k not in exclude}

        salt_val = self._salt.get()
        view_sql = _generate_masked_view_sql(
            table_ref=table_ref,
            view_ref=view_ref,
            sample=sample,
            reports=reports,
            salt=salt_val,
            dialect=self._dialect,
            replace=replace,
        )

        if dry_run:
            return {"view": view, "sql": view_sql, "status": "dry_run", "columns_masked": list(reports)}

        _require_sqlalchemy(self._engine)
        with self._engine.begin() as conn:
            conn.execute(_text(view_sql))

        logger.info("create_masked_view | view=%s | cols=%s", view, list(reports))
        return {
            "view":            view,
            "sql":             view_sql,
            "columns_masked":  list(reports),
            "status":          "success",
        }

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def tables(self, schema: Optional[str] = None) -> List[str]:
        """Lista as tabelas disponíveis no banco."""
        try:
            from sqlalchemy import inspect as _sa_inspect
            insp = _sa_inspect(self._engine)
            return insp.get_table_names(schema=schema)
        except Exception:
            return []

    def columns(self, table: str, schema: Optional[str] = None) -> List[Dict]:
        """Retorna metadados das colunas de uma tabela."""
        try:
            from sqlalchemy import inspect as _sa_inspect
            insp = _sa_inspect(self._engine)
            return insp.get_columns(table, schema=schema)
        except Exception:
            return []

    def clear_cache(self) -> None:
        """Limpa o cache de schemas detectados."""
        self._reports_cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._reports_cache)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params: Optional[Any], **kwargs) -> pd.DataFrame:
        try:
            return pd.read_sql(sql, self._engine, params=params, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Erro ao executar query: {type(exc).__name__}: {exc}") from None

    def _mask(
        self, df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        reports = self._detect_with_cache(df)
        if columns:
            reports = {k: v for k, v in reports.items() if k in columns}
        if exclude:
            reports = {k: v for k, v in reports.items() if k not in exclude}
        if not reports:
            return df.copy()
        engine = _MaskingEngine(salt=self._salt.get(), random_state=self._random_state)
        return engine.apply(df, reports)

    def _detect_with_cache(self, df: pd.DataFrame) -> Dict:
        if not self._cache_schema:
            return self._detector.detect_dict(df)
        fp = _schema_fingerprint(df)
        if fp not in self._reports_cache:
            self._reports_cache[fp] = self._detector.detect_dict(df)
        return self._reports_cache[fp]

    def __repr__(self) -> str:
        return (
            f"SecureDBAdapter(dialect={self._dialect!r}, "
            f"salt={self._salt!r}, cache_size={self.cache_size})"
        )


# ---------------------------------------------------------------------------
# SQL generators for in-DB masking
# ---------------------------------------------------------------------------

def _generate_in_db_update_sqls(
    table_ref: str,
    reports: Dict,
    salt: str,
    dialect: Dialect,
) -> List[str]:
    """Gera UPDATE SQLs para mascaramento in-DB por dialeto."""
    sqls = []
    for col, report in reports.items():
        expr = _mask_expr(col, report, salt, dialect)
        if expr:
            sqls.append(f'UPDATE {table_ref} SET "{col}" = {expr};')
    return sqls


def _generate_masked_view_sql(
    table_ref: str,
    view_ref: str,
    sample: pd.DataFrame,
    reports: Dict,
    salt: str,
    dialect: Dialect,
    replace: bool = True,
) -> str:
    """Gera CREATE VIEW com colunas mascaradas inline."""
    create = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
    select_parts = []
    for col in sample.columns:
        if col in reports:
            expr = _mask_expr(col, reports[col], salt, dialect)
            select_parts.append(f'  {expr} AS "{col}"')
        else:
            select_parts.append(f'  "{col}"')
    cols_sql = ",\n".join(select_parts)
    return f"{create} {view_ref} AS\nSELECT\n{cols_sql}\nFROM {table_ref};"


def _mask_expr(col: str, report: Any, salt: str, dialect: Dialect) -> Optional[str]:
    """Gera expressão SQL de mascaramento para uma coluna."""
    strategy = report.mask_strategy
    c = f'"{col}"'
    salt_escaped = salt.replace("'", "''")  # escape SQL

    if strategy == MaskStrategy.HASH:
        return _hash_expr(c, salt_escaped, dialect)

    if strategy == MaskStrategy.REDACT:
        return "'REDACTED'"

    if strategy == MaskStrategy.TRUNCATE:
        # CEP: primeiros 5 dígitos + -XXX
        if dialect == "postgresql":
            return f"CASE WHEN {c} ~ '^[0-9]{{8}}$' THEN substring(regexp_replace({c}, '[^0-9]', '', 'g'), 1, 5) || '-XXX' ELSE {c} END"
        return f"CONCAT(LEFT(REGEXP_REPLACE({c}, '[^0-9]', ''), 5), '-XXX')"

    if strategy == MaskStrategy.SUPPRESS:
        return "NULL"

    if strategy == MaskStrategy.GENERALIZE_DATE:
        if dialect == "postgresql":
            return f"CASE WHEN {c} IS NOT NULL THEN CONCAT(((EXTRACT(YEAR FROM {c}::date)::int / 10) * 10)::text, '-', ((EXTRACT(YEAR FROM {c}::date)::int / 10) * 10 + 9)::text) ELSE NULL END"
        return f"CASE WHEN {c} IS NOT NULL THEN CONCAT(FLOOR(YEAR({c}) / 10) * 10, '-', FLOOR(YEAR({c}) / 10) * 10 + 9) ELSE NULL END"

    if strategy == MaskStrategy.MOCK_NUM:
        # Adiciona ruído uniforme de ±10% do valor
        if dialect == "postgresql":
            return f"ROUND(({c} * (0.9 + RANDOM() * 0.2))::numeric, 2)"
        return f"ROUND({c} * (0.9 + RAND() * 0.2), 2)"

    return None  # PASSTHROUGH, MOCK_CAT (sem suporte nativo), etc.


def _hash_expr(col_expr: str, salt: str, dialect: Dialect) -> str:
    """Expressão SQL de hash por dialeto."""
    if dialect == "postgresql":
        return (
            f"encode(hmac({col_expr}::text::bytea, '{salt}'::bytea, 'sha256'), 'hex')"
            # Fallback sem pgcrypto:
            # f"encode(digest('{salt}' || {col_expr}::text, 'sha256'), 'hex')"
        )
    if dialect == "mysql":
        return f"SHA2(CONCAT('{salt}', COALESCE({col_expr}, '')), 256)"
    if dialect == "sqlserver":
        return f"CONVERT(NVARCHAR(64), HASHBYTES('SHA2_256', CONCAT(N'{salt}', COALESCE({col_expr}, N''))), 2)"
    if dialect == "sqlite":
        # SQLite não tem SHA256 nativo — usa hex(randomblob()) como proxy
        return f"lower(hex(randomblob(8)))"
    if dialect == "bigquery":
        return f"TO_HEX(SHA256(CONCAT('{salt}', COALESCE(CAST({col_expr} AS STRING), ''))))"
    # oracle / fallback
    return f"STANDARD_HASH('{salt}' || {col_expr}, 'SHA256')"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_engine(engine: Any) -> Any:
    """Aceita URL string ou engine SQLAlchemy. Retorna engine."""
    if isinstance(engine, str):
        try:
            from sqlalchemy import create_engine
            return create_engine(engine)
        except ImportError:
            raise ImportError(
                "Para usar URL de conexão, instale SQLAlchemy: pip install sqlalchemy"
            ) from None
    return engine


def _detect_dialect(engine: Any) -> Dialect:
    """Auto-detecta o dialeto SQL a partir do engine SQLAlchemy."""
    try:
        name = engine.dialect.name.lower()
        if "postgres" in name or "pg" in name:
            return "postgresql"
        if "mysql" in name or "mariadb" in name:
            return "mysql"
        if "sqlite" in name:
            return "sqlite"
        if "mssql" in name or "sqlserver" in name:
            return "sqlserver"
        if "oracle" in name:
            return "oracle"
        if "bigquery" in name:
            return "bigquery"
    except AttributeError:
        pass
    return "postgresql"  # default seguro


def _require_sqlalchemy(engine: Any) -> None:
    if not hasattr(engine, "begin"):
        raise TypeError(
            "Esta operação requer SQLAlchemy Engine. "
            "Crie com: from sqlalchemy import create_engine; engine = create_engine(url)"
        )


def _text(sql: str) -> Any:
    from sqlalchemy import text
    return text(sql)


def _schema_fingerprint(df: pd.DataFrame) -> str:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()
