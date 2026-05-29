"""
adapters/sql_adapter.py
=======================
Gerador Automático de Views SQL Mascaradas e Dynamic Data Masking (DDM).

O que este módulo faz
---------------------
Recebe os resultados do PIIDetector (sabe que 'cpf' é identificador direto,
'cep' é dado geográfico, 'nome' é texto sensível) e gera automaticamente
scripts SQL com Views mascaradas para PostgreSQL, SQL Server, MySQL e BigQuery.

O DBA simplesmente:
  1. Importa os dados reais na tabela `clientes` (acesso restrito)
  2. Executa o script gerado → cria a VIEW `clientes_masked`
  3. Revoga SELECT na tabela base para o role de desenvolvimento
  4. Desenvolvedores consultam apenas `clientes_masked` — veem dados descaracterizados

Estratégias por tipo de PII:
  - CPF / CNPJ / RG / Email / IP → HMAC-SHA256(salt, valor) — pseudonimização
  - Nome / Telefone / Cartão      → 'REDACTED'
  - CEP                           → LEFT(digits, 5) || '-XXX'
  - Numérico quasi-identifier     → ruído uniforme via RANDOM()

NOTA IMPORTANTE sobre HMAC no SQL:
  Nem todos os bancos expõem HMAC nativamente. Onde não há HMAC, o script
  gera SHA256 concatenado com aviso explícito no comentário SQL, para que
  o DBA saiba que aquele trecho requer revisão antes de ir a produção.
  - PostgreSQL: HMAC() via pgcrypto (preferido) com fallback encode(digest())
  - SQL Server: HASHBYTES('SHA2_256', ...) — sem HMAC nativo, aviso incluído
  - MySQL:      SHA2(..., 256) — sem HMAC nativo, aviso incluído
  - BigQuery:   SHA256() — sem HMAC nativo, aviso incluído

Bancos suportados:
  - PostgreSQL (padrão) — HMAC completo via pgcrypto
  - SQL Server (T-SQL)
  - MySQL / MariaDB
  - BigQuery (Google Cloud)

Uso:
    from datalock.adapters.sql_adapter import SQLAdapter
    from datalock.detectors.pii_detector import PIIDetector

    reports = PIIDetector().detect_dict(df)
    adapter = SQLAdapter(source_table="clientes", salt="chave-minimo-16-chars!")
    script  = adapter.generate(reports, dialect="postgresql")
    adapter.save(script, "views/clientes_masked.sql")
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from datalock.detectors.pii_detector import (
    ColumnReport, PIIType, MaskStrategy,
)
from datalock.maskers.hashing import _validate_salt

logger = logging.getLogger(__name__)


class SecurityWarning(UserWarning):
    """Aviso de prática insegura detectada no pipeline."""

# Dialetos suportados
SUPPORTED_DIALECTS = ("postgresql", "sqlserver", "mysql", "bigquery")


class SQLAdapter:
    """
    Gerador de scripts SQL com Views mascaradas.

    AVISO DE SEGURANÇA — Salt no script SQL:
    O salt HMAC é embutido no script SQL gerado. Scripts SQL frequentemente
    vão para repositórios git, Confluence, tickets e e-mails.
    Use `salt_as_variable=True` (padrão) para gerar o salt como variável
    de sessão PostgreSQL que deve ser injetada em runtime pelo vault:

        SET datalock.salt = '${DATALOCK_SALT}';  -- vault injeta em runtime
        -- Nunca comita o valor real do salt

    Para dialetos que não suportam variável de sessão, o salt é embutido
    com aviso explícito no script gerado.
    """
    """
    Gera scripts SQL com Views mascaradas a partir de relatórios do PIIDetector.

    Exemplo completo:
        detector = PIIDetector()
        reports  = detector.detect_dict(df)

        adapter = SQLAdapter(
            source_table="schema_raw.clientes",
            view_name="schema_dev.clientes_masked",
            salt="2024-lgpd-key",
        )
        script = adapter.generate(reports, dialect="postgresql")
        adapter.save(script, "output/clientes_masked.sql")
        logger.debug("SQL:\n%s", script)

    Parâmetros:
        source_table: Nome da tabela de origem (com schema se necessário).
        view_name: Nome da view a criar. Se None, usa source_table + '_masked'.
        salt: Salt para hashing SQL. Tratado como literal de string no SQL
              gerado — use chaves curtas e sem aspas simples internas.
        schema: Schema SQL onde a view será criada (opcional).
    """

    def __init__(
        self,
        source_table: str,
        view_name: Optional[str] = None,
        salt: str = "lgpd_salt",
        schema: Optional[str] = None,
        salt_as_variable: bool = True,
    ):
        # Valida força do salt — o mesmo padrão do DeterministicHasher Python
        try:
            _validate_salt(salt)
        except ValueError as exc:
            raise ValueError(
                f"SQLAdapter: salt inválido para geração de scripts SQL. {exc}"
            ) from exc

        if not salt_as_variable:
            import warnings
            warnings.warn(
                "SQLAdapter: salt_as_variable=False embute o salt em texto claro no script SQL. "
                "Scripts SQL frequentemente são commitados em repositórios git. "
                "Use salt_as_variable=True (padrão) e injete o valor real via vault em runtime. "
                "Para PostgreSQL: SET datalock.salt = '<valor-vault>';",
                SecurityWarning,
                stacklevel=2,
            )

        self.source_table    = source_table
        self.view_name       = view_name or f"{source_table}_masked"
        self.salt            = salt
        self.schema          = schema
        self.salt_as_variable = salt_as_variable

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(
        self,
        reports: Dict[str, ColumnReport],
        all_columns: Optional[list] = None,
        dialect: str = "postgresql",
        or_replace: bool = True,
    ) -> str:
        """
        Gera o script SQL completo com a View mascarada.

        Args:
            reports: Dicionário {coluna: ColumnReport} do PIIDetector.
            all_columns: Lista de todas as colunas da tabela (para incluir as
                         colunas sem risco no SELECT). Se None, usa apenas as
                         colunas detectadas + passthrough implícito.
            dialect: "postgresql" | "sqlserver" | "mysql" | "bigquery"
            or_replace: Usa CREATE OR REPLACE VIEW (PostgreSQL/BigQuery/MySQL)
                        ou DROP + CREATE (SQL Server).

        Returns:
            String com o script SQL completo, pronto para execução.
        """
        if dialect not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"Dialeto '{dialect}' não suportado. "
                f"Escolha: {SUPPORTED_DIALECTS}"
            )

        builder = _get_builder(dialect)
        columns_sql = self._build_select_columns(reports, all_columns, builder)
        ddl = builder.create_view(
            view_name=self.view_name,
            source_table=self.source_table,
            columns_sql=columns_sql,
            or_replace=or_replace,
        )
        header = self._build_header(reports, dialect)
        grants = builder.grants(self.view_name)
        return f"{header}\n{ddl}\n{grants}"

    def generate_grant_revoke(
        self,
        dev_role: str = "dev_role",
        dialect: str = "postgresql",
    ) -> str:
        """
        Gera instruções adicionais de GRANT na view e REVOKE na tabela base.
        Reforça Zero Trust: devs só acessam a view mascarada.
        """
        builder = _get_builder(dialect)
        return builder.grant_revoke(
            view_name=self.view_name,
            table_name=self.source_table,
            role=dev_role,
        )

    def save(self, script: str, path: str) -> None:
        """Salva o script SQL em arquivo."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(script, encoding="utf-8")
        logger.info("Script SQL salvo: %s", path)
        logger.info("Script SQL salvo em: %s", path)

    # ------------------------------------------------------------------
    # Construção das colunas do SELECT
    # ------------------------------------------------------------------

    def _build_select_columns(
        self,
        reports: Dict[str, ColumnReport],
        all_columns: Optional[list],
        builder: "_DialectBuilder",
    ) -> list[str]:
        col_expressions = []

        # Colunas com risco — aplicar mascaramento
        for col, report in reports.items():
            expr = self._column_expression(col, report, builder)
            col_expressions.append(f"    {expr} AS {_quote(col)}")

        # Colunas sem risco — passthrough
        if all_columns:
            masked_cols = set(reports.keys())
            for col in all_columns:
                if col not in masked_cols:
                    col_expressions.append(f"    {_quote(col)}")

        return col_expressions

    def _column_expression(
        self,
        col: str,
        report: ColumnReport,
        builder: "_DialectBuilder",
    ) -> str:
        strategy = report.mask_strategy
        q = _quote(col)

        if strategy == MaskStrategy.HASH:
            # salt_as_variable=True (padrão): emite current_setting('datalock.salt')
            # O DBA injeta o valor real via SET datalock.salt = '<vault-value>';
            # antes de executar a view — o salt nunca aparece no .sql.
            #
            # salt_as_variable=False: embute o literal (SecurityWarning emitido
            # no __init__ quando esta opção é usada explicitamente).
            if self.salt_as_variable:
                salt_sql = builder.salt_variable_expr()
            else:
                salt_sql = f"'{self.salt}'"
            return builder.hash_expr(q, salt_sql, col)

        elif strategy == MaskStrategy.TRUNCATE:
            return builder.cep_truncate_expr(q, col)

        elif strategy == MaskStrategy.REDACT:
            return f"'REDACTED'"

        elif strategy == MaskStrategy.MOCK_NUM:
            lo = report.col_min or 0.0
            hi = report.col_max or 1.0
            return builder.numeric_mock_expr(q, lo, hi, col)

        elif strategy == MaskStrategy.MOCK_CAT:
            # SQL não sorteia categorias facilmente — usa REDACTED como fallback seguro
            logger.info(
                "Coluna '%s' (MOCK_CAT) → 'CATEGORIA_MOCK' na view SQL. "
                "Para sorteio real use a camada Python (pandas_adapter).", col,
            )
            return f"'CATEGORIA_MOCK'"

        elif strategy == MaskStrategy.SUPPRESS:
            return "NULL"

        else:  # PASSTHROUGH
            return q

    def _build_header(self, reports: Dict[str, ColumnReport], dialect: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        high  = sum(1 for r in reports.values() if r.risk_level.value == "high")
        med   = sum(1 for r in reports.values() if r.risk_level.value == "medium")
        return (
            f"-- ============================================================\n"
            f"-- Privacy Framework — View Mascarada LGPD\n"
            f"-- Gerado em: {now}\n"
            f"-- Tabela origem   : {self.source_table}\n"
            f"-- View destino    : {self.view_name}\n"
            f"-- Dialeto         : {dialect}\n"
            f"-- Colunas HIGH    : {high}\n"
            f"-- Colunas MEDIUM  : {med}\n"
            f"-- ============================================================\n"
        )

    def __repr__(self) -> str:
        return (
            f"SQLAdapter(source='{self.source_table}', "
            f"view='{self.view_name}', salt='{self.salt[:4]}...')"
        )


# ---------------------------------------------------------------------------
# Builders por dialeto SQL
# ---------------------------------------------------------------------------

class _DialectBuilder:
    """Interface base para builders de SQL por dialeto."""

    def salt_variable_expr(self) -> str:
        """
        Retorna a expressão SQL que lê o salt de uma variável de sessão.

        O salt nunca aparece no script gerado — o DBA o injeta em runtime
        via mecanismo específico do banco antes de executar queries na view.
        """
        raise NotImplementedError

    def hash_expr(self, col_quoted: str, salt_sql: str, col_name: str) -> str:
        """
        Gera a expressão SQL de hash.

        Parâmetros:
            col_quoted: Coluna entre aspas (ex: "cpf").
            salt_sql:   Expressão SQL do salt — pode ser literal ('abc') ou
                        referência a variável de sessão (current_setting('datalock.salt')).
            col_name:   Nome da coluna (sem aspas), para comentários.
        """
        raise NotImplementedError

    def cep_truncate_expr(self, col_quoted: str, col_name: str) -> str:
        raise NotImplementedError

    def numeric_mock_expr(
        self, col_quoted: str, lo: float, hi: float, col_name: str
    ) -> str:
        raise NotImplementedError

    def create_view(
        self,
        view_name: str,
        source_table: str,
        columns_sql: list[str],
        or_replace: bool,
    ) -> str:
        raise NotImplementedError

    def grants(self, view_name: str) -> str:
        return (
            f"\n-- Conceda acesso apenas à view mascarada:\n"
            f"-- GRANT SELECT ON {view_name} TO <dev_role>;\n"
            f"-- REVOKE SELECT ON <source_table> FROM <dev_role>;\n"
        )

    def grant_revoke(self, view_name: str, table_name: str, role: str) -> str:
        return (
            f"GRANT  SELECT ON {view_name}  TO   {role};\n"
            f"REVOKE SELECT ON {table_name} FROM {role};\n"
        )


class _PostgreSQLBuilder(_DialectBuilder):
    """
    PostgreSQL — HMAC-SHA256 via pgcrypto.

    pgcrypto expõe hmac(data, key, type) nativamente, tornando o PostgreSQL
    o único dos 4 dialetos com suporte completo a HMAC no SQL.
    Ref: https://www.postgresql.org/docs/current/pgcrypto.html

    Com salt_as_variable=True (padrão), o DBA injeta o salt assim:
        -- Uma vez por sessão, antes de qualquer SELECT na view:
        SET datalock.salt = '<valor-do-vault>';
        SELECT * FROM schema.clientes_masked;

    Para injeção permanente via postgresql.conf ou ALTER ROLE:
        ALTER ROLE dev_role SET datalock.salt = '<valor>';
    """

    def salt_variable_expr(self) -> str:
        # current_setting() lê parâmetro de sessão definido por SET datalock.salt = '...'
        # Lança erro se não definido — comportamento desejado (falha explícita).
        return "current_setting('datalock.salt')"

    def hash_expr(self, col_quoted: str, salt_sql: str, col_name: str) -> str:
        # hmac(data, key, 'sha256') — RFC 2104, immune to length extension
        # salt_sql pode ser literal 'abc' ou current_setting('datalock.salt')
        return (
            f"LEFT(\n"
            f"      encode(\n"
            f"        hmac(COALESCE({col_quoted}::text, ''), {salt_sql}, 'sha256'),\n"
            f"        'hex'\n"
            f"      ),\n"
            f"      16\n"
            f"    )"
        )

    def cep_truncate_expr(self, col_quoted: str, col_name: str) -> str:
        return (
            f"CASE\n"
            f"      WHEN {col_quoted} ~ '^\\d{{5}}-?\\d{{3}}$'\n"
            f"      THEN LEFT(REGEXP_REPLACE({col_quoted}, '[^0-9]', '', 'g'), 5) || '-XXX'\n"
            f"      ELSE {col_quoted}\n"
            f"    END"
        )

    def numeric_mock_expr(
        self, col_quoted: str, lo: float, hi: float, col_name: str
    ) -> str:
        return (
            f"ROUND(\n"
            f"      ({lo:.4f} + RANDOM() * ({hi:.4f} - {lo:.4f}))::numeric,\n"
            f"      2\n"
            f"    )"
        )

    def create_view(
        self, view_name, source_table, columns_sql, or_replace
    ) -> str:
        prefix = "CREATE OR REPLACE VIEW" if or_replace else "CREATE VIEW"
        cols = ",\n".join(columns_sql)
        return (
            f"-- PRÉ-REQUISITO: habilitar pgcrypto (uma vez, como superuser):\n"
            f"-- CREATE EXTENSION IF NOT EXISTS pgcrypto;\n"
            f"-- pgcrypto expõe hmac() — HMAC-SHA256 completo (RFC 2104).\n\n"
            f"{prefix} {view_name} AS\n"
            f"SELECT\n{cols}\n"
            f"FROM {source_table};\n"
        )


class _SQLServerBuilder(_DialectBuilder):
    """
    Microsoft SQL Server / Azure SQL — T-SQL.

    SQL Server não expõe HMAC nativamente. HASHBYTES('SHA2_256', key+msg)
    é SHA256 concatenado (length extension attack possível em teoria).
    Para ambientes de alta segurança, use uma CLR function com System.Security
    .Cryptography.HMACSHA256, ou aplique o mascaramento na camada Python.

    Com salt_as_variable=True (padrão), o DBA injeta o salt assim:
        -- Uma vez por sessão (ou via CONTEXT_INFO):
        EXEC sp_set_session_context 'logus_salt', N'<valor-do-vault>';
        SELECT * FROM schema.clientes_masked;

    Referência: https://docs.microsoft.com/sql/t-sql/functions/hashbytes-transact-sql
    """

    def salt_variable_expr(self) -> str:
        # SQL Server usa SESSION_CONTEXT() para variáveis de sessão tipadas
        return "CAST(SESSION_CONTEXT(N'logus_salt') AS NVARCHAR(256))"

    def hash_expr(self, col_quoted: str, salt_sql: str, col_name: str) -> str:
        return (
            f"-- AVISO: SQL Server não tem HMAC nativo. SHA256 concatenado abaixo.\n"
            f"    -- Para HMAC completo, implemente via CLR (System.Security.Cryptography).\n"
            f"    -- O salt é injetado em runtime via: EXEC sp_set_session_context 'logus_salt', N'<vault>';\n"
            f"    LEFT(\n"
            f"      LOWER(CONVERT(VARCHAR(64),\n"
            f"        HASHBYTES('SHA2_256', {salt_sql} + COALESCE(CAST({col_quoted} AS NVARCHAR(MAX)), '')),\n"
            f"        2)),\n"
            f"      16\n"
            f"    )"
        )

    def cep_truncate_expr(self, col_quoted: str, col_name: str) -> str:
        return (
            f"CASE\n"
            f"      WHEN {col_quoted} LIKE '[0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9]'\n"
            f"        OR {col_quoted} LIKE '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'\n"
            f"      THEN LEFT(REPLACE({col_quoted}, '-', ''), 5) + '-XXX'\n"
            f"      ELSE {col_quoted}\n"
            f"    END"
        )

    def numeric_mock_expr(
        self, col_quoted: str, lo: float, hi: float, col_name: str
    ) -> str:
        return (
            f"ROUND(\n"
            f"      {lo:.4f} + (RAND(CHECKSUM(NEWID())) * ({hi:.4f} - {lo:.4f})),\n"
            f"      2\n"
            f"    )"
        )

    def create_view(
        self, view_name, source_table, columns_sql, or_replace
    ) -> str:
        cols = ",\n".join(columns_sql)
        drop = (
            f"-- AVISO: SQL Server usa SHA256 concatenado (sem HMAC nativo).\n"
            f"-- Para HMAC completo, use CLR com System.Security.Cryptography.HMACSHA256.\n"
            f"IF OBJECT_ID('{view_name}', 'V') IS NOT NULL DROP VIEW {view_name};\nGO\n\n"
        )
        return (
            f"{drop if or_replace else ''}"
            f"CREATE VIEW {view_name} AS\n"
            f"SELECT\n{cols}\n"
            f"FROM {source_table};\nGO\n"
        )


class _MySQLBuilder(_DialectBuilder):
    """
    MySQL / MariaDB.

    MySQL não tem HMAC nativo. SHA2(CONCAT(salt, msg), 256) é SHA256
    concatenado. Para HMAC completo, use UDF (User-Defined Function)
    ou aplique o mascaramento na camada Python (pandas_adapter).

    Com salt_as_variable=True (padrão), o DBA injeta o salt assim:
        -- Uma vez por conexão, antes de usar a view:
        SET @logus_salt = '<valor-do-vault>';
        SELECT * FROM clientes_masked;

    Referência: https://dev.mysql.com/doc/refman/8.0/en/encryption-functions.html
    """

    def salt_variable_expr(self) -> str:
        # MySQL usa user-defined variables (@variavel) com escopo de sessão
        return "@logus_salt"

    def hash_expr(self, col_quoted: str, salt_sql: str, col_name: str) -> str:
        return (
            f"/* AVISO: MySQL/MariaDB não tem HMAC nativo. SHA256 concatenado abaixo.\n"
            f"       O salt é injetado via: SET @logus_salt = '<vault-value>';\n"
            f"       Para HMAC completo, implemente via UDF ou use a camada Python. */\n"
            f"    LEFT(\n"
            f"      SHA2(CONCAT(IFNULL({salt_sql}, ''), IFNULL({col_quoted}, '')), 256),\n"
            f"      16\n"
            f"    )"
        )

    def cep_truncate_expr(self, col_quoted: str, col_name: str) -> str:
        return (
            f"CASE\n"
            f"      WHEN {col_quoted} REGEXP '^[0-9]{{5}}-?[0-9]{{3}}$'\n"
            f"      THEN CONCAT(LEFT(REPLACE({col_quoted}, '-', ''), 5), '-XXX')\n"
            f"      ELSE {col_quoted}\n"
            f"    END"
        )

    def numeric_mock_expr(
        self, col_quoted: str, lo: float, hi: float, col_name: str
    ) -> str:
        return f"ROUND({lo:.4f} + (RAND() * ({hi:.4f} - {lo:.4f})), 2)"

    def create_view(
        self, view_name, source_table, columns_sql, or_replace
    ) -> str:
        prefix = "CREATE OR REPLACE VIEW" if or_replace else "CREATE VIEW"
        cols = ",\n".join(columns_sql)
        return (
            f"-- AVISO: MySQL/MariaDB usa SHA256 concatenado (sem HMAC nativo).\n"
            f"-- Para HMAC completo, use UDF ou aplique mascaramento no Python.\n\n"
            f"{prefix} {view_name} AS\n"
            f"SELECT\n{cols}\n"
            f"FROM {source_table};\n"
        )


class _BigQueryBuilder(_DialectBuilder):
    """
    Google BigQuery (Standard SQL).

    BigQuery não tem HMAC nativo no Standard SQL. SHA256() com CONCAT
    é usado. Para HMAC completo, use BigQuery Remote Functions com Cloud Run.

    Com salt_as_variable=True (padrão), o salt é referenciado como
    parâmetro de query ou via Session System Variables (BigQuery Enterprise):
        SET @@logus_salt = '<valor-do-vault>';
        SELECT * FROM `project.dataset.clientes_masked`;

    Para Standard BigQuery sem Enterprise, passe o salt como parâmetro
    via BigQuery client SDK (@logus_salt em named parameters).

    Referência: https://cloud.google.com/bigquery/docs/reference/standard-sql/hash_functions
    """

    def salt_variable_expr(self) -> str:
        # BigQuery Enterprise: session system variable
        # Standard BigQuery: named query parameter
        return "@@logus_salt"

    def hash_expr(self, col_quoted: str, salt_sql: str, col_name: str) -> str:
        return (
            f"/* AVISO: BigQuery não tem HMAC no Standard SQL. SHA256 concatenado abaixo.\n"
            f"       O salt é injetado via: SET @@logus_salt = '<vault-value>';\n"
            f"       Para HMAC completo, use Remote Functions com Cloud Run. */\n"
            f"    LEFT(\n"
            f"      TO_HEX(SHA256(CONCAT(CAST({salt_sql} AS STRING), COALESCE(CAST({col_quoted} AS STRING), '')))),\n"
            f"      16\n"
            f"    )"
        )

    def cep_truncate_expr(self, col_quoted: str, col_name: str) -> str:
        return (
            f"CASE\n"
            f"      WHEN REGEXP_CONTAINS({col_quoted}, r'^\\d{{5}}-?\\d{{3}}$')\n"
            f"      THEN CONCAT(SUBSTR(REGEXP_REPLACE({col_quoted}, r'[^0-9]', ''), 1, 5), '-XXX')\n"
            f"      ELSE {col_quoted}\n"
            f"    END"
        )

    def numeric_mock_expr(
        self, col_quoted: str, lo: float, hi: float, col_name: str
    ) -> str:
        return f"ROUND({lo:.4f} + (RAND() * ({hi:.4f} - {lo:.4f})), 2)"

    def create_view(
        self, view_name, source_table, columns_sql, or_replace
    ) -> str:
        prefix = "CREATE OR REPLACE VIEW" if or_replace else "CREATE VIEW"
        cols = ",\n".join(columns_sql)
        return (
            f"-- AVISO: BigQuery usa SHA256 concatenado (sem HMAC nativo no Standard SQL).\n"
            f"-- Para HMAC: https://cloud.google.com/bigquery/docs/remote-functions\n\n"
            f"{prefix} `{view_name}` AS\n"
            f"SELECT\n{cols}\n"
            f"FROM `{source_table}`;\n"
        )


def _get_builder(dialect: str) -> _DialectBuilder:
    return {
        "postgresql": _PostgreSQLBuilder(),
        "sqlserver":  _SQLServerBuilder(),
        "mysql":      _MySQLBuilder(),
        "bigquery":   _BigQueryBuilder(),
    }[dialect]


def _quote(col: str) -> str:
    """
    Envolve nome de coluna em aspas duplas (padrão ANSI SQL).

    Sanitização obrigatória:
    - Duplica aspas duplas internas (escape ANSI SQL padrão).
    - Rejeita nomes com caracteres de controle (prevenção de SQL injection).

    Raises:
        ValueError: Se o nome contiver caracteres de controle (< ord 32).
    """
    if any(ord(c) < 32 for c in col):
        raise ValueError(
            f"Nome de coluna com caractere de controle inválido: {repr(col)}. "
            "Possível tentativa de SQL injection."
        )
    sanitized = col.replace('"', '""')
    return f'"{sanitized}"'
