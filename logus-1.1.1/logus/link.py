"""
logus.link — Conectores para bancos de dados e geração de SQL.

    from datalock import link

    # Conecta ao banco (PostgreSQL, MySQL, SQLite, SQL Server, etc.)
    adapter = link.db("postgresql://user:pass@host/db", salt="chave")

    # 1. Lê e mascara localmente
    df = adapter.query("SELECT * FROM clientes")
    df = adapter.query_table("clientes", where="uf = 'SP'", limit=10000)

    # 2. Escreve DataFrame mascarado de volta
    df_safe = lg.mask(df, salt="chave")
    adapter.write(df_safe, "clientes_masked")

    # 3. In-DB masking — dados nunca saem do banco (mais seguro)
    result = adapter.in_db_mask("clientes", dry_run=True)   # revisa SQLs
    result = adapter.in_db_mask("clientes")                 # executa

    # 4. View mascarada — original intacto, dev vê apenas a view
    result = adapter.create_masked_view("clientes")
    # Agora: SELECT * FROM clientes_masked

    # 5. Gera script SQL com views mascaradas
    reports = lg.scan(df)
    script = link.sql(df, reports, table="clientes", dialect="postgresql")
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

from datalock.adapters.db_adapter import SecureDBAdapter
from datalock.adapters.sql_adapter import SQLAdapter

__all__ = [
    "db", "sql", "SecureDBAdapter", "SQLAdapter",
]


def db(
    engine: Union[str, Any],
    salt: str,
    dialect: Optional[str] = None,
    **kwargs,
) -> SecureDBAdapter:
    """
    Cria um SecureDBAdapter conectado ao banco.

    Args:
        engine:  URL de conexão ou SQLAlchemy Engine.
                 URLs suportadas:
                   "postgresql://user:pass@host:5432/db"
                   "mysql+pymysql://user:pass@host/db"
                   "sqlite:///arquivo.db"
                   "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17"
        salt:    Chave HMAC para pseudonimização (≥ 16 chars).
        dialect: Auto-detectado se engine for SQLAlchemy.
                 Forçar: "postgresql" | "mysql" | "sqlite" | "sqlserver"
        **kwargs: Repassado ao SecureDBAdapter (cache_schema, detector_kwargs, etc.)

    Returns:
        SecureDBAdapter com métodos query(), write(), in_db_mask(), create_masked_view()

    Exemplos:
        # PostgreSQL
        adapter = link.db("postgresql://user:pass@localhost/mydb", salt="chave")
        df = adapter.query("SELECT * FROM clientes LIMIT 100")

        # SQLite local
        adapter = link.db("sqlite:///dados.db", salt="chave")

        # MySQL
        adapter = link.db("mysql+pymysql://user:pass@host/db", salt="chave")

        # SQL Server
        adapter = link.db(
            "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server",
            salt="chave"
        )
    """
    return SecureDBAdapter(engine=engine, salt=salt, dialect=dialect, **kwargs)


def sql(
    df: Any,
    reports: dict,
    table: str,
    dialect: str = "postgresql",
    view_name: Optional[str] = None,
    salt: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Gera script SQL com CREATE VIEW mascarada.

    Args:
        df:        DataFrame de referência (para inferir tipos).
        reports:   dict[str, ColumnReport] do lg.scan().
        table:     Nome da tabela de origem.
        dialect:   "postgresql" | "sqlserver" | "mysql" | "bigquery".
        view_name: Nome da view (padrão: {table}_masked).
        salt:      Salt HMAC (embutido na view — use variável de ambiente em prod).

    Returns:
        Script SQL como string.

    Exemplo:
        reports = lg.scan(df)
        script = link.sql(df, reports, table="clientes", dialect="postgresql")
        print(script)
        # → CREATE OR REPLACE VIEW clientes_masked AS ...
    """
    adapter = SQLAdapter(
        source_table=table,
        view_name=view_name or f"{table}_masked",
        dialect=dialect,
        salt=salt or "",
        **kwargs,
    )
    return adapter.generate(reports)
