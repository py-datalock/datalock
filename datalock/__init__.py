"""
datalock — Privacy-by-Design para dados tabulares. LGPD compliance em Python.

Uso rápido:
    import datalock as lg

    df = dd.read("clientes.csv")
    df_safe = dd.mask(df, salt=SALT)
    dd.store(df_safe, "clientes.dlk", key=KEY)

    # Pipeline fluente
    result = (
        dd.pipe("clientes.parquet")
        .where(uf="SP", tipo_pessoa="PF")
        .add_column(imposto=dd.col("renda_mensal") * 0.27)
        .mask(salt=SALT)
        .collect()
    )
"""
from __future__ import annotations

__version__ = "1.1.2"

# Backward compat: "import logus as lg" still works
from datalock._logus_compat import *  # noqa: F401, F403

# ---------------------------------------------------------------------------
# Imports internos — ordem importa para evitar circular
# ---------------------------------------------------------------------------

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl

from datalock import core as _core
from datalock.detectors.pii_detector import (
    PIIDetector,
    PIIType,
    MaskStrategy,
    RiskLevel,
    ColumnReport,
)
from datalock.utils.salt import generate_salt, generate_salt_hex
from datalock.analytics import write as _analytics_write
from datalock.secure_file import SecureFile
from datalock.adapters.pandas_adapter import IdempotencyError

# dd.col = pl.col — acesso a todos os métodos nativos Polars (rank, over, str.*, dt.*, etc.)
# dd.when, dd.lit, dd.concat_str também são wrappers Polars nativos
col        = pl.col
lit        = pl.lit
concat_str = pl.concat_str

# dd.when vem de analytics — wrapper que funciona sem import polars
from datalock.analytics import when

# Todas as funções analíticas
from datalock.analytics import (
    # Filtragem
    where, filter_, query,
    # Seleção / reshape
    select, drop, rename,
    # Ordenação / limite
    sort, order_by, head, tail, limit, sample,
    # Agrupamento
    groupby, group_by, top_n,
    # Deduplicação
    unique, distinct, drop_duplicates,
    # Combinação
    concat, union_all, pivot, melt, unpivot,
    # Nulos
    fill_null, fillna, coalesce,
    # Tipo / transformação
    cast, add_column, with_column, assign, apply, clip,
    # Séries temporais
    shift, lag, lead, explode,
    # Séries temporais e dados nested
    shift, lag, lead, explode,
    # Informação / schema
    describe, schema, dtypes, info, shape,
    count, count_nulls, null_counts, nunique, isnull,
    value_counts, corr, cols,
    # Conversão de tipo
    to_pandas, to_polars,
)

# Aliases SQL/pandas
q = where      # curto: dd.q(df, uf="SP")
save  = None   # definido após store() abaixo
load  = None   # definido após read() abaixo

# Sub-namespaces
from datalock.adapters import sql_adapter as _sql_mod
from datalock import check, link
from datalock import lineage
from datalock.viewer import view
from datalock.canary import canary_check, canary_info as _canary_info_fn, save_to_manifest as _save_canary_manifest

def canary_info(header: dict) -> Any:
    meta = header.get('metadata', {}) if isinstance(header, dict) else {}
    return (header.get('canary') if isinstance(header, dict) else None) or meta.get('canary')
from datalock.scan_directory import scan_directory, DirectoryInventory
from datalock.generators.synthetic import SyntheticGenerator
from datalock import asymmetric
from datalock.contract import contract, DataContract, FieldSpec, ContractDiff
from datalock.reports.compliance_report import build_compliance_report as _build_cr
from datalock.validate import validate_schema, save_rules, load_rules
from datalock.asymmetric import (
    generate_keypair, save_keypair, load_private_key, load_public_key,
    public_key_to_pem, public_key_from_pem,
)
from datalock.secure_file import ExpiredFileError
from datalock.io_big import (
    read_partial as _read_partial,
    build_csv_index, load_csv_index,
    DatabaseConnection,
)
from datalock.processor import process, ProcessResult
from datalock import validate as _validate_module
from datalock import sql_transpiler as _sql_transpiler_module
from datalock import privacy_score as _privacy_score_module
from datalock.validate import validate, expect, ValidationReport
from datalock.sql_transpiler import mask_sql, generate_view

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# __all__ — API pública completa em um único lugar
# ---------------------------------------------------------------------------

__all__ = [
    # Privacidade
    "scan", "mask", "diff", "profile", "join",
    # I/O
    "read", "store", "write", "stream", "open",
    "inspect", "rekey", "read_db",
    "save", "load",
    # Segurança
    "generate_salt", "generate_salt_hex", "configure",
    # Expressões
    "col", "lit", "concat_str", "when",
    # Texto
    "scan_text", "mask_text",
    # Sintético
    "clone", "train", "sandbox",
    # Classes
    "SecureFile", "PIIDetector", "PIIType", "MaskStrategy",
    "RiskLevel", "ColumnReport", "IdempotencyError", "DlkFile",
    # Sub-namespaces
    "pl", "check", "link",
    # Filtragem
    "where", "filter_", "query", "q",
    # Seleção
    "select", "drop", "rename", "cols", "add_column", "with_column",
    "assign", "cast", "fill_null", "fillna", "coalesce", "clip", "apply",
    # Ordenação
    "sort", "order_by", "head", "tail", "limit", "sample",
    # Agrupamento
    "groupby", "group_by", "top_n", "unique", "distinct",
    "drop_duplicates", "concat", "union_all", "pivot", "melt", "unpivot",
    # Info
    "describe", "schema", "dtypes", "info", "shape",
    "count", "count_nulls", "null_counts", "nunique", "isnull",
    "value_counts", "corr",
    # Conversão
    "to_pandas", "to_polars",
    # Pipeline e SQL
    "pipe", "sql",
    # v1.1.2 (datalock) — canary, scan_directory, synthetic, audit webhook
    'view', 'canary_check', 'canary_info', 'scan_directory', 'DirectoryInventory', 'SyntheticGenerator',
    # v1.1.0 — 10 new features
    'contract', 'DataContract', 'FieldSpec', 'ContractDiff',
    'validate_schema', 'save_rules', 'load_rules',
    'compliance_report',
    'asymmetric', 'generate_keypair', 'save_keypair',
    'load_private_key', 'load_public_key',
    'public_key_to_pem', 'public_key_from_pem',
    'ExpiredFileError',
    'shift', 'lag', 'lead', 'explode',
    # v1.0.5
    'db', 'DatabaseConnection',
    'build_csv_index', 'load_csv_index',
    # v1.2.0
    'process', 'ProcessResult',
    # Novas features v1.1.0
    "validate", "expect", "mask_sql", "generate_view",
    "lineage", "ValidationReport",
]


# ---------------------------------------------------------------------------
# scan() — detecta PII
# ---------------------------------------------------------------------------

def scan(
    source: Union[str, pd.DataFrame, pl.DataFrame],
    *,
    key: Optional[str] = None,
    sample_size: int = 500,
    threshold: float = 0.5,
    sensitive: bool = False,
) -> Union[Dict, tuple]:
    """
    Detecta e classifica colunas com dados pessoais (PII).

    Aceita pd.DataFrame, pl.DataFrame ou caminho de arquivo
    (.csv, .parquet, .xlsx, .json, .dlk).

    Args:
        source:      DataFrame ou caminho de arquivo.
        key:         Chave de decriptação (necessária para .dlk cifrado).
        sample_size: Linhas amostradas para detecção PII (padrão 500).
        threshold:   Match ratio mínimo para classificar como PII.
        sensitive:   Se True, também executa SensitiveDataDetector.

    Returns:
        Dict[str, ColumnReport] — uma entrada por coluna PII detectada.
        Ou (Dict, findings) se sensitive=True.

    Exemplos:
        reports = dd.scan(df)
        for col, r in reports.items():
            print(f"{col}: tipo={r.pii_type.value} risco={r.risk_level.value}")

        reports = dd.scan("clientes.parquet")
        reports = dd.scan("clientes.dlk", key=KEY)
    """
    if isinstance(source, (str, Path)):
        p = Path(str(source))
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {source}")
        if p.suffix.lower() == ".dlk":
            raw = read(p, key=key, raw=True)
            if isinstance(raw, dict):
                raw = pd.concat(list(raw.values()), ignore_index=True)
            df = pl.from_pandas(raw) if isinstance(raw, pd.DataFrame) else raw
        else:
            df = _core.read_file(p)
    elif isinstance(source, pd.DataFrame):
        df = pl.from_pandas(source)
    else:
        df = source  # já é pl.DataFrame

    from datalock.detectors.fast_scan import FastPIIScanner as _FastScanner
    reports = _FastScanner(sample_size=sample_size, match_threshold=threshold).detect_dict(df)

    if sensitive:
        from datalock.detectors.sensitive_detector import SensitiveDataDetector
        findings = SensitiveDataDetector().detect(df.to_pandas())
        return reports, findings
    return reports


# ---------------------------------------------------------------------------
# mask() — mascara PII
# ---------------------------------------------------------------------------

def mask(
    df: Union[pd.DataFrame, pl.DataFrame, "pl.LazyFrame"],
    *,
    salt: Optional[str] = None,
    random_state: int = 42,
    strict: bool = True,
    verbose: bool = False,
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    risk: Optional[str] = None,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Aplica mascaramento PII. Engine Polars internamente — vetorizado e rápido.

    Preserva o tipo de entrada: pd.DataFrame → pd.DataFrame, pl.DataFrame → pl.DataFrame.

    Args:
        df:           DataFrame a mascarar.
        salt:         Chave HMAC (256 bits mínimo). None → salt aleatório + aviso.
        random_state: Semente para mockers numéricos e categóricos.
        strict:       Se True, levanta IdempotencyError se coluna já parecer mascarada.
        verbose:      Imprime relatório de detecção PII.
        columns:      Mascara apenas estas colunas (None = todas PII detectadas).
        exclude:      Não mascara estas colunas.

    Returns:
        DataFrame mascarado, mesmo tipo do input.

    Determinismo:
        O mesmo valor + mesmo salt sempre gera o mesmo token HMAC-SHA256.
        Essencial para JOINs entre tabelas mascaradas em momentos distintos.

    Normalização:
        CPF: "111.444.777-35", "11144477735", "111-444-777.35" → mesmo token.

    risk:
        Mascaramento risk-aware — aplica estratégia por nível de risco sem precisar
        especificar colunas explicitamente:
          'high'   → suprime (null) identificadores de alto risco
          'medium' → hash para risco médio, truncate/redact para baixo
          'low'    → truncate/generalize — preserva utilidade

    Exemplos:
        df_safe = dd.mask(df, salt=SALT)
        df_safe = dd.mask(df, salt=SALT, columns=["cpf", "email"])
        df_safe = dd.mask(df, salt=SALT, exclude=["uf", "tipo_pessoa"])
        df_pl_safe = dd.mask(df_pl, salt=SALT)   # retorna pl.DataFrame
        df_safe = dd.mask(df, salt=SALT, risk="high")   # suprime alto risco
    """
    if not salt:
        # Verifica DEFAULT_SALT configurado via dd.configure(default_salt=...)
        import datalock._defaults as _defs
        if _defs.DEFAULT_SALT:
            salt = _defs.DEFAULT_SALT
        else:
            import secrets as _sec
            warnings.warn(
                "dd.mask(salt=None): salt aleatório gerado — hashes NÃO serão reprodutíveis "
                "entre execuções. Joins entre tabelas mascaradas em momentos distintos vão quebrar. "
                "Use dd.configure(default_salt=dd.generate_salt()) ou passe salt= explicitamente.",
                UserWarning,
                stacklevel=2,
            )
            salt = _sec.token_hex(32)

    # Risk-aware masking: sobrepõe estratégia por nível de risco
    if risk is not None:
        from datalock.detectors.pii_detector import RiskLevel, MaskStrategy
        risk_level = risk.lower().strip()
        _risk_override: Dict[str, str] = {}  # col → strategy override

        # Detecta PII primeiro para saber o risco de cada coluna
        _reports = _core.detect_pii(df)
        _risk_map = {
            "high":   MaskStrategy.SUPPRESS,       # dados de alto risco = suprimir
            "medium": MaskStrategy.HASH,            # médio = hash determinístico
            "low":    MaskStrategy.TRUNCATE,        # baixo = truncate (preserva utilidade)
        }
        _strategy = _risk_map.get(risk_level, MaskStrategy.HASH)

        # Filtra colunas pelo nível de risco pedido
        _filter_level = {
            "high":   RiskLevel.HIGH,
            "medium": RiskLevel.MEDIUM,
            "low":    RiskLevel.LOW,
        }.get(risk_level)

        if _filter_level:
            columns = [
                c for c, r in _reports.items()
                if r.risk_level == _filter_level
            ]
        return _core.mask_frame(
            df,
            salt=salt,
            random_state=random_state,
            strict_idempotency=strict,
            verbose=verbose,
            columns=columns or list(_reports.keys()),
            exclude=exclude,
        )

    # ── LazyFrame: sample for detection, apply exprs lazily, stays lazy ──
    if isinstance(df, pl.LazyFrame):
        import datalock._defaults as _defs
        _salt = salt or _defs.DEFAULT_SALT
        if not _salt:
            import secrets as _sec
            warnings.warn(
                "dd.mask(lf, salt=None): salt aleatório — hashes não reprodutíveis.",
                UserWarning, stacklevel=2,
            )
            _salt = _sec.token_hex(32)
        return _core.mask_lazyframe(
            df, salt=_salt,
            random_state=random_state,
            columns=columns,
            exclude=exclude,
            verbose=verbose,
        )

    return _core.mask_frame(
        df,
        salt=salt,
        random_state=random_state,
        strict_idempotency=strict,
        verbose=verbose,
        columns=columns,
        exclude=exclude,
    )


# ---------------------------------------------------------------------------
# read() — leitura unificada
# ---------------------------------------------------------------------------

def read(
    source: Union[str, pd.DataFrame, pl.DataFrame, "DatabaseConnection"],
    table_or_sql: Optional[str] = None,
    *,
    key: Optional[str] = None,
    private_key: Optional[Any] = None,
    salt: Optional[str] = None,
    raw: bool = False,
    frame: Optional[str] = None,
    columns: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    size: Optional[int] = None,
    verbose: bool = False,
    # ── Big data / partial read ──────────────────────────────────────
    header_only: bool = False,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    n_chunks: Optional[int] = None,
    chunks: Optional[List[int]] = None,
    sample: Optional[int] = None,
    sample_seed: int = 42,
    iter_chunks: bool = False,
    **kwargs,
) -> Union[pl.DataFrame, pd.DataFrame, Dict, Any]:
    """
    Leitura unificada de qualquer formato tabular.

    Auto-detecta formato pela extensão. Auto-detecta encoding (UTF-8, Latin-1, CP-1252).
    Sempre retorna pl.DataFrame para arquivos, preserva tipo para DataFrames em memória.

    Formatos suportados:
        .csv .tsv .txt           → Polars nativo (auto-encoding)
        .parquet                 → Polars nativo
        .json .ndjson .jsonl     → Polars nativo
        .feather .ipc .arrow     → Polars nativo (zero-copy)
        .avro .orc               → Polars nativo
        .xlsx .xls .ods          → pandas → converte para pl.DataFrame
        .xml .html .dta          → pandas → converte para pl.DataFrame
        .sas7bdat .xpt           → pandas → converte para pl.DataFrame
        .sav .zsav               → pandas → converte para pl.DataFrame
        .pkl .hdf .h5            → pandas → converte para pl.DataFrame
        .dlk                     → datalock (AES-256-GCM, multi-frame)

    Args:
        source:    Caminho, pd.DataFrame ou pl.DataFrame.
        key:       Chave AES (obrigatória para .dlk cifrado).
        salt:      Salt HMAC. Se fornecido → mascara PII automaticamente.
        raw:       Para .dlk: retorna sem mascaramento adicional.
        frame:     Frame a extrair de .dlk multi-frame.
        chunksize: Lê CSV em chunks (gerador de pd.DataFrame).
        size:      Linhas a gerar de modelo generativo em .dlk.
        verbose:   Relatório de detecção PII (quando salt= fornecido).
        **kwargs:  Repassados ao leitor (sep=, encoding=, etc.).

    Returns:
        pl.DataFrame | pd.DataFrame | dict[str, pd.DataFrame]

    Exemplos:
        df = dd.read("clientes.csv")
        df = dd.read("clientes.parquet")
        df = dd.read("clientes.dlk", key=KEY)
        df = dd.read("clientes.dlk", key=KEY, salt=SALT)  # descriptografa + mascara
        df = dd.read("clientes.csv", salt=SALT)           # lê + mascara
        df = dd.read("clientes.csv", sep=";", encoding="latin-1")
    """
    # ── DatabaseConnection — objeto de conexão dd.db() ──────────────────
    if isinstance(source, DatabaseConnection):
        if table_or_sql is None:
            raise ValueError(
                "dd.read(banco, ...) requer o nome da tabela ou SQL como segundo argumento. "
                "Exemplo: dd.read(banco, 'clientes')"
            )
        return source.read(
            table_or_sql,
            columns=columns,
            sample=sample,
            head=head,
            salt=salt,
            verbose=verbose,
        )

    # ── DataFrames em memória ──────────────────────────────────────────────
    if isinstance(source, pd.DataFrame):
        return mask(source, salt=salt, verbose=verbose) if salt else source.copy()

    if isinstance(source, pl.DataFrame):
        return mask(source, salt=salt, verbose=verbose) if salt else source.clone()

    # ── LazyFrame ─────────────────────────────────────────────────────────
    if isinstance(source, pl.LazyFrame):
        df = source.collect()
        return mask(df, salt=salt, verbose=verbose) if salt else df

    # ── Arquivo ───────────────────────────────────────────────────────────
    p = Path(str(source))
    ext = p.suffix.lower()

    if ext == ".dlk":
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {source}")
        if key is None:
            return SecureFile.load_open(p)

        content_type = _peek_lgs_content_type(p, key)

        if size is not None or content_type == "generative_model":
            return _read_lgs_model(p, key, size or 1000)

        if content_type == SecureFile.CONTENT_TYPE_MULTI:
            if frame is not None:
                return SecureFile.load_frame(p, key=key, frame=frame,
                                              salt_masking=salt, verbose=verbose)
            return SecureFile.load_frames(p, key=key, salt_masking=salt, verbose=verbose)

        if raw or not salt:
            return SecureFile.load_raw(p, key=key, columns=columns)
        return SecureFile.load(p, key=key, salt_masking=salt, verbose=verbose)

    # ── Outros formatos ───────────────────────────────────────────────────
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source}")

    if chunksize is not None:
        return _read_chunked(str(source), salt=salt, chunksize=chunksize,
                              verbose=verbose, **kwargs)

    # ── raw= is only meaningful for .dlk files ────────────────────────────
    if raw and ext != ".dlk":
        warnings.warn(
            f"dd.read(raw=True) não tem efeito em arquivos {ext!r}. "
            "raw= é usado apenas para .dlk (evita mascaramento automático). "
            "Para outros formatos, o dado é sempre retornado sem transformação.",
            UserWarning,
            stacklevel=2,
        )

    # ── Big-data partial read ─────────────────────────────────────────────
    _big_data_params = any([
        header_only, head is not None, tail is not None,
        n_chunks is not None, chunks is not None,
        sample is not None, iter_chunks,
    ])
    if _big_data_params:
        result = _read_partial(
            p,
            header_only=header_only, head=head, tail=tail,
            columns=columns, n_chunks=n_chunks, chunks=chunks,
            sample=sample, sample_seed=sample_seed,
            iter_chunks=iter_chunks, **kwargs,
        )
        # Apply masking if salt provided (only for materialized DataFrames)
        if salt and isinstance(result, pl.DataFrame):
            return _core.mask_frame(result, salt=salt, verbose=verbose)
        return result

    df = _core.read_file(p, **kwargs)

    if columns:
        existing = [c for c in columns if c in df.columns]
        if existing:
            df = df.select(existing) if isinstance(df, pl.DataFrame) else df[existing]

    if salt:
        return _core.mask_frame(df, salt=salt, verbose=verbose)
    return df


# ---------------------------------------------------------------------------
# store() — salva como .lgs
# ---------------------------------------------------------------------------

def store(
    source: Union[str, pd.DataFrame, pl.DataFrame, Dict, bytes],
    output_path: str,
    *,
    key: Optional[str] = None,
    salt: Optional[str] = None,
    anonymize: bool = False,
    raw: bool = False,
    label: str = "",
    compress: bool = True,
    overwrite: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    expires_at: Optional[str] = None,
    public_key: Optional[Any] = None,
    public_keys: Optional[List[Any]] = None,
    canary: bool = False,
    canary_n_rows: int = 3,
    pipeline_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Salva dados no formato .dlk (AES-256-GCM + Parquet/zstd internamente).

    Args:
        source:    DataFrame, dict[str, DataFrame], bytes ou caminho de arquivo.
        key:       Chave mestre AES-256. None → arquivo sem criptografia (v4).
        salt:      Salt HMAC. Se fornecido → mascara PII antes de gravar.
        anonymize: Se True → mascara PII antes de gravar (sem salt explícito).
        raw:       Se True + key → grava sem mascaramento.
        label:     Rótulo para auditoria.
        compress:  True=zstd (menor), False=lz4 (mais rápido).
        overwrite: Sobrescreve arquivo existente.
        metadata:  Dict de metadados arbitrários {str: str}.

    Exemplos:
        dd.store(df, "dados.dlk", key=KEY)                      # cifrado
        dd.store(df, "dados.dlk", key=KEY, salt=SALT)           # mascara + cifra
        dd.store(df, "dados.dlk", anonymize=True)               # mascara, sem key
        dd.store({"a": df1, "b": df2}, "multi.dlk", key=KEY)   # multi-frame
    """
    from datalock.utils.frames import to_pandas as _to_pd

    out = output_path if output_path.endswith(".dlk") else output_path + ".dlk"

    # Normaliza pl.DataFrame → pd.DataFrame (SecureFile trabalha com pandas internamente)
    if isinstance(source, pl.DataFrame):
        source = source.to_pandas()
    elif isinstance(source, dict):
        source = {
            k: v.to_pandas() if isinstance(v, pl.DataFrame) else v
            for k, v in source.items()
        }

    # Arquivo sem criptografia (v4/open)
    if key is None:
        if isinstance(source, dict):
            raise TypeError(
                "store() multi-frame requer key=. "
                "Arquivos sem criptografia não suportam multi-frame."
            )
        if not isinstance(source, pd.DataFrame):
            raise TypeError(
                f"store() sem key aceita apenas pd.DataFrame "
                f"(recebido {type(source).__name__})."
            )
        if not anonymize:
            _warn_if_pii(source)
        return SecureFile.pack_open(
            source, out, anonymize=anonymize, salt_masking=salt,
            label=label, compress=compress, overwrite=overwrite,
        )

    _validate_key_salt_distinct(key, salt)

    # salt= implica anonimização
    if salt and not anonymize:
        anonymize = True

    # Multi-frame
    if isinstance(source, dict):
        _validate_frames_dict(source)
        return SecureFile.pack_frames(
            source, out, key=key, label=label,
            compress=compress, overwrite=overwrite, metadata=metadata,
        )

    # DataFrame único
    if isinstance(source, pd.DataFrame):
        if anonymize and salt and not raw:
            source = mask(source, salt=salt)
            ct = SecureFile.CONTENT_TYPE_MASKED
        else:
            ct = (SecureFile.CONTENT_TYPE_MASKED if not raw and _df_is_masked(source)
                  else SecureFile.CONTENT_TYPE_RAW)
        return SecureFile.pack_dataframe(
            source, out, key=key, content_type=ct,
            label=label, compress=compress, overwrite=overwrite, metadata=metadata,
            expires_at=expires_at, canary=canary,
            canary_n_rows=canary_n_rows, pipeline_id=pipeline_id,
        )

    # Bytes brutos
    if isinstance(source, bytes):
        return SecureFile.pack_bytes(
            source, out, key=key, content_type="bytes",
            label=label, compress=compress, overwrite=overwrite,
        )

    return SecureFile.pack(str(source), out, key=key, label=label,
                           compress=compress, overwrite=overwrite, expires_at=expires_at)


# ---------------------------------------------------------------------------
# inspect() / rekey()
# ---------------------------------------------------------------------------

def inspect(path: Union[str, Path], *, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Lê metadados de .dlk sem descriptografar o payload.

    Retorna: content_type, shape, label, created_at, encryption, metadata, frame_names.

    Exemplos:
        info = dd.inspect("clientes.dlk", key=KEY)
        print(info["shape"], info["content_type"])
    """
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    ok, info = SecureFile.verify(str(path), key=key)
    if not ok and not info:
        raise ValueError(f"Não foi possível ler metadados de '{path}'. Chave incorreta?")
    return info or {}


def rekey(
    path: str,
    *,
    old_key: str,
    new_key: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Rotaciona a chave de um arquivo .dlk cifrado sem expor dados em disco.

    AVISO: dados brutos ficam na heap durante a operação.
    Use apenas em ambientes com controles de acesso adequados.

    Exemplos:
        dd.rekey("dados.dlk", old_key=KEY_ANTIGO, new_key=KEY_NOVO)
    """
    import shutil, tempfile, os

    src = Path(str(path))
    ok, info = SecureFile.verify(str(src), key=old_key)
    if not ok:
        raise ValueError(f"old_key inválida ou arquivo corrompido: {info.get('error')}")

    content_type = info.get("content_type", "raw_dataframe")
    label = info.get("label", "")

    if output_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".dlk", dir=src.parent)
        os.close(fd)
        dst, atomic = Path(tmp), True
    else:
        dst = Path(str(output_path))
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} já existe. Use overwrite=True.")
        atomic = False

    if content_type == SecureFile.CONTENT_TYPE_MULTI:
        frames = SecureFile.load_frames(str(src), key=old_key)
        result = SecureFile.pack_frames(frames, str(dst), key=new_key,
                                        label=label, overwrite=True)
    elif content_type == SecureFile.CONTENT_TYPE_BYTES:
        payload = SecureFile.load_bytes(str(src), key=old_key)
        result = SecureFile.pack_bytes(payload, str(dst), key=new_key,
                                       label=label, overwrite=True)
    else:
        df_plain = SecureFile.load_raw(str(src), key=old_key)
        result = SecureFile.pack_dataframe(df_plain, str(dst), new_key,
                                           content_type=content_type, label=label,
                                           overwrite=True)
        del df_plain

    if atomic:
        shutil.move(str(dst), str(src))
        result["output_path"] = str(src)
    return result



def write(
    df,
    path_or_conn,
    table: Optional[str] = None,
    **kw,
) -> None:
    """
    Escreve DataFrame em arquivo ou banco de dados.

    Detecta automaticamente o destino pelo tipo de path_or_conn:
    - str / Path → arquivo (extensão define o formato)
    - DatabaseConnection → banco de dados (table= obrigatório)

    Args:
        df:           pl.DataFrame ou pd.DataFrame.
        path_or_conn: Caminho de arquivo (str/Path) ou DatabaseConnection (dd.db()).
        table:        Nome da tabela — obrigatório quando path_or_conn é banco.
        **kw:         Para arquivo: opções do escritor (sep=, compression=, etc.)
                      Para banco: if_exists=, schema=, chunk_size=

    Exemplos:
        dd.write(df, "resultado.parquet")
        dd.write(df, "resultado.csv")
        dd.write(df, "resultado.xlsx")
        dd.write(df, banco, "clientes_masked")   # banco = dd.db("...")
    """
    if isinstance(path_or_conn, DatabaseConnection):
        if table is None:
            raise ValueError(
                "dd.write(df, banco, ...) requer o nome da tabela como terceiro argumento. "
                "Exemplo: dd.write(df, banco, 'clientes_masked')"
            )
        return path_or_conn.write(df, table, **kw)
    # Arquivo — delega para analytics.write
    _analytics_write(df, str(path_or_conn), **kw)

# ---------------------------------------------------------------------------
# stream() — leitura em chunks sem OOM
# ---------------------------------------------------------------------------

def stream(
    source: Union[str, Path],
    *,
    salt: Optional[str] = None,
    chunksize: int = 10_000,
    on_progress=None,
    verbose: bool = False,
    **kwargs,
):
    """
    Lê e mascara em modo gerador (streaming por chunks) — ideal para big data.

    Suporta CSV e Parquet. Não carrega o arquivo inteiro em memória.

    Args:
        source:      Caminho do arquivo CSV ou Parquet.
        salt:        Salt HMAC para mascaramento determinístico.
        chunksize:   Linhas por chunk (padrão: 10.000).
        on_progress: Callback(chunk_n, rows_done, rows_total_estimate).
        verbose:     Exibe relatório PII no primeiro chunk.

    Yields:
        pd.DataFrame — um chunk por vez, já mascarado.

    Exemplos:
        for chunk in dd.stream("grande.csv", salt=SALT, chunksize=50_000):
            salvar_no_banco(chunk)
    """
    from datalock.detectors.fast_scan import FastPIIScanner

    scanner = FastPIIScanner(sample_size=500)
    reports: Optional[Dict] = None
    chunk_n = rows_done = 0

    p = Path(str(source))
    suffix = p.suffix.lower()

    try:
        rows_total_estimate = p.stat().st_size // 100
    except Exception:
        rows_total_estimate = -1

    # ── Parquet: PyArrow iter_batches — true streaming, zero-copy Arrow ──
    if suffix == ".parquet":
        import pyarrow.parquet as _pq
        import pyarrow as _pa
        pf = _pq.ParquetFile(source)
        if hasattr(pf, "metadata"):
            rows_total_estimate = pf.metadata.num_rows
        def _parquet_iter():
            for batch in pf.iter_batches(batch_size=chunksize):
                yield pl.from_arrow(batch)
        iterator = _parquet_iter()
        is_polars_iter = True

    # ── CSV: pl.read_csv_batched — no full file load, true streaming ──
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = kwargs.pop("sep", "," if suffix == ".csv" else "	")
        sep = kwargs.pop("separator", sep)
        # Use scan_csv with collect_batches (read_csv_batched deprecated in Polars 1.x)
        clean_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ("encoding", "dtype", "usecols", "skiprows", "index_col")}
        lf_csv = pl.scan_csv(source, separator=sep, **clean_kwargs)
        def _csv_iter():
            for batch in lf_csv.collect_batches(chunk_size=chunksize):
                yield batch
        iterator = _csv_iter()
        is_polars_iter = True

    # ── Fallback: other formats via pandas ──────────────────────────────
    else:
        iterator = (pl.from_pandas(chunk) for chunk in pd.read_csv(source, chunksize=chunksize, **kwargs))
        is_polars_iter = True

    for chunk_pl in iterator:
        if not isinstance(chunk_pl, pl.DataFrame):
            chunk_pl = pl.from_pandas(chunk_pl)
        chunk_n += 1

        # Detect PII on first chunk only (schema is constant across chunks)
        if reports is None:
            reports = scanner.detect_dict(chunk_pl)
            if verbose and reports:
                from datalock.adapters.pandas_adapter import _print_detection_report
                _print_detection_report(reports)

        if reports and salt:
            chunk_safe = _core.mask_frame(chunk_pl, salt=salt, verbose=False)
        else:
            chunk_safe = chunk_pl

        rows_done += len(chunk_safe)

        if on_progress is not None:
            try:
                on_progress(chunk_n, rows_done, rows_total_estimate)
            except Exception:
                pass

        yield chunk_safe


# ---------------------------------------------------------------------------
# sql() — SQL direto via DuckDB
# ---------------------------------------------------------------------------

def sql(
    query: str,
    *,
    salt: Optional[str] = None,
    key: Optional[str] = None,
    **frames,
) -> pl.DataFrame:
    """
    Executa SQL em DataFrames, arquivos ou .dlk via DuckDB (zero-copy Arrow).

    Requer: pip install 'datalock[sql]'  (duckdb)

    Args:
        query:    Instrução SQL.
        salt:     Se fornecido, mascara PII no resultado.
        key:      Chave AES para .dlk referenciados por caminho.
        **frames: DataFrames ou caminhos nomeados para o SQL.

    Returns:
        pl.DataFrame com o resultado.

    Exemplos:
        result = dd.sql(
            "SELECT uf, AVG(renda_mensal) AS media, COUNT(*) AS n "
            "FROM df GROUP BY uf HAVING n > 100",
            df=df
        )
        result = dd.sql(
            "SELECT * FROM read_parquet('dados.parquet') WHERE uf='SP'"
        )
        result = dd.sql(
            "SELECT c.uf, p.valor FROM clientes c JOIN pedidos p ON c.cpf = p.cpf",
            clientes=df_c, pedidos=df_p,
        )
    """
    try:
        import duckdb as _ddb
    except ImportError:
        raise ImportError(
            "dd.sql() requer duckdb. Instale com: pip install 'datalock[sql]'"
        ) from None

    conn = _ddb.connect()

    # Registra DataFrames ou carrega arquivos nomeados
    for name, frame_src in frames.items():
        if isinstance(frame_src, (str, Path)):
            p = Path(str(frame_src))
            if p.suffix.lower() == ".dlk":
                raw = read(p, key=key, raw=True)
                df_pd = raw if isinstance(raw, pd.DataFrame) else pd.concat(list(raw.values()))
                conn.register(name, pl.from_pandas(df_pd).to_arrow())
            else:
                conn.register(name, _core.read_file(p).to_arrow())
        elif isinstance(frame_src, pl.DataFrame):
            conn.register(name, frame_src.to_arrow())
        elif isinstance(frame_src, pd.DataFrame):
            conn.register(name, pl.from_pandas(frame_src).to_arrow())

    result_arrow = conn.execute(query).arrow()
    result = pl.from_arrow(result_arrow)

    if salt:
        result = _core.mask_frame(result, salt=salt)
    return result


# ---------------------------------------------------------------------------
# scan_text() / mask_text()
# ---------------------------------------------------------------------------

def scan_text(text: str) -> List[Dict[str, Any]]:
    """
    Detecta PII em texto livre.

    Retorna lista de {type, value, start, end}.

    Exemplos:
        dd.scan_text("Meu CPF é 111.444.777-35, email: ana@empresa.com")
    """
    from datalock.detectors.text_detector import TextPIIDetector
    spans = TextPIIDetector().scan(text)
    return [{"type": s.pii_type.value, "value": s.text, "start": s.start, "end": s.end}
            for s in spans]


def mask_text(text: str, *, salt: Optional[str] = None, strategy: str = "redact",
    seed: Optional[int] = None,
) -> str:
    """
    Mascara PII em texto livre.

    Args:
        text:     Texto a mascarar.
        salt:     Salt HMAC (necessário para strategy='hash').
        strategy: 'redact' (padrão) ou 'hash'.

    Exemplos:
        safe = dd.mask_text("CPF: 111.444.777-35, email: ana@empresa.com")
        # "CPF: REDACTED, email: REDACTED"
    """
    from datalock.maskers.text_masker import TextMasker
    masker = TextMasker(salt=salt, strategy=strategy)
    return masker.mask(text)


# ---------------------------------------------------------------------------
# diff() — compara original vs mascarado
# ---------------------------------------------------------------------------

def diff(
    original: Union[pd.DataFrame, pl.DataFrame],
    masked: Union[pd.DataFrame, pl.DataFrame],
    *,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """
    Compara DataFrame original e mascarado — mostra o que mudou e como.

    Args:
        original:    DataFrame antes do mascaramento.
        masked:      DataFrame após o mascaramento.
        sample_size: Exemplos por coluna no relatório.

    Returns:
        Dict com: columns_changed, columns_unchanged, per_column, summary.

    Exemplos:
        df_safe = dd.mask(df, salt=SALT)
        report = dd.diff(df, df_safe)
        print(report["summary"])
    """
    from datalock.utils.frames import to_pandas as _tp

    df_orig   = _tp(original)
    df_masked = _tp(masked)

    if df_orig.shape != df_masked.shape:
        raise ValueError(
            f"DataFrames têm shapes diferentes: "
            f"original={df_orig.shape}, masked={df_masked.shape}."
        )

    changed: List[str] = []
    unchanged: List[str] = []
    per_col: Dict[str, Any] = {}

    for col_name in df_orig.columns:
        if col_name not in df_masked.columns:
            continue
        orig_s   = df_orig[col_name]
        masked_s = df_masked[col_name]
        try:
            equal = (orig_s == masked_s) | (orig_s.isna() & masked_s.isna())
            changed_pct = round(1 - equal.mean(), 4)
        except Exception:
            changed_pct = 0.0

        if changed_pct > 0.0:
            changed.append(col_name)
            strategy = _infer_mask_strategy(orig_s, masked_s)
            examples = []
            for i in range(min(sample_size, len(df_orig))):
                bef, aft = str(orig_s.iloc[i]), str(masked_s.iloc[i])
                if bef != aft:
                    examples.append({"before": bef[:30], "after": aft[:30]})
                    if len(examples) >= 3:
                        break
            per_col[col_name] = {
                "strategy":    strategy,
                "changed_pct": f"{changed_pct:.0%}",
                "examples":    examples,
            }
        else:
            unchanged.append(col_name)
            per_col[col_name] = {"strategy": "unchanged", "changed_pct": "0%", "examples": []}

    strat_groups: Dict[str, List[str]] = {}
    for c in changed:
        s = per_col[c]["strategy"]
        strat_groups.setdefault(s, []).append(c)

    parts = [f"{len(changed)} coluna(s) mascarada(s):"]
    for strat, clist in sorted(strat_groups.items()):
        parts.append(f"  {strat}: {', '.join(clist)}")
    if unchanged:
        parts.append(f"{len(unchanged)} coluna(s) inalterada(s): {', '.join(unchanged)}")

    return {
        "columns_changed":   changed,
        "columns_unchanged": unchanged,
        "per_column":        per_col,
        "summary":           "\n".join(parts),
    }


# ---------------------------------------------------------------------------
# profile() — diagnóstico integrado
# ---------------------------------------------------------------------------

def profile(
    source: Union[str, pd.DataFrame, pl.DataFrame],
    *,
    key: Optional[str] = None,
    sample_size: int = 500,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Diagnóstico rápido — PII, nulos, tipos, distribuições. Resultado JSON-serializable.

    Args:
        source:      DataFrame ou caminho de arquivo.
        key:         Chave de decriptação para .dlk.
        sample_size: Linhas amostradas para detecção PII.
        threshold:   Match ratio mínimo para classificar como PII.

    Returns:
        Dict JSON-serializable: shape, pii_columns, pii_risk_summary,
        null_counts, nunique, dtypes, describe, sample.

    Exemplos:
        report = dd.profile(df)
        report = dd.profile("clientes.parquet")
        import json; print(json.dumps(report, ensure_ascii=False, indent=2))
    """
    from datalock.utils.frames import to_pandas as _tp

    if isinstance(source, (str, Path)):
        p = Path(str(source))
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {source}")
        if p.suffix.lower() == ".dlk":
            raw = read(p, key=key, raw=True)
            df_pd = (pd.concat(list(raw.values()), ignore_index=True)
                     if isinstance(raw, dict) else raw)
        else:
            df_pd = _core.read_file(p).to_pandas()
    else:
        df_pd = _tp(source)

    reports = scan(df_pd, sample_size=sample_size, threshold=threshold)
    risk_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in reports.values():
        risk_counts[r.risk_level.value] += 1

    nc = count_nulls(df_pd)
    total_cells = df_pd.shape[0] * df_pd.shape[1]
    total_nulls = int(nc.sum())

    import math
    def _safe(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    return {
        "shape":            list(df_pd.shape),
        "n_pii_columns":    len(reports),
        "pii_columns":      list(reports.keys()),
        "pii_risk_summary": (f"{risk_counts['high']}🔴 "
                             f"{risk_counts['medium']}🟡 "
                             f"{risk_counts['low']}🟢"),
        "pii_reports": {
            c: {
                "type":        r.pii_type.value,
                "risk":        r.risk_level.value,
                "strategy":    r.mask_strategy.value,
                "match_ratio": round(r.match_ratio, 3),
                "unique_ratio": round(r.unique_ratio, 3),
                "notes":       r.notes,
            }
            for c, r in reports.items()
        },
        "null_counts": nc.to_dict(),
        "total_nulls": total_nulls,
        "null_pct":    round(total_nulls / max(total_cells, 1) * 100, 2),
        "nunique":     nunique(df_pd).to_dict(),
        "dtypes":      dtypes(df_pd),
        "describe":    {
            k: {kk: _safe(vv) for kk, vv in v.items()}
            for k, v in describe(df_pd).to_dict().items()
        },
        "sample":      df_pd.head(3).to_dict(orient="records"),
        "privacy_score": _compute_privacy_score(df_pd, reports),
    }


# ---------------------------------------------------------------------------
# join() — JOIN seguro com tokens compatíveis
# ---------------------------------------------------------------------------

def join(
    left: Union[pd.DataFrame, pl.DataFrame],
    right: Union[pd.DataFrame, pl.DataFrame],
    on: Union[str, List[str]],
    *,
    salt: Optional[str] = None,
    how: str = "inner",
    validate_salt: bool = True,
    suffixes: tuple = ("_x", "_y"),
) -> pd.DataFrame:
    """
    JOIN seguro garantindo integridade referencial de tokens HMAC.

    Se os DataFrames não estão mascarados, aplica o mesmo salt em ambos
    antes do join — garantindo que os tokens sejam idênticos.

    Args:
        left:          DataFrame esquerdo.
        right:         DataFrame direito.
        on:            Coluna(s) de join (devem ser PII).
        salt:          Salt HMAC compartilhado (obrigatório se não mascarados).
        how:           Tipo: "inner" | "left" | "right" | "outer".
        validate_salt: Verifica compatibilidade dos tokens.
        suffixes:      Sufixos para colunas duplicadas.

    Returns:
        pd.DataFrame resultado do join mascarado.

    Exemplos:
        # Dados brutos — aplica mesmo salt nos dois antes do join
        result = dd.join(df_clientes, df_pedidos, on="cpf", salt=SALT)

        # Dados já mascarados com o mesmo salt
        result = dd.join(df_c_safe, df_p_safe, on="cpf")
    """
    import re as _re
    from datalock.utils.frames import to_pandas as _tp

    left_pd  = _tp(left)
    right_pd = _tp(right)
    on_list  = [on] if isinstance(on, str) else list(on)
    _hex16   = _re.compile(r"^[0-9a-f]{16}$")

    def _looks_masked(df_: pd.DataFrame, cols: List[str]) -> bool:
        for c in cols:
            if c not in df_.columns:
                return False
            sample = df_[c].dropna().astype(str).head(20)
            if len(sample) < 3:
                continue
            if sample.map(lambda v: bool(_hex16.match(v))).mean() < 0.8:
                return False
        return True

    left_masked  = _looks_masked(left_pd, on_list)
    right_masked = _looks_masked(right_pd, on_list)

    if not left_masked or not right_masked:
        if salt is None:
            raise ValueError(
                "join(): os DataFrames não parecem mascarados. "
                "Forneça salt= para que datalock aplique o mesmo mascaramento em ambos."
            )
        left_pd  = mask(left_pd,  salt=salt, columns=on_list)
        right_pd = mask(right_pd, salt=salt, columns=on_list)

    elif validate_salt:
        for c in on_list:
            if c not in left_pd.columns or c not in right_pd.columns:
                continue
            lt = set(left_pd[c].dropna().astype(str).head(100))
            rt = set(right_pd[c].dropna().astype(str).head(100))
            if lt and rt and len(lt & rt) / max(len(lt), 1) == 0.0:
                raise ValueError(
                    f"join(): nenhum token em comum na coluna '{c}'. "
                    f"Os DataFrames provavelmente foram mascarados com salts diferentes."
                )

    return left_pd.merge(right_pd, on=on_list, how=how, suffixes=suffixes)


# ---------------------------------------------------------------------------
# configure() — configuração global
# ---------------------------------------------------------------------------

def configure(
    *,
    audit: Optional[Any] = None,
    audit_path: Optional[str] = None,
    default_salt: Optional[str] = None,
    canary_salt: Optional[str] = None,
    wm_salt: Optional[str] = None,
    load_dotenv: bool = False,
    dotenv_path: Optional[str] = None,
    audit_webhook: Optional[str] = None,
) -> None:
    """
    Configura parâmetros globais do datalock.

    Args:
        audit:        AuditReport para trilha de auditoria automática.
        audit_path:   Diretório para auto-criar AuditReport com gravação em arquivo.
        default_salt: Salt padrão para mascaramento. Nunca commite o valor real no código.
        canary_salt:  Salt secreto para geração de fingerprints canary (tabular).
                      Substitui o valor padrão público hardcoded no source.
                      Em produção, use os.environ["DATALOCK_CANARY_SALT"] ou .env.
                      Com um salt secreto, adversários não conseguem pré-calcular
                      os fingerprints para remover canary rows antes de um breach.
        wm_salt:      Salt secreto para fingerprints de watermarking textual.
                      Substitui o valor padrão público. Use DATALOCK_WM_SALT no .env.
        load_dotenv:  Se True, carrega variáveis de um arquivo .env (requer python-dotenv).
                      Lê DATALOCK_SALT, DATALOCK_CANARY_SALT e DATALOCK_WM_SALT automaticamente.
        audit_webhook: URL para receber eventos de auditoria via HTTP POST (JSON).
                       Cada operação dd.mask(), dd.scan(), dd.store() dispara um POST.
                       Suporta Slack webhooks, SIEM, Datadog, ou qualquer endpoint HTTP.
        dotenv_path:  Caminho para o .env. None = procura ".env" no diretório atual.

    Exemplos:
        # Carrega .env automaticamente (recomendado — salts nunca no código)
        dd.configure(load_dotenv=True)

        # Configuração explícita (valores de variáveis de ambiente, não literals)
        dd.configure(
            default_salt=os.environ["DATALOCK_SALT"],
            canary_salt=os.environ["DATALOCK_CANARY_SALT"],
        )

        from datalock.reports.audit_report import AuditReport
        dd.configure(audit=AuditReport())
        dd.configure(audit_path="./audit/")
    """
    # Auto-load .env if requested
    if load_dotenv:
        try:
            from dotenv import load_dotenv as _load_dotenv
            _load_dotenv(dotenv_path=dotenv_path, override=False)
            import os as _os
            if default_salt is None:
                _env_salt = _os.environ.get("DATALOCK_SALT")
                if _env_salt:
                    default_salt = _env_salt
            if canary_salt is None:
                _env_canary = _os.environ.get("DATALOCK_CANARY_SALT")
                if _env_canary:
                    canary_salt = _env_canary
            if wm_salt is None:
                _env_wm = _os.environ.get("DATALOCK_WM_SALT")
                if _env_wm:
                    wm_salt = _env_wm
        except ImportError:
            warnings.warn(
                "dd.configure(load_dotenv=True) requer python-dotenv. "
                "Instale com: pip install python-dotenv",
                UserWarning, stacklevel=2,
            )

    from datalock.adapters import pandas_adapter as _pa

    # Audit webhook — stores globally for AuditReport to dispatch
    if audit_webhook is not None:
        import datalock._defaults as _defs_cfg
        _defs_cfg.AUDIT_WEBHOOK = audit_webhook

    if audit_path is not None and audit is None:
        from datalock.reports.audit_report import AuditReport
        audit = AuditReport(output_dir=audit_path)

    _pa._GLOBAL_AUDIT = audit

    if default_salt is not None:
        import datalock._defaults as _d
        _d.DEFAULT_SALT = default_salt
        from datalock.adapters import pandas_adapter as _pa2
        _pa2._DEFAULT_SALT = default_salt

    if canary_salt is not None:
        import datalock._defaults as _d2
        _d2.CANARY_SALT = canary_salt
        import datalock.canary as _canary
        _canary._CANARY_SALT_OVERRIDE = canary_salt

    if wm_salt is not None:
        import datalock._defaults as _d3
        _d3.WM_SALT = wm_salt
        import datalock.canary as _canary2
        _canary2._WM_SALT_OVERRIDE = wm_salt


# ---------------------------------------------------------------------------
# read_db() — lê de banco de dados
# ---------------------------------------------------------------------------

def db(
    uri: str,
    *,
    salt: Optional[str] = None,
    dialect: Optional[str] = None,
    pool_size: int = 5,
    connect_timeout: int = 30,
) -> "DatabaseConnection":
    """
    Cria um objeto de conexão reutilizável para banco de dados.

    Integra-se com dd.read() e dd.write() para leitura/escrita unificada.
    Usa ConnectorX (Rust, zero-copy Arrow) quando disponível, com fallback
    para SQLAlchemy. Credenciais são armazenadas como SecretStr (nunca
    aparecem em repr ou logs).

    Args:
        uri:             URL de conexão (SQLAlchemy format).
                         "postgresql://user:pass@host:5432/db"
                         "mysql+pymysql://user:pass@host/db"
                         "sqlite:///arquivo.db"
                         "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17"
                         "duckdb:///:memory:"
        salt:            Salt HMAC. Se fornecido, mascara automaticamente em
                         todo dd.read(banco, ...). Omita para exploração sem mascaramento.
        dialect:         Auto-detectado pela URI. Force: "postgresql" | "mysql" |
                         "sqlite" | "sqlserver" | "bigquery" | "snowflake".
        pool_size:       Conexões no pool (padrão 5).
        connect_timeout: Timeout em segundos (padrão 30).

    Returns:
        DatabaseConnection — passível a dd.read(), dd.write() e context manager.

    Exemplos:
        # Exploração sem mascaramento
        banco = dd.db("postgresql://user:pass@host/db")
        print(banco.tables())
        df_sample = banco.sample_table("clientes")

        # Leitura com mascaramento automático
        banco = dd.db("postgresql://user:pass@host/db", salt=SALT)
        df = dd.read(banco, "clientes")
        df = dd.read(banco, "clientes", sample=10_000)
        df = dd.read(banco, "SELECT * FROM clientes WHERE uf = 'SP'")

        # Escrita
        banco.write(df_safe, "clientes_masked")
        dd.write(banco, df_safe, "clientes_masked")

        # Context manager
        with dd.db("postgresql://...", salt=SALT) as banco:
            df = dd.read(banco, "clientes")
    """
    return DatabaseConnection(
        uri,
        salt=salt,
        dialect=dialect,
        pool_size=pool_size,
        connect_timeout=connect_timeout,
    )


def compliance_report(
    df: Any,
    reports: Dict,
    *,
    audit: Optional[Any] = None,
    title: str = "Relatório de Conformidade LGPD",
    organization: str = "",
    dataset_name: str = "dataset",
    extra_notes: str = "",
) -> Any:
    """
    Gera relatório formal de conformidade LGPD para DPOs e auditores.

    Args:
        df:           DataFrame analisado.
        reports:      Dict[str, ColumnReport] do dd.scan().
        audit:        AuditReport para incluir trilha de auditoria.
        title:        Título do documento.
        organization: Nome da organização.
        dataset_name: Identificação do dataset.
        extra_notes:  Observações do DPO.

    Returns:
        ComplianceReport com métodos:
          .to_html("relatorio.html")    — HTML formatado
          .to_pdf("relatorio.pdf")     — PDF (requer weasyprint)
          .to_text()                   — texto simples, sempre disponível
          .to_json("relatorio.json")   — JSON serializado
          .to_dict()                   — dict Python

    Exemplos:
        reports = dd.scan(df)
        report  = dd.compliance_report(df, reports, dataset_name="Clientes Q1 2025")
        report.to_html("lgpd_relatorio.html")
        report.to_pdf("lgpd_relatorio.pdf")   # pip install weasyprint
        print(report.to_text())
    """
    return _build_cr(
        df, reports,
        audit=audit, title=title,
        organization=organization, dataset_name=dataset_name,
        extra_notes=extra_notes,
    )


def read_db(
    connection: Union[str, Any],
    sql_or_table: str,
    *,
    salt: str,
    params: Optional[Any] = None,
    table: bool = False,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
    chunksize: Optional[int] = None,
    columns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Lê dados de banco relacional com mascaramento automático.

    Args:
        connection:    URL de conexão SQLAlchemy ou engine.
        sql_or_table:  SQL ou nome de tabela (se table=True).
        salt:          Chave HMAC para mascaramento.
        params:        Parâmetros da query SQL.
        table:         Se True, sql_or_table é nome de tabela.
        where_clause:  Cláusula WHERE (apenas se table=True).
        limit:         Limite de linhas (apenas se table=True).
        chunksize:     Lê em chunks para tabelas grandes.
        columns:       Mascara apenas estas colunas.
        exclude:       Não mascara estas colunas.

    Exemplos:
        df = dd.read_db("postgresql://u:p@h/db", "clientes", salt=SALT, table=True)
        df = dd.read_db("postgresql://u:p@h/db",
                        "SELECT * FROM clientes WHERE uf = %s",
                        salt=SALT, params=("SP",))
    """
    from datalock.adapters.db_adapter import SecureDBAdapter

    adapter = SecureDBAdapter(engine=connection, salt=salt, **kwargs)

    if table:
        return adapter.query_table(
            sql_or_table, where=where_clause, limit=limit,
            mask_columns=columns, exclude_mask=exclude,
        )
    if chunksize:
        return adapter.query_chunked(sql_or_table, chunksize=chunksize,
                                     params=params, columns=columns, exclude=exclude)
    return adapter.query(sql_or_table, params=params, columns=columns, exclude=exclude)


# ---------------------------------------------------------------------------
# open() — context manager para .lgs
# ---------------------------------------------------------------------------

from datalock.dlk import DlkFile, LGSFile, DlkFile


def open(
    path: Union[str, Path],
    *,
    key: Optional[str] = None,
    salt: Optional[str] = None,
    compress: bool = True,
) -> DlkFile:
    """
    Abre um arquivo .dlk como objeto pythônico (suporta context manager).

    Args:
        path:     Caminho para o arquivo .dlk.
        key:      Chave de criptografia AES-256.
        salt:     Salt HMAC para mascaramento na leitura.
        compress: Compressão ao escrever (True=zstd, False=lz4).

    Returns:
        DlkFile com métodos read(), write(), frames(), info(), valid(), etc.

    Exemplos:
        with dd.open("clientes.dlk", key=KEY) as f:
            df = f.read()
            print(f.info())
            f.add_frame("novos", df_novos)

        df = dd.open("clientes.dlk", key=KEY).read()
    """
    return DlkFile(path, key=key, salt=salt, compress=compress)


# ---------------------------------------------------------------------------
# Sintético (opcional)
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame, *, n: int = 1000, random_state: int = 42, **kwargs) -> Any:
    """
    Treina modelo generativo sintético (requer: pip install 'datalock[synthetic]').
    Retorna modelo que pode ser armazenado em .dlk via dd.store().
    """
    try:
        from datalock.generators.tabular_generative import TabularGenerativeModel
    except ImportError:
        raise ImportError(
            "Geração sintética requer dependências extras. "
            "Instale com: pip install 'datalock[synthetic]'"
        ) from None
    model = TabularGenerativeModel(**kwargs)
    model.fit(df)
    return model


def clone(
    df: pd.DataFrame,
    n: Optional[int] = None,
    *,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Gera DataFrame sintético com mesmas distribuições do original.
    Atalho para train() + model.generate().
    """
    model = train(df, n=n or len(df), random_state=random_state, **kwargs)
    return model.generate(n or len(df))


def sandbox(
    df: pd.DataFrame,
    *,
    n: int = 1000,
    salt: Optional[str] = None,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Gera dados sintéticos mascarados para uso em desenvolvimento e testes.
    """
    synthetic = clone(df, n=n, random_state=random_state, **kwargs)
    if salt:
        return mask(synthetic, salt=salt)
    return synthetic


# ---------------------------------------------------------------------------
# Aliases para compatibilidade
# ---------------------------------------------------------------------------

save = store   # dd.save() == dd.store()
load = read    # dd.load() == dd.read()


# ---------------------------------------------------------------------------
# Namespace dd.pl — sub-API Polars exposta via datalock
# ---------------------------------------------------------------------------

class _PolarsNamespace:
    """
    dd.pl — acesso ao Polars via namespace datalock.

    Permite usar Polars sem importar polars diretamente.

    Exemplos:
        df = dd.pl.read_parquet("dados.parquet")
        df = dd.pl.from_pandas(df_pd)
        schema = dd.pl.Schema({"nome": dd.pl.String, "idade": dd.pl.Int32})
    """

    # Tipos
    Int8       = pl.Int8
    Int16      = pl.Int16
    Int32      = pl.Int32
    Int64      = pl.Int64
    UInt8      = pl.UInt8
    UInt16     = pl.UInt16
    UInt32     = pl.UInt32
    UInt64     = pl.UInt64
    Float32    = pl.Float32
    Float64    = pl.Float64
    String     = pl.String
    Boolean    = pl.Boolean
    Date       = pl.Date
    Datetime   = pl.Datetime
    Duration   = pl.Duration
    Categorical = pl.Categorical
    Null       = pl.Null

    # Funções
    col        = staticmethod(pl.col)
    lit        = staticmethod(pl.lit)
    concat_str = staticmethod(pl.concat_str)
    concat     = staticmethod(pl.concat)
    from_pandas = staticmethod(pl.from_pandas)
    from_arrow  = staticmethod(pl.from_arrow)

    # Leitores
    read_csv      = staticmethod(pl.read_csv)
    read_parquet  = staticmethod(pl.read_parquet)
    read_json     = staticmethod(pl.read_json)
    read_ndjson   = staticmethod(pl.read_ndjson)
    read_ipc      = staticmethod(pl.read_ipc)
    read_avro     = staticmethod(pl.read_avro)
    scan_csv      = staticmethod(pl.scan_csv)
    scan_parquet  = staticmethod(pl.scan_parquet)

    DataFrame  = pl.DataFrame
    LazyFrame  = pl.LazyFrame
    Series     = pl.Series
    Schema     = pl.Schema

    def __repr__(self) -> str:
        return f"datalock.pl (Polars {pl.__version__})"


# Singleton
pl_ns = _PolarsNamespace()


# ---------------------------------------------------------------------------
# __dir__ e __getattr__ — namespace limpo
# ---------------------------------------------------------------------------

def __dir__():
    return sorted(__all__)


def __getattr__(name: str):
    """Resolve dd.pl, dd.check, dd.link via __getattr__."""
    if name == "pl":
        return pl_ns
    if name == "check":
        return check
    if name == "link":
        return link
    if name in ("save", "load"):
        return store if name == "save" else read
    raise AttributeError(f"módulo 'datalock' não tem atributo '{name!r}'")


# ---------------------------------------------------------------------------
# Pipeline fluente
# ---------------------------------------------------------------------------

class _Pipeline:
    """
    Pipeline fluente para processar dados sem variáveis temporárias.

    Criado via dd.pipe(source) ou dd.pipe().

    Exemplos:
        result = (
            dd.pipe("clientes.parquet")
            .where(uf="SP", tipo_pessoa="PF")
            .add_column(
                imposto=dd.col("renda_mensal") * 0.27,
                faixa=dd.when(dd.col("renda_mensal") > 10000, "alta")
                        .when(dd.col("renda_mensal") > 5000, "media")
                        .otherwise("baixa"),
            )
            .mask(salt=SALT)
            .groupby("faixa", {"n": ("*", "count"), "media": ("renda_mensal", "mean")})
            .sort("media", desc=True)
            .collect()
        )
    """

    def __init__(self, source=None, *, key=None, salt=None):
        self._df  = None
        self._key  = key
        self._salt = salt
        if source is not None:
            self._df = read(source, key=key)

    # ── Entrada ───────────────────────────────────────────────────────────

    def read(self, source, **kwargs) -> "_Pipeline":
        self._df = read(source, key=self._key, salt=self._salt, **kwargs)
        return self

    def sql(self, query_str: str, **frames) -> "_Pipeline":
        self._df = sql(query_str, key=self._key, **frames)
        return self

    # ── Transformação ─────────────────────────────────────────────────────

    def where(self, expr=None, **kwargs) -> "_Pipeline":
        self._ensure()
        self._df = where(self._df, expr, **kwargs)
        return self

    def mask(self, *, salt: Optional[str] = None,
             columns=None, exclude=None) -> "_Pipeline":
        self._ensure()
        s = salt or self._salt
        self._df = _core.mask_frame(self._df, salt=s, columns=columns, exclude=exclude)
        return self

    def select(self, cols_) -> "_Pipeline":
        self._ensure()
        self._df = select(self._df, cols_)
        return self

    def drop(self, cols_) -> "_Pipeline":
        self._ensure()
        self._df = drop(self._df, cols_)
        return self

    def rename(self, mapping: dict) -> "_Pipeline":
        self._ensure()
        self._df = rename(self._df, mapping)
        return self

    def sort(self, by, *, desc: bool = False, ascending=None) -> "_Pipeline":
        self._ensure()
        self._df = sort(self._df, by, desc=desc, ascending=ascending)
        return self

    def groupby(self, by, agg, **kwargs) -> "_Pipeline":
        self._ensure()
        self._df = groupby(self._df, by, agg, **kwargs)
        return self

    def add_column(self, **col_exprs) -> "_Pipeline":
        self._ensure()
        self._df = add_column(self._df, **col_exprs)
        return self

    def cast(self, schema_: dict) -> "_Pipeline":
        self._ensure()
        self._df = cast(self._df, schema_)
        return self

    def fill_null(self, value) -> "_Pipeline":
        self._ensure()
        self._df = fill_null(self._df, value)
        return self

    def unique(self, subset=None, *, keep="first") -> "_Pipeline":
        self._ensure()
        self._df = unique(self._df, subset=subset, keep=keep)
        return self

    def head(self, n: int = 5) -> "_Pipeline":
        self._ensure()
        self._df = head(self._df, n)
        return self

    def tail(self, n: int = 5) -> "_Pipeline":
        self._ensure()
        self._df = tail(self._df, n)
        return self

    # ── Saída ─────────────────────────────────────────────────────────────

    def store(self, path: str, *, key: Optional[str] = None,
              salt: Optional[str] = None, overwrite: bool = True,
              **kwargs) -> None:
        """Salva o resultado como .dlk."""
        self._ensure()
        store(self._df, path, key=key or self._key,
              salt=salt or self._salt, overwrite=overwrite, **kwargs)

    def write(self, path: str, **kwargs) -> None:
        """Escreve em formato detectado pela extensão (csv, parquet, xlsx...)."""
        self._ensure()
        from datalock.analytics import write as _w
        _w(self._df, str(path), **kwargs)

    def collect(self) -> Union[pl.DataFrame, pd.DataFrame]:
        """Retorna o DataFrame resultante."""
        self._ensure()
        return self._df

    def to_pandas(self) -> pd.DataFrame:
        """Retorna como pd.DataFrame."""
        self._ensure()
        if isinstance(self._df, pl.DataFrame):
            return self._df.to_pandas()
        return self._df

    def to_polars(self) -> pl.DataFrame:
        """Retorna como pl.DataFrame."""
        self._ensure()
        if isinstance(self._df, pd.DataFrame):
            return pl.from_pandas(self._df)
        return self._df

    def __repr__(self) -> str:
        shape = self._df.shape if self._df is not None else None
        return f"Pipeline(shape={shape})"

    def _ensure(self):
        if self._df is None:
            raise RuntimeError(
                "Pipeline sem dados. Use dd.pipe('arquivo.parquet'), "
                ".read('arquivo') ou .sql('SELECT ...')."
            )


def pipe(source=None, *, key: Optional[str] = None, salt: Optional[str] = None) -> _Pipeline:
    """
    Inicia um pipeline fluente de processamento de dados.

    Args:
        source: Arquivo, DataFrame ou None (começa vazio para usar .sql()).
        key:    Chave AES padrão para arquivos .dlk no pipeline.
        salt:   Salt HMAC padrão para mascaramento no pipeline.

    Returns:
        _Pipeline com métodos encadeáveis: .where().mask().groupby().collect()

    Exemplos:
        # Lê → filtra → mascara → salva
        (dd.pipe("clientes.parquet")
           .where(uf="SP")
           .mask(salt=SALT)
           .store("clientes_sp.dlk", key=KEY))

        # Pipeline com SQL
        result = (
            dd.pipe()
            .sql("SELECT * FROM read_parquet('dados.parquet') WHERE uf='SP'")
            .mask(salt=SALT)
            .collect()
        )
    """
    return _Pipeline(source, key=key, salt=salt)


# ---------------------------------------------------------------------------
# Helpers internos (não exportados)
# ---------------------------------------------------------------------------

def _peek_lgs_content_type(path: Path, key: str) -> str:
    ok, info = SecureFile.verify(str(path), key=key)
    return info.get("content_type", "raw_dataframe") if ok else "raw_dataframe"


def _read_lgs_model(path: Path, key: str, size: int) -> pd.DataFrame:
    try:
        from datalock.generators.tabular_generative import TabularGenerativeModel
        model = TabularGenerativeModel.load(str(path), key=key)
        return model.generate(size)
    except Exception as exc:
        raise RuntimeError(f"Não foi possível carregar modelo generativo: {exc}") from exc


def _read_chunked(path: str, *, salt, chunksize, verbose, **kwargs):
    """Gerador de chunks para CSV (quando chunksize= passado a dd.read())."""
    return stream(path, salt=salt, chunksize=chunksize, verbose=verbose, **kwargs)


def _validate_frames_dict(source: Dict) -> None:
    if not source:
        raise ValueError("store(): dict de frames está vazio.")
    for name, df_item in source.items():
        if not isinstance(df_item, pd.DataFrame):
            raise TypeError(
                f"store(): valor para frame '{name}' deve ser pd.DataFrame, "
                f"recebido {type(df_item).__name__}."
            )


def _validate_key_salt_distinct(key: Optional[str], salt: Optional[str]) -> None:
    if key and salt and key == salt:
        raise ValueError(
            "key= e salt= não devem ser iguais. "
            "Use valores distintos gerados com dd.generate_salt()."
        )


def _warn_if_pii(df_: pd.DataFrame) -> None:
    try:
        det = PIIDetector()
        df_pl = pl.from_pandas(df_)
        reports = det.detect_sampled(df_pl)
        if reports:
            cols_detected = list(reports.keys())
            warnings.warn(
                f"store() sem key: arquivo não cifrado contém possível PII "
                f"nas colunas {cols_detected}. "
                f"Use key= para criptografar ou anonymize=True para mascarar.",
                UserWarning,
                stacklevel=4,
            )
    except Exception:
        pass


def _df_is_masked(df_: pd.DataFrame) -> bool:
    """Heurística: verifica se o DF parece mascarado (tokens hex-16)."""
    import re as _re
    _hex16 = _re.compile(r"^[0-9a-f]{16}$")
    try:
        str_cols = df_.select_dtypes(include=["object", "string"]).columns[:3]
        if not len(str_cols):
            return False
        for c in str_cols:
            sample = df_[c].dropna().astype(str).head(10)
            if sample.map(lambda v: bool(_hex16.match(v))).mean() > 0.8:
                return True
    except Exception:
        pass
    return False


def _compute_privacy_score(df_pd, reports):
    """Computa privacy score para inclusion em profile()."""
    try:
        from datalock.privacy_score import calculate as _ps_calc
        score = _ps_calc(df_pd, reports)
        return score.to_dict()
    except Exception:
        return None


def _infer_mask_strategy(orig: pd.Series, masked: pd.Series) -> str:
    """Heurística: infere a estratégia de mascaramento pelo padrão de valores."""
    import re as _re
    sample = masked.dropna().astype(str).head(20)
    if sample.empty:
        return "unknown"
    if (sample == "REDACTED").mean() > 0.8:
        return "redact"
    if sample.map(lambda v: bool(_re.match(r"^[0-9a-f]{16}$", v))).mean() > 0.8:
        return "hash"
    if sample.map(lambda v: bool(_re.match(r"^\d{4}-\d{4}$", v))).mean() > 0.5:
        return "generalize_date"
    if sample.map(lambda v: bool(_re.match(r"^\d{5}-XXX$", v))).mean() > 0.5:
        return "truncate_cep"
    if masked.isna().mean() > 0.8:
        return "suppress"
    if pd.api.types.is_numeric_dtype(orig) and pd.api.types.is_numeric_dtype(masked):
        return "mock_numeric"
    return "mock_category"
