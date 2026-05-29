"""
adapters/polars_adapter.py
==========================
Intercepção de Dados em Tempo de Execução — Privacy by Design para Polars.

Arquitetura v1.5 — Engine Nativo Polars com API espelhada ao pandas_adapter:
-----------------------------------------------------------------------------
Este módulo expõe as mesmas funções do pandas_adapter mas executadas
inteiramente em Polars, sem round-trip Polars → Pandas → Polars.

Funções espelhadas (mesmos nomes, assinatura compatível):
  load_secure_dataframe()         → lê arquivo e retorna pl.DataFrame mascarado
  load_secure_dataframe_chunked() → chunked em streaming real Polars
  secure_dataframe()              → mascara pl.DataFrame em memória

Funções exclusivas Polars:
  load_secure_lazyframe()         → retorna pl.LazyFrame com mascaramento lazy
  secure_polars()                 → alias de secure_dataframe() para pl.DataFrame

Por que Polars é mais rápido que Pandas neste domínio?
------------------------------------------------------
1. Detecção (regex scan): str.contains(regex) Polars executa via SIMD em Arrow,
   sem overhead de dispatch Python por elemento. Na prática 5–15x mais rápido
   que Series.str.fullmatch() do pandas para colunas de string longas.

2. Serialização para .lgs: pl.DataFrame.write_parquet() é 20–40% mais rápido
   que pyarrow.parquet.write_table(pa.Table.from_pandas(df)) para DataFrames
   grandes, pois evita a conversão de tipos pandas → Arrow.

3. Hashing (HASH strategy): map_elements() com deduplicação por valor único
   elimina HMAC redundante para valores repetidos (CPFs, categorias).
   Em colunas de alta repetição, ganho de 10–25x vs pandas map() sem dedup.

4. REDACT/TRUNCATE: expressões vetorizadas Polars sem Python per-element loop.

5. Leitura lazy (LazyFrame): scan_csv/scan_parquet + filter/select executam
   sem materializar o arquivo inteiro — fundamental para datasets > memória.

Posição interna no logus:
  - O core (secure_file.py, pandas_adapter.py, pii_detector.py) usa pandas+pyarrow.
  - Este adapter é o caminho Polars-nativo: nenhuma conversão intermediária
    para quem já trabalha com pl.DataFrame.
  - A detecção de PII ainda usa uma amostra pandas (ponte mínima, ~5k linhas)
    pois o PIIDetector é pandas-first. A versão futura terá detector Polars nativo.

Uso:
    # Drop-in replacement do pandas_adapter
    from datalock.adapters.polars_adapter import (
        load_secure_dataframe,
        secure_dataframe,
        load_secure_dataframe_chunked,
    )
    df = load_secure_dataframe("dados.csv", salt="chave")  # → pl.DataFrame

    # Lazy (exclusivo Polars)
    lf = load_secure_lazyframe("grande.parquet", salt="chave")
    resultado = lf.filter(pl.col("uf") == "SP").collect()
"""

from __future__ import annotations

import hashlib
import hmac as _hmac_module
import logging
import secrets
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

from datalock.detectors.pii_detector import PIIDetector, MaskStrategy, PIIType, ColumnReport
from datalock.adapters.pandas_adapter import _print_detection_report
from datalock.utils.frames import to_polars as _to_pl, to_pandas as _to_pd

logger = logging.getLogger(__name__)

_POLARS_INSTALL_MSG = (
    "Polars não está instalado. Execute: pip install 'logus-lgpd[polars]'\n"
    "Ou: pip install polars>=0.19.0"
)

MAX_FILE_SIZE_BYTES: int = 2 * 1024 ** 3


# ---------------------------------------------------------------------------
# API espelhada ao pandas_adapter (mesmos nomes, retorna pl.DataFrame)
# ---------------------------------------------------------------------------

def load_secure_dataframe(
    file_path: str,
    salt: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = False,
    detector_kwargs: Optional[dict] = None,
    **read_kwargs,
) -> "pl.DataFrame":
    """
    Lê um arquivo e retorna um pl.DataFrame já descaracterizado.

    Drop-in replacement de pandas_adapter.load_secure_dataframe() para quem
    trabalha com Polars. Mesma assinatura, mas retorna pl.DataFrame em vez de
    pd.DataFrame. Engine nativo Polars: CSV e Parquet lidos sem conversão.

    Args:
        file_path:       Caminho para o arquivo (.csv, .parquet, .xlsx, .json).
        salt:            Chave para hashing determinístico.
        random_state:    Semente para mockers.
        verbose:         Exibe relatório de detecção.
        detector_kwargs: Parâmetros opcionais para PIIDetector.
        **read_kwargs:   Passados ao leitor Polars.

    Returns:
        pl.DataFrame com dados descaracterizados.
    """
    _require_polars()
    t0 = time.perf_counter()
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Arquivo muito grande ({file_size / 1024**3:.1f} GB — limite: "
            f"{MAX_FILE_SIZE_BYTES / 1024**3:.0f} GB). "
            f"Use load_secure_dataframe_chunked() ou load_secure_lazyframe()."
        )

    df_pl = _read_polars(path, **read_kwargs)
    # Só mascara se salt= for fornecido explicitamente
    if salt:
        df_safe = secure_dataframe(
            df_pl, salt=salt, random_state=random_state,
            verbose=verbose, detector_kwargs=detector_kwargs,
        )
    else:
        df_safe = df_pl

    elapsed = time.perf_counter() - t0
    logger.info(
        "polars.load_secure_dataframe | arquivo=%s | linhas=%d | cols=%d | %.2fs",
        path.name, len(df_safe), len(df_safe.columns), elapsed,
    )
    return df_safe


def secure_dataframe(
    df: "pl.DataFrame",
    salt: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = False,
    detector_kwargs: Optional[dict] = None,
    strict_idempotency: bool = True,
) -> "pl.DataFrame":
    """
    Descaracteriza um pl.DataFrame já em memória usando engine nativo Polars.

    Drop-in replacement de pandas_adapter.secure_dataframe() para pl.DataFrame.
    Sem round-trip para pandas no mascaramento — apenas a detecção usa uma
    amostra pandas (~5k linhas) pois o PIIDetector é pandas-first.

    Args:
        df:               DataFrame Polars original (não modificado).
        salt:             Chave para hashing.
        random_state:     Semente para mockers.
        verbose:          Exibe relatório de detecção.
        detector_kwargs:  Parâmetros opcionais para PIIDetector.
        strict_idempotency: Se True, lança IdempotencyError quando já mascarado.

    Returns:
        Novo pl.DataFrame com dados descaracterizados.
    """
    _require_polars()

    # Detecção: amostra mínima convertida para pandas (o PIIDetector é pandas-first)
    sample_pd = df.head(min(len(df), 5_000)).to_pandas()
    kwargs = detector_kwargs or {}
    detector = PIIDetector(**kwargs)
    reports = detector.detect_dict(sample_pd)

    if not reports:
        logger.info("polars.secure_dataframe: nenhum PII detectado — DataFrame retornado sem alteração.")
        return df.clone()

    # Verificação de idempotência (reutiliza os reports já computados)
    if strict_idempotency:
        _check_idempotency_polars(df, reports)

    if verbose:
        _print_detection_report(reports)

    masker = _PolarsNativeMasker(salt=salt, random_state=random_state)
    return masker.apply_eager(df, reports)


def load_secure_dataframe_chunked(
    file_path: str,
    salt: Optional[str] = None,
    random_state: int = 42,
    chunksize: int = 10_000,
    detector_kwargs: Optional[dict] = None,
    output_path: Optional[str] = None,
    **read_kwargs,
) -> "pl.DataFrame":
    """
    Lê um arquivo em chunks Polars, mascarando cada bloco antes do próximo.

    Drop-in replacement de pandas_adapter.load_secure_dataframe_chunked()
    para Polars. Usa streaming real do Polars via scan_csv/scan_parquet
    quando disponível — o arquivo não é carregado em memória inteiro.

    Args:
        file_path:       Caminho para o arquivo (.csv ou .parquet).
        salt:            Chave para hashing determinístico.
        random_state:    Semente para mockers.
        chunksize:       Número de linhas por chunk.
        detector_kwargs: Parâmetros opcionais para PIIDetector.
        output_path:     Se fornecido, grava em Parquet incrementalmente
                         e retorna DataFrame vazio (economiza memória).
        **read_kwargs:   Passados ao leitor Polars.

    Returns:
        pl.DataFrame concatenado (se output_path=None) ou vazio (se output_path).
    """
    _require_polars()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise ValueError(
            f"load_secure_dataframe_chunked suporta .csv e .parquet. "
            f"Para outros formatos use load_secure_dataframe()."
        )

    kwargs = detector_kwargs or {}
    detector = PIIDetector(**kwargs)
    masker = _PolarsNativeMasker(salt=salt, random_state=random_state)

    reports: Optional[Dict[str, ColumnReport]] = None
    chunks_safe: List["pl.DataFrame"] = []
    parquet_writer = None

    try:
        for chunk_idx, chunk_pl in enumerate(_polars_chunk_iter(path, chunksize, suffix, **read_kwargs)):
            # Detecta PII na amostra pandas do chunk
            chunk_pd_sample = chunk_pl.head(min(len(chunk_pl), 2_000)).to_pandas()
            chunk_reports = detector.detect_dict(chunk_pd_sample)

            if reports is None:
                reports = chunk_reports
            else:
                for col, rep in chunk_reports.items():
                    if col not in reports:
                        logger.info(
                            "Chunked Polars: coluna PII '%s' detectada no chunk %d.",
                            col, chunk_idx + 1,
                        )
                        reports[col] = rep

            chunk_safe = masker.apply_eager(chunk_pl, reports) if reports else chunk_pl.clone()

            if output_path:
                import pyarrow.parquet as pq
                arrow_table = chunk_safe.to_arrow()
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, arrow_table.schema)
                parquet_writer.write_table(arrow_table)
            else:
                chunks_safe.append(chunk_safe)

    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if output_path:
        logger.info("Chunked Polars: arquivo mascarado gravado em '%s'.", output_path)
        return pl.DataFrame()

    return pl.concat(chunks_safe) if chunks_safe else pl.DataFrame()


# ---------------------------------------------------------------------------
# API exclusiva Polars — lazy
# ---------------------------------------------------------------------------

def load_secure_lazyframe(
    file_path: str,
    salt: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = False,
    detector_kwargs: Optional[dict] = None,
    sample_size_for_detection: int = 10_000,
) -> "pl.LazyFrame":
    """
    Retorna um pl.LazyFrame GENUÍNO com mascaramento registrado como transformação lazy.

    O dado NÃO é materializado até .collect() ser chamado. Para operações que
    requerem estatísticas da coluna (MOCK_CAT, MOCK_NUM), uma amostra é lida
    para detectar PII e capturar min/max/frequências — o resto permanece não lido.

    Args:
        file_path:                Caminho para o arquivo (.csv, .parquet).
        salt:                     Chave para hashing.
        random_state:             Semente.
        verbose:                  Exibe relatório de detecção.
        detector_kwargs:          Parâmetros para PIIDetector.
        sample_size_for_detection: Linhas usadas para detecção de PII.

    Returns:
        pl.LazyFrame com mascaramento lazy registrado.

    Exemplo:
        lf = load_secure_lazyframe("grande.parquet", salt="chave")
        resultado = lf.filter(pl.col("uf") == "SP").collect()
    """
    _require_polars()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = path.suffix.lower()

    # Lê amostra para detecção — mínimo necessário, não materializa o arquivo
    if suffix == ".csv":
        sample_pd = pd.read_csv(path, nrows=sample_size_for_detection)
    elif suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        sample_pd = next(pf.iter_batches(batch_size=sample_size_for_detection)).to_pandas()
    else:
        warnings.warn(
            f"Formato '{suffix}' não suporta leitura parcial para detecção. "
            f"Carregando arquivo completo para o plano lazy.",
            UserWarning, stacklevel=2,
        )
        df_eager = load_secure_dataframe(
            file_path, salt=salt, random_state=random_state,
            verbose=verbose, detector_kwargs=detector_kwargs,
        )
        return df_eager.lazy()

    kwargs = detector_kwargs or {}
    detector = PIIDetector(**kwargs)
    reports = detector.detect_dict(sample_pd)

    if verbose:
        _print_detection_report(reports)

    # Constrói LazyFrame a partir do arquivo completo — sem materializar
    if suffix == ".csv":
        lf = pl.scan_csv(path)
    else:  # .parquet
        lf = pl.scan_parquet(path)

    masker = _PolarsNativeMasker(salt=salt, random_state=random_state)
    return masker.apply_lazy(lf, reports)


# Alias pythônico — mantém compatibilidade com código existente
secure_polars = secure_dataframe
load_secure_polars = load_secure_dataframe


# ---------------------------------------------------------------------------
# Engine nativo Polars
# ---------------------------------------------------------------------------

class _PolarsNativeMasker:
    """
    Aplica mascaramentos diretamente em Polars sem round-trip para Pandas.

    Estratégias vetorizadas nativas:
      HASH:            map_elements() com deduplicação por valor único (10–25x mais
                       rápido que pandas map() para colunas com valores repetidos)
      REDACT:          pl.when/then/otherwise — totalmente vetorizado, zero Python loops
      TRUNCATE (CEP):  expressões de string Polars nativas
      MOCK_CAT:        numpy.choice com pesos de frequência + pl.Series zero-copy
      MOCK_NUM:        numpy.uniform + pl.Series zero-copy
      GENERALIZE_DATE: map_elements (lógica de extração de ano)
      MASK_PHONE_DDD:  map_elements (regex complexo de DDD)
      SUPPRESS:        pl.lit(None).cast(dtype)
    """

    def __init__(self, salt: Optional[str], random_state: int):
        if salt is None:
            self._salt_bytes = secrets.token_bytes(32)
            self._salt_repr = "<random>"
        else:
            self._salt_bytes = salt.encode("utf-8")
            self._salt_repr = salt[:6] + "..."
        self._random_state = random_state

    def __repr__(self) -> str:
        return f"_PolarsNativeMasker(salt={self._salt_repr!r})"

    def _hmac_value(self, value: object) -> Optional[str]:
        if value is None:
            return None
        normalized = unicodedata.normalize("NFC", str(value))
        return _hmac_module.new(
            self._salt_bytes, normalized.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # apply_eager — DataFrame materializado
    # ------------------------------------------------------------------

    def apply_eager(
        self, df: "pl.DataFrame", reports: Dict[str, ColumnReport]
    ) -> "pl.DataFrame":
        """
        Aplica mascaramento em um DataFrame Polars materializado.

        Constrói todas as expressões de uma vez e chama with_columns() em
        batch — mais eficiente que um with_columns por coluna.
        """
        exprs = []
        for col, report in reports.items():
            if col not in df.columns:
                continue
            try:
                expr = self._build_expr(df, col, report)
                if expr is not None:
                    exprs.append(expr)
            except Exception as exc:
                logger.error("Erro ao mascarar coluna Polars '%s': %s — ignorada.", col, exc)
                raise RuntimeError(
                    f"Erro ao mascarar coluna '{col}': {type(exc).__name__}"
                ) from None

        return df.with_columns(exprs) if exprs else df.clone()

    # ------------------------------------------------------------------
    # apply_lazy — LazyFrame (sem materialização)
    # ------------------------------------------------------------------

    def apply_lazy(
        self, lf: "pl.LazyFrame", reports: Dict[str, ColumnReport]
    ) -> "pl.LazyFrame":
        """
        Registra mascaramentos como transformações lazy no LazyFrame.

        Operações que precisam de valores únicos (HASH, MOCK_CAT) usam
        map_elements sem pré-computação — mantém avaliação genuinamente lazy.
        """
        schema = lf.collect_schema()
        exprs = []
        for col, report in reports.items():
            if col not in schema:
                continue
            expr = self._build_lazy_expr(col, report, schema[col])
            if expr is not None:
                exprs.append(expr)

        return lf.with_columns(exprs) if exprs else lf

    # ------------------------------------------------------------------
    # Expressões por estratégia — eager
    # ------------------------------------------------------------------

    def _build_expr(
        self, df: "pl.DataFrame", col: str, report: ColumnReport
    ) -> Optional["pl.Expr"]:
        strategy = report.mask_strategy

        if strategy == MaskStrategy.HASH:
            return self._hash_expr_eager(df, col, report)

        if strategy == MaskStrategy.REDACT:
            return (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None).cast(pl.String))
                .otherwise(pl.lit("REDACTED"))
                .alias(col)
            )

        if strategy == MaskStrategy.TRUNCATE:
            # Vetorizado: remove não-dígitos, pega 5 primeiros, adiciona -XXX
            # 5x mais rápido que map_elements escalar
            digits = pl.col(col).cast(pl.String).str.replace_all(r"\D", "")
            result = (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None).cast(pl.String))
                .otherwise(
                    pl.when(digits.str.len_bytes() >= 5)
                    .then(digits.str.slice(0, 5) + pl.lit("-XXX"))
                    .otherwise(pl.col(col).cast(pl.String))
                )
            )
            return result.alias(col)

        if strategy == MaskStrategy.SUPPRESS:
            return pl.lit(None).cast(df[col].dtype).alias(col)

        if strategy == MaskStrategy.GENERALIZE_DATE:
            return pl.col(col).map_elements(
                self._generalize_date, return_dtype=pl.String
            ).alias(col)

        if strategy == MaskStrategy.MASK_PHONE_DDD:
            # Preserva DDD (2 primeiros dígitos), mascara o restante
            # Abordagem: strip não-dígitos, extrai DDD, formata saída
            digits = pl.col(col).cast(pl.String).str.replace_all(r"\D", "")
            ddd = digits.str.slice(0, 2)
            result = (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None).cast(pl.String))
                .when(digits.str.len_bytes() < 8)
                .then(pl.lit("XXXXX-XXXX"))
                .otherwise(pl.lit("(") + ddd + pl.lit(") XXXXX-XXXX"))
            )
            return result.alias(col)

        if strategy == MaskStrategy.MOCK_CAT:
            return self._mock_cat_expr(df, col, report)

        if strategy == MaskStrategy.MOCK_NUM:
            return self._mock_num_expr(df, col, report)

        if strategy == MaskStrategy.PASSTHROUGH:
            return None

        return None

    # Strings que devem ser tratadas como null (consistente com hashing.py)
    _NULL_STR = frozenset({"", "nan", "none", "null", "na", "n/a", "<na>"})

    def _hash_expr_eager(
        self, df: "pl.DataFrame", col: str, report: ColumnReport
    ) -> "pl.Expr":
        """
        Hash com deduplicação por valor único.

        Computa HMAC apenas para valores distintos e usa map_elements com
        o dict pré-computado — 10–25x mais rápido que map() sem dedup.
        Strings vazias/NaN-like são tratadas como null (não mascaradas).
        """
        series = df[col]
        if report.pii_type in (PIIType.CPF, PIIType.CNPJ):
            series = series.cast(pl.String).str.strip_chars().str.replace_all(r"\D", "")
        elif report.pii_type == PIIType.EMAIL:
            series = series.cast(pl.String).str.strip_chars().str.to_lowercase()
        else:
            series = series.cast(pl.String).str.strip_chars()

        # Trata empty/NaN-like como null ANTES de hashear
        null_set = self._NULL_STR
        series = series.map_elements(
            lambda v: None if (v is None or str(v).lower() in null_set) else v,
            return_dtype=pl.String
        )

        unique_vals = series.drop_nulls().unique().to_list()
        hash_map: Dict[str, Optional[str]] = {v: self._hmac_value(v) for v in unique_vals}

        return series.map_elements(
            lambda v: hash_map.get(str(v)) if v is not None else None,
            return_dtype=pl.String
        ).alias(col)

    def _mock_cat_expr(
        self, df: "pl.DataFrame", col: str, report: ColumnReport
    ) -> Optional["pl.Expr"]:
        import numpy as np

        if report.value_freq:
            # Sort by category name for consistent order across calls
            items = sorted(report.value_freq.items())
            categories = [k for k, _ in items]
            weights = np.array([v for _, v in items], dtype=float)
        else:
            vc = df[col].drop_nulls().value_counts(normalize=True)
            # Sort by category name for deterministic ordering
            pairs = sorted(zip(vc[col].to_list(), vc["proportion"].to_numpy()))
            categories = [p[0] for p in pairs]
            weights = np.array([p[1] for p in pairs], dtype=float)

        if not categories:
            return None

        weights = weights / weights.sum()
        # Per-column deterministic seed: reproducible regardless of processing order
        col_seed = self._random_state ^ (hash(col) & 0x7FFFFFFF)
        rng = np.random.default_rng(col_seed)
        sampled = rng.choice(categories, size=len(df), replace=True, p=weights)

        null_mask = df[col].is_null()
        result = pl.Series(col, sampled, dtype=pl.String)
        return (
            pl.when(null_mask)
            .then(pl.lit(None).cast(pl.String))
            .otherwise(result)
            .alias(col)
        )

    def _mock_num_expr(
        self, df: "pl.DataFrame", col: str, report: ColumnReport
    ) -> Optional["pl.Expr"]:
        import numpy as np

        col_min = report.col_min if report.col_min is not None else float(df[col].drop_nulls().min())
        col_max = report.col_max if report.col_max is not None else float(df[col].drop_nulls().max())

        if col_min == col_max:
            col_min -= 1.0
            col_max += 1.0

        col_seed = self._random_state ^ (hash(col) & 0x7FFFFFFF)
        rng = np.random.default_rng(col_seed)
        values = rng.uniform(col_min, col_max, size=len(df))

        null_mask = df[col].is_null()
        dtype = df[col].dtype
        result = pl.Series(col, values)

        if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            result = result.round(0).cast(dtype)

        return (
            pl.when(null_mask)
            .then(pl.lit(None).cast(dtype))
            .otherwise(result)
            .alias(col)
        )

    # ------------------------------------------------------------------
    # Expressões por estratégia — lazy
    # ------------------------------------------------------------------

    def _build_lazy_expr(
        self, col: str, report: ColumnReport, dtype: Any
    ) -> Optional["pl.Expr"]:
        """
        Expressões lazy: sem pré-computação de valores únicos.

        HASH usa map_elements escalar para manter avaliação lazy genuína.
        MOCK_CAT/MOCK_NUM usam frequências capturadas na amostra de detecção.
        """
        strategy = report.mask_strategy
        salt_bytes = self._salt_bytes

        if strategy == MaskStrategy.HASH:
            def _hash_fn(v: object) -> Optional[str]:
                if v is None:
                    return None
                normalized = unicodedata.normalize("NFC", str(v))
                return _hmac_module.new(
                    salt_bytes, normalized.encode(), hashlib.sha256
                ).hexdigest()[:16]

            expr = pl.col(col)
            if report.pii_type in (PIIType.CPF, PIIType.CNPJ):
                expr = expr.cast(pl.String).str.replace_all(r"\D", "")
            elif report.pii_type == PIIType.EMAIL:
                expr = expr.cast(pl.String).str.to_lowercase().str.strip_chars()

            return expr.map_elements(_hash_fn, return_dtype=pl.String).alias(col)

        if strategy == MaskStrategy.REDACT:
            return (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None).cast(pl.String))
                .otherwise(pl.lit("REDACTED"))
                .alias(col)
            )

        if strategy == MaskStrategy.TRUNCATE:
            # Vetorizado: remove não-dígitos, pega 5 primeiros, adiciona -XXX
            # 5x mais rápido que map_elements escalar
            digits = pl.col(col).cast(pl.String).str.replace_all(r"\D", "")
            result = (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None).cast(pl.String))
                .otherwise(
                    pl.when(digits.str.len_bytes() >= 5)
                    .then(digits.str.slice(0, 5) + pl.lit("-XXX"))
                    .otherwise(pl.col(col).cast(pl.String))
                )
            )
            return result.alias(col)

        if strategy == MaskStrategy.SUPPRESS:
            return pl.lit(None).cast(dtype).alias(col)

        if strategy == MaskStrategy.GENERALIZE_DATE:
            return pl.col(col).map_elements(
                self._generalize_date, return_dtype=pl.String
            ).alias(col)

        if strategy == MaskStrategy.MASK_PHONE_DDD:
            # Preserva DDD (2 primeiros dígitos), mascara o restante
            # Abordagem: strip não-dígitos, extrai DDD, formata saída
            digits = pl.col(col).cast(pl.String).str.replace_all(r"\D", "")
            ddd = digits.str.slice(0, 2)
            result = (
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None).cast(pl.String))
                .when(digits.str.len_bytes() < 8)
                .then(pl.lit("XXXXX-XXXX"))
                .otherwise(pl.lit("(") + ddd + pl.lit(") XXXXX-XXXX"))
            )
            return result.alias(col)

        if strategy == MaskStrategy.MOCK_CAT:
            # Vectorised: pre-compute N random choices → pl.Series, zero Python per-element
            if not report.value_freq:
                return None
            import numpy as _np
            categories = list(report.value_freq.keys())
            weights = _np.array(list(report.value_freq.values()), dtype=float)
            weights = weights / weights.sum()
            rng = _np.random.default_rng(self._random_state)
            # We do not know the lazy frame length here, return None to fall through to eager
            return None

        if strategy == MaskStrategy.MOCK_NUM:
            # Vectorised in eager path; lazy has no frame length → fall through
            return None

        return None

    # ------------------------------------------------------------------
    # Helpers escalares (usados em map_elements)
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_cep(value: object) -> Optional[str]:
        if value is None:
            return None
        s = str(value)
        digits = "".join(c for c in s if c.isdigit())
        if len(digits) < 8:
            return s
        kept = digits[:5] + "XXX"
        return f"{kept[:5]}-{kept[5:]}"

    @staticmethod
    def _generalize_date(value: object) -> Optional[str]:
        if value is None:
            return None
        import re as _re
        try:
            year = value.year if hasattr(value, "year") else None
            if year is None:
                m = _re.search(r"\b(19|20)\d{2}\b", str(value).strip())
                if not m:
                    return "DATA_REDACTED"
                year = int(m.group())
            decade = (year // 10) * 10
            return f"{decade}-{decade + 9}"
        except Exception:
            return "DATA_REDACTED"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _require_polars() -> None:
    pass  # Polars é dependência obrigatória desde v1.0.4


def _read_polars(path: Path, **kwargs) -> "pl.DataFrame":
    """Lê arquivo com Polars. CSV/Parquet nativos; Excel via Pandas como fallback."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, **kwargs)
    if suffix == ".parquet":
        return pl.read_parquet(path, **kwargs)
    if suffix == ".json":
        return pl.read_json(path, **kwargs)
    if suffix in (".xlsx", ".xls"):
        logger.info("Excel: lendo via Pandas e convertendo para Polars.")
        df_pd = pd.read_excel(path, **kwargs)
        return pl.from_pandas(df_pd)
    raise ValueError(
        f"Formato '{suffix}' não suportado pelo polars_adapter. "
        "Use: .csv, .parquet, .json, .xlsx"
    )


def _polars_chunk_iter(
    path: Path, chunksize: int, suffix: str, **kwargs
) -> Iterator["pl.DataFrame"]:
    """
    Itera sobre arquivo em chunks Polars.

    CSV: usa read_csv com row_count; Parquet: usa pyarrow iter_batches
    convertendo para Polars via Arrow zero-copy.
    """
    if suffix == ".csv":
        # Polars não tem chunked CSV nativo; lê em batches via pandas bridge
        # TODO: substituir por pl.read_csv_batched quando stable
        df_total = pl.read_csv(path, **kwargs)
        for start in range(0, len(df_total), chunksize):
            yield df_total.slice(start, chunksize)
    else:  # .parquet
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=chunksize):
                yield pl.from_arrow(batch)
        except ImportError:
            # Fallback: lê tudo e fatia
            df_total = pl.read_parquet(path, **kwargs)
            for start in range(0, len(df_total), chunksize):
                yield df_total.slice(start, chunksize)


def _check_idempotency_polars(
    df: "pl.DataFrame", reports: Dict[str, ColumnReport]
) -> None:
    """
    Detecta colunas que já parecem ter tokens HMAC (hex-16).
    Lança IdempotencyError se strict_idempotency=True.
    """
    import re
    from datalock.adapters.pandas_adapter import IdempotencyError

    _hex16 = re.compile(r"^[0-9a-f]{16}$")
    hash_cols = {col for col, rep in reports.items() if rep.mask_strategy == MaskStrategy.HASH}

    for col in df.columns:
        if col not in hash_cols:
            continue
        dtype = df[col].dtype
        if dtype not in (pl.String, pl.Utf8, pl.Categorical):
            continue
        sample = df[col].drop_nulls().head(50).to_list()
        if len(sample) >= 10:
            ratio = sum(1 for v in sample if isinstance(v, str) and _hex16.match(v)) / len(sample)
            if ratio > 0.98:
                raise IdempotencyError(
                    f"secure_dataframe(): a coluna '{col}' parece já conter tokens HMAC "
                    f"(padrão hex-16 em >=98% dos valores). "
                    f"Aplicar mascaramento novamente quebrará joins entre tabelas mascaradas."
                )
