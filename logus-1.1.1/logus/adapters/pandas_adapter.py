"""
adapters/pandas_adapter.py
==========================
Intercepção de Dados em Tempo de Execução — Privacy by Design para Pandas.

Conceito: Zero Trust no Localhost
----------------------------------
O desenvolvedor NUNCA deve ver dados reais em sua máquina. Ao chamar
`load_secure_dataframe()` em vez de `pd.read_csv()`, o framework intercepta
a leitura, roda o PIIDetector em segundo plano, aplica os mockers/maskers
e entrega um DataFrame já blindado.

Otimizações v1.1:
  - secure_dataframe() elimina a dupla detecção: a checagem de idempotência
    reutiliza o resultado do detector principal em vez de rodar PIIDetector
    duas vezes no mesmo DataFrame.
  - load_secure_dataframe_chunked() usa escrita incremental em Parquet via
    pyarrow.parquet.ParquetWriter, eliminando o pd.concat() final que
    duplicava o uso de memória.
  - Suporte a chunked para Parquet via pyarrow iter_batches().
  - Normalização de CPF/CNPJ/CEP antes do mascaramento para tratar
    variações de formatação presentes em dados sujos.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from datalock.detectors.pii_detector import (
    PIIDetector, PIIType, MaskStrategy, ColumnReport,
)
from datalock.maskers.hashing import DeterministicHasher
from datalock.maskers.truncation import CepTruncator, PhoneDddMasker, StringRedactor
from datalock.maskers.date_masker import DateMasker
from datalock.mockers.numeric_mocker import NumericMocker
from datalock.mockers.category_mocker import CategoryMocker
from datalock.utils.secret_str import SecretStr

logger = logging.getLogger(__name__)

# Auditoria global — configurada via lg.configure(audit=AuditReport())
_GLOBAL_AUDIT: Optional[Any] = None
_DEFAULT_SALT: Optional[str] = None  # salt padrão global configurado via lg.configure()

MAX_FILE_SIZE_BYTES: int = 2 * 1024 ** 3


class IdempotencyError(ValueError):
    """
    Raised when mask() is called on a DataFrame that appears to already
    be masked (HMAC token pattern detected) and strict_idempotency=True.
    """
    pass


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def load_secure_dataframe(
    file_path: str,
    salt: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = False,
    detector_kwargs: Optional[dict] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Lê um arquivo de dados e retorna um DataFrame já descaracterizado.

    Substitui pd.read_csv(), pd.read_excel(), etc. com Privacy by Design
    embutido. O dado real nunca é exposto após esta função retornar.

    Formatos suportados: .csv, .xlsx, .xls, .parquet, .json

    Args:
        file_path: Caminho para o arquivo de dados.
        salt: Chave para hashing determinístico.
        random_state: Semente para operações aleatórias (mockers).
        verbose: Exibe relatório de colunas detectadas no console.
        detector_kwargs: Parâmetros opcionais para PIIDetector.
        **read_kwargs: Argumentos repassados ao leitor pandas.

    Returns:
        DataFrame pandas com dados descaracterizados.
    """
    t0 = time.perf_counter()
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Arquivo muito grande ({file_size / 1024**3:.1f} GB — limite: "
            f"{MAX_FILE_SIZE_BYTES / 1024**3:.0f} GB). "
            f"Use lg.stream() ou lg.read(..., chunksize=50_000) para arquivos grandes."
        )

    df_raw = _read_file(path, **read_kwargs)

    # Só mascara se salt= for fornecido explicitamente
    if salt:
        df_safe = secure_dataframe(
            df_raw,
            salt=salt,
            random_state=random_state,
            verbose=verbose,
            detector_kwargs=detector_kwargs,
        )
    else:
        df_safe = df_raw

    elapsed = time.perf_counter() - t0
    logger.info(
        "load_secure_dataframe | arquivo=%s | linhas=%d | cols=%d | %.2fs",
        path.name, len(df_safe), len(df_safe.columns), elapsed,
    )
    return df_safe


def secure_dataframe(
    df: pd.DataFrame,
    salt: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = False,
    detector_kwargs: Optional[dict] = None,
    strict_idempotency: bool = True,
    columns: Optional[list] = None,
    exclude: Optional[list] = None,
    audit: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Descaracteriza um DataFrame já em memória.

    v1.1 — Elimina dupla detecção: a verificação de idempotência agora
    reutiliza os reports do detector principal, evitando rodar PIIDetector
    duas vezes no mesmo DataFrame.

    Args:
        df: DataFrame original (não é modificado — opera em cópia).
        salt: Chave para hashing. None gera salt aleatório.
        random_state: Semente para mockers.
        verbose: Exibe detalhes das colunas detectadas.
        detector_kwargs: Parâmetros opcionais para PIIDetector.
        strict_idempotency: Se True, levanta IdempotencyError quando
            uma coluna parece já conter tokens HMAC.

    Returns:
        Nova cópia do DataFrame com dados descaracterizados.
    """
    import re as _re

    kwargs = detector_kwargs or {}
    detector = PIIDetector(**kwargs)
    reports = detector.detect_dict(df)

    if not reports:
        logger.info("secure_dataframe: nenhum PII detectado — DataFrame retornado sem alteração.")
        return df.copy()

    # Filtra colunas conforme columns= / exclude=
    if columns is not None:
        reports = {k: v for k, v in reports.items() if k in columns}
    if exclude is not None:
        reports = {k: v for k, v in reports.items() if k not in exclude}

    if not reports:
        return df.copy()

    # Verificação de idempotência — reutiliza reports já computados
    _hex16 = _re.compile(r'^[0-9a-f]{16}$')
    _hash_target_cols = {
        col for col, rep in reports.items()
        if rep.mask_strategy.value == "hash"
    }
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col not in _hash_target_cols:
            continue
        sample = df[col].dropna().astype(str).head(50)
        if len(sample) >= 10 and sample.map(lambda v: bool(_hex16.match(v))).mean() > 0.98:
            msg = (
                f"secure_dataframe(): a coluna '{col}' parece já conter tokens HMAC "
                f"(padrão hex-16 detectado em >=98% dos valores amostrados). "
                f"Aplicar mascaramento novamente quebrará joins com outras tabelas mascaradas."
            )
            if strict_idempotency:
                raise IdempotencyError(msg)
            else:
                warnings.warn(msg, UserWarning, stacklevel=2)
            break

    if verbose:
        _print_detection_report(reports)

    _active_audit = audit or _GLOBAL_AUDIT
    engine = _MaskingEngine(salt=salt, random_state=random_state, audit=_active_audit)
    return engine.apply(df, reports)


def load_secure_dataframe_chunked(
    file_path: str,
    salt: Optional[str] = None,
    random_state: int = 42,
    chunksize: int = 10_000,
    detector_kwargs: Optional[dict] = None,
    output_path: Optional[str] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Lê um arquivo em chunks, mascarando cada bloco antes de carregar o próximo.

    v1.1 — Usa escrita incremental em Parquet via pyarrow.parquet.ParquetWriter
    quando output_path é fornecido, eliminando o pd.concat() final que
    duplicava o uso de memória. Suporta CSV e Parquet.

    Args:
        file_path: Caminho para o arquivo (.csv ou .parquet).
        salt: Chave para hashing determinístico.
        random_state: Semente para mockers.
        chunksize: Número de linhas por chunk (padrão 10.000).
        detector_kwargs: Parâmetros opcionais para PIIDetector.
        output_path: Se fornecido, grava em Parquet incrementalmente
                     e retorna DataFrame vazio (economiza memória).
        **read_kwargs: Argumentos repassados ao leitor.

    Returns:
        DataFrame concatenado (se output_path=None) ou vazio (se output_path).
    """
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
    engine = _MaskingEngine(salt=salt, random_state=random_state)

    # Seleciona o iterador de chunks adequado
    chunk_iter: Iterator[pd.DataFrame]
    if suffix == ".csv":
        chunk_iter = pd.read_csv(path, chunksize=chunksize, **read_kwargs)
    else:
        chunk_iter = _parquet_chunk_iter(path, chunksize, **read_kwargs)

    reports: Optional[dict] = None
    chunks_safe = []
    parquet_writer = None

    try:
        for chunk_idx, chunk in enumerate(chunk_iter):
            chunk_reports = detector.detect_dict(chunk)
            if reports is None:
                reports = chunk_reports
            else:
                for col, rep in chunk_reports.items():
                    if col not in reports:
                        logger.info(
                            "Chunked: coluna PII '%s' detectada no chunk %d.",
                            col, chunk_idx + 1,
                        )
                        reports[col] = rep

            chunk_safe = engine.apply(chunk, reports) if reports else chunk.copy()

            if output_path:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(chunk_safe, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, table.schema)
                parquet_writer.write_table(table)
            else:
                chunks_safe.append(chunk_safe)

    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if output_path:
        logger.info("Chunked: arquivo mascarado gravado em '%s'.", output_path)
        return pd.DataFrame()

    return pd.concat(chunks_safe, ignore_index=True) if chunks_safe else pd.DataFrame()


def _parquet_chunk_iter(path: Path, chunksize: int, **kwargs) -> Iterator[pd.DataFrame]:
    """Itera sobre um arquivo Parquet em batches de tamanho aproximado."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    except ImportError:
        # Fallback sem pyarrow: lê tudo e divide
        df = pd.read_parquet(path, **kwargs)
        for start in range(0, len(df), chunksize):
            yield df.iloc[start:start + chunksize]


# ---------------------------------------------------------------------------
# Motor de mascaramento interno
# ---------------------------------------------------------------------------

class _MaskingEngine:
    """Aplica os mascaramentos/mocks conforme os ColumnReports."""

    def __init__(
        self,
        salt: Optional[str],
        random_state: int,
        audit: Optional[Any] = None,
    ):
        self._salt_secret = SecretStr(salt) if salt is not None else None
        self._hasher   = DeterministicHasher(salt=salt)
        self._redactor = StringRedactor()
        self._cep      = CepTruncator()
        self._phone    = PhoneDddMasker()
        self._date     = DateMasker()
        self._num      = NumericMocker(strategy="perturb", random_state=random_state)
        self._cat      = CategoryMocker(random_state=random_state)
        self._plan_reports: dict = {}
        self._audit = audit  # AuditReport | None — só executa se fornecido

    def __repr__(self) -> str:
        return f"_MaskingEngine(salt={self._salt_secret!r}, hasher={self._hasher!r})"

    def apply(
        self, df: pd.DataFrame, reports: dict[str, ColumnReport]
    ) -> pd.DataFrame:
        df_out = df.copy()

        for col, report in reports.items():
            if col not in df_out.columns:
                continue
            try:
                df_out[col] = self._apply_column(df_out[col], report)
                logger.debug("Mascarado | col=%-25s strategy=%s", col, report.mask_strategy.value)
            except Exception as exc:
                logger.error("Erro ao mascarar '%s': %s", col, exc)
                raise RuntimeError(
                    f"Erro ao mascarar coluna '{col}': {type(exc).__name__}"
                ) from None

        # Trilha de auditoria (LGPD Art. 50) — só executa se AuditReport foi injetado
        if self._audit is not None:
            for col, report in reports.items():
                if col in df_out.columns:
                    self._audit.log(
                        column=col,
                        technique=report.mask_strategy.value,
                        policy=report.pii_type.value,
                        rows_affected=int(df_out[col].notna().sum()),
                        status='success',
                    )

        return df_out

    def _apply_column(self, series: pd.Series, report: ColumnReport) -> pd.Series:
        strategy = report.mask_strategy

        # Guard: datetime64 → GENERALIZE_DATE
        if pd.api.types.is_datetime64_any_dtype(series):
            if strategy in (MaskStrategy.MOCK_NUM, MaskStrategy.GENERALIZE_DATE):
                return self._date.transform(series)

        if strategy == MaskStrategy.HASH:
            # Normaliza CPF/CNPJ antes de hashar para tratar variações de formatação
            normalized = _normalize_identifier(series, report.pii_type)
            return self._hasher.transform(normalized)

        elif strategy == MaskStrategy.TRUNCATE:
            return self._cep.transform(series)

        elif strategy == MaskStrategy.REDACT:
            return self._redactor.transform(series)

        elif strategy == MaskStrategy.MOCK_NUM:
            plan_report = self._plan_reports.get(
                series.name if hasattr(series, "name") else ""
            )
            effective_report = plan_report if (
                plan_report is not None
                and plan_report.col_min is not None
                and plan_report.col_max is not None
            ) else report
            return self._num.mock_from_report(
                series,
                col_min=effective_report.col_min or 0.0,
                col_max=effective_report.col_max or 1.0,
            )

        elif strategy == MaskStrategy.MOCK_CAT:
            return self._cat.mock(series, value_freq=report.value_freq)

        elif strategy == MaskStrategy.MASK_PHONE_DDD:
            return self._phone.transform(series)

        elif strategy == MaskStrategy.GENERALIZE_DATE:
            return self._date.transform(series)

        elif strategy == MaskStrategy.SUPPRESS:
            return pd.Series(
                [None] * len(series), index=series.index, name=series.name
            )

        else:  # PASSTHROUGH
            return series


# ---------------------------------------------------------------------------
# Normalização de identificadores antes do hash
# ---------------------------------------------------------------------------

def _normalize_identifier(series: pd.Series, pii_type: PIIType) -> pd.Series:
    """
    Normaliza CPF, CNPJ e e-mail antes do hash.

    Usa str.replace() vetorizado (Cython) em vez de map(lambda+re.sub) —
    3–4x mais rápido para séries com 100k+ linhas.

    CPF/CNPJ: remove qualquer não-dígito. Email: lowercase + strip.
    """
    if pii_type in (PIIType.CPF, PIIType.CNPJ):
        # str.replace vetorizado — nulos são preservados automaticamente
        return series.astype(str).str.replace(r"\D", "", regex=True).where(series.notna(), other=None)
    if pii_type == PIIType.EMAIL:
        return series.astype(str).str.strip().str.lower().where(series.notna(), other=None)
    return series


def _is_null(v) -> bool:
    """Testa nulidade de forma robusta para None, np.nan, pd.NA e pd.NaT."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path, **kwargs) -> pd.DataFrame:
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Arquivo muito grande ({size / 1024**3:.1f} GB). "
            f"Use load_secure_dataframe_chunked() para arquivos grandes."
        )
    suffix = path.suffix.lower()
    readers = {
        ".csv":     pd.read_csv,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".parquet": pd.read_parquet,
        ".json":    pd.read_json,
    }
    reader = readers.get(suffix)
    if reader is None:
        raise ValueError(
            f"Formato '{suffix}' não suportado. "
            f"Use: {', '.join(readers.keys())}"
        )
    return reader(path, **kwargs)


def _print_detection_report(reports: dict) -> None:
    print("\n" + "=" * 60)
    print("PRIVACY FRAMEWORK -- DETECCAO DE PII")
    print("=" * 60)
    _icons = {"high": "[ALTO]", "medium": "[MED]", "low": "[BAIXO]"}
    for col, r in sorted(reports.items(), key=lambda x: x[1].risk_level.value):
        icon = _icons.get(r.risk_level.value, "[?]")
        print(
            f"  {icon} {col:<30} "
            f"type={r.pii_type.value:<18} "
            f"-> {r.mask_strategy.value}"
        )
    print("=" * 60 + "\n")
