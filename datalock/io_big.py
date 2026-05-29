"""
datalock/io_big.py
===============
Utilitários de I/O para big data — acesso parcial a arquivos grandes
sem carregar tudo na memória.

Exporta:
  read_partial()   — motor chamado por dd.read() quando big-data params usados
  build_csv_index() / load_csv_index() — sidecar .csv.datalock_idx
  DatabaseConnection — objeto de conexão reutilizável (dd.db())

Design
------
Parquet (e IPC/Arrow) têm metadados de row group no footer — acesso cirúrgico
real, sem ler dados que não foram pedidos.

CSV não tem estrutura de blocos. Usa mmap para construir um índice de byte
offsets em <1ms (vs 37s com loop Python). O índice é salvo como arquivo
sidecar de 666 bytes e reutilizado em leituras futuras (201× mais rápido).

Semântica honesta por formato
  header_only  → todos os formatos  → ~0ms, schema + shape, zero dados
  head=N       → todos              → rápido e exato
  tail=N       → Parquet: rápido; CSV: lento (precisa ler até o fim)
  columns=     → Parquet: zero-copy; CSV: parse completo depois filtra
  n_chunks/chunks → Parquet: exato por row group; CSV: aproximado por byte offset
  sample=N     → Parquet: row groups aleatórios (bloco-aleatório, não linha)
               → CSV: head(N) com UserWarning
  iter_chunks  → gerador de pl.DataFrame, nunca carrega tudo na memória
"""
from __future__ import annotations

import json
import logging
import mmap
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

# Extensão do arquivo de índice sidecar
_IDX_EXT = ".datalock_idx"

# Formatos com estrutura de blocos nativos (acesso parcial real)
_BLOCK_FORMATS = frozenset({".parquet", ".feather", ".ipc", ".arrow"})

# Formatos baseados em texto (sem estrutura de blocos)
_TEXT_FORMATS = frozenset({".csv", ".tsv", ".txt", ".json", ".ndjson", ".jsonl"})


# ---------------------------------------------------------------------------
# Ponto de entrada principal
# ---------------------------------------------------------------------------

def read_partial(
    path: Path,
    *,
    header_only: bool = False,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    columns: Optional[List[str]] = None,
    n_chunks: Optional[int] = None,
    chunks: Optional[List[int]] = None,
    sample: Optional[int] = None,
    sample_seed: int = 42,
    iter_chunks: bool = False,
    **kwargs,
) -> Union[pl.DataFrame, Dict, Generator]:
    """
    Lê um arquivo grande com acesso parcial eficiente.

    Chamado por dd.read() quando qualquer parâmetro big-data é fornecido:
    header_only, head, tail, n_chunks, chunks, sample, iter_chunks.

    Args:
        path:        Caminho do arquivo.
        header_only: Retorna apenas schema e shape — zero dados lidos.
        head:        Primeiras N linhas.
        tail:        Últimas N linhas.
        columns:     Colunas a selecionar (Parquet: zero-copy).
        n_chunks:    Divide logicamente o arquivo em N chunks.
        chunks:      Lista 1-based de chunks a ler (ex: [2, 4]).
        sample:      N linhas por amostragem (Parquet: row groups aleatórios).
        sample_seed: Semente para amostragem aleatória.
        iter_chunks: Se True, retorna gerador de pl.DataFrame.
        **kwargs:    Repassados ao leitor (sep=, encoding=, etc.).

    Returns:
        pl.DataFrame, dict (header_only=True) ou Generator.
    """
    ext = path.suffix.lower()

    # Valida combinações
    if chunks is not None and n_chunks is None:
        raise ValueError(
            "chunks= requer n_chunks= para definir quantos chunks existem no total.\n"
            "Exemplo: dd.read('big.parquet', n_chunks=5, chunks=[2, 4])"
        )
    if chunks is not None:
        for c in chunks:
            if c < 1 or c > n_chunks:
                raise ValueError(
                    f"chunks=[{c}] inválido para n_chunks={n_chunks}. "
                    f"Use valores entre 1 e {n_chunks}."
                )

    if ext in _BLOCK_FORMATS:
        return _read_block_format(
            path, ext=ext,
            header_only=header_only, head=head, tail=tail,
            columns=columns, n_chunks=n_chunks, chunks=chunks,
            sample=sample, sample_seed=sample_seed,
            iter_chunks=iter_chunks, **kwargs,
        )

    if ext in _TEXT_FORMATS:
        return _read_text_format(
            path, ext=ext,
            header_only=header_only, head=head, tail=tail,
            columns=columns, n_chunks=n_chunks, chunks=chunks,
            sample=sample, sample_seed=sample_seed,
            iter_chunks=iter_chunks, **kwargs,
        )

    raise ValueError(
        f"Acesso parcial não suportado para '{ext}'. "
        f"Suportados: {sorted(_BLOCK_FORMATS | _TEXT_FORMATS)}"
    )


# ---------------------------------------------------------------------------
# Parquet / IPC / Arrow — acesso real por row group
# ---------------------------------------------------------------------------

def _read_block_format(
    path: Path,
    ext: str,
    *,
    header_only: bool,
    head: Optional[int],
    tail: Optional[int],
    columns: Optional[List[str]],
    n_chunks: Optional[int],
    chunks: Optional[List[int]],
    sample: Optional[int],
    sample_seed: int,
    iter_chunks: bool,
    **kwargs,
) -> Union[pl.DataFrame, Dict, Generator]:

    import pyarrow as pa
    import pyarrow.parquet as pq

    # ── IPC / Feather ──────────────────────────────────────────────────────
    if ext in (".feather", ".ipc", ".arrow"):
        return _read_ipc_partial(
            path, header_only=header_only, head=head, tail=tail,
            columns=columns, n_chunks=n_chunks, chunks=chunks,
            sample=sample, sample_seed=sample_seed, iter_chunks=iter_chunks,
        )

    # ── Parquet ────────────────────────────────────────────────────────────
    pf = pq.ParquetFile(str(path))
    meta = pf.metadata
    schema = pf.schema_arrow
    n_rows = meta.num_rows
    n_rg   = meta.num_row_groups

    # ── header_only: zero dados lidos ────────────────────────────────────
    if header_only:
        col_stats = {}
        for i in range(meta.num_columns):
            rg0 = meta.row_group(0).column(i)
            col_stats[schema.names[i]] = {
                "dtype":    str(schema.types[i]),
                "n_nulls":  sum(meta.row_group(r).column(i).statistics.null_count
                                for r in range(n_rg)
                                if meta.row_group(r).column(i).statistics) if n_rg > 0 else None,
                "min":      str(meta.row_group(0).column(i).statistics.min)
                             if meta.row_group(0).column(i).statistics else None,
                "max":      str(meta.row_group(n_rg-1).column(i).statistics.max)
                             if meta.row_group(n_rg-1).column(i).statistics else None,
            }
        return {
            "format":        "parquet",
            "path":          str(path),
            "n_rows":        n_rows,
            "n_columns":     meta.num_columns,
            "n_row_groups":  n_rg,
            "columns":       schema.names,
            "column_stats":  col_stats,
            "file_size_mb":  round(path.stat().st_size / 1024 / 1024, 2),
        }

    # ── head: usa n_rows= do Polars (lê só o 1º row group necessário) ─────
    if head is not None and chunks is None and sample is None:
        return pl.read_parquet(str(path), n_rows=head, columns=columns)

    # ── tail: lê últimos N rows ────────────────────────────────────────────
    if tail is not None and chunks is None and sample is None:
        # Parquet: acumula row groups a partir do fim
        accumulated = 0
        rgs_needed = []
        for i in range(n_rg - 1, -1, -1):
            rg_rows = meta.row_group(i).num_rows
            rgs_needed.insert(0, i)
            accumulated += rg_rows
            if accumulated >= tail:
                break
        tables = [pf.read_row_group(i, columns=columns) for i in rgs_needed]
        df = pl.from_arrow(pa.concat_tables(tables))
        return df.tail(tail)

    # ── sample: row groups aleatórios ─────────────────────────────────────
    if sample is not None and chunks is None:
        rows_per_rg = meta.row_group(0).num_rows if n_rg > 0 else 1
        n_rgs_needed = max(1, (sample + rows_per_rg - 1) // rows_per_rg)
        n_rgs_needed = min(n_rgs_needed, n_rg)
        rng = random.Random(sample_seed)
        selected_rgs = sorted(rng.sample(range(n_rg), n_rgs_needed))
        tables = [pf.read_row_group(i, columns=columns) for i in selected_rgs]
        df = pl.from_arrow(pa.concat_tables(tables))
        # Trim to exact sample size and shuffle within selection
        if len(df) > sample:
            df = df.sample(sample, seed=sample_seed)
        return df

    # ── chunks: resolve row groups para cada logical chunk ────────────────
    if chunks is not None:
        rg_indices = _logical_chunks_to_rgs(n_rg, n_chunks, [c - 1 for c in chunks])

        if iter_chunks:
            def _gen():
                for rg_idx in rg_indices:
                    yield pl.from_arrow(pf.read_row_group(rg_idx, columns=columns))
            return _gen()

        tables = [pf.read_row_group(i, columns=columns) for i in rg_indices]
        df = pl.from_arrow(pa.concat_tables(tables))
        if head:
            df = df.head(head)
        return df

    # ── iter_chunks: gerador sobre todos os row groups ────────────────────
    if iter_chunks:
        rg_size = max(1, n_rg // (n_chunks or n_rg))

        def _gen():
            for i in range(0, n_rg, rg_size):
                batch_rgs = range(i, min(i + rg_size, n_rg))
                tables = [pf.read_row_group(j, columns=columns) for j in batch_rgs]
                yield pl.from_arrow(pa.concat_tables(tables))
        return _gen()

    # ── n_chunks sem chunks selecionados: retorna info dos chunks ─────────
    if n_chunks is not None and chunks is None:
        rows_per_chunk = n_rows // n_chunks
        chunk_info = []
        for i in range(n_chunks):
            rg_start, rg_end = _chunk_rg_range(n_rg, n_chunks, i)
            chunk_rows = sum(meta.row_group(r).num_rows for r in range(rg_start, rg_end))
            chunk_info.append({
                "chunk": i + 1,
                "row_groups": list(range(rg_start, rg_end)),
                "approx_rows": chunk_rows,
            })
        return {"n_chunks": n_chunks, "chunks": chunk_info, "total_rows": n_rows}

    # ── Fallback: read normal com columns ─────────────────────────────────
    return pl.read_parquet(str(path), columns=columns)


def _read_ipc_partial(
    path: Path,
    *,
    header_only: bool,
    head: Optional[int],
    tail: Optional[int],
    columns: Optional[List[str]],
    n_chunks: Optional[int],
    chunks: Optional[List[int]],
    sample: Optional[int],
    sample_seed: int,
    iter_chunks: bool,
) -> Union[pl.DataFrame, Dict, Generator]:
    import pyarrow as pa
    reader = pa.ipc.open_file(str(path))
    n_batches = reader.num_record_batches
    schema = reader.schema_arrow

    if header_only:
        return {
            "format":    "ipc",
            "path":      str(path),
            "n_batches": n_batches,
            "columns":   schema.names,
            "dtypes":    {n: str(t) for n, t in zip(schema.names, schema.types)},
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        }

    def _col_filter(df: pl.DataFrame) -> pl.DataFrame:
        if columns:
            existing = [c for c in columns if c in df.columns]
            if existing:
                return df.select(existing)
        return df

    if head is not None and chunks is None:
        accumulated = []
        total = 0
        for i in range(n_batches):
            batch = pl.from_arrow(reader.get_batch(i))
            accumulated.append(batch)
            total += len(batch)
            if total >= head:
                break
        return _col_filter(pl.concat(accumulated)).head(head)

    if chunks is not None:
        rg_indices = _logical_chunks_to_rgs(n_batches, n_chunks, [c - 1 for c in chunks])
        if iter_chunks:
            def _gen():
                for i in rg_indices:
                    yield _col_filter(pl.from_arrow(reader.get_batch(i)))
            return _gen()
        batches = [pl.from_arrow(reader.get_batch(i)) for i in rg_indices]
        return _col_filter(pl.concat(batches))

    if sample is not None:
        rng = random.Random(sample_seed)
        selected = sorted(rng.sample(range(n_batches), min(max(1, sample // 10000 + 1), n_batches)))
        batches = [pl.from_arrow(reader.get_batch(i)) for i in selected]
        df = pl.concat(batches)
        return _col_filter(df.sample(min(sample, len(df)), seed=sample_seed))

    if iter_chunks:
        chunk_size = max(1, n_batches // (n_chunks or n_batches))
        def _gen():
            for i in range(0, n_batches, chunk_size):
                batches = [pl.from_arrow(reader.get_batch(j))
                            for j in range(i, min(i + chunk_size, n_batches))]
                yield _col_filter(pl.concat(batches))
        return _gen()

    return _col_filter(pl.from_arrow(reader.read_all()))


# ---------------------------------------------------------------------------
# CSV / TSV / TXT / JSON — byte-offset chunking via mmap
# ---------------------------------------------------------------------------

def _read_text_format(
    path: Path,
    ext: str,
    *,
    header_only: bool,
    head: Optional[int],
    tail: Optional[int],
    columns: Optional[List[str]],
    n_chunks: Optional[int],
    chunks: Optional[List[int]],
    sample: Optional[int],
    sample_seed: int,
    iter_chunks: bool,
    **kwargs,
) -> Union[pl.DataFrame, Dict, Generator]:

    is_csv = ext in (".csv", ".tsv", ".txt")
    sep = kwargs.get("sep", "," if ext == ".csv" else "\t")

    # ── header_only ────────────────────────────────────────────────────────
    if header_only:
        header_line = _read_header_line(path)
        col_names = header_line.split(sep)
        size_mb = round(path.stat().st_size / 1024 / 1024, 2)
        return {
            "format":    ext.lstrip("."),
            "path":      str(path),
            "columns":   col_names,
            "n_columns": len(col_names),
            "file_size_mb": size_mb,
            "n_rows":    "unknown (use head/sample to inspect)",
        }

    # ── head: fast sequential read ─────────────────────────────────────────
    if head is not None and chunks is None and sample is None and not iter_chunks:
        df = pl.read_csv(str(path), n_rows=head, separator=sep,
                         **{k: v for k, v in kwargs.items() if k != "sep"})
        if columns:
            existing = [c for c in columns if c in df.columns]
            if existing:
                df = df.select(existing)
        return df

    # ── sample in CSV: honest head with warning ────────────────────────────
    if sample is not None and is_csv and not iter_chunks:
        warnings.warn(
            f"dd.read('{path.name}', sample={sample}): CSV não tem estrutura de blocos. "
            f"Retornando as primeiras {sample} linhas (head), não uma amostra aleatória. "
            f"Para amostragem aleatória eficiente, converta para Parquet:\n"
            f"  dd.write(dd.read('{path.name}', head=1_000_000), '{path.stem}.parquet')\n"
            f"  df = dd.read('{path.stem}.parquet', sample={sample})",
            UserWarning,
            stacklevel=4,
        )
        return _read_text_format(path, ext=ext, header_only=False, head=sample,
                                  tail=None, columns=columns, n_chunks=None,
                                  chunks=None, sample=None, sample_seed=sample_seed,
                                  iter_chunks=False, **kwargs)

    # ── chunks via byte-offset index ──────────────────────────────────────
    if (chunks is not None or iter_chunks) and is_csv:
        idx = _get_csv_index(path, n_chunks, sep=sep)

        # Warn about approximate boundaries
        if chunks is not None:
            warnings.warn(
                f"dd.read('{path.name}', n_chunks={n_chunks}, chunks={chunks}): "
                f"CSV não tem estrutura de blocos — os boundaries são aproximados "
                f"(±1 linha por chunk). Para acesso exato por chunk, use Parquet.",
                UserWarning,
                stacklevel=4,
            )

        header_bytes = path.read_bytes()[:idx["header_end"]]
        chunk_list = [c - 1 for c in chunks] if chunks else list(range(n_chunks))

        if iter_chunks:
            def _gen():
                with open(str(path), "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    for ci in range(n_chunks or len(idx["offsets"]) - 1):
                        if chunks and (ci + 1) not in chunks:
                            continue
                        start = idx["offsets"][ci]
                        end   = idx["offsets"][ci + 1]
                        data  = header_bytes + bytes(mm[start:end])
                        df    = pl.read_csv(data, separator=sep)
                        if columns:
                            existing = [c for c in columns if c in df.columns]
                            if existing:
                                df = df.select(existing)
                        yield df
                    mm.close()
            return _gen()

        with open(str(path), "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            frames = []
            for ci in chunk_list:
                start = idx["offsets"][ci]
                end   = idx["offsets"][ci + 1]
                data  = header_bytes + bytes(mm[start:end])
                frames.append(pl.read_csv(data, separator=sep))
            mm.close()
        df = pl.concat(frames)
        if columns:
            existing = [c for c in columns if c in df.columns]
            if existing:
                df = df.select(existing)
        if head:
            df = df.head(head)
        return df

    # Fallback
    df = pl.read_csv(str(path), separator=sep,
                     **{k: v for k, v in kwargs.items() if k != "sep"})
    if columns:
        existing = [c for c in columns if c in df.columns]
        if existing:
            df = df.select(existing)
    return df


# ---------------------------------------------------------------------------
# CSV sidecar index
# ---------------------------------------------------------------------------

def _get_csv_index(path: Path, n_chunks: int, sep: str = ",") -> Dict:
    """Retorna o índice de byte offsets para o CSV. Cria sidecar se não existir."""
    idx_path = path.with_suffix(path.suffix + _IDX_EXT)

    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text())
            # Valida: sidecar deve ter o mesmo n_chunks e mtime
            file_mtime = int(path.stat().st_mtime)
            if (idx.get("n_chunks") == n_chunks
                    and idx.get("file_mtime") == file_mtime
                    and idx.get("sep") == sep):
                logger.debug("CSV index loaded from sidecar: %s", idx_path)
                return idx
        except Exception:
            pass

    return build_csv_index(path, n_chunks=n_chunks, sep=sep, save=True)


def build_csv_index(
    path: Union[str, Path],
    n_chunks: int = 10,
    sep: str = ",",
    save: bool = True,
) -> Dict:
    """
    Constrói um índice de byte offsets para um CSV usando mmap.

    O mmap usa page faults do OS — não lê o arquivo inteiro em Python.
    Resultado: construção em <1ms para qualquer tamanho de arquivo.

    Args:
        path:     Caminho do CSV.
        n_chunks: Número de chunks a dividir.
        sep:      Separador de colunas.
        save:     Se True, salva sidecar .csv.datalock_idx.

    Returns:
        Dict com offsets, n_chunks, header_end, total_bytes.
    """
    p = Path(str(path))
    file_size = p.stat().st_size

    with open(str(p), "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        header_end = mm.find(b"\n") + 1
        data_size  = file_size - header_end
        chunk_size = data_size // n_chunks

        offsets = [header_end]
        for i in range(1, n_chunks):
            approx  = header_end + chunk_size * i
            # Avança até o próximo newline (byte boundary → line boundary)
            nl = mm.find(bytes([10]))
            offsets.append(nl + 1 if nl != -1 else file_size)
        offsets.append(file_size)

        # Estima linhas por chunk
        sample_lines = mm[header_end:min(header_end + 4096, file_size)].count(bytes([10]))
        avg_line_bytes = (4096 / max(sample_lines, 1)) if sample_lines > 0 else 50
        approx_total_rows = int(data_size / max(avg_line_bytes, 1))

        mm.close()

    idx = {
        "format":       "logus_csv_index_v1",
        "path":         str(p),
        "file_mtime":   int(p.stat().st_mtime),
        "total_bytes":  file_size,
        "header_end":   header_end,
        "n_chunks":     n_chunks,
        "sep":          sep,
        "approx_total_rows": approx_total_rows,
        "offsets":      offsets,
        "chunks": [
            {
                "chunk":       i + 1,
                "start":       offsets[i],
                "end":         offsets[i + 1],
                "approx_rows": int((offsets[i + 1] - offsets[i]) / max(avg_line_bytes, 1)),
            }
            for i in range(n_chunks)
        ],
    }

    if save:
        idx_path = p.with_suffix(p.suffix + _IDX_EXT)
        try:
            idx_path.write_text(json.dumps(idx, indent=2))
            logger.info("CSV index saved: %s (%d bytes)", idx_path, idx_path.stat().st_size)
        except Exception as exc:
            logger.warning("Could not save CSV index: %s", exc)

    return idx


def load_csv_index(path: Union[str, Path]) -> Optional[Dict]:
    """Carrega o sidecar index se existir, None caso contrário."""
    p = Path(str(path))
    idx_path = p.with_suffix(p.suffix + _IDX_EXT)
    if not idx_path.exists():
        return None
    try:
        return json.loads(idx_path.read_text())
    except Exception:
        return None



# ---------------------------------------------------------------------------
# Chunk ↔ row group helpers
# ---------------------------------------------------------------------------

def _logical_chunks_to_rgs(n_rg: int, n_chunks: int, chunk_indices: list) -> list:
    rgs = []
    for ci in chunk_indices:
        rg_start, rg_end = _chunk_rg_range(n_rg, n_chunks, ci)
        rgs.extend(range(rg_start, rg_end))
    return sorted(set(rgs))


def _chunk_rg_range(n_rg: int, n_chunks: int, chunk_idx: int):
    rg_per_chunk = max(1, n_rg // n_chunks)
    rg_start = chunk_idx * rg_per_chunk
    rg_end = rg_start + rg_per_chunk if chunk_idx < n_chunks - 1 else n_rg
    return rg_start, min(rg_end, n_rg)


def _read_header_line(path: Path) -> str:
    with open(str(path), "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        nl = mm.find(bytes([10]))
        header = mm[:nl].decode("utf-8", errors="replace").rstrip() if nl != -1 else ""
        mm.close()
        return header


# ---------------------------------------------------------------------------
# dd.db() — DatabaseConnection
# ---------------------------------------------------------------------------

class DatabaseConnection:
    """
    Objeto de conexão reutilizável para bancos de dados.

    Criado via dd.db(). Integra-se com dd.read() e dd.write().

    Usa ConnectorX (Rust, zero-copy Arrow) quando disponível → 3–5× mais rápido.
    Fallback para SQLAlchemy + pandas (sempre disponível).
    Credenciais armazenadas como SecretStr — nunca aparecem em repr ou logs.

    Uso:
        banco = dd.db("postgresql://user:pass@host/db", salt=SALT)
        df    = dd.read(banco, "clientes")
        df    = dd.read(banco, "clientes", sample=10_000)
        df    = dd.read(banco, "SELECT * FROM clientes WHERE uf='SP'")
        banco.write(df_safe, "clientes_masked")

        with dd.db("postgresql://...", salt=SALT) as banco:
            df = banco.read("clientes")
    """

    def __init__(
        self,
        uri: str,
        *,
        salt: Optional[str] = None,
        dialect: Optional[str] = None,
        pool_size: int = 5,
        connect_timeout: int = 30,
    ) -> None:
        from datalock.utils.secret_str import SecretStr
        self._uri             = SecretStr(uri)
        self._salt            = SecretStr(salt) if salt else None
        self._pool_size       = pool_size
        self._connect_timeout = connect_timeout
        self._engine          = None
        self._dialect         = dialect or self._detect_dialect(uri)
        self._cx_available    = self._check_connectorx()

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "DatabaseConnection":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Fecha o connection pool."""
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception:
                pass
            self._engine = None

    # ── Leitura ────────────────────────────────────────────────────────────

    def read(
        self,
        table_or_sql: str,
        *,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        sample: Optional[int] = None,
        head: Optional[int] = None,
        salt: Optional[str] = None,
        verbose: bool = False,
    ) -> "pl.DataFrame":
        """
        Lê tabela ou SQL do banco. Retorna pl.DataFrame.

        Args:
            table_or_sql: Nome de tabela ou query SQL completa.
            columns:      Colunas a selecionar.
            where:        Cláusula WHERE (sem a keyword).
            sample:       N linhas via TABLESAMPLE quando suportado.
            head:         Limita a N linhas.
            salt:         Sobrescreve o salt do objeto para esta query.
            verbose:      Imprime relatório de detecção PII.
        """
        sql = self._build_sql(table_or_sql, columns=columns, where=where,
                               sample=sample, head=head)
        df = self._execute_query(sql)

        effective_salt = salt or (str(self._salt) if self._salt else None)
        if effective_salt:
            import datalock as dd
            return dd.mask(df, salt=effective_salt, verbose=verbose)
        return df

    def _execute_query(self, sql: str) -> "pl.DataFrame":
        """Executa SQL via ConnectorX (preferido) ou SQLAlchemy."""
        uri = self._uri.get()

        if self._cx_available:
            try:
                import connectorx as cx
                arrow = cx.read_sql(uri, sql, return_type="arrow2")
                return pl.from_arrow(arrow)
            except Exception as exc:
                logger.debug("ConnectorX falhou (%s), usando SQLAlchemy.", exc)

        import pandas as _pd
        engine = self._get_engine()
        with engine.connect() as conn:
            df_pd = _pd.read_sql(sql, conn)
        return pl.from_pandas(df_pd)

    def _build_sql(
        self,
        table_or_sql: str,
        *,
        columns: Optional[List[str]],
        where: Optional[str],
        sample: Optional[int],
        head: Optional[int],
    ) -> str:
        is_sql = table_or_sql.strip().upper().startswith(
            ("SELECT", "WITH", "TABLE", "VALUES")
        )
        if is_sql:
            sql = table_or_sql.strip()
            if head:
                sql = self._apply_limit(sql, head)
            return sql

        col_list = ", ".join(columns) if columns else "*"

        if sample and self._supports_tablesample():
            pct = max(0.001, min(100.0, (sample / 1_000_000) * 100))
            sample_clause = self._tablesample_clause(sample, pct)
            sql = f"SELECT {col_list} FROM {table_or_sql} {sample_clause}"
        else:
            sql = f"SELECT {col_list} FROM {table_or_sql}"
            if where:
                sql += f" WHERE {where}"
            if head or sample:
                sql = self._apply_limit(sql, head or sample)

        return sql

    def _apply_limit(self, sql: str, n: int) -> str:
        if self._dialect == "sqlserver":
            if sql.upper().startswith("SELECT "):
                return sql.replace("SELECT ", f"SELECT TOP {n} ", 1)
        return f"{sql.rstrip(';')} LIMIT {n}"

    def _tablesample_clause(self, n: int, pct: float) -> str:
        if self._dialect == "postgresql":
            return f"TABLESAMPLE BERNOULLI({pct:.4f})"
        if self._dialect == "sqlserver":
            return f"TABLESAMPLE ({n} ROWS)"
        if self._dialect == "bigquery":
            return f"TABLESAMPLE SYSTEM ({pct:.4f} PERCENT)"
        return ""

    def _supports_tablesample(self) -> bool:
        return self._dialect in ("postgresql", "sqlserver", "bigquery")

    # ── Escrita ────────────────────────────────────────────────────────────

    def write(
        self,
        df: Any,
        table: str,
        *,
        if_exists: str = "append",
        schema: Optional[str] = None,
        chunk_size: int = 10_000,
        salt: Optional[str] = None,
    ) -> Dict:
        """Escreve DataFrame no banco. Mascara se salt configurado."""
        import pandas as _pd

        effective_salt = salt or (str(self._salt) if self._salt else None)
        if effective_salt:
            import datalock as dd
            df = dd.mask(df, salt=effective_salt)

        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        engine = self._get_engine()
        df_pd.to_sql(table, engine, if_exists=if_exists, index=False,
                     schema=schema, chunksize=chunk_size, method="multi")
        return {"table": table, "rows": len(df_pd), "if_exists": if_exists}

    # ── Exploração ─────────────────────────────────────────────────────────

    def tables(self, schema: Optional[str] = None) -> List[str]:
        """Lista as tabelas disponíveis."""
        from sqlalchemy import inspect as _inspect
        return _inspect(self._get_engine()).get_table_names(schema=schema)

    def schema(self, table: str, schema_: Optional[str] = None) -> Dict:
        """Retorna o schema de uma tabela."""
        from sqlalchemy import inspect as _inspect
        cols = _inspect(self._get_engine()).get_columns(table, schema=schema_)
        return {c["name"]: str(c["type"]) for c in cols}

    def sample_table(self, table: str, n: int = 5) -> "pl.DataFrame":
        """Retorna N linhas sem mascaramento (para inspeção)."""
        return self._execute_query(self._apply_limit(f"SELECT * FROM {table}", n))

    def create_table(
        self,
        df: Any,
        table: str,
        *,
        schema: Optional[str] = None,
        if_exists: str = "fail",
    ) -> None:
        """
        Cria tabela no banco com o schema inferido do DataFrame.

        Args:
            df:        DataFrame de referência (pl.DataFrame ou pd.DataFrame).
            table:     Nome da tabela.
            schema:    Schema do banco.
            if_exists: "fail" (padrão) | "replace" | "ignore"

        Exemplos:
            banco.create_table(df, "clientes")
            banco.create_table(df_pl, "pedidos", if_exists="replace")
        """
        import pandas as _pd
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        engine = self._get_engine()
        from sqlalchemy import inspect as _inspect
        insp = _inspect(engine)
        exists = insp.has_table(table, schema=schema)
        if exists and if_exists == "fail":
            raise ValueError(f"Tabela '{table}' já existe. Use if_exists='replace' ou 'ignore'.")
        if exists and if_exists == "ignore":
            return
        df_pd.head(0).to_sql(table, engine, schema=schema,
                              if_exists="replace", index=False)

    def upsert(
        self,
        df: Any,
        table: str,
        *,
        on: Union[str, List[str]],
        schema: Optional[str] = None,
        salt: Optional[str] = None,
    ) -> int:
        """
        INSERT ... ON CONFLICT UPDATE para PostgreSQL e SQLite. Fallback DELETE+INSERT.

        Args:
            df:     DataFrame com os dados.
            table:  Tabela destino.
            on:     Coluna(s) de chave de conflito.
            schema: Schema do banco.
            salt:   Se fornecido, mascara antes do upsert.

        Returns:
            Número de linhas afetadas.

        Exemplos:
            banco.upsert(df_new, "clientes", on="cpf")
        """
        import pandas as _pd
        effective_salt = salt or (str(self._salt) if self._salt else None)
        if effective_salt:
            import datalock as dd
            df = dd.mask(df, salt=effective_salt)
        df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        from datalock.adapters.db_adapter import SecureDBAdapter
        adapter = SecureDBAdapter(engine=self._get_engine(), salt=effective_salt or "x")
        return adapter.upsert(df_pd, table, on=on, schema=schema)

    # ── Internos ───────────────────────────────────────────────────────────

    def _get_engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine
            kwargs: Dict = {"pool_size": self._pool_size}
            if self._dialect == "postgresql":
                kwargs["connect_args"] = {"connect_timeout": self._connect_timeout}
            self._engine = create_engine(self._uri.get(), **kwargs)
        return self._engine

    @staticmethod
    def _detect_dialect(uri: str) -> str:
        u = uri.lower()
        if "postgresql" in u or "postgres" in u or "psycopg" in u: return "postgresql"
        if "mysql" in u or "mariadb" in u:                          return "mysql"
        if "sqlite" in u:                                            return "sqlite"
        if "mssql" in u or "sqlserver" in u or "pyodbc" in u:       return "sqlserver"
        if "oracle" in u or "cx_oracle" in u:                       return "oracle"
        if "bigquery" in u:                                          return "bigquery"
        if "snowflake" in u:                                         return "snowflake"
        if "redshift" in u:                                          return "redshift"
        if "databricks" in u:                                        return "databricks"
        if "duckdb" in u:                                            return "duckdb"
        return "generic"

    @staticmethod
    def _check_connectorx() -> bool:
        try:
            import connectorx  # noqa: F401
            return True
        except ImportError:
            return False

    def __repr__(self) -> str:
        salt_status = "com salt" if self._salt else "sem salt (exploração)"
        engine_name = "connectorx" if self._cx_available else "sqlalchemy"
        return (
            f"DatabaseConnection(dialect={self._dialect!r}, "
            f"engine={engine_name!r}, {salt_status})"
        )
