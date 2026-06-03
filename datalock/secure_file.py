"""
secure_file.py
==============
SecureFile — Formato de Arquivo Criptografado `.dlk`.

O que é o .dlk?
-----------------
Um contêiner binário de propósito geral para transferência e armazenamento
seguro de dados tabulares. A criptografia, a integridade e os metadados são
independentes do tipo de conteúdo — o .dlk não sabe nem precisa saber se
o payload são dados brutos, dados mascarados ou qualquer outra coisa.

Casos de uso suportados (content_type no header):
  "raw_dataframe"    → dados brutos em trânsito entre ambientes seguros
  "masked_dataframe" → dados com mascaramento HMAC/REDACT/TRUNCATE aplicado
  "multi_dataframe"  → múltiplos DataFrames num único arquivo (v3)
  "bytes"            → qualquer payload binário arbitrário (logs, modelos, etc.)
  "custom:<str>"     → tipos definidos pelo usuário

Por que não usar GPG, 7-Zip ou Parquet criptografado?
-------------------------------------------------------
| Ferramenta              | AES-256-GCM | HMAC integridade | Metadados LGPD | Schema tabular |
|-------------------------|-------------|------------------|----------------|----------------|
| GPG / age               | ✅          | ✅               | ❌             | ❌             |
| 7-Zip AES-256           | ✅          | ❌               | ❌             | ❌             |
| Parquet + S3 SSE        | ✅          | ❌               | ❌             | ✅             |
| **datalock**            | ✅          | ✅               | ✅             | ✅             |

Posição de segurança em memória:
  A janela de exposição do dado bruto é minimizada — ele existe na heap
  durante a desserialização, não como arquivo persistente acessível.
  Python não oferece zeroização determinística (GC não é imediato);
  o framework reduz a *exposição operacional*, não garante apagamento físico.

Criptografia utilizada
-----------------------
- AES-256-GCM   (AEAD — garante confidencialidade + integridade em uma operação)
- HKDF-SHA256   (RFC 5869 — deriva a DEK da master key + salt único por arquivo)
- HMAC-SHA256   (MAC sobre todo o arquivo incluindo o header JSON em claro)
- zlib          (compressão leve, sem dependências externas; zstd via extras)

Referências:
  - NIST SP 800-38D: AES-GCM recommendation
  - RFC 5869: HMAC-based Key Derivation Function (HKDF)
  - NIST SP 800-175B: Guideline for Using Cryptographic Standards

Estrutura binária do arquivo `.dlk`
---------------------------------------
Formato v2 (single-frame — datalock >= 1.3.2):
  [5  bytes]  MAGIC          = b"DLOCK"
  [1  byte ]  VERSION        = 0x02
  [1  byte ]  CIPHER         = 0x01 (AES-256-GCM) | 0x02 (ChaCha20-Poly1305)
  [32 bytes]  SALT_KDF       — salt HKDF único por arquivo
  [12 bytes]  NONCE_HEADER   — nonce para cifrar o header JSON
  [4  bytes]  HEADER_CT_LEN  — comprimento do header cifrado + auth_tag
  [N+16 bytes] HEADER_CT+TAG — header JSON cifrado com HEK + auth_tag GCM
  [12 bytes]  NONCE_PAYLOAD  — nonce para cifrar o payload
  [M+16 bytes] PAYLOAD_CT+TAG — dados Parquet/zstd cifrados + auth_tag GCM
  [32 bytes]  FILE_HMAC      — HMAC-SHA256 sobre tudo acima

Formato v3 (multi-frame — datalock >= 1.5.0):
  Idêntico ao v2, mas content_type = "multi_dataframe" e o payload
  é um ZIP em memória contendo um Parquet por frame + "index.json"
  com os nomes e tamanhos. Retrocompatível com leitores v1/v2.

Formato v1 (legado — datalock < 1.3.2, retrocompatível):
  [5  bytes]  MAGIC     = b"DLOCK"
  [1  byte ]  VERSION   = 0x01
  [4  bytes]  HEADER_LEN — tamanho do header JSON em claro
  [N  bytes]  HEADER JSON (em claro, autenticado pelo HMAC final)
  [32 bytes]  SALT_KDF
  [12 bytes]  NONCE
  [M+16 bytes] CIPHERTEXT+AUTH_TAG — AES-256-GCM
  [32 bytes]  FILE_HMAC

Uso básico
----------
    from datalock.secure_file import SecureFile

    # Empacotar dado bruto (DBA ou engenheiro de dados — tem a master key)
    SecureFile.pack(
        source_path="clientes.csv",
        output_path="clientes.dlk",
        key="chave-mestre-vault",
    )

    # Ler dado mascarado (desenvolvedor — nunca vê o CSV original)
    df_safe = SecureFile.load(
        path="clientes.dlk",
        key="chave-mestre-vault",
        salt_masking="chave-hmac-dev",
    )

    # Verificar integridade sem decifrar (auditoria)
    ok, info = SecureFile.verify(
        path="clientes.dlk",
        key="chave-mestre-vault",
    )

    # Multi-frame
    SecureFile.pack_frames(
        {"clientes": df1, "pedidos": df2},
        output_path="base.dlk",
        key="chave",
    )
    frames = SecureFile.load_frames("base.dlk", key="chave")
    df = SecureFile.load_frame("base.dlk", key="chave", frame="clientes")
"""

from __future__ import annotations

import hashlib
import hmac as _hmac_module
import io
import json
import logging
import os
import struct
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from datalock.adapters.pandas_adapter import secure_dataframe
from datalock.utils.secret_str import SecretStr
from datalock.ipc_index import (
    compute_batch_stats,
    prune_row_groups,
    apply_arrow_filters,
    normalize_filters,
    ALL_BATCHES_SENTINEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes do formato
# ---------------------------------------------------------------------------

MAGIC   = b"DLOCK"          # 5 bytes — .dlk format — identificador do formato
VERSION_V1 = 0x01            # formato original (header em claro)
VERSION_V2 = 0x02            # v1.3.2: header cifrado, DEK/KEK, cipher field
VERSION_V3 = 0x03            # v1.5.0: multi-frame (payload = ZIP de Parquets)
VERSION    = VERSION_V2      # versão padrão de escrita para single-frame
CIPHER_AES256GCM        = 0x01   # AES-256-GCM (FIPS-approved)
CIPHER_CHACHA20POLY1305 = 0x02   # ChaCha20-Poly1305 (RFC 8439)
SALT_KDF_LEN   = 32          # bytes — salt HKDF, único por arquivo
NONCE_LEN      = 12          # bytes — nonce AEAD, único por operação
AUTH_TAG_LEN   = 16          # bytes — tag de autenticação AEAD
FILE_HMAC_LEN  = 32          # bytes — HMAC-SHA256 do arquivo completo
HEADER_LEN_FMT = ">I"        # big-endian uint32 para comprimento do header cifrado

# Tamanho máximo permitido para o header JSON (cifrado ou em claro).
# Um header legítimo raramente ultrapassa alguns KB; 1 MB é generoso.
# Sem este limite, um arquivo malformado com header_len=0xFFFFFFFF (~4 GB)
# pode causar tentativa de alocação massiva de memória (DoS) — especialmente
# relevante para arquivos v4 cujo HMAC usa chave pública e pode ser forjado.
MAX_HEADER_LEN = 1 * 1024 * 1024  # 1 MB

# Mapa cipher byte → string name
_CIPHER_BYTE_TO_STR = {CIPHER_AES256GCM: "AES256GCM", CIPHER_CHACHA20POLY1305: "ChaCha20Poly1305"}
_CIPHER_STR_TO_BYTE = {v: k for k, v in _CIPHER_BYTE_TO_STR.items()}

# Multi-frame: magic marker no payload ZIP
_MULTI_FRAME_INDEX = "index.json"
# ---------------------------------------------------------------------------
# LGSInfo — resultado rico de verify() com __bool__
# ---------------------------------------------------------------------------

class LGSInfo:
    """
    Resultado de SecureFile.verify() com suporte a bool e acesso por atributo.

    Permite:
        ok = SecureFile.verify(path, key=k)   # bool check direto
        if not SecureFile.verify(path, key=k):
            raise RuntimeError("Corrompido!")

        # Acesso por atributo
        info = SecureFile.verify(path, key=k)
        print(info.shape, info.content_type, info.label)

        # Acesso por chave (retrocompatível com dict)
        print(info["shape"])
    """
    def __init__(self, valid: bool, data: dict) -> None:
        self._valid = valid
        self._data  = data

    def __bool__(self) -> bool:
        return self._valid

    def __iter__(self):
        """Support tuple unpacking: ok, info = SecureFile.verify(...)"""
        yield self._valid
        yield self._data

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_dict(self) -> dict:
        return dict(self._data)

    def __repr__(self) -> str:
        status = "válido" if self._valid else "inválido"
        ct = self._data.get("content_type", "?")
        shape = self._data.get("shape", "?")
        return f"LGSInfo({status}, type={ct!r}, shape={shape})"

    # Allow attribute access for common fields
    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"LGSInfo has no attribute {name!r}") from None




def _require_cryptography() -> Any:
    """Importa e retorna o módulo cryptography, com mensagem clara se ausente."""
    try:
        import cryptography
        return cryptography
    except ImportError:
        raise ImportError(
            "O formato .dlk requer a biblioteca 'cryptography'. "
            "Instale com: pip install cryptography>=41.0.0"
        ) from None


# ---------------------------------------------------------------------------
# Serialização interna do DataFrame — Parquet via PyArrow
# ---------------------------------------------------------------------------

def _df_to_bytes(
    df,
    parquet_compression: str = "zstd",
    row_group_size: int = 50_000,
    content_type: str = "raw_dataframe",
) -> Tuple[bytes, List[Dict]]:
    """
    Serializa DataFrame para bytes via Arrow IPC em múltiplos record batches.

    v1.2.0: serializa como stream de N record batches com estatísticas por
    batch, permitindo column pruning e predicate pushdown na leitura sem
    decifrar/alocar dados desnecessários.

    Retorna TUPLA (bytes, list[dict]) onde list[dict] são os metadados de
    cada batch: {batch_index, byte_offset, byte_length, n_rows, stats}.

    Retrocompatibilidade — callers existentes que faziam:
        payload = _df_to_bytes(df)
    devem ser atualizados para:
        payload, row_groups_meta = _df_to_bytes(df, content_type=content_type)

    Magic markers por formato:
      b'IPC1\x00' → Arrow IPC (datalock >= v1.4) ← formato atual
      b'PQ1\x00'  → Parquet  (datalock v1.3)      ← leitura retrocompat
      (sem marker) → Arrow IPC legado (datalock v1.0-v1.2) ← leitura retrocompat

    Compressões suportadas:
      "zstd" → IPC/zstd  — melhor razão tamanho/velocidade (padrão)
      "lz4"  → IPC/lz4   — máxima velocidade
      "none" → IPC sem compressão

    Args:
        df:                  pd.DataFrame ou pl.DataFrame.
        parquet_compression: "zstd" | "lz4" | "none" (alias mantido por compat).
        row_group_size:      Linhas por record batch (padrão 50 000).
        content_type:        "raw_dataframe" | "masked_dataframe" | outros.
                             Controla quais stats são incluídas nos metadados.

    Returns:
        Tuple (bytes, list[dict])

    Referências:
        Apache Arrow IPC Format Specification
        https://arrow.apache.org/docs/format/IPC.html
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    _COMPRESSION_MAP = {
        "zstd":         "zstd",
        "lz4":          "lz4",
        "snappy":       "lz4",   # snappy → lz4 (não suportado por Arrow IPC)
        "none":         None,
        "uncompressed": None,
    }
    arrow_compression = _COMPRESSION_MAP.get(parquet_compression, "zstd")

    # Converte para Arrow Table — path único independente do tipo de entrada
    if hasattr(df, "to_arrow"):
        # pl.DataFrame → Arrow direto, zero-copy quando possível
        table = df.to_arrow()
    else:
        # pd.DataFrame → Arrow via PyArrow (preserva tipos pandas corretamente)
        table = pa.Table.from_pandas(df, preserve_index=False)

    buf = io.BytesIO()
    opts = ipc.IpcWriteOptions(compression=arrow_compression)
    row_groups_meta: List[Dict] = []

    with ipc.new_stream(buf, table.schema, options=opts) as writer:
        batches = table.to_batches(max_chunksize=row_group_size)
        for batch_index, batch in enumerate(batches):
            byte_offset = buf.tell()
            writer.write_batch(batch)
            byte_length = buf.tell() - byte_offset

            stats = compute_batch_stats(batch, content_type=content_type)
            row_groups_meta.append({
                "batch_index":  batch_index,
                "byte_offset":  byte_offset,
                "byte_length":  byte_length,
                "n_rows":       batch.num_rows,
                "stats":        stats,
            })

    return b"IPC1\x00" + buf.getvalue(), row_groups_meta


def _bytes_to_df(
    data: bytes,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict] = None,
    row_groups_meta: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Desserializa bytes para DataFrame com detecção automática de formato.

    v1.2.0: aceita columns=, filters= e row_groups_meta= para column pruning
    e predicate pushdown no nível Arrow, antes de alocar memória para os dados.

    Detecta o formato pelo magic marker nos primeiros 5 bytes:

      b'IPC1\x00' → Arrow IPC (datalock >= v1.4)   ← formato atual
      b'PQ1\x00'  → Parquet  (datalock v1.3)        ← retrocompat
      outros      → Arrow IPC sem marker (datalock v1.0-v1.2) ← retrocompat legado

    Retrocompatibilidade:
      row_groups_meta=[] ou None → sem pruning, lê tudo (comportamento anterior).
      columns=None, filters=None → comportamento idêntico ao anterior.

    Args:
        data:             Bytes serializados (com magic marker).
        columns:          Lista de colunas a retornar. None = todas.
        filters:          Dict de filtros no formato dd.read(filters=...).
                          None = sem filtragem de linhas.
        row_groups_meta:  Metadados de row groups do header (para pruning).
                          None ou [] = sem pruning (lê todos os batches).
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    marker = data[:5]

    # Arquivos legados (retrocompat) — sem row group pruning
    if marker[:4] == b"PQ1\x00":
        # Parquet — datalock v1.3 (retrocompat)
        try:
            import polars as _pl
            df = _pl.read_parquet(io.BytesIO(data[4:])).to_pandas()
        except Exception:
            import pyarrow.parquet as pq
            df = pq.read_table(io.BytesIO(data[4:])).to_pandas()
        # Aplica column pruning e filtros em pandas (retrocompat — sem stats)
        if columns:
            existing = [c for c in columns if c in df.columns]
            if existing:
                df = df[existing]
        if filters:
            df = _apply_pandas_filters(df, filters)
        return df

    # Arrow IPC — formato atual e legado sem marker
    ipc_data = data[5:] if marker == b"IPC1\x00" else data
    reader = ipc.open_stream(io.BytesIO(ipc_data))

    # Determina batches relevantes via row group pruning
    relevant_batches = prune_row_groups(row_groups_meta, filters)
    use_pruning = relevant_batches != {ALL_BATCHES_SENTINEL}

    batches = []
    n_batches = reader.num_record_batches

    for i in range(n_batches):
        batch = reader.get_batch(i)

        # Row group pruning: pula batches irrelevantes
        if use_pruning and i not in relevant_batches:
            continue  # lê e descarta sem construir arrays python

        # Column pruning: seleciona apenas as colunas necessárias
        if columns:
            available = [c for c in columns if c in batch.schema.names]
            if available:
                batch = batch.select(available)
            else:
                continue  # nenhuma das colunas pedidas existe neste batch

        # Predicate pushdown: filtra linhas no nível Arrow
        if filters:
            batch = apply_arrow_filters(batch, filters)
            if batch.num_rows == 0:
                continue  # batch vazio após filtro — não materializa

        batches.append(batch)

    if not batches:
        # Nenhum batch passou os filtros — retorna DataFrame vazio com schema correto
        schema = reader.schema_arrow
        if columns:
            available = [c for c in columns if c in schema.names]
            if available:
                schema = schema.select(available)
        empty_table = pa.table({f.name: pa.array([], type=f.type) for f in schema})
        return empty_table.to_pandas()

    table = pa.Table.from_batches(batches)
    return table.to_pandas()


def _apply_pandas_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Aplica filtros em DataFrame pandas (fallback para arquivos legados sem Arrow IPC).
    """
    normalized = normalize_filters(filters)
    mask = pd.Series([True] * len(df), index=df.index)

    for col, op, value in normalized:
        if col not in df.columns:
            continue
        s = df[col]
        if op == "==":
            mask &= s == value
        elif op == "!=":
            mask &= s != value
        elif op == ">":
            mask &= s > value
        elif op == ">=":
            mask &= s >= value
        elif op == "<":
            mask &= s < value
        elif op == "<=":
            mask &= s <= value
        elif op == "in":
            mask &= s.isin(value)
        elif op == "range":
            mask &= (s >= value[0]) & (s <= value[1])

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-frame: serialização de dict[str, DataFrame] → ZIP bytes e vice-versa
# ---------------------------------------------------------------------------

def _frames_to_zip_bytes(
    frames: Dict[str, pd.DataFrame],
    parquet_compression: str = "zstd",
) -> Tuple[bytes, List[Dict]]:
    """
    Serializa dict[str, DataFrame] para um ZIP em memória.

    Estrutura do ZIP:
      index.json           → lista de {name, size_bytes, rows, cols, schema}
      clientes.parquet     → Arrow IPC de df_clientes
      pedidos.parquet      → Arrow IPC de df_pedidos
      ...

    Returns:
        (zip_bytes, index_list) onde index_list é a lista de metadados dos frames.
    """
    if not frames:
        raise ValueError("O dicionário de frames não pode ser vazio.")
    invalid = [k for k in frames if not isinstance(k, str) or not k.strip()]
    if invalid:
        raise ValueError(f"Nomes de frame inválidos (devem ser strings não-vazias): {invalid!r}")

    buf = io.BytesIO()
    index: List[Dict] = []

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for name, df in frames.items():
            parquet_bytes, _rg = _df_to_bytes(df, parquet_compression=parquet_compression)
            filename = f"{name}.parquet"
            zf.writestr(filename, parquet_bytes)
            index.append({
                "name":        name,
                "filename":    filename,
                "size_bytes":  len(parquet_bytes),
                "rows":        df.shape[0],
                "cols":        df.shape[1],
                "schema":      {c: str(t) for c, t in df.dtypes.items()},
            })
        # Grava o índice dentro do ZIP
        zf.writestr(_MULTI_FRAME_INDEX, json.dumps(index, ensure_ascii=False))

    return buf.getvalue(), index


def _zip_bytes_to_frames(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Desserializa ZIP em memória para dict[str, DataFrame].

    Lê o index.json para garantir ordem e nomes, depois extrai cada Parquet.
    """
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, mode="r") as zf:
        index: List[Dict] = json.loads(zf.read(_MULTI_FRAME_INDEX).decode("utf-8"))
        frames: Dict[str, pd.DataFrame] = {}
        for entry in index:
            raw = zf.read(entry["filename"])
            frames[entry["name"]] = _bytes_to_df(raw)
    return frames


def _zip_bytes_to_single_frame(zip_bytes: bytes, frame: str) -> pd.DataFrame:
    """
    Extrai um único frame do ZIP sem desserializar os demais.
    """
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, mode="r") as zf:
        index: List[Dict] = json.loads(zf.read(_MULTI_FRAME_INDEX).decode("utf-8"))
        entry = next((e for e in index if e["name"] == frame), None)
        if entry is None:
            available = [e["name"] for e in index]
            raise KeyError(
                f"Frame '{frame}' não encontrado. "
                f"Frames disponíveis: {available}"
            )
        return _bytes_to_df(zf.read(entry["filename"]))


# ---------------------------------------------------------------------------
# Serialização multi-frame com ACL (Access Control per Frame)
# ---------------------------------------------------------------------------

def _frames_to_acl_zip_bytes(
    frames: Dict[str, pd.DataFrame],
    frame_access_levels: Dict[str, str],
    master_key_bytes: bytes,
    salt_kdf: bytes,
    parquet_compression: str = "zstd",
    cipher_str: str = "AES256GCM",
) -> Tuple[bytes, List[Dict]]:
    """
    Serializa frames em ZIP onde cada frame é cifrado com sua própria DEK de nível.

    Estrutura do ZIP resultante:
      index.json                  → lista de metadados com access_level por frame
      clientes_nonce.bin          → nonce (12 bytes) para o frame "clientes"
      clientes_tag.bin            → auth tag GCM (16 bytes)
      clientes.parquet.enc        → Parquet cifrado com DEK_clientes
      salarios_nonce.bin          → nonce para o frame "salarios"
      salarios_tag.bin            → auth tag GCM
      salarios.parquet.enc        → Parquet cifrado com DEK_salarios
      ...

    O ZIP inteiro é depois cifrado pela DEK principal do arquivo via _pack_v2_body,
    adicionando uma segunda camada de proteção. A DEK por frame protege o conteúdo
    dentro do ZIP — um adversário que extraísse o ZIP sem decifrar o arquivo externo
    ainda precisaria das DEKs de cada nível para ler qualquer frame.

    Args:
        frames:              Dict[nome → DataFrame].
        frame_access_levels: Dict[nome → access_level]. Frames sem entrada
                             usam o nível padrão "internal".
        master_key_bytes:    Master key para derivar DEKs por nível.
        salt_kdf:            Salt único do arquivo (já gerado antes de chamar esta função).
        parquet_compression: Compressão Parquet interna.
        cipher_str:          Algoritmo AEAD.

    Returns:
        (zip_bytes_cifrado, index_list)
    """
    buf = io.BytesIO()
    index: List[Dict] = []
    _require_cryptography()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for name, df in frames.items():
            level = frame_access_levels.get(name, "internal")
            frame_dek = _derive_frame_dek(master_key_bytes, salt_kdf, level)

            parquet_bytes, _rg = _df_to_bytes(df, parquet_compression=parquet_compression)
            nonce = os.urandom(NONCE_LEN)
            ct, tag = _encrypt(frame_dek, nonce, parquet_bytes, cipher_str)

            enc_filename = f"{name}.parquet.enc"
            zf.writestr(f"{name}_nonce.bin", nonce)
            zf.writestr(f"{name}_tag.bin", tag)
            zf.writestr(enc_filename, ct)

            index.append({
                "name":         name,
                "filename":     enc_filename,
                "access_level": level,
                "size_bytes":   len(ct),
                "rows":         df.shape[0],
                "cols":         df.shape[1],
                "schema":       {c: str(t) for c, t in df.dtypes.items()},
                "acl_encrypted": True,
            })

        zf.writestr(_MULTI_FRAME_INDEX, json.dumps(index, ensure_ascii=False))

    return buf.getvalue(), index


def _acl_zip_bytes_to_frames(
    zip_bytes: bytes,
    master_key_bytes: bytes,
    salt_kdf: bytes,
    allowed_levels: Optional[List[str]] = None,
    cipher_str: str = "AES256GCM",
) -> Dict[str, pd.DataFrame]:
    """
    Desserializa ZIP com ACL, decrifrando apenas os frames autorizados.

    Args:
        zip_bytes:       ZIP em memória (payload do arquivo .dlk já decifrado).
        master_key_bytes: Master key para derivar DEKs.
        salt_kdf:        Salt do arquivo.
        allowed_levels:  Lista de níveis que o usuário pode acessar.
                         None = acesso a todos os níveis (admin).
        cipher_str:      Algoritmo AEAD.

    Returns:
        Dict[nome → DataFrame] contendo apenas os frames autorizados.
        Frames de nível não autorizado são omitidos sem erro — o chamador
        recebe apenas o que pode ver.

    Raises:
        PermissionError: Se nenhum frame estiver acessível com os níveis fornecidos.
    """
    buf = io.BytesIO(zip_bytes)
    frames: Dict[str, pd.DataFrame] = {}
    skipped: List[str] = []

    with zipfile.ZipFile(buf, mode="r") as zf:
        index: List[Dict] = json.loads(zf.read(_MULTI_FRAME_INDEX).decode("utf-8"))

        for entry in index:
            name   = entry["name"]
            level  = entry.get("access_level", "internal")
            is_acl = entry.get("acl_encrypted", False)

            # Verifica autorização — None significa acesso total
            if allowed_levels is not None and level not in allowed_levels:
                skipped.append(f"{name} (level={level})")
                logger.debug("ACL: frame '%s' ignorado (level=%s não autorizado)", name, level)
                continue

            if is_acl:
                # Frame cifrado por nível — deriva DEK específica e decifra
                frame_dek = _derive_frame_dek(master_key_bytes, salt_kdf, level)
                nonce = zf.read(f"{name}_nonce.bin")
                tag   = zf.read(f"{name}_tag.bin")
                ct    = zf.read(entry["filename"])
                try:
                    parquet_bytes = _decrypt(frame_dek, nonce, ct, tag, cipher_str)
                except RuntimeError:
                    # DEK incorreta para este nível — não deve acontecer se
                    # master_key e salt_kdf estiverem corretos
                    logger.error("ACL: falha ao decifrar frame '%s' (DEK inválida?)", name)
                    skipped.append(f"{name} (decryption_failed)")
                    continue
            else:
                # Frame sem ACL (retrocompatível com pack_frames clássico)
                parquet_bytes = zf.read(entry["filename"])

            frames[name] = _bytes_to_df(parquet_bytes)

    if skipped:
        logger.info("ACL: frames omitidos por nível de acesso: %s", skipped)

    if not frames and skipped:
        raise PermissionError(
            f"Nenhum frame acessível com os níveis fornecidos ({allowed_levels}). "
            f"Frames presentes mas não autorizados: {[s.split(' ')[0] for s in skipped]}"
        )

    return frames


def _acl_zip_bytes_to_single_frame(
    zip_bytes: bytes,
    master_key_bytes: bytes,
    salt_kdf: bytes,
    frame: str,
    allowed_levels: Optional[List[str]] = None,
    cipher_str: str = "AES256GCM",
) -> pd.DataFrame:
    """Extrai um único frame ACL do ZIP, verificando autorização."""
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, mode="r") as zf:
        index: List[Dict] = json.loads(zf.read(_MULTI_FRAME_INDEX).decode("utf-8"))
        entry = next((e for e in index if e["name"] == frame), None)
        if entry is None:
            # Não listar frames disponíveis: em contexto ACL isso revelaria
            # nomes de frames confidenciais para usuários sem acesso a eles.
            raise KeyError(
                f"Frame '{frame}' não encontrado. "
                "Use SecureFile.verify() ou load_frames() para listar os frames acessíveis."
            )

        level  = entry.get("access_level", "internal")
        is_acl = entry.get("acl_encrypted", False)

        if allowed_levels is not None and level not in allowed_levels:
            raise PermissionError(
                f"Frame '{frame}' requer nível '{level}'. "
                f"Níveis autorizados: {allowed_levels}"
            )

        if is_acl:
            frame_dek = _derive_frame_dek(master_key_bytes, salt_kdf, level)
            nonce = zf.read(f"{frame}_nonce.bin")
            tag   = zf.read(f"{frame}_tag.bin")
            ct    = zf.read(entry["filename"])
            parquet_bytes = _decrypt(frame_dek, nonce, ct, tag, cipher_str)
        else:
            parquet_bytes = zf.read(entry["filename"])

    return _bytes_to_df(parquet_bytes)


# ---------------------------------------------------------------------------
# Primitivos criptográficos
# ---------------------------------------------------------------------------

def _derive_dek(master_key: bytes, salt_kdf: bytes) -> bytes:
    """
    Deriva a Data Encryption Key (DEK) via HKDF-SHA256 (RFC 5869).

    DEK (info=b"datalock-dek-v1"): cifra o payload de dados.
    HEK (info=b"datalock-hek-v1"): cifra o header JSON.
    info= diferentes garantem DEK ≠ HEK mesmo com o mesmo salt e master_key.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_kdf,
        info=b"datalock-dek-v1",
    ).derive(master_key)


def _derive_hek(master_key: bytes, salt_kdf: bytes) -> bytes:
    """
    Deriva a Header Encryption Key (HEK) via HKDF-SHA256.

    Chave separada da DEK — garante que comprometer o payload não expõe o header.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_kdf,
        info=b"datalock-hek-v1",
    ).derive(master_key)


def _derive_mak(master_key: bytes, salt_kdf: bytes) -> bytes:
    """
    Deriva a Message Authentication Key (MAK) via HKDF-SHA256 (RFC 5869).

    Usada exclusivamente para o FILE_HMAC — autenticação do layout binário
    completo do arquivo. Separada da DEK e da HEK por domínio (info=) distinto,
    garantindo que nenhuma das três chaves possa ser derivada a partir das outras,
    mesmo com acesso ao salt_kdf em claro.

    Sem essa separação, a master_key seria usada simultaneamente como input do
    HKDF (para DEK/HEK) e como chave HMAC direta — reuso de material de chave
    em contextos diferentes, violando o princípio de separação de domínio do
    NIST SP 800-108r1 e da RFC 5869 §3.

    Referências:
      - RFC 5869 §3: "the use of the same key for different cryptographic
        operations is strongly discouraged"
      - NIST SP 800-108r1: domain separation via label/context in KDF
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_kdf,
        info=b"datalock-mak-v1",
    ).derive(master_key)


def _derive_frame_dek(master_key: bytes, salt_kdf: bytes, access_level: str) -> bytes:
    """
    Deriva uma Data Encryption Key (DEK) específica para um nível de acesso.

    Cada nível de acesso produz uma DEK distinta, derivada da mesma master_key
    mas com campo `info` diferente — separação de domínio criptográfico completa.

    Modelo de controle de acesso:
      master_key  →  HKDF(salt, info="frame-dek-v1:{level}")  →  DEK_level

    Consequências de segurança:
      1. Um usuário com acesso ao nível "user" recebe apenas DEK_user.
         DEK_admin é computacionalmente indistinguível de ruído aleatório
         para quem não possui a master_key — não pode ser derivada a partir
         de DEK_user ou de qualquer combinação de DEKs conhecidas.

      2. O comprometimento de DEK_user não expõe DEK_admin, nem a master_key,
         nem DEKs de outros níveis (propriedade de forward secrecy por nível).

      3. A master_key nunca é transmitida — apenas as DEKs derivadas são
         distribuídas por nível, tipicamente via KMS ou variável de ambiente
         específica por perfil de usuário.

    Hierarquia recomendada de níveis:
      "public"      → dados sem restrição de acesso
      "internal"    → dados internos da organização
      "confidential"→ dados sensíveis (RH, jurídico)
      "restricted"  → dados altamente sensíveis (executivos, M&A)
      "secret"      → dados pessoais de alta criticidade (LGPD art. 11)

    A hierarquia é convencional — o sistema criptográfico trata os níveis
    como strings opacas. O controle de quem recebe qual DEK é responsabilidade
    do KMS ou do sistema de distribuição de chaves da organização.

    Args:
        master_key:   Chave mestre da organização (bytes).
        salt_kdf:     Salt único por arquivo (32 bytes de os.urandom).
        access_level: String identificando o nível (ex: "user", "admin").

    Returns:
        DEK de 32 bytes única para (master_key, salt_kdf, access_level).
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    # O info= inclui o access_level como sufixo — garante que níveis distintos
    # produzam DEKs distintas mesmo com mesma master_key e salt_kdf.
    info = f"datalock-frame-dek-v1:{access_level}".encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_kdf,
        info=info,
    ).derive(master_key)


def _detect_best_cipher() -> str:
    """
    Retorna 'AES256GCM' se AES-NI estiver disponível, 'ChaCha20Poly1305' caso contrário.

    A detecção usa um benchmark de 1MB para medir throughput real em vez de
    confiar em flags de CPU (que podem não refletir o suporte do runtime OpenSSL).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

    probe  = os.urandom(1024 * 1024)  # 1 MB
    key32  = os.urandom(32)
    nonce12 = os.urandom(12)

    t0 = time.perf_counter()
    AESGCM(key32).encrypt(nonce12, probe, None)
    t_aes = time.perf_counter() - t0

    t0 = time.perf_counter()
    ChaCha20Poly1305(key32).encrypt(nonce12, probe, None)
    t_cha = time.perf_counter() - t0

    return "AES256GCM" if t_aes <= t_cha else "ChaCha20Poly1305"


_BEST_CIPHER: Optional[str] = None  # lazy — computado na primeira escrita


def _get_best_cipher() -> str:
    """Retorna (e cacheia) o melhor cipher disponível. Lazy para não custar no import."""
    global _BEST_CIPHER
    if _BEST_CIPHER is None:
        _BEST_CIPHER = _detect_best_cipher()
    return _BEST_CIPHER


def _encrypt(
    dek: bytes, nonce: bytes, plaintext: bytes, cipher: str = "AES256GCM"
) -> Tuple[bytes, bytes]:
    """
    Cifra com AES-256-GCM ou ChaCha20-Poly1305. Retorna (ciphertext, auth_tag).

    Ambos são AEAD: adulteração no ciphertext é detectada na decifração.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

    aead = AESGCM(dek) if cipher == "AES256GCM" else ChaCha20Poly1305(dek)
    combined = aead.encrypt(nonce, plaintext, None)
    return combined[:-AUTH_TAG_LEN], combined[-AUTH_TAG_LEN:]


def _decrypt(
    dek: bytes, nonce: bytes, ciphertext: bytes, auth_tag: bytes, cipher: str = "AES256GCM"
) -> bytes:
    """
    Decifra com AES-256-GCM ou ChaCha20-Poly1305.
    Verifica auth_tag antes de retornar qualquer dado.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

    aead = AESGCM(dek) if cipher == "AES256GCM" else ChaCha20Poly1305(dek)
    try:
        return aead.decrypt(nonce, ciphertext + auth_tag, None)
    except Exception:
        raise RuntimeError(
            "Falha na decifração (auth_tag inválida). "
            "O payload foi corrompido ou a master_key está incorreta."
        ) from None


# Aliases para retrocompatibilidade interna
def _aes_gcm_encrypt(dek, nonce, plaintext): return _encrypt(dek, nonce, plaintext, "AES256GCM")
def _aes_gcm_decrypt(dek, nonce, ct, tag):   return _decrypt(dek, nonce, ct, tag, "AES256GCM")


def _compute_file_hmac(master_key: bytes, data: bytes, salt_kdf: Optional[bytes] = None) -> bytes:
    """
    HMAC-SHA256 sobre os dados do arquivo (header + payload cifrado).

    Quando salt_kdf é fornecido (arquivos v2/v3), a chave HMAC é a MAK derivada
    via HKDF — separação de domínio completa entre DEK, HEK e MAK.
    Quando salt_kdf é None (arquivos v1 legados ou v4 sem key), usa master_key
    diretamente para manter retrocompatibilidade.
    """
    hmac_key = _derive_mak(master_key, salt_kdf) if salt_kdf is not None else master_key
    return _hmac_module.new(hmac_key, data, hashlib.sha256).digest()


def _verify_file_hmac(master_key: bytes, data: bytes, expected: bytes, salt_kdf: Optional[bytes] = None) -> bool:
    """Verifica HMAC em tempo constante (resistente a timing attack)."""
    return _hmac_module.compare_digest(
        _compute_file_hmac(master_key, data, salt_kdf), expected
    )


def _parse_v2_header_only(raw: bytes, master_key_bytes: bytes) -> Dict:
    """
    Decifra e retorna APENAS o cabeçalho de um arquivo v2/v3, sem tocar no payload.

    Usado por verify() e inspect() para auditoria de metadados sem carregar
    o payload de dados em memória. Garante que a propriedade de "auditabilidade
    sem exposição do conteúdo" seja real na implementação, não apenas declarada.

    O payload nunca é decifrado, descomprimido ou deserializado — existe no
    buffer raw[] como bytes cifrados inacessíveis até que load() seja chamado
    com intenção explícita de ler os dados.
    """
    offset = len(MAGIC) + 1  # skip MAGIC + VERSION
    cipher_byte = raw[offset]; offset += 1
    cipher_str  = _CIPHER_BYTE_TO_STR.get(cipher_byte, "AES256GCM")

    salt_kdf     = raw[offset:offset + SALT_KDF_LEN];  offset += SALT_KDF_LEN
    nonce_header = raw[offset:offset + NONCE_LEN];      offset += NONCE_LEN
    header_ct_len = struct.unpack(HEADER_LEN_FMT, raw[offset:offset + 4])[0]; offset += 4
    if header_ct_len > MAX_HEADER_LEN:
        raise ValueError(
            f"Header cifrado excede o tamanho máximo permitido "
            f"({header_ct_len} bytes > {MAX_HEADER_LEN} bytes). "
            "Arquivo possivelmente malformado ou corrompido."
        )
    header_ct_with_tag = raw[offset:offset + header_ct_len]
    header_ct  = header_ct_with_tag[:-AUTH_TAG_LEN]
    header_tag = header_ct_with_tag[-AUTH_TAG_LEN:]

    hek = _derive_hek(master_key_bytes, salt_kdf)
    header_plain = _decrypt(hek, nonce_header, header_ct, header_tag, cipher_str)
    return json.loads(header_plain.decode("utf-8")), salt_kdf
    # payload nunca tocado — offset após header_ct_with_tag aponta para
    # NONCE_PAYLOAD seguido do bloco cifrado, que permanece intocado.


# ---------------------------------------------------------------------------
# Versão 4 — .dlk sem criptografia (plaintext + HMAC de integridade opcional)
# ---------------------------------------------------------------------------
# Casos de uso: ambientes internos onde confidencialidade não é necessária mas
# a integridade e o formato padronizado (Parquet + metadados LGPD) ainda são úteis.
# A ausência de key é declarada explicitamente no header — não há "segurança por
# obscuridade". Arquivos sem key não devem conter dados pessoais não mascarados.
#
# Estrutura v4 (sem criptografia):
#   [5  bytes]  MAGIC     = b"DLOCK"
#   [1  byte ]  VERSION   = 0x04
#   [4  bytes]  HEADER_LEN — tamanho do header JSON em claro
#   [N  bytes]  HEADER JSON em claro (inclui "encrypted": false)
#   [M  bytes]  PAYLOAD em claro (Parquet/zstd)
#   [32 bytes]  FILE_HMAC — HMAC com chave derivada do conteúdo (ou zeros se sem HMAC)
#
# Nota: sem key, o FILE_HMAC usa chave pública fixa ("datalock-no-key-v4") que
# detecta corrupção acidental mas NÃO protege contra adulteração intencional.
# ---------------------------------------------------------------------------

VERSION_V4 = 0x04
_NO_KEY_HMAC_KEY = b"datalock-no-key-integrity-v1"


# ---------------------------------------------------------------------------
# Helpers internos de escrita/leitura da estrutura v2/v4
# ---------------------------------------------------------------------------

def _check_expiry(header: Dict) -> None:
    """Lança ExpiredFileError se o arquivo passou do prazo definido em expires_at."""
    import warnings as _warnings
    expires_at = header.get("expires_at")
    if not expires_at:
        return
    try:
        from datetime import datetime, timezone
        expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if now > expiry:
            raise ExpiredFileError(
                f"Arquivo expirado em {expiry.strftime('%Y-%m-%d')}. "
                f"Conforme a política de retenção LGPD (Art. 16), "
                f"dados após o prazo devem ser eliminados."
            )
    except ExpiredFileError:
        raise
    except Exception as e:
        # Formato de data inválido — alerta em vez de ignorar silenciosamente.
        # Um campo expires_at mal-formatado pode mascarar uma expiração real;
        # o comportamento seguro é alertar, não prosseguir sem aviso.
        _warnings.warn(
            f"Campo 'expires_at' com valor inválido ({expires_at!r}): {e}. "
            "Verificação de expiração ignorada para este arquivo. "
            "Corrija o campo para garantir a política de retenção LGPD (Art. 16).",
            UserWarning,
            stacklevel=3,
        )


class ExpiredFileError(ValueError):
    """Levantado quando um arquivo .dlk expirou (expires_at no header)."""
    pass


def _pack_v2_body(
    master_key_bytes: bytes,
    header: Dict,
    plaintext_bytes: bytes,
    cipher_str: str,
    compress: bool,
    version_byte: int = VERSION_V2,
) -> Tuple[bytes, bytes]:
    """
    Constrói o corpo de um arquivo .dlk v2/v3 (sem o FILE_HMAC final).

    version_byte: VERSION_V2 (single-frame) ou VERSION_V3 (multi-frame).
    Retorna (body, salt_kdf) — o salt_kdf é necessário para derivar a MAK
    e calcular o FILE_HMAC com separação de domínio completa.
    """
    cipher_byte = _CIPHER_STR_TO_BYTE[cipher_str]

    salt_kdf = os.urandom(SALT_KDF_LEN)
    dek = _derive_dek(master_key_bytes, salt_kdf)
    hek = _derive_hek(master_key_bytes, salt_kdf)

    nonce_payload = os.urandom(NONCE_LEN)
    ct_payload, tag_payload = _encrypt(dek, nonce_payload, plaintext_bytes, cipher_str)

    header_plain = json.dumps(header, ensure_ascii=False).encode("utf-8")
    nonce_header = os.urandom(NONCE_LEN)
    ct_header, tag_header = _encrypt(hek, nonce_header, header_plain, cipher_str)
    header_ct_with_tag = ct_header + tag_header

    body = (
        MAGIC
        + bytes([version_byte])
        + bytes([cipher_byte])
        + salt_kdf
        + nonce_header
        + struct.pack(HEADER_LEN_FMT, len(header_ct_with_tag))
        + header_ct_with_tag
        + nonce_payload
        + ct_payload
        + tag_payload
    )
    return body, salt_kdf


def _pack_v4_body(header: Dict, plaintext_bytes: bytes) -> bytes:
    """
    Constrói o corpo de um arquivo .dlk v4 sem criptografia.

    Sem key: payload Parquet em claro, header JSON em claro.
    Deve ser usado APENAS com dados já anonimizados.
    """
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return (
        MAGIC
        + bytes([VERSION_V4])
        + struct.pack(HEADER_LEN_FMT, len(header_bytes))
        + header_bytes
        + plaintext_bytes
    )


def _write_lgs(
    output: Path, body: bytes, master_key_bytes: Optional[bytes], salt_kdf: Optional[bytes] = None
) -> None:
    """
    Grava body + HMAC em disco de forma atômica (write → rename).

    salt_kdf: quando fornecido (arquivos v2/v3), o FILE_HMAC é calculado com a
    MAK derivada via HKDF — separação de domínio completa. Para v1 e v4 (sem
    salt_kdf disponível), usa master_key_bytes diretamente para retrocompatibilidade.

    Segurança de concorrência:
      O arquivo temporário usa um nome único por chamada (uuid4) em vez de um
      sufixo fixo (.dlk.tmp). Isso evita que duas threads/processos que escrevam
      no mesmo `output` simultaneamente usem o mesmo caminho temporário, o que
      resultaria em corrupção silenciosa do arquivo destino. A operação
      tmp.replace(output) continua atômica no mesmo filesystem.
    """
    import uuid as _uuid
    hmac_key_bytes = master_key_bytes if master_key_bytes is not None else _NO_KEY_HMAC_KEY
    file_hmac = _compute_file_hmac(hmac_key_bytes, body, salt_kdf)
    # Nome único por chamada: evita colisão entre escritas paralelas ao mesmo output
    tmp = output.parent / f".{output.name}.{_uuid.uuid4().hex}.tmp"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(body + file_hmac)
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _validate_key(master_key: str) -> bytes:
    """
    Valida comprimento mínimo e retorna bytes UTF-8.

    Segurança — HKDF vs KDF de senha:
      HKDF-SHA256 é um KDF de *expansão*, não de *stretching*. Ele pressupõe
      que o IKM (input key material) já é criptograficamente forte (saída de
      KMS, os.urandom, vault). Se uma senha humana for fornecida como master_key,
      um adversário com o arquivo pode testá-la a bilhões de tentativas/s com GPU.

      Para chaves derivadas de senhas humanas, aplique Argon2id ou PBKDF2-SHA256
      (≥600.000 iterações) antes de passar o resultado aqui.

      Para ambientes corporativos, gere a master_key via:
        python -c "import secrets; print(secrets.token_hex(32))"
      e armazene no vault (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault).
    """
    import math
    import warnings as _warnings

    encoded = master_key.encode("utf-8")
    if len(encoded) < 16:
        raise ValueError(
            "master_key muito curta (mínimo 16 bytes). "
            "Use uma chave de alta entropia para proteger dados pessoais."
        )

    # Heurística de entropia de Shannon para detectar senhas humanas.
    # Chaves de alta entropia (saída de KMS/urandom/token_hex) têm
    # distribuição quase uniforme → Shannon entropy próxima ao máximo.
    # Senhas humanas e frases comuns têm entropia típica < 3.5 bits/char.
    freq: dict = {}
    for c in master_key:
        freq[c] = freq.get(c, 0) + 1
    n = len(master_key)
    shannon = -sum((f / n) * math.log2(f / n) for f in freq.values())
    n_unique = len(freq)

    _WEAK_KEY_PATTERNS = frozenset({
        "password", "senha", "chave", "key", "secret", "segredo",
        "admin", "master", "test", "teste", "demo", "example", "exemplo",
        "123456", "abcdef", "qwerty",
    })
    lower = master_key.lower()
    has_weak_pattern = any(p in lower for p in _WEAK_KEY_PATTERNS)

    # Limites: chaves de 32 hex chars têm ~4 bits/char (alphabet de 16 símbolos)
    # e n_unique ≈ 16. Senhas humanas típicas têm < 3.5 bits/char ou < 10 únicos.
    low_entropy = (n_unique < 10 or shannon < 3.5) and len(master_key) < 48

    if has_weak_pattern or low_entropy:
        reasons = []
        if has_weak_pattern:
            reasons.append("contém palavra-chave fraca")
        if low_entropy:
            reasons.append(
                f"baixa entropia estimada ({shannon:.1f} bits/char, {n_unique} chars únicos)"
            )
        _warnings.warn(
            f"master_key com possível baixa entropia ({'; '.join(reasons)}). "
            "HKDF não é um KDF de senhas — use material de alta entropia "
            "(saída de KMS, secrets.token_hex(32), vault). "
            "Para senhas humanas, aplique Argon2id antes de passar a chave aqui.",
            UserWarning,
            stacklevel=3,
        )

    return encoded


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Arquivo de saída já existe: {output}. Use overwrite=True para substituir."
        )


# ---------------------------------------------------------------------------
# SecureFile
# ---------------------------------------------------------------------------

class SecureFile:
    """
    Interface estática para criar, ler e verificar arquivos `.dlk`.

    Todos os métodos são classmethod — não há instância, apenas operações.

    Métodos principais:
        pack()            → empacota arquivo de dados (CSV, xlsx, parquet, json)
        pack_dataframe()  → empacota DataFrame único diretamente
        pack_frames()     → empacota dict[str, DataFrame] (multi-frame, v3)
        pack_bytes()      → empacota payload binário arbitrário
        load()            → decifra e retorna DataFrame (aplica mascaramento se configurado)
        load_raw()        → decifra e retorna DataFrame SEM mascaramento automático
        load_bytes()      → decifra e retorna bytes brutos (para payloads não-tabulares)
        load_frames()     → decifra multi-frame → dict[str, DataFrame]
        load_frame()      → decifra multi-frame → um único DataFrame pelo nome
        verify()          → verifica integridade sem decifrar
    """

    CONTENT_TYPE_RAW        = "raw_dataframe"      # dados brutos — ainda não mascarados
    CONTENT_TYPE_MASKED     = "masked_dataframe"    # mascaramento HMAC/REDACT aplicado
    CONTENT_TYPE_ANON       = "anonymous_dataframe" # anonimizado, sem key obrigatória
    CONTENT_TYPE_MULTI      = "multi_dataframe"     # múltiplos DataFrames (v3)
    CONTENT_TYPE_BYTES      = "bytes"               # payload binário arbitrário
    CONTENT_TYPE_CUSTOM     = "custom"              # tipos definidos pelo usuário

    # ------------------------------------------------------------------
    # Escrita — DataFrame único (com ou sem criptografia)
    # ------------------------------------------------------------------

    @classmethod
    def pack_open(
        cls,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        *,
        anonymize: bool = True,
        salt_masking: Optional[str] = None,
        label: str = "",
        compress: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Empacota um DataFrame em .dlk **sem criptografia** (key opcional).

        Use quando a confidencialidade não é necessária mas o formato
        padronizado (Parquet + metadados LGPD + integridade) ainda é útil.
        Exemplos: compartilhamento de dados já anonimizados em ambiente interno,
        datasets de desenvolvimento sem PII real.

        Segurança:
          - Sem key: header e payload ficam em CLARO no arquivo.
          - O FILE_HMAC usa chave pública fixa — detecta corrupção acidental
            mas NÃO protege contra adulteração intencional.
          - Use SOMENTE com dados que já não contenham PII, ou com
            anonymize=True (mascaramento aplicado antes de gravar).

        Args:
            df:           DataFrame a empacotar.
            output_path:  Caminho do .dlk de saída.
            anonymize:    Se True (padrão), aplica mascaramento PII antes de gravar.
                          Se False, grava o dado como está (garanta que não há PII).
            salt_masking: Salt HMAC para mascaramento quando anonymize=True.
                          Se None, salt aleatório (hashes não reprodutíveis).
            label:        Rótulo livre para auditoria.
            compress:     True → Parquet/zstd; False → Parquet/lz4.
            overwrite:    Sobrescreve arquivo existente.

        Returns:
            Dict com metadados: shape, tamanho, elapsed.

        Exemplo:
            # Dados de desenvolvimento sem PII
            SecureFile.pack_open(df_anonimo, "dados_dev.dlk")

            # Aplica mascaramento automático antes de gravar
            SecureFile.pack_open(df_bruto, "dados_dev.dlk",
                                 anonymize=True, salt_masking="salt-dev")

            # Lê sem key
            df = SecureFile.load_open("dados_dev.dlk")
        """
        output = Path(output_path)
        _check_output(output, overwrite)

        t0 = time.perf_counter()
        original_shape = list(df.shape)

        if anonymize:
            from datalock.adapters.pandas_adapter import secure_dataframe as _secure
            df = _secure(df, salt=salt_masking)
            content_type = SecureFile.CONTENT_TYPE_ANON
        else:
            content_type = SecureFile.CONTENT_TYPE_ANON

        parquet_comp = "zstd" if compress else "lz4"
        plaintext_bytes, _rg_meta = _df_to_bytes(
            df, parquet_compression=parquet_comp,
            content_type=content_type,
        )

        header = {
            "format":               "lgs",
            "version":              "4.0",
            "content_type":         content_type,
            "encrypted":            False,
            "label":                label,
            "created_at":           datetime.now(timezone.utc).isoformat(),
            "created_by":           f"datalock/{_logus_version()}",
            "shape":                list(df.shape),
            "schema":               {c: str(t) for c, t in df.dtypes.items()},
            "masking_applied":      anonymize,
            "compression":          f"ipc_{parquet_comp}",
            "kdf":                  "none",
            "encryption":           "none",
            "integrity":            "HMAC-SHA256 (public key — tamper detection only)",
            "plaintext_size_bytes": len(plaintext_bytes),
        }

        body = _pack_v4_body(header, plaintext_bytes)
        _write_lgs(output, body, None)

        elapsed = time.perf_counter() - t0
        packed_size = output.stat().st_size
        logger.info(
            "SecureFile.pack_open | content_type=%s | shape=%s | encrypted=False | %.3fs",
            content_type, original_shape, elapsed,
        )
        return {
            "output_path":       str(output),
            "content_type":      content_type,
            "encrypted":         False,
            "shape":             original_shape,
            "original_size_kb":  round(len(plaintext_bytes) / 1024, 1),
            "packed_size_kb":    round(packed_size / 1024, 1),
            "compression_ratio": round(len(plaintext_bytes) / max(packed_size, 1), 3),
            "elapsed_seconds":   round(elapsed, 3),
        }

    @classmethod
    def load_open(
        cls,
        path: Union[str, Path],
        *,
        anonymize: bool = False,
        salt_masking: Optional[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Lê um arquivo .dlk sem criptografia (v4 / pack_open).

        Também funciona para arquivos v1/v2/v3 que não precisem de key
        na chamada — mas nesses casos a key é obrigatória e o erro é claro.

        Args:
            path:          Caminho para o arquivo `.dlk`.
            apply_masking: Se True, aplica mascaramento PII na leitura.
            salt_masking:  Salt para mascaramento.
            verbose:       Exibe relatório de detecção PII.

        Returns:
            DataFrame com o conteúdo decifrado.

        Raises:
            ValueError: Se o arquivo estiver criptografado (use load() com key=).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        raw = p.read_bytes()
        if not raw.startswith(MAGIC):
            raise ValueError(f"'{p.name}' não é um arquivo .dlk válido.")

        file_version = raw[len(MAGIC)]
        if file_version != VERSION_V4:
            raise ValueError(
                f"'{p.name}' está criptografado (versão {file_version:#04x}). "
                f"Use SecureFile.load(path, master_key='sua-chave') para arquivos cifrados."
            )

        # Verifica integridade com chave pública (detecta corrupção acidental)
        body, file_hmac = raw[:-FILE_HMAC_LEN], raw[-FILE_HMAC_LEN:]
        if not _verify_file_hmac(_NO_KEY_HMAC_KEY, body, file_hmac):
            raise RuntimeError(
                f"'{p.name}': HMAC de integridade falhou. "
                f"O arquivo pode estar corrompido (atenção: sem key, adulteração "
                f"intencional não é detectável)."
            )

        # Parseia header v4 (plaintext)
        offset = len(MAGIC) + 1
        header_len = struct.unpack(HEADER_LEN_FMT, body[offset:offset + 4])[0]
        offset += 4
        if header_len > MAX_HEADER_LEN:
            raise ValueError(
                f"Header JSON excede o tamanho máximo permitido "
                f"({header_len} bytes > {MAX_HEADER_LEN} bytes). "
                "Arquivo possivelmente malformado ou corrompido."
            )
        header = json.loads(body[offset:offset + header_len].decode("utf-8"))
        offset += header_len
        payload_bytes = body[offset:]

        content_type = header.get("content_type", SecureFile.CONTENT_TYPE_ANON)
        if content_type == SecureFile.CONTENT_TYPE_MULTI:
            raise TypeError(
                f"'{p.name}' é um arquivo multi-frame. "
                f"Use SecureFile.load_frames() para ler todos os frames."
            )

        df = _bytes_to_df(payload_bytes)

        if anonymize:
            from datalock.adapters.pandas_adapter import secure_dataframe as _secure
            df = _secure(df, salt=salt_masking, verbose=verbose)
        elif not header.get("masking_applied", False):
            import warnings as _w
            _w.warn(
                f"load_open('{p.name}'): dados não foram anonimizados (masking_applied=False). "
                "Use anonymize=True ou verifique se o arquivo contém PII.",
                UserWarning, stacklevel=2,
            )

        logger.info("SecureFile.load_open | %s | shape=%s", p.name, df.shape)
        return df

    @classmethod
    def pack(
        cls,
        source_path: Union[str, Path],
        output_path: Union[str, Path],
        key: str,
        master_key: Optional[str] = None,  # deprecated alias
        anonymize: bool = False,
        salt_masking: Optional[str] = None,
        label: str = "",
        compress: bool = True,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Empacota um arquivo de dados em formato `.dlk` criptografado.

        Args:
            source_path:   Arquivo de origem (.csv, .xlsx, .parquet, .json).
            output_path:   Caminho de saída (ex: "clientes.dlk").
            master_key:    Chave mestre para criptografia (≥ 16 chars).
            apply_masking: Se True, aplica mascaramento ANTES de criptografar.
            salt_masking:  Salt HMAC para mascaramento (obrigatório se apply_masking=True).
            label:         Rótulo livre para o header (auditoria).
            compress:      Se True (padrão), Parquet/zstd; False → Parquet/lz4.
            overwrite:     Se True, sobrescreve arquivo existente.

        Returns:
            Dict com metadados do arquivo criado (tamanho, shape, ratio de compressão).
        """
        key = key if key is not None else master_key  # compat
        source = Path(source_path)
        output = Path(output_path)

        if not source.exists():
            raise FileNotFoundError(f"Arquivo de origem não encontrado: {source}")
        _check_output(output, overwrite)
        master_key_bytes = _validate_key(key)

        t0 = time.perf_counter()
        df = _read_source(source)
        original_shape = list(df.shape)

        if anonymize:
            if not salt_masking:
                raise ValueError(
                    "salt_masking é obrigatório quando anonymize=True. "
                    "Use uma chave HMAC separada da master_key."
                )
            df = secure_dataframe(df, salt=salt_masking)

        content_type = (
            SecureFile.CONTENT_TYPE_MASKED if anonymize
            else SecureFile.CONTENT_TYPE_RAW
        )
        parquet_comp = "zstd" if compress else "lz4"
        plaintext_bytes, row_groups_meta = _df_to_bytes(
            df, parquet_compression=parquet_comp,
            content_type=content_type,
        )

        header = cls._build_header(
            content_type=content_type,
            label=label,
            df=df,
            parquet_comp=parquet_comp,
            cipher_str=_get_best_cipher(),
            plaintext_size=len(plaintext_bytes),
            metadata=metadata,
            expires_at=expires_at,
            row_groups_meta=row_groups_meta,
        )

        body, salt_kdf = _pack_v2_body(master_key_bytes, header, plaintext_bytes, _get_best_cipher(), compress)
        _write_lgs(output, body, master_key_bytes, salt_kdf)

        elapsed = time.perf_counter() - t0
        logger.info(
            "SecureFile.pack | source=%s | content_type=%s | shape=%s | %.3fs",
            source.name, content_type, original_shape, elapsed,
        )
        packed_size = output.stat().st_size
        return {
            "output_path":       str(output),
            "content_type":      content_type,
            "shape":             original_shape,
            "original_size_kb":  round(len(plaintext_bytes) / 1024, 1),
            "packed_size_kb":    round(packed_size / 1024, 1),
            "compression_ratio": round(len(plaintext_bytes) / max(packed_size, 1), 3),
            "elapsed_seconds":   round(elapsed, 3),
        }

    @classmethod
    def pack_dataframe(
        cls,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias
        content_type: str = "raw_dataframe",
        label: str = "",
        compress: bool = True,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
        canary: bool = False,
        canary_n_rows: int = 3,
        canary_pipeline_id: Optional[str] = None,  # nome explícito (preferido)
        pipeline_id: Optional[str] = None,          # alias legado
    ) -> Dict[str, Any]:
        """
        Empacota um DataFrame diretamente em formato .dlk criptografado.

        v2.0 (v1.3.2):
          - Header JSON cifrado com HEK separado (schema/metadados não ficam em claro)
          - Cipher auto-negociado: AES-256-GCM (com AES-NI) ou ChaCha20-Poly1305
          - Payload: Parquet/zstd ou Parquet/lz4 em memória + AEAD

        Args:
            df:           DataFrame a empacotar.
            output_path:  Caminho do .dlk de saída.
            master_key:   Chave mestre (>= 16 chars).
            content_type: CONTENT_TYPE_RAW | CONTENT_TYPE_MASKED | CONTENT_TYPE_BYTES.
            label:        Rótulo livre para auditoria (ex: "exportacao_crm_mai").
            compress:     True → Parquet/zstd (menor); False → Parquet/lz4 (mais rápido).
            overwrite:    Sobrescreve arquivo existente.

        Returns:
            Dict com metadados: shape, tamanho, cipher, compression, elapsed.
        """
        key = key if key is not None else master_key  # compat: master_key= is deprecated
        output = Path(output_path)
        _check_output(output, overwrite)
        master_key_bytes = _validate_key(key)

        t0 = time.perf_counter()
        original_shape = list(df.shape)
        parquet_comp = "zstd" if compress else "lz4"

        # Resolve pipeline_id: canary_pipeline_id tem prioridade sobre pipeline_id
        resolved_pipeline_id = canary_pipeline_id or pipeline_id

        # Canary injection (transparent — stripped on read)
        canary_meta = None
        if canary:
            from datalock.canary import inject_canary, save_to_manifest
            pid = resolved_pipeline_id or __import__("uuid").uuid4().hex[:16]
            df, canary_meta = inject_canary(df, pid, canary_n_rows)
            # Save to local manifest for dd.canary_check()
            save_to_manifest(str(output_path), canary_meta)

        plaintext_bytes, row_groups_meta = _df_to_bytes(
            df, parquet_compression=parquet_comp,
            content_type=content_type,
        )

        # Merge canary metadata into header so load() can strip rows
        merged_meta = dict(metadata or {})
        if canary_meta:
            merged_meta["canary"] = canary_meta
            merged_meta["_original_shape"] = original_shape

        header = cls._build_header(
            content_type=content_type,
            label=label,
            df=df,
            parquet_comp=parquet_comp,
            cipher_str=_get_best_cipher(),
            plaintext_size=len(plaintext_bytes),
            metadata=merged_meta if merged_meta else metadata,
            expires_at=expires_at,
            row_groups_meta=row_groups_meta,
        )

        body, salt_kdf = _pack_v2_body(master_key_bytes, header, plaintext_bytes, _get_best_cipher(), compress)
        _write_lgs(output, body, master_key_bytes, salt_kdf)

        elapsed = time.perf_counter() - t0
        packed_size = output.stat().st_size
        logger.info(
            "SecureFile.pack_dataframe | content_type=%s | shape=%s | %.3fs",
            content_type, original_shape, elapsed,
        )
        return {
            "output_path":       str(output),
            "content_type":      content_type,
            "shape":             original_shape,
            "encryption":        _get_best_cipher(),
            "original_size_kb":  round(len(plaintext_bytes) / 1024, 1),
            "packed_size_kb":    round(packed_size / 1024, 1),
            "compression_ratio": round(len(plaintext_bytes) / max(packed_size, 1), 3),
            "elapsed_seconds":   round(elapsed, 3),
        }

    # ------------------------------------------------------------------
    # Escrita — múltiplos DataFrames (multi-frame, v3)
    # ------------------------------------------------------------------

    @classmethod
    def pack_frames(
        cls,
        frames: Dict[str, pd.DataFrame],
        output_path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias
        label: str = "",
        compress: bool = True,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Empacota múltiplos DataFrames em um único arquivo .dlk (multi-frame).

        O payload é um ZIP em memória de Parquets, com um índice JSON listando
        nomes, tamanhos e schemas. O formato é retrocompatível: leitores v1/v2
        que não conhecem multi-frame verão content_type="multi_dataframe" e
        poderão reportar erro descritivo em vez de corrupção silenciosa.

        Args:
            frames:      Dict[nome → DataFrame]. A ordem é preservada.
            output_path: Caminho do .dlk de saída.
            master_key:  Chave mestre (≥ 16 chars).
            label:       Rótulo livre para auditoria.
            compress:    True → Parquet/zstd; False → Parquet/lz4.
            overwrite:   Sobrescreve arquivo existente.

        Returns:
            Dict com metadados: n_frames, nomes, shape total, tamanho, elapsed.

        Exemplo:
            SecureFile.pack_frames(
                {"clientes": df1, "pedidos": df2, "produtos": df3},
                "base.dlk", master_key="chave",
            )
        """
        key = key if key is not None else master_key  # compat: master_key= is deprecated
        output = Path(output_path)
        _check_output(output, overwrite)
        master_key_bytes = _validate_key(key)

        if not isinstance(frames, dict) or not frames:
            raise ValueError("frames deve ser um dict não-vazio de {str: DataFrame}.")

        t0 = time.perf_counter()
        parquet_comp = "zstd" if compress else "lz4"
        zip_bytes, index = _frames_to_zip_bytes(frames, parquet_compression=parquet_comp)

        total_rows = sum(e["rows"] for e in index)
        total_cols = sum(e["cols"] for e in index)

        header = {
            "format":        "lgs",
            "version":       "3.0",
            "content_type":  SecureFile.CONTENT_TYPE_MULTI,
            "label":         label,
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "created_by":    f"datalock/{_logus_version()}",
            "n_frames":      len(frames),
            "frame_names":   [e["name"] for e in index],
            "frame_index":   index,
            "compression":   f"ipc_{parquet_comp}",
            "kdf":           "HKDF-SHA256-v2",
            "encryption":    _get_best_cipher(),
            "integrity":     "HMAC-SHA256",
            "plaintext_size_bytes": len(zip_bytes),
            "metadata":      metadata or {},
        }

        body, salt_kdf = _pack_v2_body(master_key_bytes, header, zip_bytes, _get_best_cipher(), compress, version_byte=VERSION_V3)
        _write_lgs(output, body, master_key_bytes, salt_kdf)

        elapsed = time.perf_counter() - t0
        packed_size = output.stat().st_size
        logger.info(
            "SecureFile.pack_frames | n_frames=%d | names=%s | %.3fs",
            len(frames), list(frames.keys()), elapsed,
        )
        return {
            "output_path":       str(output),
            "content_type":      SecureFile.CONTENT_TYPE_MULTI,
            "n_frames":          len(frames),
            "frame_names":       [e["name"] for e in index],
            "total_rows":        total_rows,
            "total_cols":        total_cols,
            "original_size_kb":  round(len(zip_bytes) / 1024, 1),
            "packed_size_kb":    round(packed_size / 1024, 1),
            "compression_ratio": round(len(zip_bytes) / max(packed_size, 1), 3),
            "elapsed_seconds":   round(elapsed, 3),
        }

    # ------------------------------------------------------------------
    # ACL multi-frame: DEK por nível de acesso
    # ------------------------------------------------------------------

    @classmethod
    def pack_frames_acl(
        cls,
        frames: Dict[str, pd.DataFrame],
        output_path: Union[str, Path],
        key: str,
        frame_access_levels: Optional[Dict[str, str]] = None,
        label: str = "",
        compress: bool = True,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Empacota múltiplos DataFrames com controle de acesso por nível (ACL).

        Cada frame é cifrado com uma DEK derivada do seu nível de acesso.
        Usuários recebem apenas as DEKs dos níveis a que têm direito — frames
        de nível superior são ilegíveis para eles mesmo após decifrar o arquivo.

        Isso resolve o problema de RAG corporativo: um pipeline de busca semântica
        que consulta um arquivo com frames de RH, finanças e dados públicos pode
        ser configurado para que o modelo de linguagem nunca veja os frames
        confidenciais, independente da query do usuário.

        Modelo de derivação de chaves:
          master_key  →  HKDF(salt, "datalock-frame-dek-v1:{level}")  →  DEK_level

        Os níveis recomendados (convenção, não enforçados):
          "public"        → sem restrição
          "internal"      → colaboradores gerais
          "confidential"  → gestores, jurídico
          "restricted"    → diretoria, RH sênior
          "secret"        → dados pessoais LGPD art. 11

        Exemplo — RAG corporativo com múltiplos níveis:
            SecureFile.pack_frames_acl(
                frames={
                    "produtos":  df_produtos,      # público
                    "politicas": df_politicas,      # interno
                    "salarios":  df_salarios,       # restrito
                    "processos": df_processos_rh,   # confidencial
                },
                frame_access_levels={
                    "produtos":  "public",
                    "politicas": "internal",
                    "salarios":  "restricted",
                    "processos": "confidential",
                },
                output_path="base_rag.dlk",
                key="master-key-vault",
            )

            # Usuário comum — vê apenas public + internal
            frames = SecureFile.load_frames_acl(
                "base_rag.dlk", key="master-key-vault",
                allowed_levels=["public", "internal"],
            )

            # Admin — vê tudo
            frames = SecureFile.load_frames_acl(
                "base_rag.dlk", key="master-key-vault",
                allowed_levels=None,  # None = acesso total
            )

        Args:
            frames:              Dict[nome → DataFrame].
            output_path:         Caminho do .dlk de saída.
            key:                 Master key (≥ 16 chars).
            frame_access_levels: Dict[nome_frame → nível]. Padrão "internal".
            label:               Rótulo livre para auditoria.
            compress:            Compressão Parquet interna.
            overwrite:           Sobrescreve arquivo existente.
            metadata:            Metadados extras para o cabeçalho.
            expires_at:          Data de expiração ISO 8601.

        Returns:
            Dict com metadados: n_frames, frame_names, access_levels, etc.
        """
        output = Path(output_path)
        _check_output(output, overwrite)
        master_key_bytes = _validate_key(key)

        if not isinstance(frames, dict) or not frames:
            raise ValueError("frames deve ser um dict não-vazio de {str: DataFrame}.")

        acl = frame_access_levels or {}
        # Valida que todos os frames declarados em acl existem
        unknown = set(acl) - set(frames)
        if unknown:
            raise ValueError(
                f"frame_access_levels contém frames inexistentes: {unknown}. "
                f"Frames disponíveis: {set(frames.keys())}"
            )

        t0 = time.perf_counter()
        parquet_comp = "zstd" if compress else "lz4"
        cipher_str   = _get_best_cipher()

        # Gera salt_kdf aqui para passá-lo a _frames_to_acl_zip_bytes
        # (as DEKs por nível precisam do mesmo salt do arquivo)
        salt_kdf = os.urandom(SALT_KDF_LEN)

        zip_bytes, index = _frames_to_acl_zip_bytes(
            frames, acl, master_key_bytes, salt_kdf, parquet_comp, cipher_str
        )

        total_rows = sum(e["rows"] for e in index)
        access_level_map = {e["name"]: e["access_level"] for e in index}

        header = {
            "format":        "lgs",
            "version":       "3.1",          # 3.1 = multi-frame com ACL
            "content_type":  SecureFile.CONTENT_TYPE_MULTI,
            "acl_enabled":   True,
            "label":         label,
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "created_by":    f"datalock/{_logus_version()}",
            "n_frames":      len(frames),
            "frame_names":   [e["name"] for e in index],
            "frame_index":   index,
            "access_levels": access_level_map,
            "compression":   f"ipc_{parquet_comp}",
            "kdf":           "HKDF-SHA256-v2",
            "encryption":    cipher_str,
            "integrity":     "HMAC-SHA256+MAK",
            "plaintext_size_bytes": len(zip_bytes),
            "metadata":      metadata or {},
            "expires_at":    expires_at or None,
        }

        # Usa salt_kdf pré-gerado para construir o body
        # _pack_v2_body geraria um novo salt — precisamos reutilizar o mesmo
        # que foi usado nas DEKs por frame. Por isso montamos manualmente.
        hek = _derive_hek(master_key_bytes, salt_kdf)
        dek = _derive_dek(master_key_bytes, salt_kdf)

        nonce_payload = os.urandom(NONCE_LEN)
        ct_payload, tag_payload = _encrypt(dek, nonce_payload, zip_bytes, cipher_str)

        header_plain = json.dumps(header, ensure_ascii=False).encode("utf-8")
        nonce_header = os.urandom(NONCE_LEN)
        ct_header, tag_header = _encrypt(hek, nonce_header, header_plain, cipher_str)
        header_ct_with_tag = ct_header + tag_header

        cipher_byte = _CIPHER_STR_TO_BYTE[cipher_str]
        body = (
            MAGIC
            + bytes([VERSION_V3])
            + bytes([cipher_byte])
            + salt_kdf
            + nonce_header
            + struct.pack(HEADER_LEN_FMT, len(header_ct_with_tag))
            + header_ct_with_tag
            + nonce_payload
            + ct_payload
            + tag_payload
        )
        _write_lgs(output, body, master_key_bytes, salt_kdf)

        elapsed = time.perf_counter() - t0
        packed_size = output.stat().st_size
        logger.info(
            "SecureFile.pack_frames_acl | n_frames=%d | levels=%s | %.3fs",
            len(frames), list(set(acl.values())), elapsed,
        )
        return {
            "output_path":       str(output),
            "content_type":      SecureFile.CONTENT_TYPE_MULTI,
            "acl_enabled":       True,
            "n_frames":          len(frames),
            "frame_names":       [e["name"] for e in index],
            "access_levels":     access_level_map,
            "total_rows":        total_rows,
            "original_size_kb":  round(len(zip_bytes) / 1024, 1),
            "packed_size_kb":    round(packed_size / 1024, 1),
            "elapsed_seconds":   round(elapsed, 3),
        }

    @classmethod
    def load_frames_acl(
        cls,
        path: Union[str, Path],
        key: str,
        allowed_levels: Optional[List[str]] = None,
        salt_masking: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Lê um arquivo multi-frame com ACL, retornando apenas os frames autorizados.

        O sistema decifra apenas os frames cujo nível de acesso está em
        `allowed_levels`. Frames de outros níveis são silenciosamente omitidos
        do resultado — o chamador não vê nem o conteúdo nem um erro de
        "frame existe mas é proibido" (security by obscurity intencional para
        evitar enumeração de frames confidenciais).

        Args:
            path:           Caminho para o arquivo .dlk multi-frame com ACL.
            key:            Master key.
            allowed_levels: Lista de níveis autorizados para este usuário.
                            None = acesso total (admin).
                            Exemplo: ["public", "internal"]
            salt_masking:   Aplica mascaramento adicional nos frames lidos.
            verbose:        Log de PII detectada.

        Returns:
            Dict[nome_frame → DataFrame] — apenas frames autorizados.

        Raises:
            PermissionError: Se nenhum frame estiver acessível.
            TypeError:       Se o arquivo não for multi-frame.

        Exemplo:
            # Usuário padrão — vê apenas dados públicos e internos
            frames = SecureFile.load_frames_acl(
                "base_rag.dlk", key=KEY,
                allowed_levels=["public", "internal"],
            )
            # frames NÃO contém "salarios" nem "processos"
            # O modelo de linguagem que usa esses frames nunca vê dados restritos
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        t0 = time.perf_counter()
        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, salt_kdf = cls._decrypt_file(p, master_key_bytes)

        cls._assert_multi_frame(header, p)
        payload_decrypted = _decompress_payload(payload_decrypted, header)

        acl_enabled = header.get("acl_enabled", False)
        if acl_enabled:
            cipher_str = header.get("encryption", "AES256GCM")
            frames = _acl_zip_bytes_to_frames(
                payload_decrypted, master_key_bytes, salt_kdf,
                allowed_levels, cipher_str,
            )
        else:
            # Arquivo multi-frame sem ACL — usa load_frames clássico
            frames = _zip_bytes_to_frames(payload_decrypted)

        if salt_masking is not None:
            frames = {
                name: secure_dataframe(df, salt=salt_masking, verbose=verbose)
                for name, df in frames.items()
            }

        elapsed = time.perf_counter() - t0
        logger.info(
            "SecureFile.load_frames_acl | %s | allowed=%s | loaded=%s | %.3fs",
            p.name, allowed_levels, list(frames.keys()), elapsed,
        )
        return frames

    @classmethod
    def load_frame_acl(
        cls,
        path: Union[str, Path],
        key: str,
        *,
        frame: str,
        allowed_levels: Optional[List[str]] = None,
        salt_masking: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Lê um único frame de arquivo multi-frame com ACL.

        Equivalente a load_frames_acl()[frame], mas sem desserializar os
        frames não solicitados.

        Raises:
            PermissionError: Se o frame solicitado estiver num nível não autorizado.
            KeyError:        Se o frame não existir.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, salt_kdf = cls._decrypt_file(p, master_key_bytes)
        cls._assert_multi_frame(header, p)
        payload_decrypted = _decompress_payload(payload_decrypted, header)

        acl_enabled = header.get("acl_enabled", False)
        cipher_str  = header.get("encryption", "AES256GCM")

        if acl_enabled:
            df = _acl_zip_bytes_to_single_frame(
                payload_decrypted, master_key_bytes, salt_kdf,
                frame, allowed_levels, cipher_str,
            )
        else:
            df = _zip_bytes_to_single_frame(payload_decrypted, frame)

        if salt_masking is not None:
            df = secure_dataframe(df, salt=salt_masking)

        return df

    # ------------------------------------------------------------------
    # Escrita — bytes brutos
    # ------------------------------------------------------------------

    @classmethod
    def pack_bytes(
        cls,
        payload: bytes,
        output_path: Union[str, Path],
        key: str,
        content_type: str = "bytes",
        label: str = "",
        compress: bool = True,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Empacota payload binário arbitrário em .dlk.

        Uso: para arquivos que não são DataFrames — logs, modelos serializados,
        arquivos de configuração, relatórios PDF, etc.

        Args:
            payload:      Bytes a empacotar.
            output_path:  Caminho do .dlk de saída.
            key:          Chave mestre (≥ 16 chars).
            content_type: Descrição do conteúdo (ex: "bytes", "pickle_model").
            label:        Rótulo livre para o header.
            compress:     Compressão zlib antes de cifrar.
            overwrite:    Sobrescreve arquivo existente.

        AVISO DE SEGURANÇA — Pickle e formatos de serialização arbitrários:
            `pickle.loads()` executa código Python arbitrário durante a deserialização.
            Qualquer pessoa que possua a master_key pode criar um payload que execute
            código arbitrário na máquina do receptor ao chamar `pickle.loads(raw)`.

            Boas práticas para payloads pickle:
              - Nunca deserialize payload de fonte não confiável, mesmo com key correta.
              - Prefira formatos seguros: ONNX, SafeTensors, joblib com hash verificado.
              - Restrinja ao máximo quem tem acesso à master_key de arquivos pickle_model.

        Exemplo:
            import pickle
            model_bytes = pickle.dumps(gen._model)
            SecureFile.pack_bytes(
                model_bytes, "modelo_ctgan.dlk",
                key="chave", content_type="pickle_model",
                label="ctgan_clientes_v1"
            )
            raw = SecureFile.load_bytes("modelo_ctgan.dlk", key="chave")
            # AVISO: Só deserialize se confiar completamente na origem do arquivo.
            model = pickle.loads(raw)
        """
        output = Path(output_path)
        _check_output(output, overwrite)
        master_key_bytes = _validate_key(key)

        t0 = time.perf_counter()
        original_size = len(payload)
        cipher_str    = _get_best_cipher()

        # Comprime antes de cifrar — ciphertext AEAD tem distribuição uniforme,
        # qualquer compressor seria ineficaz sobre ele.
        payload_bytes     = zlib.compress(payload, level=6) if compress else payload
        compression_ratio = len(payload_bytes) / max(original_size, 1)

        header = {
            "format":               "lgs",
            "version":              "2.1",
            "content_type":         content_type,
            "label":                label,
            "created_at":           datetime.now(timezone.utc).isoformat(),
            "created_by":           f"datalock/{_logus_version()}",
            "original_size_bytes":  original_size,
            "compression":          "zlib" if compress else "none",
            "kdf":                  "HKDF-SHA256-v2",
            "encryption":           cipher_str,
            "integrity":            "HMAC-SHA256+MAK",
            "plaintext_size_bytes": len(payload_bytes),
            "metadata":             metadata or {},
        }

        # v2: header cifrado com HEK, payload cifrado com DEK,
        # FILE_HMAC calculado com MAK — separação de domínio completa.
        # Corrige a inconsistência anterior onde pack_bytes era o único
        # método que deixava metadados em claro, quebrando a propriedade
        # de auditabilidade sem exposição garantida pelo inspect().
        body, salt_kdf = _pack_v2_body(
            master_key_bytes, header, payload_bytes, cipher_str, compress=False
        )
        _write_lgs(output, body, master_key_bytes, salt_kdf)

        elapsed = time.perf_counter() - t0
        packed_size = output.stat().st_size
        logger.info(
            "SecureFile.pack_bytes | content_type=%s | size=%d bytes | %.3fs",
            content_type, original_size, elapsed,
        )
        return {
            "output_path":       str(output),
            "content_type":      content_type,
            "original_size_kb":  round(original_size / 1024, 1),
            "packed_size_kb":    round(packed_size / 1024, 1),
            "compression_ratio": round(compression_ratio, 3),
            "elapsed_seconds":   round(elapsed, 3),
        }

    # ------------------------------------------------------------------
    # Leitura — DataFrame único
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        key: str,
        salt_masking: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Lê um arquivo `.dlk` e retorna um DataFrame mascarado.

        v1.2.0: aceita columns= e filters= para column pruning e predicate
        pushdown no nível Arrow — só aloca memória para os dados necessários.

        A janela de exposição do dado bruto é minimizada: ele existe na heap
        durante a desserialização e o mascaramento, sem ser exposto como
        arquivo ou variável persistente ao código do chamador.

        Args:
            path:         Caminho para o arquivo `.dlk`.
            key:          Chave mestre (a mesma usada no `pack()`).
            salt_masking: Salt HMAC para mascaramento. Se None e o arquivo
                          não tiver mascaramento pré-aplicado, usa salt aleatório
                          (hashes não reprodutíveis entre execuções).
            random_state: Semente para mockers.
            verbose:      Exibe relatório de colunas PII detectadas.
            columns:      Lista de colunas a retornar. None = todas.
            filters:      Dict de filtros para predicate pushdown. None = sem filtro.
                          Ex: {"uf": "SP"}, {"renda": (">", 10_000)}

        Returns:
            DataFrame com dados mascarados.

        Raises:
            ValueError:  Se o arquivo não for um .dlk válido.
            RuntimeError: Se o HMAC ou a auth_tag falharem (arquivo adulterado).
            TypeError:   Se o arquivo for multi-frame (use load_frames ou load_frame).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        t0 = time.perf_counter()
        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, _salt_kdf = cls._decrypt_file(p, master_key_bytes)

        # Rejeita multi-frame com mensagem clara
        if header.get("content_type") == SecureFile.CONTENT_TYPE_MULTI:
            names = header.get("frame_names", [])
            raise TypeError(
                f"'{p.name}' é um arquivo multi-frame com {len(names)} frames: {names}. "
                f"Use SecureFile.load_frames() ou SecureFile.load_frame(frame='...')."
            )

        payload_decrypted = _decompress_payload(payload_decrypted, header)
        df_raw = _bytes_to_df(
            payload_decrypted,
            columns=columns,
            filters=filters,
            row_groups_meta=header.get("row_groups") or None,
        )

        # Strip canary rows do Nível 1 (arquivo) — transparente ao usuário
        canary_meta_header = header.get("canary") or header.get("metadata", {}).get("canary")
        if canary_meta_header or _CANARY_COL in (df_raw.columns if hasattr(df_raw, "columns") else []):
            from datalock.canary import strip_canary
            df_raw = strip_canary(df_raw)

        # Injeta canary rows do Nível 2 (sessão de leitura — insider threat).
        # As linhas são inseridas no DataFrame entregue ao usuário, de forma que
        # qualquer exportação subsequente (CSV, banco, e-mail) carregue os fingerprints.
        # O arquivo .dlk em disco não é modificado.
        pipeline_id_for_read = (
            canary_meta_header.get("pipeline_id") if canary_meta_header
            else header.get("pipeline_id") or header.get("label") or p.stem
        )
        if pipeline_id_for_read:
            from datalock.canary import inject_canary_on_read
            df_raw, _ = inject_canary_on_read(df_raw, pipeline_id_for_read)

        already_masked = (
            header.get("masking_applied", False)
            or header.get("content_type") == SecureFile.CONTENT_TYPE_MASKED
        )
        if already_masked:
            logger.info(
                "SecureFile.load | %s | mascaramento pré-aplicado | shape=%s",
                p.name, df_raw.shape,
            )
            return df_raw

        df_safe = secure_dataframe(
            df_raw,
            salt=salt_masking,
            random_state=random_state,
            verbose=verbose,
        )
        del df_raw

        elapsed = time.perf_counter() - t0
        logger.info("SecureFile.load | %s | shape=%s | %.3fs", p.name, df_safe.shape, elapsed)
        return df_safe

    @classmethod
    def load_raw(
        cls,
        path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Decifra um .dlk e retorna o DataFrame SEM aplicar mascaramento automático.

        v1.2.0: aceita filters= para predicate pushdown no nível Arrow.
        O payload inteiro é decifrado (AES-GCM exige isso), mas apenas os
        batches e colunas relevantes são materializados em memória.

        AVISO: este método retorna dados potencialmente não mascarados.
        Use apenas em ambientes com controles de acesso adequados.

        Args:
            path:       Caminho para o arquivo `.dlk`.
            key:        Chave mestre.
            columns:    Lista de colunas a retornar. None = todas.
            filters:    Dict de filtros para predicate pushdown. None = sem filtro.

        Returns:
            DataFrame com o conteúdo decifrado, sem mascaramento adicional.

        Raises:
            TypeError: Se o arquivo for multi-frame ou bytes.
        """
        key = key if key is not None else master_key  # compat: master_key= is deprecated
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, _salt_kdf = cls._decrypt_file(p, master_key_bytes)

        content_type = header.get("content_type", "raw_dataframe")
        if content_type == SecureFile.CONTENT_TYPE_BYTES:
            raise TypeError(
                f"Este arquivo contém content_type='{content_type}'. "
                f"Use SecureFile.load_bytes() para payloads binários."
            )
        if content_type == SecureFile.CONTENT_TYPE_MULTI:
            names = header.get("frame_names", [])
            raise TypeError(
                f"Este arquivo é multi-frame ({names}). "
                f"Use SecureFile.load_frames() ou SecureFile.load_frame()."
            )

        payload_decrypted = _decompress_payload(payload_decrypted, header)
        df = _bytes_to_df(
            payload_decrypted,
            columns=columns,
            filters=filters,
            row_groups_meta=header.get("row_groups") or None,
        )

        # Strip canary rows do Nível 1 (arquivo).
        # load_raw() não injeta canary de Nível 2: o chamador usa load_raw()
        # com intenção explícita de receber dados brutos (ex: rekey, testes).
        if "canary" in header or "canary" in header.get("metadata", {}):
            from datalock.canary import strip_canary
            df = strip_canary(df)

        logger.info(
            "SecureFile.load_raw | %s | content_type=%s | shape=%s",
            p.name, content_type, df.shape,
        )
        return df

    @classmethod
    def load_bytes(
        cls,
        path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias
    ) -> bytes:
        """
        Decifra um .dlk e retorna o payload binário bruto.

        Use para arquivos empacotados com `pack_bytes()` — modelos serializados,
        logs, PDFs, ou qualquer payload não-tabular.

        Args:
            path:       Caminho para o arquivo `.dlk`.
            master_key: Chave mestre.

        Returns:
            Bytes decifrados (sem desserialização).
        """
        key = key if key is not None else master_key  # compat: master_key= is deprecated
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, _salt_kdf = cls._decrypt_file(p, master_key_bytes)
        payload_decrypted = _decompress_payload(payload_decrypted, header)
        logger.info("SecureFile.load_bytes | %s | %d bytes", p.name, len(payload_decrypted))
        return payload_decrypted

    # ------------------------------------------------------------------
    # Leitura — múltiplos DataFrames (multi-frame)
    # ------------------------------------------------------------------

    @classmethod
    def load_frames(
        cls,
        path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias
        salt_masking: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Lê um arquivo multi-frame e retorna todos os DataFrames.

        Args:
            path:         Caminho para o arquivo `.dlk` multi-frame.
            master_key:   Chave mestre.
            salt_masking: Se fornecido, aplica mascaramento em cada frame.
            verbose:      Exibe relatório de detecção PII por frame.

        Returns:
            Dict[nome_do_frame → DataFrame].

        Raises:
            TypeError: Se o arquivo não for multi-frame.

        Exemplo:
            frames = SecureFile.load_frames("base.dlk", master_key="chave")
            df_clientes = frames["clientes"]
        """
        key = key if key is not None else master_key  # compat: master_key= is deprecated
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        t0 = time.perf_counter()
        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, _salt_kdf = cls._decrypt_file(p, master_key_bytes)

        cls._assert_multi_frame(header, p)
        payload_decrypted = _decompress_payload(payload_decrypted, header)
        frames = _zip_bytes_to_frames(payload_decrypted)

        if salt_masking is not None:
            frames = {
                name: secure_dataframe(df, salt=salt_masking, verbose=verbose)
                for name, df in frames.items()
            }

        elapsed = time.perf_counter() - t0
        logger.info(
            "SecureFile.load_frames | %s | n=%d | frames=%s | %.3fs",
            p.name, len(frames), list(frames.keys()), elapsed,
        )
        return frames

    @classmethod
    def load_frame(
        cls,
        path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias
        *,
        frame: str,
        salt_masking: Optional[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Lê um único frame de um arquivo multi-frame.

        Mais eficiente que load_frames() quando só um frame é necessário:
        os demais não são desserializados.

        Args:
            path:         Caminho para o arquivo `.dlk` multi-frame.
            master_key:   Chave mestre.
            frame:        Nome do frame a extrair.
            salt_masking: Se fornecido, aplica mascaramento.
            verbose:      Exibe relatório de detecção PII.

        Returns:
            DataFrame do frame solicitado.

        Raises:
            KeyError:  Se o frame não existir.
            TypeError: Se o arquivo não for multi-frame.

        Exemplo:
            df = SecureFile.load_frame("base.dlk", master_key="chave", frame="clientes")
        """
        key = key if key is not None else master_key  # compat: master_key= is deprecated
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        t0 = time.perf_counter()
        master_key_bytes = key.encode("utf-8")
        header, payload_decrypted, _salt_kdf = cls._decrypt_file(p, master_key_bytes)

        cls._assert_multi_frame(header, p)
        payload_decrypted = _decompress_payload(payload_decrypted, header)
        df = _zip_bytes_to_single_frame(payload_decrypted, frame)

        if salt_masking is not None:
            df = secure_dataframe(df, salt=salt_masking, verbose=verbose)

        elapsed = time.perf_counter() - t0
        logger.info(
            "SecureFile.load_frame | %s | frame=%s | shape=%s | %.3fs",
            p.name, frame, df.shape, elapsed,
        )
        return df

    # ------------------------------------------------------------------
    # Verificação de integridade
    # ------------------------------------------------------------------

    @classmethod
    def verify(
        cls,
        path: Union[str, Path],
        key: Optional[str] = None,
        master_key: Optional[str] = None,  # deprecated alias for key=
    ) -> "LGSInfo":
        """
        Verifica a integridade de um arquivo `.dlk` sem expor o conteúdo.

        Para arquivos criptografados (v1/v2/v3): master_key obrigatória.
        Para arquivos abertos (v4 / pack_open): master_key desnecessária.

        Args:
            path:       Caminho para o arquivo `.dlk`.
            master_key: Chave mestre (obrigatória para arquivos cifrados;
                        None para arquivos v4/pack_open).

        Returns:
            Tupla (válido: bool, info: dict).

        Exemplo:
            # Arquivo cifrado
            ok, info = SecureFile.verify("clientes.dlk", master_key="chave")

            # Arquivo aberto (sem key)
            ok, info = SecureFile.verify("dados_dev.dlk")
        """
        # Backward compat: master_key= is deprecated alias for key=
        if master_key is not None and key is None:
            key = master_key
        p = Path(path)
        if not p.exists():
            return LGSInfo(False, {"error": f"Arquivo não encontrado: {path}"})

        try:
            raw = p.read_bytes()

            if not raw.startswith(MAGIC):
                return LGSInfo(False, {"error": "Magic bytes inválidos — não é um arquivo .dlk"})

            file_version = raw[len(MAGIC)]
            SUPPORTED_VERSIONS = {VERSION_V1, VERSION_V2, VERSION_V3, VERSION_V4}
            if file_version not in SUPPORTED_VERSIONS:
                return LGSInfo(False, {"error": f"Versão não suportada: {file_version:#04x}"})

            body, file_hmac = raw[:-FILE_HMAC_LEN], raw[-FILE_HMAC_LEN:]

            # v4: sem criptografia — usa HMAC público para integridade
            if file_version == VERSION_V4:
                hmac_ok = _verify_file_hmac(_NO_KEY_HMAC_KEY, body, file_hmac)
                if not hmac_ok:
                    return LGSInfo(False, {"error": "HMAC inválido — arquivo corrompido"})
                offset = len(MAGIC) + 1
                header_len = struct.unpack(HEADER_LEN_FMT, body[offset:offset + 4])[0]
                offset += 4
                if header_len > MAX_HEADER_LEN:
                    return LGSInfo(False, {
                        "error": f"Header excede tamanho máximo ({header_len} > {MAX_HEADER_LEN} bytes)"
                    })
                header = json.loads(body[offset:offset + header_len].decode("utf-8"))
                return LGSInfo(True, {
                    "valid":          True,
                    "format_version": header.get("version", "4.0"),
                    "content_type":   header.get("content_type", SecureFile.CONTENT_TYPE_ANON),
                    "encrypted":      False,
                    "label":          header.get("label", ""),
                    "created_at":     header.get("created_at"),
                    "created_by":     header.get("created_by"),
                    "shape":          header.get("shape"),
                    "masking_applied": header.get("masking_applied", False),
                })

            # v1/v2/v3: usa HMAC com master_key
            if key is None:
                return LGSInfo(False, {"error": "master_key obrigatória para arquivos criptografados"})
            master_key_bytes = key.encode("utf-8")

            # Para v2/v3: extrair salt_kdf do corpo para verificar HMAC com MAK
            # (separação de domínio completa). Para v1: salt_kdf não disponível
            # antes do HMAC, usa master_key diretamente (retrocompat).
            salt_kdf_for_hmac: Optional[bytes] = None
            if file_version in (VERSION_V2, VERSION_V3):
                # salt_kdf está em posição fixa após MAGIC(5) + VERSION(1) + CIPHER(1)
                salt_kdf_for_hmac = raw[7:7 + SALT_KDF_LEN]

            if not _verify_file_hmac(master_key_bytes, body, file_hmac, salt_kdf_for_hmac):
                return LGSInfo(False, {"error": "HMAC inválido — arquivo foi modificado após a criação"})

            # Decifra APENAS o cabeçalho — payload nunca tocado
            if file_version in (VERSION_V2, VERSION_V3):
                header, _ = _parse_v2_header_only(raw, master_key_bytes)
            else:
                # v1: header em claro, sem custo de decifração
                offset_v1 = len(MAGIC) + 1
                header_len_v1 = struct.unpack(HEADER_LEN_FMT, raw[offset_v1:offset_v1 + 4])[0]
                offset_v1 += 4
                header = json.loads(raw[offset_v1:offset_v1 + header_len_v1].decode("utf-8"))

            info: Dict[str, Any] = {
                "valid":             True,
                "format_version":    header.get("version", "1.0"),
                "version":           header.get("version", "1.0"),
                "content_type":      header.get("content_type", "raw_dataframe"),
                "label":             header.get("label", ""),
                "created_at":        header.get("created_at"),
                "created_by":        header.get("created_by"),
                "original_filename": header.get("original_filename"),
                "masking_applied":   header.get("masking_applied", False),
                "compression":       header.get("compression"),
                "encryption":        header.get("encryption"),
                "metadata":          header.get("metadata", {}),
            }
            # Campos específicos por content_type
            if header.get("content_type") == SecureFile.CONTENT_TYPE_MULTI:
                info["n_frames"]    = header.get("n_frames", 0)
                info["frame_names"] = header.get("frame_names", [])
                info["frame_index"] = header.get("frame_index", [])
            else:
                info["shape"]         = header.get("shape")
                info["columns"]       = header.get("columns")
                info["column_stats"]  = header.get("column_stats", {})
                info["schema"]        = header.get("schema", {})
                info["kdf"]           = header.get("kdf")
                info["integrity"]     = header.get("integrity")
            return LGSInfo(True, info)

        except Exception as exc:
            return LGSInfo(False, {"error": f"Erro ao verificar: {type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------------
    # Helpers estáticos internos
    # ------------------------------------------------------------------

    @classmethod
    def _build_header(
        cls,
        content_type: str,
        label: str,
        df: pd.DataFrame,
        parquet_comp: str,
        cipher_str: str,
        plaintext_size: int,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[str] = None,
        row_groups_meta: Optional[List[Dict]] = None,
    ) -> Dict:
        import numpy as _np
        # Estatísticas de coluna acessíveis via dd.inspect() sem decifrar o payload.
        # Para raw_dataframe (dados não mascarados), apenas o dtype é exposto:
        #   - min/max podem revelar distribuições de dados sensíveis (faixa de renda, etc.)
        #   - n_unique pode revelar cardinalidade de campos PII (ex: 1 único = coluna constante)
        #   - n_nulls pode revelar completude de campos sensíveis (ex: 0 nulos = CPF preenchido
        #     para todos os registros, confirmando presença de dado pessoal identificável)
        # Para masked_dataframe, todas as estatísticas são seguras: os valores são hashes HMAC.
        expose_stats = (content_type != SecureFile.CONTENT_TYPE_RAW)
        col_stats = {}
        for col in df.columns:
            try:
                s = df[col]
                stat: Dict[str, Any] = {"dtype": str(df[col].dtype)}
                if expose_stats:
                    stat["n_nulls"]  = int(s.isna().sum())
                    stat["n_unique"] = int(s.nunique())
                    if hasattr(s, "dtype") and str(s.dtype) in (
                        "float64", "float32", "int64", "int32", "int16", "int8"
                    ):
                        stat["min"] = float(s.min()) if not s.empty else None
                        stat["max"] = float(s.max()) if not s.empty else None
                col_stats[col] = stat
            except Exception:
                col_stats[col] = {"dtype": str(df[col].dtype) if col in df.columns else "unknown"}

        return {
            "format":               "lgs",
            "version":              "2.1",
            "format_version":       "3.0",
            "content_type":         content_type,
            "label":                label,
            "created_at":           datetime.now(timezone.utc).isoformat(),
            "created_by":           f"datalock/{_logus_version()}",
            "shape":                list(df.shape),
            "schema":               {c: str(t) for c, t in df.dtypes.items()},
            "column_stats":         col_stats,
            "columns":              list(df.columns),
            "masking_applied":      content_type == SecureFile.CONTENT_TYPE_MASKED,
            "compression":          f"ipc_{parquet_comp}",
            "kdf":                  "HKDF-SHA256-v2",
            "encryption":           cipher_str,
            "integrity":            "HMAC-SHA256+MAK",
            "plaintext_size_bytes": plaintext_size,
            "metadata":             metadata or {},
            "expires_at":           expires_at or None,
            "row_groups":           row_groups_meta or [],
        }

    @classmethod
    def _assert_multi_frame(cls, header: Dict, p: Path) -> None:
        """Lança TypeError se o arquivo não for multi-frame."""
        ct = header.get("content_type")
        if ct != SecureFile.CONTENT_TYPE_MULTI:
            raise TypeError(
                f"'{p.name}' não é um arquivo multi-frame (content_type='{ct}'). "
                f"Use SecureFile.load() para DataFrames únicos."
            )

    @classmethod
    def _decrypt_file(
        cls, path: Path, master_key_bytes: bytes
    ) -> Tuple[Dict, bytes, Optional[bytes]]:
        """
        Lê, verifica e decifra um arquivo .dlk.

        Suporta:
          v1 (0x01): header em claro, AES-256-GCM fixo.
          v2 (0x02): header cifrado, cipher negociado, DEK/HEK separados.
          v3 (0x03): igual ao v2, mas content_type=multi_dataframe.

        Retorna (header_dict, payload_decrypted_bytes, salt_kdf_or_None).

        O salt_kdf é retornado diretamente do raw já lido, eliminando a
        necessidade de releitura do arquivo nos paths ACL (fix TOCTOU).
        Para v1, salt_kdf é None (não disponível no offset fixo).
        Lança RuntimeError em qualquer falha de integridade.
        """
        raw = path.read_bytes()

        if not raw.startswith(MAGIC):
            raise ValueError(f"'{path.name}' não é um arquivo .dlk válido.")

        file_version = raw[len(MAGIC)]
        if file_version not in (VERSION_V1, VERSION_V2, VERSION_V3, VERSION_V4):
            raise ValueError(
                f"Versão {file_version:#04x} não suportada. "
                f"Atualize datalock para ler este arquivo."
            )

        if file_version == VERSION_V4:
            raise ValueError(
                f"'{path.name}' não está criptografado (versão 4). "
                f"Use SecureFile.load_open() para arquivos sem criptografia."
            )

        # Verifica HMAC antes de qualquer decifração (Verify-then-Decrypt).
        # Para v2/v3: usa MAK derivada do salt_kdf em posição fixa no cabeçalho,
        # garantindo separação de domínio completa entre DEK, HEK e MAK.
        # Para v1: salt_kdf vem depois do header em claro, usa master_key diretamente.
        salt_kdf_for_hmac: Optional[bytes] = None
        if file_version in (VERSION_V2, VERSION_V3):
            salt_kdf_for_hmac = raw[7:7 + SALT_KDF_LEN]  # MAGIC(5)+VERSION(1)+CIPHER(1)

        body, file_hmac = raw[:-FILE_HMAC_LEN], raw[-FILE_HMAC_LEN:]
        if not _verify_file_hmac(master_key_bytes, body, file_hmac, salt_kdf_for_hmac):
            raise RuntimeError(
                "Falha de integridade (HMAC inválido). "
                "Arquivo modificado após criação ou master_key incorreta."
            )

        if file_version == VERSION_V1:
            header, payload = cls._parse_v1(raw, master_key_bytes)
            return header, payload, None  # salt_kdf não disponível em offset fixo no v1

        # v2 e v3 têm estrutura idêntica
        header, payload = cls._parse_v2(raw, master_key_bytes)
        return header, payload, salt_kdf_for_hmac  # reutiliza salt já extraído

    @staticmethod
    def _parse_v1(raw: bytes, master_key_bytes: bytes) -> Tuple[Dict, bytes]:
        """Parseia formato v1 (header em plaintext)."""
        offset = len(MAGIC) + 1  # skip MAGIC + VERSION
        header_len = struct.unpack(HEADER_LEN_FMT, raw[offset:offset + 4])[0]
        offset += 4
        if header_len > MAX_HEADER_LEN:
            raise ValueError(
                f"Header JSON excede o tamanho máximo permitido "
                f"({header_len} bytes > {MAX_HEADER_LEN} bytes). "
                "Arquivo possivelmente malformado ou corrompido."
            )
        header = json.loads(raw[offset:offset + header_len].decode("utf-8"))
        offset += header_len
        salt_kdf  = raw[offset:offset + SALT_KDF_LEN];  offset += SALT_KDF_LEN
        nonce     = raw[offset:offset + NONCE_LEN];      offset += NONCE_LEN
        auth_tag_start = len(raw) - FILE_HMAC_LEN - AUTH_TAG_LEN
        ciphertext = raw[offset:auth_tag_start]
        auth_tag   = raw[auth_tag_start:auth_tag_start + AUTH_TAG_LEN]
        dek = _derive_dek(master_key_bytes, salt_kdf)
        payload = _decrypt(dek, nonce, ciphertext, auth_tag, "AES256GCM")
        return header, payload

    @staticmethod
    def _parse_v2(raw: bytes, master_key_bytes: bytes) -> Tuple[Dict, bytes]:
        """Parseia formato v2/v3 (header cifrado, cipher negociado)."""
        offset = len(MAGIC) + 1  # skip MAGIC + VERSION
        cipher_byte = raw[offset]; offset += 1
        cipher_str  = _CIPHER_BYTE_TO_STR.get(cipher_byte, "AES256GCM")

        salt_kdf      = raw[offset:offset + SALT_KDF_LEN];  offset += SALT_KDF_LEN
        nonce_header  = raw[offset:offset + NONCE_LEN];     offset += NONCE_LEN
        header_ct_len = struct.unpack(HEADER_LEN_FMT, raw[offset:offset + 4])[0]
        offset += 4
        if header_ct_len > MAX_HEADER_LEN:
            raise ValueError(
                f"Header cifrado excede o tamanho máximo permitido "
                f"({header_ct_len} bytes > {MAX_HEADER_LEN} bytes). "
                "Arquivo possivelmente malformado ou corrompido."
            )
        header_ct_with_tag = raw[offset:offset + header_ct_len]; offset += header_ct_len
        header_ct  = header_ct_with_tag[:-AUTH_TAG_LEN]
        header_tag = header_ct_with_tag[-AUTH_TAG_LEN:]

        nonce_payload  = raw[offset:offset + NONCE_LEN]; offset += NONCE_LEN
        auth_tag_start = len(raw) - FILE_HMAC_LEN - AUTH_TAG_LEN
        ciphertext = raw[offset:auth_tag_start]
        auth_tag   = raw[auth_tag_start:auth_tag_start + AUTH_TAG_LEN]

        dek = _derive_dek(master_key_bytes, salt_kdf)
        hek = _derive_hek(master_key_bytes, salt_kdf)

        header_plain = _decrypt(hek, nonce_header, header_ct, header_tag, cipher_str)
        header = json.loads(header_plain.decode("utf-8"))
        # Check expiration before decrypting payload (LGPD Art. 16)
        _check_expiry(header)
        payload = _decrypt(dek, nonce_payload, ciphertext, auth_tag, cipher_str)

        return header, payload


# ---------------------------------------------------------------------------
# Helpers de leitura de arquivo de origem
# ---------------------------------------------------------------------------

def _read_source(path: Path) -> pd.DataFrame:
    """Lê um arquivo de dados de origem para empacotamento."""
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
            f"Formato '{suffix}' não suportado para empacotamento. "
            f"Formatos aceitos: {', '.join(readers)}"
        )
    return reader(path)


def _decompress_payload(payload: bytes, header: Dict) -> bytes:
    """
    Aplica descompressão ao payload conforme indicado pelo header.

    A compressão nos formatos ipc_* e parquet_* é interna ao serializer
    (Arrow IPC ou Parquet) — não há camada de compressão externa além do
    serializer. Esta função só atua em arquivos legados v1.0-v1.2 que
    usavam zlib externo antes da serialização.

    Valores do campo "compression" por versão:
      "zlib"         → compressão zlib externa  (v1.0-v1.2, legado)
      "parquet_zstd" → Parquet/zstd interno     (v1.3, retrocompat)
      "parquet_lz4"  → Parquet/lz4 interno      (v1.3, retrocompat)
      "ipc_zstd"     → Arrow IPC/zstd interno   (v1.4+, atual)
      "ipc_lz4"      → Arrow IPC/lz4 interno    (v1.4+, atual)
      "ipc_none"     → Arrow IPC sem compressão (v1.4+, atual)
      "none"         → sem compressão           (v4 sem key)
    """
    compression = header.get("compression", "none")
    if compression == "zlib":
        # Legado v1.0-v1.2: zlib externo antes do Arrow IPC sem marker
        return zlib.decompress(payload)
    # ipc_*, parquet_*, none: compressão interna ao serializer — sem ação extra
    return payload


def _logus_version() -> str:
    """Retorna a versão do datalock sem importação circular."""
    try:
        import datalock
        return datalock.__version__
    except Exception:
        return "unknown"