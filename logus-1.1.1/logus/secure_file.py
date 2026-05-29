"""
secure_file.py
==============
SecureFile — Formato de Arquivo Criptografado `.dlk`.

O que é o .lgs?
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
| **logus**               | ✅          | ✅               | ✅             | ✅             |

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
Formato v2 (single-frame — logus >= 1.3.2):
  [5  bytes]  MAGIC          = b"LOGUS"
  [1  byte ]  VERSION        = 0x02
  [1  byte ]  CIPHER         = 0x01 (AES-256-GCM) | 0x02 (ChaCha20-Poly1305)
  [32 bytes]  SALT_KDF       — salt HKDF único por arquivo
  [12 bytes]  NONCE_HEADER   — nonce para cifrar o header JSON
  [4  bytes]  HEADER_CT_LEN  — comprimento do header cifrado + auth_tag
  [N+16 bytes] HEADER_CT+TAG — header JSON cifrado com HEK + auth_tag GCM
  [12 bytes]  NONCE_PAYLOAD  — nonce para cifrar o payload
  [M+16 bytes] PAYLOAD_CT+TAG — dados Parquet/zstd cifrados + auth_tag GCM
  [32 bytes]  FILE_HMAC      — HMAC-SHA256 sobre tudo acima

Formato v3 (multi-frame — logus >= 1.5.0):
  Idêntico ao v2, mas content_type = "multi_dataframe" e o payload
  é um ZIP em memória contendo um Parquet por frame + "index.json"
  com os nomes e tamanhos. Retrocompatível com leitores v1/v2.

Formato v1 (legado — logus < 1.3.2, retrocompatível):
  [5  bytes]  MAGIC     = b"LOGUS"
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes do formato
# ---------------------------------------------------------------------------

MAGIC   = b"LOGUS"          # 5 bytes — identificador do formato
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

def _df_to_bytes(df, parquet_compression: str = "zstd") -> bytes:
    """
    Serializa DataFrame para bytes via Parquet em memória.

    Usa Polars nativamente quando possível — 2-2.5x mais rápido que
    pa.Table.from_pandas() + pq.write_table() e produz arquivos ~20% menores
    (Polars aplica estatísticas de coluna melhores antes da compressão).

    Aceita pd.DataFrame e pl.DataFrame transparentemente.

    Formato do payload: b'PQ1\\x00' + bytes_parquet
    Magic marker garante retrocompatibilidade com _bytes_to_df para arquivos
    criados com Arrow IPC (logus < v1.3).

    Args:
        df:                  pd.DataFrame ou pl.DataFrame.
        parquet_compression: 'zstd' (padrão, menor tamanho),
                             'lz4'  (mais rápido, +20% tamanho),
                             'snappy' (+65% tamanho),
                             'none'  (sem compressão, máxima velocidade).
    """
    import polars as _pl

    buf = io.BytesIO()

    # Polars nativo — caminho mais rápido e menor footprint
    if isinstance(df, _pl.DataFrame):
        comp = parquet_compression if parquet_compression != "none" else "uncompressed"
        df.write_parquet(buf, compression=comp)
        return b"PQ1\x00" + buf.getvalue()

    # pd.DataFrame → converte para Polars (evita pa.Table.from_pandas overhead)
    try:
        df_pl = _pl.from_pandas(df)
        comp = parquet_compression if parquet_compression != "none" else "uncompressed"
        df_pl.write_parquet(buf, compression=comp)
        return b"PQ1\x00" + buf.getvalue()
    except Exception:
        # Fallback pyarrow para DataFrames com tipos não suportados por Polars
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        compression = parquet_compression if parquet_compression != "none" else None
        pq.write_table(table, buf, compression=compression, write_statistics=False)
        return b"PQ1\x00" + buf.getvalue()


def _bytes_to_df(data: bytes) -> pd.DataFrame:
    """
    Desserializa bytes para DataFrame.

    Usa Polars nativamente para leitura Parquet — 2x mais rápido que
    pq.read_table().to_pandas(). Converte para pd.DataFrame no final
    para manter compatibilidade com o restante do SecureFile (que opera
    em pandas internamente).

    Detecta automaticamente o formato pelo magic marker:
      b'PQ1\\x00' → Parquet (logus v1.3+, lê via Polars)
      outros      → Arrow IPC (retrocompat com logus v1.0-v1.2)
    """
    if data[:4] == b"PQ1\x00":
        try:
            import polars as _pl
            return _pl.read_parquet(io.BytesIO(data[4:])).to_pandas()
        except Exception:
            # Fallback pyarrow se Polars não conseguir ler o arquivo
            import pyarrow.parquet as pq
            return pq.read_table(io.BytesIO(data[4:])).to_pandas()
    else:
        # Legado: Arrow IPC (arquivos criados antes da v1.3)
        import pyarrow as pa
        reader = pa.ipc.open_stream(io.BytesIO(data))
        return reader.read_pandas()


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
      clientes.parquet     → Parquet de df_clientes
      pedidos.parquet      → Parquet de df_pedidos
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
            parquet_bytes = _df_to_bytes(df, parquet_compression=parquet_compression)
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
# Primitivos criptográficos
# ---------------------------------------------------------------------------

def _derive_dek(master_key: bytes, salt_kdf: bytes) -> bytes:
    """
    Deriva a Data Encryption Key (DEK) via HKDF-SHA256 (RFC 5869).

    DEK (info=b"logus-dek-v2"): cifra o payload de dados.
    HEK (info=b"logus-hek-v2"): cifra o header JSON.
    info= diferentes garantem DEK ≠ HEK mesmo com o mesmo salt e master_key.
    """
    _require_cryptography()
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_kdf,
        info=b"logus-dek-v2",
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
        info=b"logus-hek-v2",
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


def _compute_file_hmac(master_key: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 sobre os dados do arquivo (header + payload cifrado)."""
    return _hmac_module.new(master_key, data, hashlib.sha256).digest()


def _verify_file_hmac(master_key: bytes, data: bytes, expected: bytes) -> bool:
    """Verifica HMAC em tempo constante (resistente a timing attack)."""
    return _hmac_module.compare_digest(_compute_file_hmac(master_key, data), expected)


# ---------------------------------------------------------------------------
# Versão 4 — .dlk sem criptografia (plaintext + HMAC de integridade opcional)
# ---------------------------------------------------------------------------
# Casos de uso: ambientes internos onde confidencialidade não é necessária mas
# a integridade e o formato padronizado (Parquet + metadados LGPD) ainda são úteis.
# A ausência de key é declarada explicitamente no header — não há "segurança por
# obscuridade". Arquivos sem key não devem conter dados pessoais não mascarados.
#
# Estrutura v4 (sem criptografia):
#   [5  bytes]  MAGIC     = b"LOGUS"
#   [1  byte ]  VERSION   = 0x04
#   [4  bytes]  HEADER_LEN — tamanho do header JSON em claro
#   [N  bytes]  HEADER JSON em claro (inclui "encrypted": false)
#   [M  bytes]  PAYLOAD em claro (Parquet/zstd)
#   [32 bytes]  FILE_HMAC — HMAC com chave derivada do conteúdo (ou zeros se sem HMAC)
#
# Nota: sem key, o FILE_HMAC usa chave pública fixa ("logus-no-key-v4") que
# detecta corrupção acidental mas NÃO protege contra adulteração intencional.
# ---------------------------------------------------------------------------

VERSION_V4 = 0x04
_NO_KEY_HMAC_KEY = b"logus-no-key-integrity-v4"


# ---------------------------------------------------------------------------
# Helpers internos de escrita/leitura da estrutura v2/v4
# ---------------------------------------------------------------------------

def _check_expiry(header: Dict) -> None:
    """Lança ExpiredFileError se o arquivo passou do prazo definido em expires_at."""
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
                f"Conforme a política de retenção LGPD (Art. 16), "                f"dados após o prazo devem ser eliminados."
            )
    except ExpiredFileError:
        raise
    except Exception:
        pass  # Formato de data inválido — ignora silenciosamente


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
) -> bytes:
    """
    Constrói o corpo de um arquivo .dlk v2/v3 (sem o FILE_HMAC final).

    version_byte: VERSION_V2 (single-frame) ou VERSION_V3 (multi-frame).
    Retorna o body SEM o FILE_HMAC — o chamador adiciona o HMAC.
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

    return (
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
    output: Path, body: bytes, master_key_bytes: Optional[bytes]
) -> None:
    """Grava body + HMAC em disco de forma atômica (write → rename)."""
    hmac_key = master_key_bytes if master_key_bytes is not None else _NO_KEY_HMAC_KEY
    file_hmac = _compute_file_hmac(hmac_key, body)
    tmp = output.with_suffix == ".dlk".tmp")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(body + file_hmac)
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _validate_key(master_key: str) -> bytes:
    """Valida comprimento mínimo e retorna bytes UTF-8."""
    encoded = master_key.encode("utf-8")
    if len(encoded) < 16:
        raise ValueError(
            "master_key muito curta (mínimo 16 bytes). "
            "Use uma chave de alta entropia para proteger dados pessoais."
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
        plaintext_bytes = _df_to_bytes(df, parquet_compression=parquet_comp)

        header = {
            "format":               "lgs",
            "version":              "4.0",
            "content_type":         content_type,
            "encrypted":            False,
            "label":                label,
            "created_at":           datetime.now(timezone.utc).isoformat(),
            "created_by":           f"logus/{_logus_version()}",
            "shape":                list(df.shape),
            "schema":               {c: str(t) for c, t in df.dtypes.items()},
            "masking_applied":      anonymize,
            "compression":          f"parquet_{parquet_comp}",
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
        plaintext_bytes = _df_to_bytes(df, parquet_compression=parquet_comp)

        header = cls._build_header(
            content_type=content_type,
            label=label,
            df=df,
            parquet_comp=parquet_comp,
            cipher_str=_get_best_cipher(),
            plaintext_size=len(plaintext_bytes),
            metadata=metadata,
            expires_at=expires_at,
        )

        body = _pack_v2_body(master_key_bytes, header, plaintext_bytes, _get_best_cipher(), compress)
        _write_lgs(output, body, master_key_bytes)

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
        plaintext_bytes = _df_to_bytes(df, parquet_compression=parquet_comp)

        header = cls._build_header(
            content_type=content_type,
            label=label,
            df=df,
            parquet_comp=parquet_comp,
            cipher_str=_get_best_cipher(),
            plaintext_size=len(plaintext_bytes),
            metadata=metadata,
            expires_at=expires_at,
        )

        body = _pack_v2_body(master_key_bytes, header, plaintext_bytes, _get_best_cipher(), compress)
        _write_lgs(output, body, master_key_bytes)

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
            "created_by":    f"logus/{_logus_version()}",
            "n_frames":      len(frames),
            "frame_names":   [e["name"] for e in index],
            "frame_index":   index,
            "compression":   f"parquet_{parquet_comp}",
            "kdf":           "HKDF-SHA256-v2",
            "encryption":    _get_best_cipher(),
            "integrity":     "HMAC-SHA256",
            "plaintext_size_bytes": len(zip_bytes),
            "metadata":      metadata or {},
        }

        body = _pack_v2_body(master_key_bytes, header, zip_bytes, _get_best_cipher(), compress, version_byte=VERSION_V3)
        _write_lgs(output, body, master_key_bytes)

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
        Empacota payload binário arbitrário em .lgs.

        Uso: para arquivos que não são DataFrames — logs, modelos serializados
        (pickle), arquivos de configuração, relatórios PDF, etc.

        Args:
            payload:      Bytes a empacotar.
            output_path:  Caminho do .dlk de saída.
            master_key:   Chave mestre (≥ 16 chars).
            content_type: Descrição do conteúdo (ex: "bytes", "pickle_model").
            label:        Rótulo livre para o header.
            compress:     Compressão zlib antes de cifrar.
            overwrite:    Sobrescreve arquivo existente.

        Exemplo:
            import pickle
            model_bytes = pickle.dumps(gen._model)
            SecureFile.pack_bytes(
                model_bytes, "modelo_ctgan.dlk",
                master_key="chave", content_type="pickle_model",
                label="ctgan_clientes_v1"
            )
            raw = SecureFile.load_bytes("modelo_ctgan.dlk", master_key="chave")
            model = pickle.loads(raw)
        """
        output = Path(output_path)
        _check_output(output, overwrite)
        master_key_bytes = _validate_key(key)

        t0 = time.perf_counter()
        original_size = len(payload)

        payload_bytes = zlib.compress(payload, level=6) if compress else payload
        compression_ratio = len(payload_bytes) / max(original_size, 1)

        salt_kdf = os.urandom(SALT_KDF_LEN)
        nonce    = os.urandom(NONCE_LEN)
        dek      = _derive_dek(master_key_bytes, salt_kdf)
        ciphertext, auth_tag = _aes_gcm_encrypt(dek, nonce, payload_bytes)

        header = {
            "format":        "lgs",
            "version":       "1.0",
            "content_type":  content_type,
            "label":         label,
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "created_by":    f"logus/{_logus_version()}",
            "original_size_bytes": original_size,
            "compression":   "zlib" if compress else "none",
            "kdf":           "HKDF-SHA256",
            "encryption":    "AES-256-GCM",
            "integrity":     "HMAC-SHA256",
            "plaintext_size_bytes": len(payload_bytes),
            "metadata":      metadata or {},
        }
        header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
        # pack_bytes usa v1 (header em claro) por simplicidade — payload já está
        # protegido por AES-GCM + HMAC. A promoção para v2 pode ser feita futuramente.
        body = (
            MAGIC + bytes([VERSION_V1])
            + struct.pack(HEADER_LEN_FMT, len(header_bytes))
            + header_bytes + salt_kdf + nonce + ciphertext + auth_tag
        )
        _write_lgs(output, body, master_key_bytes)

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
    ) -> pd.DataFrame:
        """
        Lê um arquivo `.dlk` e retorna um DataFrame mascarado.

        A janela de exposição do dado bruto é minimizada: ele existe na heap
        durante a desserialização e o mascaramento, sem ser exposto como
        arquivo ou variável persistente ao código do chamador.

        Args:
            path:         Caminho para o arquivo `.dlk`.
            master_key:   Chave mestre (a mesma usada no `pack()`).
            salt_masking: Salt HMAC para mascaramento. Se None e o arquivo
                          não tiver mascaramento pré-aplicado, usa salt aleatório
                          (hashes não reprodutíveis entre execuções).
            random_state: Semente para mockers.
            verbose:      Exibe relatório de colunas PII detectadas.

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
        header, payload_decrypted = cls._decrypt_file(p, master_key_bytes)

        # Rejeita multi-frame com mensagem clara
        if header.get("content_type") == SecureFile.CONTENT_TYPE_MULTI:
            names = header.get("frame_names", [])
            raise TypeError(
                f"'{p.name}' é um arquivo multi-frame com {len(names)} frames: {names}. "
                f"Use SecureFile.load_frames() ou SecureFile.load_frame(frame='...')."
            )

        payload_decrypted = _decompress_payload(payload_decrypted, header)
        df_raw = _bytes_to_df(payload_decrypted)

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
    ) -> pd.DataFrame:
        """
        Decifra um .dlk e retorna o DataFrame SEM aplicar mascaramento automático.

        AVISO: este método retorna dados potencialmente não mascarados.
        Use apenas em ambientes com controles de acesso adequados.

        Args:
            path:       Caminho para o arquivo `.dlk`.
            master_key: Chave mestre.

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
        header, payload_decrypted = cls._decrypt_file(p, master_key_bytes)

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
        df = _bytes_to_df(payload_decrypted)
        if columns:
            existing = [c for c in columns if c in df.columns]
            if existing:
                df = df[existing]
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
        header, payload_decrypted = cls._decrypt_file(p, master_key_bytes)
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
        header, payload_decrypted = cls._decrypt_file(p, master_key_bytes)

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
        header, payload_decrypted = cls._decrypt_file(p, master_key_bytes)

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

            if not _verify_file_hmac(master_key_bytes, body, file_hmac):
                return LGSInfo(False, {"error": "HMAC inválido — arquivo foi modificado após a criação"})

            header, _ = cls._decrypt_file(p, master_key_bytes)

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
    ) -> Dict:
        import numpy as _np
        # Column statistics (disponíveis sem descriptografar via lg.inspect())
        col_stats = {}
        for col in df.columns:
            try:
                s = df[col]
                stat = {
                    "dtype":      str(df[col].dtype),
                    "n_nulls":    int(s.isna().sum()),
                    "n_unique":   int(s.nunique()),
                }
                if hasattr(s, "dtype") and str(s.dtype) in ("float64","float32","int64","int32","int16","int8"):
                    stat["min"]  = float(s.min()) if not s.empty else None
                    stat["max"]  = float(s.max()) if not s.empty else None
                col_stats[col] = stat
            except Exception:
                col_stats[col] = {"dtype": str(df[col].dtype) if col in df.columns else "unknown"}

        return {
            "format":               "lgs",
            "version":              "2.1",
            "content_type":         content_type,
            "label":                label,
            "created_at":           datetime.now(timezone.utc).isoformat(),
            "created_by":           f"logus/{_logus_version()}",
            "shape":                list(df.shape),
            "schema":               {c: str(t) for c, t in df.dtypes.items()},
            "column_stats":         col_stats,
            "columns":              list(df.columns),
            "masking_applied":      content_type == SecureFile.CONTENT_TYPE_MASKED,
            "compression":          f"parquet_{parquet_comp}",
            "kdf":                  "HKDF-SHA256-v2",
            "encryption":           cipher_str,
            "integrity":            "HMAC-SHA256",
            "plaintext_size_bytes": plaintext_size,
            "metadata":             metadata or {},
            "expires_at":           expires_at or None,
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
    ) -> Tuple[Dict, bytes]:
        """
        Lê, verifica e decifra um arquivo .lgs.

        Suporta:
          v1 (0x01): header em claro, AES-256-GCM fixo.
          v2 (0x02): header cifrado, cipher negociado, DEK/HEK separados.
          v3 (0x03): igual ao v2, mas content_type=multi_dataframe.

        Retorna (header_dict, payload_decrypted_bytes).
        Lança RuntimeError em qualquer falha de integridade.
        """
        raw = path.read_bytes()

        if not raw.startswith(MAGIC):
            raise ValueError(f"'{path.name}' não é um arquivo .dlk válido.")

        file_version = raw[len(MAGIC)]
        if file_version not in (VERSION_V1, VERSION_V2, VERSION_V3, VERSION_V4):
            raise ValueError(
                f"Versão {file_version:#04x} não suportada. "
                f"Atualize logus para ler este arquivo."
            )

        if file_version == VERSION_V4:
            raise ValueError(
                f"'{path.name}' não está criptografado (versão 4). "
                f"Use SecureFile.load_open() para arquivos sem criptografia."
            )

        # Verifica HMAC antes de qualquer decifração (Verify-then-Decrypt)
        body, file_hmac = raw[:-FILE_HMAC_LEN], raw[-FILE_HMAC_LEN:]
        if not _verify_file_hmac(master_key_bytes, body, file_hmac):
            raise RuntimeError(
                "Falha de integridade (HMAC inválido). "
                "Arquivo modificado após criação ou master_key incorreta."
            )

        if file_version == VERSION_V1:
            return cls._parse_v1(raw, master_key_bytes)

        # v2 e v3 têm estrutura idêntica
        return cls._parse_v2(raw, master_key_bytes)

    @staticmethod
    def _parse_v1(raw: bytes, master_key_bytes: bytes) -> Tuple[Dict, bytes]:
        """Parseia formato v1 (header em plaintext)."""
        offset = len(MAGIC) + 1  # skip MAGIC + VERSION
        header_len = struct.unpack(HEADER_LEN_FMT, raw[offset:offset + 4])[0]
        offset += 4
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

    Suporta retrocompatibilidade com arquivos legados (zlib externo v1.0-v1.2)
    e o formato atual (compressão interna do Parquet — sem descompressão extra).
    """
    compression = header.get("compression", "none")
    if compression == "zlib":
        return zlib.decompress(payload)
    # "parquet_zstd", "parquet_lz4", "none" — compressão é tratada pelo Parquet
    return payload


def _logus_version() -> str:
    """Retorna a versão do logus sem importação circular."""
    try:
        import datalock
        return logus.__version__
    except Exception:
        return "unknown"
