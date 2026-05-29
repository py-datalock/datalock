"""
maskers/hashing.py
==================
Pseudonimização determinística via HMAC-SHA256.

Por que HMAC-SHA256 e não SHA256 puro?
----------------------------------------
SHA256(salt || valor) é vulnerável ao *length extension attack* (RFC 6194).
HMAC(k, m) = H(k_outer || H(k_inner || m)) é imune por construção.

Por que validar o comprimento do salt?
-----------------------------------------
CPFs têm ~1B combinações válidas. Uma RTX 4090 computa ~10B HMAC-SHA256/s.
Com salt de 3 chars, brute-force leva ~0,1s. Salt mínimo de 16 bytes (128 bits)
torna isso computacionalmente intratável.

Otimizações (v1.1):
  - transform() com deduplicação: para N linhas com K valores únicos (K << N),
    computa K hashes e aplica via map(dict). Ganho: proporcional a N/K.
  - Normalização Unicode NFC garante que "José" (NFC) e "Jose\u0301" (NFD)
    gerem sempre o mesmo token.

Referências:
  - ENISA: Pseudonymisation Techniques and Best Practices, 2019, §3.2
  - NIST SP 800-108r1: KDF using HMAC as PRF, 2022
  - RFC 2104: HMAC: Keyed-Hashing for Message Authentication
"""

from __future__ import annotations

import hashlib
import hmac as _hmac_module
import logging
import re as _re
import secrets
import unicodedata
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SALT_MIN_BYTES = 16

_WEAK_SALT_PATTERNS = frozenset({
    "test", "teste", "exemplo", "example", "salt", "senha",
    "password", "chave", "key", "dev", "debug", "123", "abc",
    "lgpd", "framework", "demo", "local", "staging", "homolog",
    "producao", "production", "empresa", "company", "corp",
})
_YEAR_IN_SALT = _re.compile(r"(19|20)\d{2}")


SALT_MIN_UNIQUE_CHARS = 6   # mínimo de caracteres distintos
SALT_MIN_LENGTH       = 16  # alias de SALT_MIN_BYTES

def _char_entropy(salt: str) -> float:
    """Shannon entropy (bits) do salt — mede diversidade de caracteres."""
    import math
    if not salt: return 0.0
    freq = {}
    for c in salt:
        freq[c] = freq.get(c, 0) + 1
    n = len(salt)
    return -sum((f/n) * math.log2(f/n) for f in freq.values())


def _validate_salt(salt: str) -> None:
    """
    Valida força do salt. Regras:
      1. Mínimo 16 bytes (128 bits) → ValueError se menor
      2. Palavras de dicionário/ambiente → UserWarning
      3. Menos de 6 caracteres distintos → UserWarning (ex: 'aaaa...', '1111...')
      4. Entropia Shannon < 2.0 bits → UserWarning (caracteres muito repetitivos)
    """
    encoded = salt.encode("utf-8")
    if len(encoded) < SALT_MIN_BYTES:
        raise ValueError(
            f"Salt muito curto ({len(encoded)} bytes — mínimo: {SALT_MIN_BYTES}). "
            f"CPFs têm ~1B combinações válidas; GPUs modernas fazem 10B SHA256/s. "
            f"Use: import datalock as lg; salt = dd.generate_salt()"
        )
    lower = salt.lower()
    has_weak = any(p in lower for p in _WEAK_SALT_PATTERNS)
    has_year = bool(_YEAR_IN_SALT.search(salt))
    n_unique = len(set(salt))
    entropy  = _char_entropy(salt)
    low_entropy = n_unique < SALT_MIN_UNIQUE_CHARS or entropy < 2.0

    if has_weak or has_year or low_entropy:
        reason = []
        if has_weak:
            reason.append("contém palavra de dicionário ou termo de ambiente")
        if has_year:
            reason.append("contém ano (reduz espaço de busca)")
        if low_entropy:
            reason.append(
                f"baixa diversidade de caracteres "
                f"({n_unique} únicos, entropia={entropy:.1f} bits — mínimo recomendado: 6 únicos e 2.0 bits)"
            )
        warnings.warn(
            f"Salt fraco ({'; '.join(reason)}). "
            f"Use: import datalock as lg; salt = dd.generate_salt()  "
            f"[gera 48 chars, ~240 bits de entropia]",
            UserWarning,
            stacklevel=3,
        )


def generate_salt(n_bytes: int = 32) -> str:
    if n_bytes < SALT_MIN_BYTES:
        raise ValueError(f"n_bytes={n_bytes} insuficiente. Mínimo: {SALT_MIN_BYTES} bytes.")
    return secrets.token_hex(n_bytes)


class DeterministicHasher:
    """
    HMAC-SHA256 para pseudonimização determinística.

    Mesmo valor + mesmo salt → mesmo token (preserva integridade referencial
    para joins entre tabelas mascaradas pelo mesmo pipeline).

    Aviso de salt=None:
      Usa warnings.warn (não logger.warning) para que o aviso apareça
      automaticamente em notebooks e CLIs sem configuração de handlers.
    """

    def __init__(
        self,
        salt: Optional[str] = None,
        truncate: int = 16,
        prefix: str = "",
    ):
        if salt is None:
            self._salt_bytes = secrets.token_bytes(32)
            self._salt_repr = self._salt_bytes[:3].hex() + "..."
            # warnings.warn (não logger) — aparece sem configuração de handlers
            warnings.warn(
                "DeterministicHasher(salt=None): salt aleatório gerado — "
                "hashes NÃO serão reprodutíveis entre execuções. "
                "Joins entre tabelas mascaradas em momentos distintos vão quebrar. "
                "Use: dd.generate_salt() para gerar um salt fixo.",
                UserWarning,
                stacklevel=5,  # bubbles up through MaskingEngine → secure_dataframe → mask() → user code
            )
        else:
            _validate_salt(salt)
            self._salt_bytes = salt.encode("utf-8")
            self._salt_repr = salt[:6] + "..."

        self.truncate = truncate
        self.prefix = prefix

    # String representations of null that should be treated as null
    _NULL_STRINGS: frozenset = frozenset({"", "nan", "none", "null", "na", "n/a", "<na>"})

    def hash_value(self, value: object) -> Optional[str]:
        """
        HMAC-SHA256 de um valor escalar. None/NaN/vazio → None.

        Tratamento de nulos:
          None, np.nan, pd.NA, pd.NaT → None (não mascarado)
          '' (string vazia)           → None (não mascarado)
          'NaN', 'None', 'null', 'NA' → None (não mascarado)
        """
        try:
            if value is None or pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        s = str(value).strip()
        if s.lower() in self._NULL_STRINGS:
            return None
        normalized = unicodedata.normalize("NFC", s)
        digest = _hmac_module.new(
            self._salt_bytes, normalized.encode("utf-8"), hashlib.sha256
        ).hexdigest()[: self.truncate]
        return f"{self.prefix}{digest}"

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Aplica HMAC-SHA256 com deduplicação adaptativa.

        < 95% únicos → computa K hashes únicos e aplica via dict (O(K) vs O(N)).
        ≥ 95% únicos → list comprehension direto (evita overhead do dict).
        """
        non_null = series.dropna()
        n_non_null = len(non_null)

        if n_non_null == 0:
            return series.copy()

        unique_vals = non_null.unique()
        n_unique = len(unique_vals)

        if n_unique > n_non_null * 0.95:
            result = pd.Series(
                [self.hash_value(v) for v in series.tolist()],
                index=series.index,
                name=series.name,
            )
        else:
            hash_map = {v: self.hash_value(v) for v in unique_vals}
            result = pd.Series(
                [hash_map.get(v) for v in series.tolist()],
                index=series.index,
                name=series.name,
            )

        logger.debug(
            "HMAC-SHA256 | col=%s | salt=%s | truncate=%d | unique=%d/%d",
            series.name, self._salt_repr, self.truncate, n_unique, n_non_null,
        )
        return result

    def verify(self, value: object, token: str) -> bool:
        """Verifica em tempo constante (resistente a timing attacks)."""
        expected = self.hash_value(value)
        if expected is None or token is None:
            return False
        return _hmac_module.compare_digest(expected, token)

    def __repr__(self) -> str:
        return (
            f"DeterministicHasher(salt={self._salt_repr}, "
            f"truncate={self.truncate}, prefix='{self.prefix}')"
        )
