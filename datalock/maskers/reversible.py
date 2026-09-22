"""
maskers/reversible.py
======================
Pseudonimização REVERSÍVEL determinística via AES-SIV (RFC 5297).

Diferença crítica em relação a hashing.py (DeterministicHasher)
-----------------------------------------------------------------
DeterministicHasher usa HMAC-SHA256, uma função de mão única: não existe
"HMAC inverso". Isso é intencional — é o que torna o resultado uma
ANONIMIZAÇÃO de verdade (LGPD Art. 12 / GDPR Art. 4(5) c/ Recital 26):
uma vez mascarado, o dado deixa de ser dado pessoal porque não há como
voltar ao valor original, nem mesmo por quem gerou o token.

Este módulo resolve um problema DIFERENTE: PSEUDONIMIZAÇÃO reversível.
Reversible.encrypt_value() é uma cifra determinística (mesma entrada + mesmo
salt → mesmo token, preservando joins), mas quem possui o salt PODE voltar
ao valor original via Reversible.decrypt_value(). Isso é o que o GDPR chama
de "pseudonymisation" (Art. 4(5)) — o dado CONTINUA sendo dado pessoal e
CONTINUA sujeito à LGPD/GDPR, só que protegido em repouso/trânsito.

    ┌─────────────────────┬────────────────────┬─────────────────────────┐
    │                      │ DeterministicHasher│ ReversibleCipher        │
    ├─────────────────────┼────────────────────┼─────────────────────────┤
    │ Reversível?          │ Não (HMAC)          │ Sim, com o salt         │
    │ Categoria legal      │ Dado anonimizado    │ Dado pseudonimizado     │
    │ Continua "dado       │ Não (LGPD Art. 12)  │ Sim                     │
    │ pessoal"?            │                     │                         │
    │ Uso típico           │ Compartilhar com    │ Armazenar em repouso,   │
    │                      │ terceiros, dev/test │ reverter para suporte,  │
    │                      │                     │ auditoria, correção     │
    └─────────────────────┴────────────────────┴─────────────────────────┘

NUNCA use ReversibleCipher quando o objetivo for anonimização para fins de
LGPD Art. 12 (torna-se anônimo, não se aplica a lei) — para isso use
DeterministicHasher (dd.mask() padrão) ou MaskStrategy.SUPPRESS. Use
ReversibleCipher (strategy="encrypt" / dd.mask(..., reversible=True)) apenas
quando o requisito de negócio for "proteger em repouso, mas alguém autorizado
precisa poder recuperar o valor original".

Por que AES-SIV (RFC 5297) e não AES-GCM com nonce aleatório?
-----------------------------------------------------------------
AES-GCM exige nonce único por mensagem — um nonce aleatório quebraria o
determinismo (mesma entrada geraria tokens diferentes, quebrando joins).
Um nonce fixo com AES-GCM reutilizado entre mensagens diferentes quebra a
segurança do GCM catastroficamente (recupera a chave de autenticação).

AES-SIV (Synthetic IV, RFC 5297) foi desenhado exatamente para isto: é uma
cifra autenticada DETERMINÍSTICA e segura contra reuso de nonce/IV por
construção — o "IV sintético" é derivado do próprio texto claro via CMAC,
então mesma entrada + mesma chave → sempre o mesmo ciphertext, com
autenticação (detecta adulteração), sem os riscos de reusar nonce em GCM.

Referências:
    - RFC 5297: Synthetic Initialization Vector (SIV) Authenticated
      Encryption Using AES
    - NIST SP 800-38F (key wrap) cita SIV como base para cifra determinística
    - GDPR Art. 4(5): definição de "pseudonymisation"
    - LGPD Art. 12: dado anonimizado não é considerado dado pessoal
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac_module
import logging
import secrets
import unicodedata
import warnings
from typing import Optional

import pandas as pd

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV
    _HAS_AESSIV = True
except ImportError:  # cryptography < 37 não tem AESSIV
    _HAS_AESSIV = False
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

SALT_MIN_BYTES = 16
_TOKEN_PREFIX = "enc:"          # marca visual — diferencia de tokens HASH (hex puro)
_HKDF_INFO = b"datalock-reversible-mask-v1"


def _derive_key(salt: str, n_bytes: int = 64) -> bytes:
    """
    Deriva uma chave AES-SIV (2×256 bits) a partir do salt via HKDF-SHA256.

    AES-SIV usa uma chave do DOBRO do tamanho normal (metade para CMAC,
    metade para CTR) — por isso 64 bytes para "AES-256-SIV" (2×32).
    Domain separation (info=) garante que esta chave nunca colide com as
    chaves de mascaramento HMAC (hashing.py) ou de criptografia do .dlk
    (secure_file.py), mesmo reaproveitando o mesmo salt.
    """
    salt_bytes = salt.encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=n_bytes,
        salt=None,  # o "salt" HKDF é o próprio secret do usuário via `info`+IKM
        info=_HKDF_INFO,
    ).derive(salt_bytes)


class ReversibleCipher:
    """
    Pseudonimização determinística e REVERSÍVEL via AES-SIV.

    Mesmo valor + mesmo salt → sempre o mesmo token (preserva joins, como
    DeterministicHasher). Diferente do hash, quem tem o salt pode chamar
    decrypt_value()/inverse_transform() e recuperar o valor original.

    Exemplo:
        cipher = ReversibleCipher(salt=SALT)
        token  = cipher.encrypt_value("111.444.777-35")
        # → "enc:AAECAwQFBgcICQoLDA0ODw=="
        original = cipher.decrypt_value(token)
        # → "111.444.777-35"
    """

    def __init__(self, salt: str, associated_data: Optional[str] = None):
        if not salt:
            raise ValueError(
                "ReversibleCipher requer salt explícito — sem ele o token "
                "não poderia ser gerado nem revertido de forma alguma. "
                "Use: dd.generate_salt()."
            )
        encoded = salt.encode("utf-8")
        if len(encoded) < SALT_MIN_BYTES:
            raise ValueError(
                f"Salt muito curto ({len(encoded)} bytes — mínimo: {SALT_MIN_BYTES}). "
                f"Use: dd.generate_salt()."
            )
        self._salt_repr = salt[:6] + "..."
        key = _derive_key(salt)
        # associated_data (AAD): opcional, amarra o token a um contexto (ex: nome
        # da coluna) — o mesmo valor em colunas diferentes gera tokens diferentes
        # e decrypt com AAD errado falha a autenticação (detecta token "movido").
        self._aad = associated_data.encode("utf-8") if associated_data else b""

        if _HAS_AESSIV:
            self._aead = AESSIV(key)
            self._mode = "aes-siv"
        else:
            # Fallback: AES-256-GCM com nonce determinístico derivado via
            # HMAC-SHA256(key, valor)[:12]. Seguro para reuso porque o nonce
            # depende do próprio plaintext (mesmo padrão usado por SIV/deterministic
            # AEAD) — nunca reutiliza (chave, nonce) para plaintexts diferentes.
            self._aead = AESGCM(key[:32])
            self._nonce_key = key[32:64]
            self._mode = "aes-gcm-deterministic"

    _NULL_STRINGS: frozenset = frozenset({"", "nan", "none", "null", "na", "n/a", "<na>"})

    def _nonce_for(self, plaintext: bytes) -> bytes:
        return _hmac_module.new(self._nonce_key, plaintext, hashlib.sha256).digest()[:12]

    def encrypt_value(self, value: object) -> Optional[str]:
        """Cifra um valor escalar. None/NaN/vazio → None (não mascarado)."""
        try:
            if value is None or pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        s = str(value).strip()
        if s.lower() in self._NULL_STRINGS:
            return None
        normalized = unicodedata.normalize("NFC", s)
        plaintext = normalized.encode("utf-8")

        if self._mode == "aes-siv":
            ct = self._aead.encrypt(plaintext, [self._aad] if self._aad else None)
        else:
            nonce = self._nonce_for(plaintext)
            ct = nonce + self._aead.encrypt(nonce, plaintext, self._aad or None)

        return _TOKEN_PREFIX + base64.urlsafe_b64encode(ct).decode("ascii")

    def decrypt_value(self, token: object) -> Optional[str]:
        """
        Reverte um token para o valor original. None/vazio → None.

        Raises:
            ValueError: token corrompido, adulterado, ou cifrado com outro
                        salt/associated_data (falha de autenticação AEAD).
        """
        try:
            if token is None or pd.isna(token):
                return None
        except (TypeError, ValueError):
            pass
        s = str(token).strip()
        if not s or s.lower() in self._NULL_STRINGS:
            return None
        if not s.startswith(_TOKEN_PREFIX):
            raise ValueError(
                f"Token {s[:20]!r}... não tem o prefixo esperado ({_TOKEN_PREFIX!r}). "
                "Este valor não foi gerado por ReversibleCipher/strategy='encrypt' "
                "— não é possível reverter (ex.: é um hash HMAC, que é irreversível "
                "por design)."
            )
        raw = base64.urlsafe_b64decode(s[len(_TOKEN_PREFIX):].encode("ascii"))
        try:
            if self._mode == "aes-siv":
                pt = self._aead.decrypt(raw, [self._aad] if self._aad else None)
            else:
                nonce, ct = raw[:12], raw[12:]
                pt = self._aead.decrypt(nonce, ct, self._aad or None)
        except Exception as exc:
            raise ValueError(
                "Falha ao reverter token: salt incorreto, associated_data "
                "(coluna) incorreta, ou token corrompido/adulterado. "
                f"Detalhe: {type(exc).__name__}"
            ) from exc
        return pt.decode("utf-8")

    def transform(self, series: pd.Series) -> pd.Series:
        """Cifra uma coluna inteira (com deduplicação — mesma lógica de hashing.py)."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return series.copy()
        unique_vals = non_null.unique()
        n_unique = len(unique_vals)
        n_non_null = len(non_null)

        if n_unique > n_non_null * 0.95:
            result = pd.Series(
                [self.encrypt_value(v) for v in series.tolist()],
                index=series.index, name=series.name,
            )
        else:
            enc_map = {v: self.encrypt_value(v) for v in unique_vals}
            result = pd.Series(
                [enc_map.get(v) for v in series.tolist()],
                index=series.index, name=series.name,
            )
        logger.debug(
            "AES-SIV reversible | col=%s | salt=%s | unique=%d/%d",
            series.name, self._salt_repr, n_unique, n_non_null,
        )
        return result

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Reverte uma coluna inteira de tokens 'enc:...' para os valores originais."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return series.copy()
        unique_vals = non_null.unique()
        dec_map = {v: self.decrypt_value(v) for v in unique_vals}
        return pd.Series(
            [dec_map.get(v) if pd.notna(v) else None for v in series.tolist()],
            index=series.index, name=series.name,
        )

    def __repr__(self) -> str:
        return f"ReversibleCipher(salt={self._salt_repr}, mode={self._mode})"


def is_reversible_token(value: object) -> bool:
    """Heurística: valor parece um token gerado por ReversibleCipher?"""
    try:
        if value is None or pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and value.startswith(_TOKEN_PREFIX)
