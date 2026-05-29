"""
datalock/asymmetric.py
===================
Criptografia assimétrica para arquivos .dlk — Feature 3.

Permite compartilhar arquivos .dlk sem expor a chave simétrica:
  - Remetente: cifra com a CHAVE PÚBLICA do destinatário
  - Destinatário: decifra com sua CHAVE PRIVADA
  - Nunca há troca de segredo entre as partes

Mecanismo (padrão ECIES / RSA-OAEP):
  1. Gera uma DEK aleatória (AES-256)
  2. Cifra a DEK com a chave pública RSA/EC do destinatário
  3. Cifra o payload com a DEK via AES-256-GCM (igual ao .dlk simétrico)
  4. Armazena a DEK cifrada no header do .dlk

Uso:
    # Gera par de chaves (uma vez por usuário/serviço)
    priv, pub = dd.asymmetric.generate_keypair()
    dd.asymmetric.save_keypair(priv, "minha_chave.pem", pub, "minha_chave_publica.pem")

    # Remetente: cifra com a chave pública do destinatário
    pub = dd.asymmetric.load_public_key("destinatario_pub.pem")
    dd.store(df, "dados.dlk", public_key=pub)

    # Destinatário: decifra com sua chave privada
    priv = dd.asymmetric.load_private_key("minha_chave.pem")
    df   = dd.read("dados.dlk", private_key=priv)

    # Multi-recipient: cifra uma vez para N destinatários
    pubs = [pub_ana, pub_bruno, pub_carlos]
    dd.store(df, "dados.dlk", public_keys=pubs)
    df = dd.read("dados.dlk", private_key=priv_ana)
"""
from __future__ import annotations

import base64
import json
import os
import struct
from pathlib import Path
from typing import List, Optional, Tuple, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey, EllipticCurvePublicKey,
    ECDH, SECP256R1,
)
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey, RSAPublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# Type alias
PrivateKey = Union[RSAPrivateKey, EllipticCurvePrivateKey]
PublicKey  = Union[RSAPublicKey, EllipticCurvePublicKey]


# ---------------------------------------------------------------------------
# Geração de chaves
# ---------------------------------------------------------------------------

def generate_keypair(
    algorithm: str = "ec",
    key_size:  int = 256,
) -> Tuple[PrivateKey, PublicKey]:
    """
    Gera um par de chaves assimétricas.

    Args:
        algorithm: "ec" (padrão, recomendado) ou "rsa".
        key_size:  Para EC: 256 (P-256) ou 384 (P-384).
                   Para RSA: 2048, 3072 ou 4096.

    Returns:
        (private_key, public_key)

    Exemplos:
        priv, pub = dd.asymmetric.generate_keypair()
        priv, pub = dd.asymmetric.generate_keypair("rsa", 4096)
    """
    if algorithm.lower() in ("ec", "ecdh", "ecdsa"):
        curve = SECP256R1() if key_size == 256 else ec.SECP384R1()
        priv  = ec.generate_private_key(curve, default_backend())
    elif algorithm.lower() == "rsa":
        priv = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size, backend=default_backend()
        )
    else:
        raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use 'ec' ou 'rsa'.")
    return priv, priv.public_key()


def save_keypair(
    private_key: PrivateKey,
    private_path: Union[str, Path],
    public_key: PublicKey,
    public_path: Union[str, Path],
    password: Optional[bytes] = None,
) -> None:
    """
    Salva par de chaves em arquivos PEM.

    Args:
        private_key:  Chave privada.
        private_path: Caminho para a chave privada (.pem).
        public_key:   Chave pública.
        public_path:  Caminho para a chave pública (.pem).
        password:     Senha opcional para proteger a chave privada.
    """
    enc = (serialization.BestAvailableEncryption(password)
           if password else serialization.NoEncryption())
    Path(str(private_path)).write_bytes(
        private_key.private_bytes(serialization.Encoding.PEM,
                                   serialization.PrivateFormat.PKCS8, enc)
    )
    Path(str(public_path)).write_bytes(
        public_key.public_bytes(serialization.Encoding.PEM,
                                 serialization.PublicFormat.SubjectPublicKeyInfo)
    )


def load_private_key(
    path: Union[str, Path],
    password: Optional[bytes] = None,
) -> PrivateKey:
    """Carrega chave privada de arquivo PEM."""
    raw = Path(str(path)).read_bytes()
    return serialization.load_pem_private_key(raw, password=password,
                                               backend=default_backend())


def load_public_key(path: Union[str, Path]) -> PublicKey:
    """Carrega chave pública de arquivo PEM."""
    raw = Path(str(path)).read_bytes()
    return serialization.load_pem_public_key(raw, backend=default_backend())


def public_key_to_pem(key: PublicKey) -> str:
    """Serializa chave pública como string PEM."""
    return key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def public_key_from_pem(pem: str) -> PublicKey:
    """Desserializa chave pública de string PEM."""
    return serialization.load_pem_public_key(pem.encode(), backend=default_backend())


# ---------------------------------------------------------------------------
# Envelope encryption: cifra DEK com chave pública
# ---------------------------------------------------------------------------

def encrypt_dek(dek: bytes, public_key: PublicKey) -> bytes:
    """
    Cifra uma DEK (Data Encryption Key) de 32 bytes com a chave pública.

    EC: ECDH efêmero + HKDF para derivar a chave de envoltório.
    RSA: RSA-OAEP com SHA-256.

    Returns:
        Bytes cifrados (incluem material efêmero para EC).
    """
    if isinstance(public_key, RSAPublicKey):
        return public_key.encrypt(
            dek,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    elif isinstance(public_key, EllipticCurvePublicKey):
        # ECIES: gera par efêmero, ECDH, deriva KEK via HKDF, cifra DEK
        ephem_priv = ec.generate_private_key(public_key.curve, default_backend())
        ephem_pub  = ephem_priv.public_key()
        shared     = ephem_priv.exchange(ECDH(), public_key)
        kek        = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None,
            info=b"datalock-dlk-dek-wrap", backend=default_backend()
        ).derive(shared)
        nonce      = os.urandom(12)
        ct_tag     = AESGCM(kek).encrypt(nonce, dek, None)
        ephem_bytes= ephem_pub.public_bytes(serialization.Encoding.X962,
                                             serialization.PublicFormat.UncompressedPoint)
        ephem_len  = struct.pack(">H", len(ephem_bytes))
        return ephem_len + ephem_bytes + nonce + ct_tag
    else:
        raise TypeError(f"Tipo de chave não suportado: {type(public_key)}")


def decrypt_dek(encrypted_dek: bytes, private_key: PrivateKey) -> bytes:
    """
    Decifra uma DEK cifrada com encrypt_dek().

    Returns:
        DEK de 32 bytes.
    """
    if isinstance(private_key, RSAPrivateKey):
        return private_key.decrypt(
            encrypted_dek,
            padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    elif isinstance(private_key, EllipticCurvePrivateKey):
        ephem_len   = struct.unpack(">H", encrypted_dek[:2])[0]
        ephem_bytes = encrypted_dek[2:2 + ephem_len]
        nonce       = encrypted_dek[2 + ephem_len:2 + ephem_len + 12]
        ct_tag      = encrypted_dek[2 + ephem_len + 12:]
        ephem_pub   = ec.EllipticCurvePublicKey.from_encoded_point(
            private_key.curve, ephem_bytes
        )
        shared = private_key.exchange(ECDH(), ephem_pub)
        kek = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=None,
            info=b"datalock-dlk-dek-wrap", backend=default_backend()
        ).derive(shared)
        return AESGCM(kek).decrypt(nonce, ct_tag, None)
    else:
        raise TypeError(f"Tipo de chave não suportado: {type(private_key)}")


# ---------------------------------------------------------------------------
# Multi-recipient support
# ---------------------------------------------------------------------------

def encrypt_dek_multi(
    dek: bytes,
    public_keys: List[PublicKey],
) -> List[str]:
    """
    Cifra a DEK para múltiplos destinatários.

    Cada entrada na lista é a DEK cifrada com a chave pública do respectivo
    destinatário, codificada em base64.

    Returns:
        List[str] — uma entrada por destinatário (base64-encoded).
    """
    return [
        base64.b64encode(encrypt_dek(dek, pk)).decode("ascii")
        for pk in public_keys
    ]


def decrypt_dek_from_list(
    encrypted_deks: List[str],
    private_key: PrivateKey,
) -> bytes:
    """
    Tenta decifrar a DEK usando a chave privada do destinatário.

    Itera pela lista até encontrar uma entrada que decifre com sucesso.

    Returns:
        DEK decifrada.

    Raises:
        ValueError: Se nenhuma entrada puder ser decifrada com a chave fornecida.
    """
    for enc_b64 in encrypted_deks:
        try:
            return decrypt_dek(base64.b64decode(enc_b64), private_key)
        except Exception:
            continue
    raise ValueError(
        "Nenhuma entrada na lista de destinatários pôde ser decifrada com a chave fornecida. "
        "Verifique se a chave privada corresponde a um dos destinatários."
    )


# ---------------------------------------------------------------------------
# Helpers para integração com SecureFile
# ---------------------------------------------------------------------------

def pack_asymmetric_header(
    public_key: Union[PublicKey, List[PublicKey]],
) -> Tuple[bytes, Dict]:
    """
    Gera DEK aleatória, cifra com a(s) chave(s) pública(s) e retorna
    (dek, header_fields) para integração com SecureFile.

    Args:
        public_key: Uma PublicKey ou lista de PublicKey (multi-recipient).

    Returns:
        (dek: bytes, extra_header: dict com 'encrypted_dek' e 'asymmetric')
    """
    dek = os.urandom(32)

    if isinstance(public_key, list):
        enc_deks = encrypt_dek_multi(dek, public_key)
        extra = {
            "asymmetric":      True,
            "key_algorithm":   _key_type(public_key[0]),
            "n_recipients":    len(public_key),
            "encrypted_deks":  enc_deks,
        }
    else:
        enc_dek = base64.b64encode(encrypt_dek(dek, public_key)).decode("ascii")
        extra = {
            "asymmetric":      True,
            "key_algorithm":   _key_type(public_key),
            "n_recipients":    1,
            "encrypted_deks":  [enc_dek],
        }

    return dek, extra


def resolve_dek_from_header(header: Dict, private_key: PrivateKey) -> bytes:
    """
    Extrai e decifra a DEK do header de um arquivo assimétrico.

    Returns:
        DEK de 32 bytes.
    """
    enc_deks = header.get("encrypted_deks") or [header.get("encrypted_dek", "")]
    return decrypt_dek_from_list([d for d in enc_deks if d], private_key)


def _key_type(key: Union[PublicKey, PrivateKey]) -> str:
    if isinstance(key, (RSAPublicKey, RSAPrivateKey)):
        return "RSA"
    if isinstance(key, (EllipticCurvePublicKey, EllipticCurvePrivateKey)):
        return f"EC-{key.curve.name}"
    return "UNKNOWN"
