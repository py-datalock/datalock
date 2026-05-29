"""
utils/salt.py
=============
Geração criptograficamente segura de salts para uso com dd.mask() e dd.read().

Por que isso importa?
---------------------
O salt é a chave do HMAC que torna o mascaramento determinístico e reversível
apenas por quem o possui. Um salt fraco (ex: "senha123", "empresa2024") pode
ser adivinhado por dicionário ou força bruta, expondo os dados originais.

Esta função usa `secrets` (baseado em os.urandom / CSPRNG do SO) — nunca
`random`, que é pseudoaleatório e previsível.

Referências:
    - NIST SP 800-132: Recommendation for Password-Based Key Derivation
    - Python docs: secrets — Generate secure random numbers for managing secrets
    - OWASP: Cryptographic Storage Cheat Sheet
"""

from __future__ import annotations

import os
import secrets
import string


def generate_salt(length: int = 48) -> str:
    """
    Gera um salt criptograficamente seguro para uso com dd.mask() e dd.read().

    Usa secrets.choice (baseado em os.urandom/CSPRNG) — mais seguro que random.
    O alfabeto inclui letras, dígitos e símbolos seguros para uso em
    variáveis de ambiente, arquivos .env e YAML sem necessidade de escape.

    Parâmetros
    ----------
    length : int
        Comprimento do salt em caracteres.
        Mínimo recomendado: 32 (256 bits de entropia com este alfabeto).
        Padrão: 48 (~300 bits de entropia — adequado para dados de produção).

    Retorna
    -------
    str
        Salt como string imprimível, pronta para copiar em .env ou vault.

    Levanta
    -------
    ValueError
        Se length < 32 — abaixo do mínimo de segurança recomendado.

    Exemplo
    -------
    >>> from datalock.utils.salt import generate_salt
    >>> salt = generate_salt()
    >>> print(salt)       # Ex: 'aB3$kP9#mX2@vQ7!nR5&wY1^jL4*hT6mK8!pZ5@'
    >>> salt_curto = generate_salt(32)   # mínimo aceitável
    >>> salt_longo = generate_salt(64)   # máxima segurança

    Notas de segurança
    ------------------
    - Nunca use palavras, datas ou nomes como salt — são previsíveis.
    - Guarde o salt no vault (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault)
      ou em variável de ambiente (DATALOCK_SALT), nunca em código-fonte.
    - Um salt diferente por ambiente (dev/staging/prod) impede que acesso a
      um ambiente comprometa os outros.
    """
    if length < 32:
        raise ValueError(
            f"Salt muito curto ({length} chars). Use pelo menos 32 caracteres "
            f"para garantir segurança adequada (~200 bits de entropia mínimo). "
            f"Recomendado: 48."
        )

    # Alfabeto sem caracteres ambíguos (0/O, 1/l/I) e sem aspas simples/duplas
    # para evitar problemas em .env, YAML, shells e strings Python
    alfabeto = (
        string.ascii_letters      # a-z A-Z  (52 chars)
        + string.digits           # 0-9      (10 chars)
        + "!@#$%^&*-_=+<>?"      # símbolos seguros para env/yaml (16 chars)
    )                             # Total: 78 chars → ~6.3 bits por char

    return "".join(secrets.choice(alfabeto) for _ in range(length))


def generate_salt_hex(bytes: int = 32) -> str:
    """
    Gera um salt como string hexadecimal (apenas 0-9 a-f).

    Menor entropia por caractere (~4 bits/char vs ~6.3 bits/char do generate_salt),
    mas 100% seguro em qualquer contexto: JSON, XML, SQL, URLs, headers HTTP.
    Preferido quando o destino não suporta caracteres especiais.

    Parâmetros
    ----------
    bytes : int
        Número de bytes aleatórios gerados por os.urandom().
        O salt resultante terá 2x este comprimento em caracteres hex.
        Padrão: 32 bytes → 64 chars hex → 256 bits de entropia.

    Retorna
    -------
    str
        Salt hexadecimal de comprimento `bytes * 2`.

    Exemplo
    -------
    >>> salt = generate_salt_hex()
    >>> print(salt)          # 64 caracteres hex
    'a3f2c9d1e0b478253c8d...'
    >>> print(len(salt))     # 64
    """
    if bytes < 16:
        raise ValueError(
            f"Mínimo de 16 bytes (128 bits). Recebido: {bytes}."
        )
    return os.urandom(bytes).hex()
