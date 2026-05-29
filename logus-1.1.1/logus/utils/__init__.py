"""privacy_framework.utils — utilitários internos de segurança."""

from __future__ import annotations

from .secret_str import SecretStr
from .salt import generate_salt, generate_salt_hex

__all__ = ["SecretStr", "generate_salt", "generate_salt_hex"]
