"""
utils/secret_str.py
===================
SecretStr — Wrapper que impede vazamento do salt em logs e tracebacks.

Qualquer representação de string (print, f-string, repr, logger.debug, Sentry)
que acidentalmente capture o objeto retornará "***" em vez do valor real.
O valor real só é acessível via .get(), tornando o acesso intencional e explícito.

Inspirado no SecretStr do Pydantic e no SecretValue do Django.

Exemplo:
    salt = SecretStr("minha-chave-secreta")
    print(salt)              # ***
    logger.debug(salt)       # ***
    f"salt={salt}"           # "salt=***"
    repr(salt)               # SecretStr('***')
    salt.get()               # "minha-chave-secreta"  (acesso explícito)
"""

from __future__ import annotations

from typing import Optional


class SecretStr:
    """
    String que se auto-redige em qualquer contexto de representação.

    Impede que o salt do HMAC apareça em:
      - Logs (logging.debug, logging.info, etc.)
      - Stack traces (traceback, Sentry, Datadog)
      - f-strings e format()
      - repr() e str()
      - print()
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(
                f"SecretStr requer uma string, recebeu {type(value).__name__}."
            )
        object.__setattr__(self, "_value", value)

    # Impede atribuição acidental de novos atributos
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SecretStr é imutável.")

    def get(self) -> str:
        """Retorna o valor real — use apenas onde necessário."""
        return object.__getattribute__(self, "_value")

    # Todos os métodos de representação retornam ***
    def __repr__(self) -> str:
        return "SecretStr('***')"

    def __str__(self) -> str:
        return "***"

    def __format__(self, format_spec: str) -> str:
        return "***"

    def __len__(self) -> int:
        # Retorna comprimento para validação sem expor o valor
        return len(self.get())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            import hmac
            return hmac.compare_digest(self.get(), other.get())
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.get())

    def __bool__(self) -> bool:
        return bool(self.get())

    @classmethod
    def from_optional(cls, value: Optional[str]) -> Optional["SecretStr"]:
        """Converte str | None para SecretStr | None."""
        return cls(value) if value is not None else None
