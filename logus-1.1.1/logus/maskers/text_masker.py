"""
maskers/text_masker.py
=======================
TextMasker — Mascaramento de PII em texto livre.

Aplica mascaramento inline em strings arbitrárias, preservando o contexto
ao redor da PII (ao contrário do mascaramento de coluna que substitui tudo).

Estratégias disponíveis:
    "redact"  → substitui por [TIPO_PII] (ex: [CPF], [EMAIL])
    "hash"    → substitui por HMAC-SHA256 truncado (16 hex chars), determinístico
    "partial" → mantém prefixo + mascara o restante (ex: 529.***.**-**)

Uso:
    from datalock.maskers.text_masker import TextMasker

    masker = TextMasker(salt="chave-segura", strategy="redact")
    safe = masker.mask("CPF 529.982.247-25, contato joao@empresa.com")
    # "CPF [CPF], contato [EMAIL]"

    # Hash determinístico (preserva joins por token)
    masker_hash = TextMasker(salt="chave", strategy="hash")
    safe = masker_hash.mask("CPF 529.982.247-25")
    # "CPF a3f2c9d1e0b47825"
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import List, Optional, Union

from datalock.detectors.text_detector import PIISpan, TextPIIDetector
from datalock.detectors.pii_detector import PIIType


class TextMasker:
    """
    Mascara PII em texto livre preservando o contexto textual.

    Parâmetros:
        salt:     Chave HMAC para strategy="hash". Obrigatório nessa estratégia.
        strategy: "redact" (padrão) | "hash" | "partial"
        detector: TextPIIDetector customizado. None usa o padrão.

    Exemplo:
        masker = TextMasker(salt="chave", strategy="hash")
        safe = masker.mask("João Silva, CPF 529.982.247-25, joao@co.com")
        # "João Silva, CPF a3f2c9d1e0b47825, 9e1d3c7f2a845b61"
    """

    def __init__(
        self,
        salt: Optional[str] = None,
        strategy: str = "redact",
        detector: Optional[TextPIIDetector] = None,
    ) -> None:
        if strategy not in ("redact", "hash", "partial"):
            raise ValueError(
                f"strategy deve ser 'redact', 'hash' ou 'partial'. Recebido: {strategy!r}"
            )
        if strategy == "hash" and not salt:
            raise ValueError(
                "TextMasker com strategy='hash' requer salt. "
                "Gere com: import secrets; secrets.token_hex(32)"
            )
        self._salt     = salt
        self._strategy = strategy
        self._detector = detector or TextPIIDetector()

    def mask(self, text: str) -> str:
        """
        Mascara todos os trechos de PII em um texto livre.

        Parâmetros:
            text: String a mascarar.

        Retorna:
            String com PII substituída, preservando contexto ao redor.

        Exemplo:
            masker.mask("Cliente João, CPF 529.982.247-25 — aprovado")
            # strategy="redact" → "Cliente João, [CPF] — aprovado"
            # strategy="hash"   → "Cliente João, a3f2c9d1e0b47825 — aprovado"
        """
        if not text:
            return text

        spans = self._detector.scan(text)
        if not spans:
            return text

        # Rebuild text replacing spans from right to left (preserves positions)
        parts = []
        prev_end = 0
        for span in sorted(spans, key=lambda s: s.start):
            parts.append(text[prev_end:span.start])
            parts.append(self._replace(span))
            prev_end = span.end
        parts.append(text[prev_end:])
        return "".join(parts)

    def mask_batch(self, texts: List[str]) -> List[str]:
        """Mascara uma lista de strings. Retorna lista de mesma extensão."""
        return [self.mask(t) for t in texts]

    def _replace(self, span: PIISpan) -> str:
        if self._strategy == "redact":
            return f"[{span.pii_type.value.upper()}]"
        elif self._strategy == "hash":
            return self._hmac_truncated(span.text)
        elif self._strategy == "partial":
            return self._partial_mask(span)
        return "[REDACTED]"

    def _hmac_truncated(self, value: str) -> str:
        """Returns a 16-hex-char deterministic HMAC-SHA256 token."""
        key = (self._salt or "").encode("utf-8")
        digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return digest[:16]

    def _partial_mask(self, span: PIISpan) -> str:
        """Keeps a short prefix, masks the rest with *."""
        text = span.text
        digits_only = re.sub(r'\D', '', text)
        if len(digits_only) >= 4:
            # Keep first 3 chars, mask rest preserving separators
            keep = 3
            result = list(text)
            char_idx = 0
            masked = 0
            for i, ch in enumerate(text):
                if ch.isdigit() or ch.isalpha():
                    if char_idx >= keep:
                        result[i] = '*'
                        masked += 1
                    char_idx += 1
            return "".join(result)
        return "*" * len(text)
