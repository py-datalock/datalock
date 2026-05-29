"""
detectors/text_detector.py
===========================
TextPIIDetector — Detecção de PII em texto livre (strings, logs, PDFs, chats).

Diferente do PIIDetector (que trabalha em colunas de DataFrames), este módulo
busca PII em qualquer posição dentro de strings arbitrárias.

Uso:
    from datalock.detectors.text_detector import TextPIIDetector, PIISpan

    detector = TextPIIDetector()
    spans = detector.scan("Cliente João Silva, CPF 123.456.789-09, email joao@emp.com")
    for span in spans:
        print(span.pii_type, span.text, span.start, span.end)

Integração com dd.scan_text() e dd.mask_text() — ver datalock/__init__.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from datalock.detectors.pii_detector import PIIType, _PATTERNS, _validate_cpf, _validate_cnpj


@dataclass
class PIISpan:
    """Representa uma ocorrência de PII encontrada em texto livre."""
    pii_type:  PIIType   # Tipo de PII detectado
    text:      str        # Texto original que deu match
    start:     int        # Posição inicial na string (inclusive)
    end:       int        # Posição final na string (exclusive)
    validated: bool = False  # True se passou por validação de dígito verificador

    def __repr__(self) -> str:
        return (
            f"PIISpan(type={self.pii_type.value!r}, "
            f"text={self.text!r}, pos={self.start}:{self.end}, "
            f"validated={self.validated})"
        )


# Patterns without anchors (^ $) for in-text search.
# We reuse _PATTERNS from pii_detector but strip anchors.
def _build_text_patterns() -> dict:
    """Rebuilds PII patterns without ^ and $ anchors for free-text search."""
    patterns = {}
    for pii_type, pattern in _PATTERNS.items():
        src = pattern.pattern
        # Strip anchors that would prevent mid-string matching
        src = re.sub(r'^\^', '', src)
        src = re.sub(r'\$$', '', src)
        try:
            patterns[pii_type] = re.compile(src, re.IGNORECASE | re.UNICODE)
        except re.error:
            # If stripping breaks the regex, use original (may not match mid-string)
            patterns[pii_type] = pattern
    return patterns


_TEXT_PATTERNS = _build_text_patterns()

# Priority order — more specific types checked first to avoid overlap
_SCAN_ORDER = [
    PIIType.CPF,
    PIIType.CNPJ,
    PIIType.RG,
    PIIType.EMAIL,
    PIIType.TELEFONE,
    PIIType.CEP,
    PIIType.NOME,
    PIIType.DATA_NASCIMENTO,
    PIIType.IP,
    PIIType.CARTAO_CREDITO,
]


class TextPIIDetector:
    """
    Detecta PII em texto livre usando os mesmos padrões do PIIDetector,
    mas sem âncoras — permite busca em qualquer posição da string.

    Parâmetros:
        validate_checksums: Se True (padrão), valida dígitos verificadores
                            para CPF e CNPJ, reduzindo falsos positivos.
        min_confidence:     Não usado internamente ainda — reservado para
                            versões futuras com scoring de confiança.
    """

    def __init__(
        self,
        validate_checksums: bool = True,
    ) -> None:
        self._validate = validate_checksums

    def scan(self, text: str) -> List[PIISpan]:
        """
        Detecta todos os trechos de PII em um texto livre.

        Parâmetros:
            text: String a ser inspecionada.

        Retorna:
            Lista de PIISpan ordenada por posição de início.
            Spans sobrepostos são removidos (o mais específico prevalece).

        Exemplo:
            spans = detector.scan("CPF: 529.982.247-25 — João Silva")
            # [PIISpan(type='cpf', text='529.982.247-25', pos=5:19, validated=True),
            #  PIISpan(type='name', text='João Silva', pos=22:32, validated=False)]
        """
        if not text:
            return []

        raw_spans: List[PIISpan] = []

        for pii_type in _SCAN_ORDER:
            pattern = _TEXT_PATTERNS.get(pii_type)
            if pattern is None:
                continue
            for m in pattern.finditer(text):
                matched = m.group(0)
                validated = False
                # Checksum validation for CPF and CNPJ
                if self._validate:
                    if pii_type == PIIType.CPF:
                        digits = re.sub(r'\D', '', matched)
                        if len(digits) != 11 or not _validate_cpf(digits):
                            continue
                        validated = True
                    elif pii_type == PIIType.CNPJ:
                        digits = re.sub(r'\D', '', matched)
                        if len(digits) != 14 or not _validate_cnpj(digits):
                            continue
                        validated = True

                raw_spans.append(PIISpan(
                    pii_type=pii_type,
                    text=matched,
                    start=m.start(),
                    end=m.end(),
                    validated=validated,
                ))

        return _deduplicate_spans(raw_spans)

    def scan_batch(self, texts: List[str]) -> List[List[PIISpan]]:
        """
        Detecta PII em uma lista de strings.

        Retorna:
            Lista de listas de PIISpan — uma por string de entrada.
        """
        return [self.scan(t) for t in texts]

    def has_pii(self, text: str) -> bool:
        """Retorna True se o texto contiver ao menos um trecho de PII."""
        return bool(self.scan(text))

    def pii_types_found(self, text: str) -> List[PIIType]:
        """Retorna lista de tipos de PII únicos encontrados no texto."""
        return list({s.pii_type for s in self.scan(text)})


def _deduplicate_spans(spans: List[PIISpan]) -> List[PIISpan]:
    """
    Remove spans sobrepostos mantendo o de maior comprimento (mais específico).
    Ordena por posição de início depois.
    """
    if not spans:
        return []
    # Sort by start, then by length descending (longer = more specific)
    spans_sorted = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    result: List[PIISpan] = []
    last_end = -1
    for span in spans_sorted:
        if span.start >= last_end:
            result.append(span)
            last_end = span.end
    return result
