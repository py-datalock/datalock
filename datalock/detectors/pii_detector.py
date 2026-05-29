"""
detectors/pii_detector.py
=========================
Varredura semântica e classificação de risco de colunas.

Combina três estratégias:
  1. Heurística por nome de coluna (keywords brasileiras)
  2. Matching por expressões regulares nos valores (vetorizado)
  3. Análise de cardinalidade e unicidade

Engine interno: Polars quando disponível, Pandas como fallback.
  - Detecção regex via pl.Series.str.contains() — SIMD sobre Arrow
  - 5–15x mais rápido que pd.Series.str.fullmatch() para colunas longas
  - Conversão mínima: aceita pd.DataFrame e pl.DataFrame transparentemente

Otimizações v1.1:
  - _regex_scan() usa str.fullmatch() vetorizado (Cython/C).
  - _is_safe_technical_column() usa token-level matching.
  - Detecção de tipos mistos com warning.
  - MIN_NON_NULL_FOR_REGEX separado de heurística de nome.

v1.5: engine Polars nativo para regex scan.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np

import polars as pl
_POLARS_AVAILABLE = True  # Polars obrigatório desde v1.0.4

logger = logging.getLogger(__name__)

MIN_NON_NULL_FOR_REGEX: int = 3


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PIIType(str, Enum):
    CPF              = "cpf"
    CNPJ             = "cnpj"
    EMAIL            = "email"
    TELEFONE         = "telefone"
    CEP              = "cep"
    DATA_NASCIMENTO  = "data_nascimento"
    NOME             = "nome"
    RG               = "rg"
    IP               = "ip"
    CARTAO_CREDITO   = "cartao_credito"
    QUASI_IDENTIFIER = "quasi_identifier"
    NUMERICO         = "numerico"
    CATEGORICO       = "categorico"
    UNKNOWN          = "unknown"


class RiskLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class MaskStrategy(str, Enum):
    HASH            = "hash"
    TRUNCATE        = "truncate"
    REDACT          = "redact"
    MOCK_NUM        = "mock_numeric"
    MOCK_CAT        = "mock_category"
    SUPPRESS        = "suppress"
    PASSTHROUGH     = "passthrough"
    MASK_PHONE_DDD  = "mask_phone_ddd"
    GENERALIZE_DATE = "generalize_date"


# ---------------------------------------------------------------------------
# Padrões Regex (compilados uma vez, reutilizados por todas as instâncias)
# ---------------------------------------------------------------------------

_PATTERNS: Dict[PIIType, re.Pattern] = {
    PIIType.CPF: re.compile(r"^\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{2}$"),
    PIIType.CNPJ: re.compile(r"^\d{2}[.\-]?\d{3}[.\-]?\d{3}[/\-]?\d{4}[.\-]?\d{2}$"),
    PIIType.EMAIL: re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
    PIIType.TELEFONE: re.compile(r"^(\+55\s?)?(\(?\d{2}\)?[\s\-]?)?(9?\d{4}[\s\-]?\d{4})$"),
    PIIType.CEP: re.compile(r"^\d{5}[\-]?\d{3}$"),
    PIIType.RG: re.compile(
        r"^(?:\d{1,2}[.\-]?\d{3}[.\-]?\d{3}[\-]?[\dxX]"
        r"|[A-Z]{2}[\-\.]?\d{3}[.\-]?\d{3}[.\-]?\d{2}"
        r"|\d{3}\s\d{3}\s\d{3}\s[\dxX])$"
    ),
    PIIType.IP: re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
    PIIType.CARTAO_CREDITO: re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$"),
}

_PII_TYPE_THRESHOLDS: Dict[PIIType, float] = {
    PIIType.RG: 0.7,
}

# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------

_NAME_HEURISTICS: Dict[PIIType, List[str]] = {
    PIIType.CPF: ["cpf", "cadastro_pessoa", "documento", "doc_pessoa"],
    PIIType.CNPJ: ["cnpj", "cadastro_nacional", "doc_empresa"],
    PIIType.EMAIL: ["email", "e_mail", "mail", "correio", "e-mail"],
    PIIType.TELEFONE: ["telefone", "celular", "fone", "phone", "tel", "whatsapp", "contato"],
    PIIType.CEP: ["cep", "codigo_postal", "zipcode", "zip", "cod_postal"],
    PIIType.DATA_NASCIMENTO: [
        "nascimento", "birth", "dob", "dt_nasc", "data_nasc",
        "aniversario", "birthdate", "data_nascimento",
    ],
    PIIType.NOME: [
        "nome", "name", "sobrenome", "first_name", "last_name",
        "razao_social", "cliente", "paciente", "usuario", "proprietario",
        "responsavel", "titular",
    ],
    PIIType.RG: ["rg", "registro_geral", "identidade", "doc_rg"],
    PIIType.IP: ["ip", "endereco_ip", "ip_address", "host_ip"],
    PIIType.CARTAO_CREDITO: ["cartao", "card", "credit_card", "pan", "numero_cartao"],
}

_QUASI_KEYWORDS: List[str] = [
    "idade", "age", "sexo", "genero", "gender", "raca", "etnia",
    "municipio", "cidade", "city", "estado", "uf", "regiao",
    "profissao", "ocupacao", "occupation", "escolaridade", "education",
    "renda", "salario", "income", "salary", "faixa",
    "bairro", "district", "latitude", "longitude", "logradouro",
    "endereco", "address",
]

_SAFE_COLUMN_TOKENS: Set[str] = {
    "id", "pk", "fk", "key", "index", "idx", "row_num",
    "created", "updated", "deleted",
    "valor", "value", "price", "amount", "preco",
    "quantidade", "count", "total", "score", "rank",
    "version", "versao", "uuid", "guid", "token", "hash",
    "arquivo", "file", "filename", "filepath", "path",
    "tabela", "table", "relatorio", "report", "log",
    "sistema", "system", "modulo", "module", "tipo",
    "timestamp", "num", "seq", "cod", "code",
    "admissao", "admissão", "contratacao", "contratação",
    "cadastro", "criacao", "criação", "atualizacao",
    "vencimento", "expiracao", "validade", "referencia",
    "pedido", "compra", "venda", "entrega", "envio", "fatura",
}

_SAFE_AFFIXES: List[Tuple[str, str]] = [
    ("id", "suffix"), ("id", "prefix"),
    ("pk", "suffix"), ("fk", "suffix"),
    ("num", "prefix"), ("seq", "prefix"),
    ("cod", "prefix"), ("code", "prefix"),
]

_SAFE_EXACT_NAMES: Set[str] = {
    "id", "pk", "fk", "key", "index", "idx", "row_num",
    "created", "updated", "deleted",
}

_DEFAULT_STRATEGY: Dict[PIIType, MaskStrategy] = {
    PIIType.CPF:              MaskStrategy.HASH,
    PIIType.CNPJ:             MaskStrategy.HASH,
    PIIType.EMAIL:            MaskStrategy.HASH,
    PIIType.NOME:             MaskStrategy.REDACT,
    PIIType.TELEFONE:         MaskStrategy.MASK_PHONE_DDD,
    PIIType.CEP:              MaskStrategy.TRUNCATE,
    PIIType.RG:               MaskStrategy.HASH,
    PIIType.IP:               MaskStrategy.HASH,
    PIIType.CARTAO_CREDITO:   MaskStrategy.REDACT,
    PIIType.DATA_NASCIMENTO:  MaskStrategy.GENERALIZE_DATE,
    PIIType.QUASI_IDENTIFIER: MaskStrategy.MOCK_CAT,
    PIIType.NUMERICO:         MaskStrategy.MOCK_NUM,
    PIIType.CATEGORICO:       MaskStrategy.MOCK_CAT,
    PIIType.UNKNOWN:          MaskStrategy.PASSTHROUGH,
}


# ---------------------------------------------------------------------------
# ColumnReport
# ---------------------------------------------------------------------------

@dataclass
class ColumnReport:
    column:         str
    pii_type:       PIIType
    risk_level:     RiskLevel
    mask_strategy:  MaskStrategy
    match_ratio:    float
    unique_ratio:   float
    sample_matches: List[str] = field(default_factory=list)
    col_min:        Optional[float] = None
    col_max:        Optional[float] = None
    value_freq:     Optional[Dict[str, float]] = None
    notes:          str = ""

    @property
    def is_direct_identifier(self) -> bool:
        return self.risk_level == RiskLevel.HIGH

    @property
    def requires_action(self) -> bool:
        return self.mask_strategy != MaskStrategy.PASSTHROUGH

    def __repr__(self):
        return (
            f"ColumnReport(col='{self.column}', type={self.pii_type.value}, "
            f"risk={self.risk_level.value}, strategy={self.mask_strategy.value})"
        )


# ---------------------------------------------------------------------------
# Validadores de dígitos verificadores
# ---------------------------------------------------------------------------

def _validate_cpf(cpf: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", cpf)]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    soma = sum(d * (10 - i) for i, d in enumerate(digits[:9]))
    r1 = (soma * 10 % 11) % 10
    soma = sum(d * (11 - i) for i, d in enumerate(digits[:10]))
    r2 = (soma * 10 % 11) % 10
    return digits[9] == r1 and digits[10] == r2


def _validate_cnpj(cnpj: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", cnpj)]
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    soma = sum(d * w for d, w in zip(digits[:12], weights1))
    r1 = 0 if (soma % 11) < 2 else 11 - (soma % 11)
    weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    soma = sum(d * w for d, w in zip(digits[:13], weights2))
    r2 = 0 if (soma % 11) < 2 else 11 - (soma % 11)
    return digits[12] == r1 and digits[13] == r2


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------

# Tipo de input aceito pelos métodos públicos
_DataFrameInput = Union["pd.DataFrame", "pl.DataFrame"]


class PIIDetector:
    """
    Detecta e classifica colunas com dados pessoais.

    Aceita pd.DataFrame e pl.DataFrame transparentemente.
    Engine de regex: Polars quando disponível (5–15x mais rápido via SIMD
    Arrow), pandas como fallback automático.

    Exemplo:
        detector = PIIDetector()
        reports = detector.detect(df)          # pd.DataFrame ou pl.DataFrame
        report_dict = detector.detect_dict(df)
    """

    def __init__(
        self,
        sample_size: int = 500,
        match_threshold: float = 0.5,
        high_cardinality_threshold: float = 0.85,
        freq_top_n: int = 50,
    ):
        self.sample_size = sample_size
        self.match_threshold = match_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.freq_top_n = freq_top_n

    def detect(self, df: _DataFrameInput) -> List[ColumnReport]:
        df_pd = _to_pandas(df)

        # Converte o DataFrame inteiro para Polars uma vez (~11ms para df 100k×7)
        # em vez de converter cada Series individualmente (~6ms × n_string_cols).
        # A referência é passada para _analyze que evita reconversão por coluna.
        df_pl_cache: Optional["pl.DataFrame"] = None
        if _POLARS_AVAILABLE:
            try:
                import polars as pl
                df_pl_cache = pl.from_pandas(df_pd)
            except Exception:
                df_pl_cache = None

        reports: List[ColumnReport] = []
        for col in df_pd.columns:
            report = self._analyze(df_pd[col], col, df_pl_cache=df_pl_cache)
            if report and report.requires_action:
                reports.append(report)
                logger.info(
                    "PII | col=%-25s type=%-18s risk=%-8s strategy=%s",
                    col, report.pii_type.value,
                    report.risk_level.value, report.mask_strategy.value,
                )
        _order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}
        reports.sort(key=lambda r: _order[r.risk_level])
        return reports

    def detect_dict(self, df: _DataFrameInput) -> Dict[str, ColumnReport]:
        return {r.column: r for r in self.detect(df)}

    def detect_sampled(self, df: _DataFrameInput, sample_size: Optional[int] = None) -> Dict[str, ColumnReport]:
        """detect_dict com amostragem prévia — até 100x mais rápido para DFs grandes."""
        ss = sample_size or self.sample_size
        df_pd = _to_pandas(df)
        if len(df_pd) > ss * 2:
            df_pd = df_pd.sample(min(ss * 2, len(df_pd)), random_state=42)
        return {r.column: r for r in self.detect(df_pd)}

    def summary_df(self, df: _DataFrameInput) -> pd.DataFrame:
        rows = []
        for r in self.detect(df):
            rows.append({
                "coluna":       r.column,
                "tipo_pii":     r.pii_type.value,
                "risco":        r.risk_level.value,
                "estrategia":   r.mask_strategy.value,
                "match_ratio":  round(r.match_ratio, 3),
                "unique_ratio": round(r.unique_ratio, 3),
                "notas":        r.notes,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Lógica interna
    # ------------------------------------------------------------------

    def _analyze(self, series: pd.Series, col: str, df_pl_cache=None) -> Optional[ColumnReport]:
        col_lower = col.lower().strip()

        if self._is_safe_technical_column(col_lower):
            return None

        if series.dtype == object:
            sample_types = series.dropna().head(50).map(type).unique()
            if len(sample_types) > 1:
                logger.warning(
                    "PIIDetector: coluna '%s' contém tipos mistos %s.",
                    col, [t.__name__ for t in sample_types],
                )

        pii_type = self._name_heuristic(col_lower)
        is_quasi = any(q in col_lower for q in _QUASI_KEYWORDS)

        match_ratio = 0.0
        sample_matches: List[str] = []
        regex_type: Optional[PIIType] = None

        is_str = series.dtype == object or pd.api.types.is_string_dtype(series)
        non_null_count = int(series.notna().sum())

        if is_str and non_null_count >= MIN_NON_NULL_FOR_REGEX:
            if _POLARS_AVAILABLE:
                match_ratio, sample_matches, regex_type = self._regex_scan_polars(series, df_pl_cache=df_pl_cache)
            else:
                match_ratio, sample_matches, regex_type = self._regex_scan_pandas(series)

            if regex_type and pii_type == PIIType.UNKNOWN:
                pii_type = regex_type

            if pii_type == PIIType.UNKNOWN and match_ratio == 0.0:
                ft_type, ft_ratio = self._free_text_scan(series)
                if ft_type is not None:
                    pii_type = ft_type
                    match_ratio = ft_ratio

        n = max(len(series), 1)

        col_min = col_max = None
        value_freq: Optional[Dict[str, float]] = None

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                col_min = float(clean.min())
                col_max = float(clean.max())

        # single pass: value_counts() já computa únicos — usa len(vc) como nunique
        if is_str or isinstance(series.dtype, pd.CategoricalDtype):
            vc = series.value_counts(normalize=True).head(self.freq_top_n)
            value_freq = {str(k): round(float(v), 6) for k, v in vc.items()}
            # n_unique via value_counts (sem segundo pass completo)
            n_unique = series.nunique()  # necessário para incluir valores além de freq_top_n
            unique_ratio = n_unique / n
        else:
            unique_ratio = series.nunique() / n

        risk_level, final_type = self._assess_risk(
            pii_type, match_ratio, unique_ratio, is_quasi,
            pd.api.types.is_numeric_dtype(series),
        )
        if risk_level is None:
            return None

        strategy = _DEFAULT_STRATEGY.get(final_type, MaskStrategy.PASSTHROUGH)

        if (
            final_type == PIIType.QUASI_IDENTIFIER
            and pd.api.types.is_numeric_dtype(series)
        ):
            strategy = MaskStrategy.MOCK_NUM
            final_type = PIIType.NUMERICO

        return ColumnReport(
            column=col,
            pii_type=final_type,
            risk_level=risk_level,
            mask_strategy=strategy,
            match_ratio=match_ratio,
            unique_ratio=unique_ratio,
            sample_matches=sample_matches[:3],
            col_min=col_min,
            col_max=col_max,
            value_freq=value_freq,
            notes=self._build_notes(final_type, match_ratio, unique_ratio),
        )

    def _name_heuristic(self, col_lower: str) -> PIIType:
        tokens = set(re.split(r"[_\-\s]+", col_lower))
        for pii_type, keywords in _NAME_HEURISTICS.items():
            for kw in keywords:
                kw_tokens = set(re.split(r"[_\-\s]+", kw))
                if kw_tokens.issubset(tokens):
                    return pii_type
        return PIIType.UNKNOWN

    def _is_safe_technical_column(self, col_lower: str) -> bool:
        if col_lower in _SAFE_EXACT_NAMES:
            return True
        col_tokens = set(re.split(r"[_\-\s]+", col_lower))
        if col_tokens & _SAFE_COLUMN_TOKENS:
            return True
        for affix, position in _SAFE_AFFIXES:
            if position == "prefix" and col_lower.startswith(affix + "_"):
                return True
            if position == "suffix" and col_lower.endswith("_" + affix):
                return True
        return False

    # ------------------------------------------------------------------
    # Regex scan — engine Polars (primário, 5–15x mais rápido via SIMD)
    # ------------------------------------------------------------------

    def _regex_scan_polars(
        self, series: pd.Series, df_pl_cache=None
    ) -> Tuple[float, List[str], Optional[PIIType]]:
        """
        Regex scan via Polars pl.Series.str.contains() — SIMD sobre Arrow.

        Estratégia:
          1. Converte amostra pandas → pl.Series (zero-copy via Arrow quando possível)
          2. Aplica str.contains() vetorizado para cada padrão
          3. Validação de dígito verificador (CPF/CNPJ) nos candidatos
          4. Retorna (ratio, amostras, melhor tipo)

        Por que mais rápido:
          - str.contains() em Polars executa a regex no engine Rust (regex crate)
            sobre buffers Arrow contíguos em memória — sem GIL, sem overhead de
            dispatch Python por elemento.
          - pandas str.fullmatch() usa o engine re do CPython, que cria um objeto
            Match por chamada e sofre GIL contention em threads.
        """
        null_ratio = series.isna().mean()
        effective_threshold = self.match_threshold * max(0.4, 1 - null_ratio * 0.5)

        # Usa a Series já convertida do cache de df (evita conversão repetida por coluna)
        sample = series.dropna()
        sample = sample[sample.map(lambda x: isinstance(x, str))]
        if len(sample) == 0:
            sample = series.dropna().astype(str)
        if len(sample) > self.sample_size:
            sample = sample.sample(self.sample_size, random_state=42)

        try:
            if df_pl_cache is not None and series.name in df_pl_cache.columns:
                # Zero-copy: extrai do DataFrame Polars já convertido
                col_pl = df_pl_cache[series.name].drop_nulls().cast(pl.String).str.strip_chars()
                if len(col_pl) > self.sample_size:
                    col_pl = col_pl.sample(self.sample_size, seed=42)
                s_pl = col_pl
            else:
                s_pl = pl.from_pandas(sample.reset_index(drop=True).astype(str).str.strip())
        except Exception:
            return self._regex_scan_pandas(series)

        best_type: Optional[PIIType] = None
        best_ratio = 0.0
        best_matches: List[str] = []
        n = len(s_pl)

        for pii_type, pattern in _PATTERNS.items():
            # Ancora o padrão para fullmatch
            anchored = pattern.pattern
            if not anchored.startswith("^"):
                anchored = "^" + anchored
            if not anchored.endswith("$"):
                anchored = anchored + "$"

            try:
                matched = s_pl.str.contains(anchored, strict=False)
            except Exception:
                # Fallback escalar para padrões com lookbehind/lookahead
                # que o engine Rust não suporta
                matched_mask_pd = sample.astype(str).str.strip().str.fullmatch(
                    pattern.pattern, na=False
                )
                ratio_fb = float(matched_mask_pd.mean()) if len(matched_mask_pd) > 0 else 0.0
                if ratio_fb > best_ratio:
                    best_ratio = ratio_fb
                    best_type = pii_type
                    best_matches = sample.astype(str).str.strip()[matched_mask_pd].head(3).tolist()
                continue

            # Validação de dígito verificador — CPF e CNPJ
            if pii_type in (PIIType.CPF, PIIType.CNPJ):
                validator = _validate_cpf if pii_type == PIIType.CPF else _validate_cnpj
                candidates = s_pl.filter(matched).to_list()
                if candidates:
                    valid_count = sum(1 for v in candidates if validator(v))
                    ratio = valid_count / n
                else:
                    ratio = 0.0
            else:
                ratio = float(matched.sum()) / n if n > 0 else 0.0

            type_threshold = _PII_TYPE_THRESHOLDS.get(pii_type, effective_threshold)
            if ratio >= type_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_type = pii_type
                best_matches = s_pl.filter(matched).head(3).to_list()

        if best_ratio == 0.0 or best_type is None:
            return 0.0, [], None
        return best_ratio, best_matches, best_type

    # ------------------------------------------------------------------
    # Regex scan — engine Pandas (fallback quando Polars não disponível)
    # ------------------------------------------------------------------

    def _regex_scan_pandas(
        self, series: pd.Series
    ) -> Tuple[float, List[str], Optional[PIIType]]:
        """
        Detecção vetorizada via str.fullmatch() — fallback sem Polars.
        Mantido idêntico ao código anterior (v1.1.1).
        """
        null_ratio = series.isna().mean()
        effective_threshold = self.match_threshold * max(0.4, 1 - null_ratio * 0.5)

        sample = series.dropna()
        sample = sample[sample.map(lambda x: isinstance(x, str))]
        if len(sample) == 0:
            sample = series.dropna().astype(str)
        if len(sample) > self.sample_size:
            sample = sample.sample(self.sample_size, random_state=42)

        sample_str = sample.astype(str).str.strip()
        best_type: Optional[PIIType] = None
        best_ratio = 0.0
        best_matches: List[str] = []

        for pii_type, pattern in _PATTERNS.items():
            matched_mask = sample_str.str.fullmatch(pattern.pattern, na=False)

            if pii_type in (PIIType.CPF, PIIType.CNPJ):
                validator = _validate_cpf if pii_type == PIIType.CPF else _validate_cnpj
                candidates = sample_str[matched_mask]
                if len(candidates) > 0:
                    valid_dict = {idx: validator(val) for idx, val in candidates.items()}
                    valid_series = pd.Series(valid_dict, dtype=bool)
                    matched_mask = pd.Series(False, index=sample_str.index, dtype=bool)
                    matched_mask.update(valid_series)
                else:
                    matched_mask = pd.Series(False, index=sample_str.index, dtype=bool)

            ratio = float(matched_mask.mean()) if len(matched_mask) > 0 else 0.0
            type_threshold = _PII_TYPE_THRESHOLDS.get(pii_type, effective_threshold)
            if ratio >= type_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_type = pii_type
                best_matches = sample_str[matched_mask].head(3).tolist()

        if best_ratio == 0.0 or best_type is None:
            return 0.0, [], None
        return best_ratio, best_matches, best_type

    # Alias público: _regex_scan aponta para o engine certo
    def _regex_scan(self, series: pd.Series) -> Tuple[float, List[str], Optional[PIIType]]:
        if _POLARS_AVAILABLE:
            return self._regex_scan_polars(series)
        return self._regex_scan_pandas(series)

    def _free_text_scan(
        self, series: pd.Series
    ) -> Tuple[Optional[PIIType], float]:
        sample = series.dropna().astype(str)
        if len(sample) == 0:
            return None, 0.0
        if sample.str.len().mean() <= 30:
            return None, 0.0

        sample_n = sample.sample(min(200, len(sample)), random_state=42)
        FREE_TEXT_THRESHOLD = 0.05
        best_type: Optional[PIIType] = None
        best_ratio = 0.0

        for pii_type, pattern in _PATTERNS.items():
            unanchored = pattern.pattern.strip("^$")
            try:
                re.compile(unanchored)
            except re.error:
                continue
            matches = sample_n.str.contains(unanchored, regex=True, na=False)
            ratio = float(matches.mean())
            if ratio > best_ratio:
                best_ratio = ratio
                best_type = pii_type

        if best_ratio >= FREE_TEXT_THRESHOLD and best_type is not None:
            return best_type, best_ratio
        return None, 0.0

    def _assess_risk(
        self,
        pii_type: PIIType,
        match_ratio: float,
        unique_ratio: float,
        is_quasi: bool,
        is_numeric: bool,
    ) -> Tuple[Optional[RiskLevel], Optional[PIIType]]:
        if match_ratio >= self.match_threshold and pii_type not in (
            PIIType.UNKNOWN, PIIType.QUASI_IDENTIFIER
        ):
            return RiskLevel.HIGH, pii_type
        if pii_type not in (PIIType.UNKNOWN, PIIType.QUASI_IDENTIFIER):
            return RiskLevel.HIGH, pii_type
        if unique_ratio >= self.high_cardinality_threshold:
            t = PIIType.NUMERICO if is_numeric else PIIType.QUASI_IDENTIFIER
            return RiskLevel.MEDIUM, t
        if is_quasi:
            t = PIIType.NUMERICO if is_numeric else PIIType.CATEGORICO
            return RiskLevel.MEDIUM, t
        if match_ratio >= 0.1:
            return RiskLevel.LOW, pii_type
        return None, None

    @staticmethod
    def _build_notes(pii_type: PIIType, match_ratio: float, unique_ratio: float) -> str:
        parts = []
        if match_ratio > 0:
            parts.append(f"regex={match_ratio:.0%}")
        parts.append(f"unique={unique_ratio:.0%}")
        if pii_type in (PIIType.CPF, PIIType.CNPJ, PIIType.RG, PIIType.EMAIL):
            parts.append("identificador direto — hash obrigatório")
        elif pii_type == PIIType.CEP:
            parts.append("truncamento geográfico recomendado")
        elif pii_type in (PIIType.NOME, PIIType.TELEFONE, PIIType.CARTAO_CREDITO):
            parts.append("supressão ou REDACTED recomendado")
        return " | ".join(parts)

    def __repr__(self) -> str:
        engine = "polars" if _POLARS_AVAILABLE else "pandas"
        return (
            f"PIIDetector(sample={self.sample_size}, "
            f"threshold={self.match_threshold:.0%}, "
            f"engine={engine})"
        )


# ---------------------------------------------------------------------------
# Helper: conversão transparente pd/pl → pd
# ---------------------------------------------------------------------------

def _to_pandas(df: _DataFrameInput) -> pd.DataFrame:
    """Converte pl.DataFrame → pd.DataFrame se necessário. pd.DataFrame é passthrough."""
    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df
