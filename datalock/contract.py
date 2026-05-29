"""
datalock/contract.py
=================
dd.contract() — Contrato de dados: tipos + PII + mascaramento + validação
em uma única declaração versionável.

Problema resolvido:
  Hoje validate() faz qualidade, scan() faz PII, mask() mascara — três
  chamadas separadas, sem fonte única de verdade. Em pipelines de produção
  isso leva a inconsistências: a regra de validação diz "not_null" mas o
  mascaramento suprime o campo; ou detecta CPF mas o validate não sabe.

dd.contract() unifica as três em uma declaração JSON-exportável que pode
ser versionada junto com os dados (commitada, comparada, evoluída).

Uso:
    contrato = dd.contract({
        "cpf":          {"type": "str",   "not_null": True, "unique": True,
                         "pii": "CPF",    "mask": "hash"},
        "renda_mensal": {"type": "float", "min": 0, "max": 500_000,
                         "pii": "numerico", "mask": "mock_numeric"},
        "uf":           {"type": "str",   "in": LISTA_UFS, "pii": None},
        "data_nasc":    {"type": "str",   "not_null": True,
                         "pii": "data_nascimento", "mask": "generalize_date"},
    })

    # Valida dados recebidos
    val = contrato.validate(df)
    val.print_report()
    val.raise_if_failed()

    # Mascara de acordo com o contrato
    df_safe = contrato.mask(df, salt=SALT)

    # Aplica tudo em sequência: valida → mascara → retorna
    result = contrato.apply(df, salt=SALT)

    # Exporta como JSON Schema (para documentação / DPO)
    schema_json = contrato.to_json_schema()

    # Detecta breaking changes entre versões
    diff = contrato.diff(contrato_v2)
    if diff.has_breaking_changes:
        print(diff.report())

    # Persiste e recarrega
    contrato.save("schema/clientes_v1.contract.json")
    contrato2 = DataContract.load("schema/clientes_v1.contract.json")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapa: nome da estratégia → MaskStrategy
# ---------------------------------------------------------------------------

_MASK_STRATEGY_MAP = {
    "hash":            "HASH",
    "redact":          "REDACT",
    "truncate":        "TRUNCATE",
    "generalize_date": "GENERALIZE_DATE",
    "mask_phone_ddd":  "MASK_PHONE_DDD",
    "mock_numeric":    "MOCK_NUM",
    "mock_category":   "MOCK_CAT",
    "suppress":        "SUPPRESS",
    "passthrough":     "PASSTHROUGH",
}

_DTYPE_ALIASES = {
    "str": "str", "string": "str", "text": "str", "varchar": "str", "object": "str",
    "int": "int", "integer": "int", "int64": "int", "int32": "int",
    "float": "float", "float64": "float", "double": "float", "decimal": "float",
    "bool": "bool", "boolean": "bool",
    "date": "date", "datetime": "datetime", "timestamp": "datetime",
}


# ---------------------------------------------------------------------------
# FieldSpec — especificação de uma coluna
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """Especificação completa de uma coluna no contrato."""
    # Tipo
    type:     Optional[str] = None      # str, int, float, bool, date, datetime
    nullable: bool          = True      # se False → equivale a not_null=True

    # Validação de qualidade
    not_null:   bool         = False
    unique:     bool         = False
    min:        Optional[Any]= None
    max:        Optional[Any]= None
    min_length: Optional[int]= None
    max_length: Optional[int]= None
    in_values:  Optional[List] = None   # campo "in" do validate()
    not_in:     Optional[List] = None
    matches:    Optional[str]  = None   # regex
    contains:   Optional[str]  = None

    # PII e mascaramento
    pii:        Optional[str]  = None   # None = não é PII; "CPF", "email", etc.
    mask:       Optional[str]  = None   # estratégia de mascaramento

    # Metadados
    description: str = ""
    tags:        List[str] = field(default_factory=list)
    examples:    List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> "FieldSpec":
        """Constrói FieldSpec a partir do dict do usuário."""
        fs = cls()
        fs.type       = _DTYPE_ALIASES.get(str(d.get("type","")).lower(), d.get("type"))
        fs.nullable   = not d.get("not_null", False)
        fs.not_null   = bool(d.get("not_null", False))
        fs.unique     = bool(d.get("unique", False))
        fs.min        = d.get("min")
        fs.max        = d.get("max")
        fs.min_length = d.get("min_length")
        fs.max_length = d.get("max_length")
        fs.in_values  = d.get("in")
        fs.not_in     = d.get("not_in")
        fs.matches    = d.get("matches")
        fs.contains   = d.get("contains")
        fs.pii        = d.get("pii")
        fs.mask       = d.get("mask")
        fs.description= d.get("description", "")
        fs.tags       = d.get("tags", [])
        fs.examples   = d.get("examples", [])
        return fs

    def to_validation_rules(self) -> Dict[str, Any]:
        """Converte para dict de regras compatível com dd.validate()."""
        rules: Dict[str, Any] = {}
        if self.not_null:          rules["not_null"]   = True
        if self.unique:            rules["unique"]     = True
        if self.min is not None:   rules["min"]        = self.min
        if self.max is not None:   rules["max"]        = self.max
        if self.min_length:        rules["min_length"] = self.min_length
        if self.max_length:        rules["max_length"] = self.max_length
        if self.in_values:         rules["in"]         = self.in_values
        if self.not_in:            rules["not_in"]     = self.not_in
        if self.matches:           rules["matches"]    = self.matches
        if self.contains:          rules["contains"]   = self.contains
        if self.type:              rules["dtype"]      = self.type
        return rules

    def to_dict(self) -> Dict:
        d = {}
        if self.type:              d["type"]        = self.type
        if not self.nullable:      d["not_null"]    = True
        if self.not_null:          d["not_null"]    = True
        if self.unique:            d["unique"]      = True
        if self.min is not None:   d["min"]         = self.min
        if self.max is not None:   d["max"]         = self.max
        if self.min_length:        d["min_length"]  = self.min_length
        if self.max_length:        d["max_length"]  = self.max_length
        if self.in_values:         d["in"]          = self.in_values
        if self.not_in:            d["not_in"]      = self.not_in
        if self.matches:           d["matches"]     = self.matches
        if self.contains:          d["contains"]    = self.contains
        if self.pii is not None:   d["pii"]         = self.pii
        if self.mask:              d["mask"]        = self.mask
        if self.description:       d["description"] = self.description
        if self.tags:              d["tags"]        = self.tags
        if self.examples:          d["examples"]    = self.examples
        return d


# ---------------------------------------------------------------------------
# ContractDiff — diferenças entre duas versões
# ---------------------------------------------------------------------------

@dataclass
class ContractDiff:
    """Diferenças entre duas versões de um contrato."""
    added_columns:          List[str] = field(default_factory=list)
    removed_columns:        List[str] = field(default_factory=list)
    type_changes:           Dict[str, tuple] = field(default_factory=dict)
    pii_changes:            Dict[str, tuple] = field(default_factory=dict)
    mask_changes:           Dict[str, tuple] = field(default_factory=dict)
    nullability_changes:    Dict[str, tuple] = field(default_factory=dict)
    validation_changes:     Dict[str, str]   = field(default_factory=dict)

    @property
    def has_breaking_changes(self) -> bool:
        """
        Breaking changes = removed columns, type narrowing, nullable→not_null,
        or PII type change (could affect masking in existing pipelines).
        """
        return bool(
            self.removed_columns
            or self.type_changes
            or self.nullability_changes
            or self.pii_changes
        )

    def report(self) -> str:
        lines = [f"Contract Diff — {'BREAKING' if self.has_breaking_changes else 'non-breaking'}\n"]
        if self.added_columns:
            lines.append(f"  + Colunas adicionadas: {self.added_columns}")
        if self.removed_columns:
            lines.append(f"  - Colunas removidas:   {self.removed_columns}  ← BREAKING")
        for col, (old, new) in self.type_changes.items():
            lines.append(f"  ~ Tipo de '{col}': {old} → {new}  ← BREAKING")
        for col, (old, new) in self.pii_changes.items():
            lines.append(f"  ~ PII de '{col}': {old!r} → {new!r}  ← BREAKING")
        for col, (old, new) in self.nullability_changes.items():
            lines.append(f"  ~ Nullable '{col}': {old} → {new}  ← BREAKING")
        for col, (old, new) in self.mask_changes.items():
            lines.append(f"  ~ Máscara '{col}': {old!r} → {new!r}")
        for col, desc in self.validation_changes.items():
            lines.append(f"  ~ Validação '{col}': {desc}")
        if len(lines) == 1:
            lines.append("  Sem diferenças.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ContractApplyResult — resultado de contrato.apply()
# ---------------------------------------------------------------------------

@dataclass
class ContractApplyResult:
    """Resultado de DataContract.apply()."""
    df:          Any     = None   # DataFrame mascarado
    validation:  Any     = None   # ValidationReport
    pii_columns: List[str] = field(default_factory=list)
    masked_columns: List[str] = field(default_factory=list)
    passed:      bool    = True
    elapsed_ms:  float   = 0.0

    def raise_if_failed(self) -> None:
        if self.validation:
            self.validation.raise_if_failed()


# ---------------------------------------------------------------------------
# DataContract — objeto principal
# ---------------------------------------------------------------------------

class DataContract:
    """
    Contrato de dados: tipos + PII + mascaramento + validação em uma declaração.

    Args:
        fields:  Dict[nome_coluna → dict de regras]. Exemplo:
                 {"cpf": {"type":"str","not_null":True,"pii":"CPF","mask":"hash"}}
        name:    Nome do contrato (para documentação e diff).
        version: Versão semântica do contrato.
        description: Descrição livre.

    Exemplos:
        contrato = dd.contract({
            "cpf":    {"type": "str",   "not_null": True, "unique": True,
                       "pii": "CPF",    "mask": "hash"},
            "renda":  {"type": "float", "min": 0, "max": 500_000,
                       "pii": "numerico", "mask": "mock_numeric"},
            "uf":     {"type": "str",   "in": ["SP","RJ","MG"]},
        })

        result = contrato.apply(df, salt=SALT)
        result.validation.print_report()
        contrato.save("clientes_v1.contract.json")
    """

    def __init__(
        self,
        fields: Dict[str, Union[Dict, FieldSpec]],
        name:        str = "contrato",
        version:     str = "1.0",
        description: str = "",
    ) -> None:
        self.name        = name
        self.version     = version
        self.description = description
        self.created_at  = datetime.now(timezone.utc).isoformat()

        self._fields: Dict[str, FieldSpec] = {}
        for col, spec in fields.items():
            if isinstance(spec, FieldSpec):
                self._fields[col] = spec
            else:
                self._fields[col] = FieldSpec.from_dict(spec)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: Any,
        *,
        severity: str = "error",
        warn_only: bool = False,
        check_schema: bool = True,
    ) -> "Any":  # ValidationReport
        """
        Valida o DataFrame contra todas as regras do contrato.

        Args:
            df:           pd.DataFrame ou pl.DataFrame.
            severity:     "error" | "warning"
            warn_only:    Se True, emite warnings em vez de falhar.
            check_schema: Se True, verifica colunas esperadas presentes.

        Returns:
            ValidationReport com resultados por coluna e por esquema.
        """
        from datalock.validate import validate as _validate, RuleResult, ValidationReport

        df_pd = _to_pd(df)

        rules = {col: spec.to_validation_rules()
                 for col, spec in self._fields.items()
                 if spec.to_validation_rules()}

        # Schema-level: todas as colunas esperadas existem?
        extra_results = []
        if check_schema:
            expected = set(self._fields)
            present  = set(df_pd.columns)
            missing_cols = expected - present
            extra_cols   = present - expected

            for col in missing_cols:
                extra_results.append(RuleResult(
                    column=col, rule="column_present", passed=False,
                    detail=f"Coluna esperada pelo contrato não encontrada no DataFrame.",
                    severity=severity,
                ))
            if extra_cols:
                extra_results.append(RuleResult(
                    column="__schema__", rule="no_extra_columns", passed=True,
                    detail=f"Colunas extras (não no contrato): {sorted(extra_cols)}",
                    severity="warning",
                ))

        report = _validate(df, rules, severity=severity, warn_only=warn_only)

        # Injeta os resultados de schema no report
        all_rules = extra_results + report.rules
        n_passed  = sum(1 for r in all_rules if r.passed)
        n_failed  = sum(1 for r in all_rules if not r.passed and r.severity == "error")
        n_warn    = sum(1 for r in all_rules if not r.passed and r.severity == "warning")
        score     = n_passed / max(len(all_rules), 1)
        bad_cols  = sorted({r.column for r in all_rules if not r.passed})

        return ValidationReport(
            passed=n_failed == 0,
            score=score,
            n_rules=len(all_rules),
            n_passed=n_passed,
            n_failed=n_failed,
            n_warnings=n_warn,
            rules=all_rules,
            shape=list(df_pd.shape),
            columns_with_issues=bad_cols,
        )

    # ------------------------------------------------------------------
    # Mascaramento
    # ------------------------------------------------------------------

    def mask(self, df: Any, *, salt: str) -> Any:
        """
        Mascara as colunas PII conforme as estratégias definidas no contrato.

        Diferença de dd.mask(): usa as estratégias EXPLÍCITAS do contrato em vez
        de auto-detectar. Isso garante que a mesma coluna seja sempre mascarada
        do mesmo jeito independentemente dos dados.

        Args:
            df:   pd.DataFrame ou pl.DataFrame.
            salt: Salt HMAC.

        Returns:
            DataFrame mascarado, mesmo tipo do input.
        """
        from datalock.detectors.pii_detector import (
            ColumnReport, PIIType, RiskLevel, MaskStrategy,
        )

        pii_cols = {col: spec for col, spec in self._fields.items()
                    if spec.pii and spec.mask}
        if not pii_cols:
            return df

        # Constrói reports artificiais a partir do contrato (sem auto-detecção)
        reports = {}
        for col, spec in pii_cols.items():
            pii_str      = spec.pii.upper()
            mask_str     = _MASK_STRATEGY_MAP.get(spec.mask.lower(), "HASH")

            try:
                pii_type = PIIType[pii_str]
            except KeyError:
                pii_type = PIIType.NUMERICO if "numer" in pii_str.lower() else PIIType.CATEGORICO

            try:
                mask_strategy = MaskStrategy[mask_str]
            except KeyError:
                mask_strategy = MaskStrategy.HASH

            risk = (RiskLevel.HIGH   if pii_type in (PIIType.CPF, PIIType.CNPJ, PIIType.EMAIL,
                                                       PIIType.TELEFONE, PIIType.RG,
                                                       PIIType.CARTAO_CREDITO)
                    else RiskLevel.MEDIUM if pii_type in (PIIType.NOME, PIIType.DATA_NASCIMENTO)
                    else RiskLevel.LOW)

            reports[col] = ColumnReport(
                column=col, pii_type=pii_type, risk_level=risk,
                mask_strategy=mask_strategy, match_ratio=1.0, unique_ratio=0.5,
            )

        from datalock.core import mask_frame as _mask_frame
        return _mask_frame(df, salt=salt, columns=list(reports.keys()))

    # ------------------------------------------------------------------
    # apply() — valida + mascara + retorna result
    # ------------------------------------------------------------------

    def apply(
        self,
        df: Any,
        *,
        salt: str,
        strict: bool = True,
        verbose: bool = False,
    ) -> ContractApplyResult:
        """
        Aplica o contrato completo: valida → mascara → retorna resultado.

        Args:
            df:      pd.DataFrame ou pl.DataFrame.
            salt:    Salt HMAC para mascaramento.
            strict:  Se True, levanta ValueError se validação falhar.
            verbose: Imprime relatório de validação.

        Returns:
            ContractApplyResult com df mascarado, ValidationReport e métricas.
        """
        import time
        t0 = time.perf_counter()

        validation = self.validate(df)
        if verbose:
            validation.print_report()
        if strict:
            validation.raise_if_failed()

        df_safe = self.mask(df, salt=salt)

        pii_cols    = [c for c, s in self._fields.items() if s.pii]
        masked_cols = [c for c, s in self._fields.items() if s.pii and s.mask]

        return ContractApplyResult(
            df=df_safe,
            validation=validation,
            pii_columns=pii_cols,
            masked_columns=masked_cols,
            passed=validation.passed,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serializa o contrato como dict (JSON-serializable)."""
        return {
            "name":        self.name,
            "version":     self.version,
            "description": self.description,
            "created_at":  self.created_at,
            "generator":   "datalock",
            "fields":      {col: spec.to_dict() for col, spec in self._fields.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializa como JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_json_schema(self) -> Dict:
        """
        Exporta como JSON Schema (draft-07) para documentação e integração.

        Útil para compartilhar com equipes não-Python, gerar documentação
        automática ou validar dados em outros sistemas.
        """
        _type_map = {
            "str": "string", "int": "integer", "float": "number",
            "bool": "boolean", "date": "string", "datetime": "string",
        }
        props = {}
        required = []

        for col, spec in self._fields.items():
            prop: Dict[str, Any] = {}
            if spec.type:
                prop["type"] = _type_map.get(spec.type, "string")
                if spec.type in ("date",):
                    prop["format"] = "date"
                elif spec.type == "datetime":
                    prop["format"] = "date-time"
            if spec.min is not None:
                prop["minimum"] = spec.min
            if spec.max is not None:
                prop["maximum"] = spec.max
            if spec.min_length:
                prop["minLength"] = spec.min_length
            if spec.max_length:
                prop["maxLength"] = spec.max_length
            if spec.in_values:
                prop["enum"] = spec.in_values
            if spec.matches:
                prop["pattern"] = spec.matches
            if spec.description:
                prop["description"] = spec.description
            if spec.examples:
                prop["examples"] = spec.examples
            if spec.pii:
                prop["x-pii-type"]      = spec.pii
                prop["x-mask-strategy"] = spec.mask
            props[col] = prop
            if spec.not_null:
                required.append(col)

        return {
            "$schema":    "http://json-schema.org/draft-07/schema#",
            "title":      self.name,
            "version":    self.version,
            "description":self.description,
            "type":       "object",
            "properties": props,
            "required":   required,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Salva o contrato em arquivo JSON."""
        p = Path(str(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        logger.info("DataContract salvo: %s", p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataContract":
        """Carrega um contrato salvo com .save()."""
        p = Path(str(path))
        data = json.loads(p.read_text(encoding="utf-8"))
        fields = {col: FieldSpec.from_dict(spec) for col, spec in data["fields"].items()}
        c = cls(fields, name=data.get("name","contrato"),
                version=data.get("version","1.0"),
                description=data.get("description",""))
        c.created_at = data.get("created_at", c.created_at)
        return c

    # ------------------------------------------------------------------
    # Diff entre versões
    # ------------------------------------------------------------------

    def diff(self, other: "DataContract") -> ContractDiff:
        """
        Compara este contrato com outro e retorna as diferenças.

        Useful for detecting breaking changes when updating schema versions.

        Args:
            other: Outra instância de DataContract para comparar.

        Returns:
            ContractDiff com todas as mudanças identificadas.
        """
        d = ContractDiff()
        self_cols  = set(self._fields)
        other_cols = set(other._fields)

        d.added_columns   = sorted(other_cols - self_cols)
        d.removed_columns = sorted(self_cols  - other_cols)

        for col in self_cols & other_cols:
            s1 = self._fields[col]
            s2 = other._fields[col]

            if s1.type != s2.type and s1.type and s2.type:
                d.type_changes[col] = (s1.type, s2.type)
            if s1.pii != s2.pii:
                d.pii_changes[col] = (s1.pii, s2.pii)
            if s1.mask != s2.mask:
                d.mask_changes[col] = (s1.mask, s2.mask)
            if s1.not_null != s2.not_null:
                d.nullability_changes[col] = (s1.not_null, s2.not_null)

            # Validation rule changes (textual description)
            r1 = s1.to_validation_rules()
            r2 = s2.to_validation_rules()
            changes = []
            for k in set(r1) | set(r2):
                if r1.get(k) != r2.get(k):
                    changes.append(f"{k}: {r1.get(k)!r} → {r2.get(k)!r}")
            if changes:
                d.validation_changes[col] = ", ".join(changes)

        return d

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def columns(self) -> List[str]:
        return list(self._fields)

    @property
    def pii_columns(self) -> List[str]:
        return [c for c, s in self._fields.items() if s.pii]

    @property
    def required_columns(self) -> List[str]:
        return [c for c, s in self._fields.items() if s.not_null]

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, col: str) -> bool:
        return col in self._fields

    def __repr__(self) -> str:
        return (f"DataContract(name={self.name!r}, version={self.version!r}, "
                f"fields={len(self._fields)}, pii={len(self.pii_columns)})")


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def contract(
    fields: Dict[str, Union[Dict, FieldSpec]],
    *,
    name:        str = "contrato",
    version:     str = "1.0",
    description: str = "",
) -> DataContract:
    """
    Cria um DataContract — contrato de dados com tipos, validação e PII.

    Args:
        fields:      Dict[nome_coluna → regras]. Regras disponíveis:
                       type:        str | int | float | bool | date | datetime
                       not_null:    True → proíbe nulos
                       unique:      True → proíbe duplicatas
                       min / max:   limites numéricos
                       in:          lista de valores permitidos
                       matches:     regex
                       contains:    substring obrigatória
                       pii:         tipo PII (CPF, email, CNPJ, etc.) ou None
                       mask:        estratégia de mascaramento
                       description: texto livre
        name:        Nome do contrato (para export e diff).
        version:     Versão semântica.
        description: Descrição longa.

    Returns:
        DataContract com métodos validate(), mask(), apply(), diff(),
        to_json_schema(), save(), load().

    Exemplos:
        contrato = dd.contract({
            "cpf":    {"type": "str",   "not_null": True, "unique": True,
                       "pii": "CPF",    "mask": "hash"},
            "renda":  {"type": "float", "min": 0, "max": 500_000,
                       "pii": "numerico", "mask": "mock_numeric"},
            "uf":     {"type": "str",   "in": ["SP","RJ","MG","RS","BA"]},
            "email":  {"type": "str",   "contains": "@",
                       "pii": "email",  "mask": "hash"},
        }, name="clientes_v2", version="2.0")

        result = contrato.apply(df, salt=SALT)
        result.raise_if_failed()

        contrato.save("schema/clientes_v2.contract.json")
        schema = contrato.to_json_schema()
    """
    return DataContract(fields, name=name, version=version, description=description)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pd(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pl.LazyFrame):
        return df.collect().to_pandas()
    return pd.DataFrame(df)
