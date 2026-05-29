"""
logus/validate.py
=================
Data Quality integrado — lg.validate() e lg.expect()

Complementa privacidade com qualidade: "dado ruim + privacidade ruim = desastre regulatório".

Filosofia:
  - Regras declarativas (sem subclasses)
  - Integração nativa com PII: valida ANTES de mascarar
  - Resultados JSON-serializable
  - Inspirado em Great Expectations, mas sem o overhead

Uso:
    result = lg.validate(df, {
        "cpf":         {"not_null": True, "unique": True, "matches": r"^\\d{11}$"},
        "email":       {"not_null": True, "contains": "@"},
        "renda_mensal":{"min": 0, "max": 500_000, "not_null": True},
        "uf":          {"in": ["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE"]},
        "data_nasc":   {"not_null": True},
    })
    result.print_report()
    result.passed  # True / False
    result.score   # 0.0 → 1.0

    # Expectations fluentes
    lg.expect(df, "cpf").not_null().unique().matches(r"^\\d{11}$").validate()
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Resultado por regra
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    column:      str
    rule:        str
    passed:      bool
    n_violations: int = 0
    pct_violations: float = 0.0
    detail:      str = ""
    severity:    str = "error"   # "error" | "warning"


@dataclass
class ValidationReport:
    """Resultado completo de uma validação."""
    passed:       bool
    score:        float                  # 0.0 (tudo falhou) → 1.0 (tudo passou)
    n_rules:      int
    n_passed:     int
    n_failed:     int
    n_warnings:   int
    rules:        List[RuleResult]
    shape:        List[int]
    columns_with_issues: List[str]

    def print_report(self) -> None:
        """Imprime relatório formatado."""
        status = "✅ PASSOU" if self.passed else "❌ FALHOU"
        print(f"\nValidação: {status}  |  Score: {self.score:.0%}  "
              f"|  {self.n_passed}/{self.n_rules} regras OK\n")

        if not self.rules:
            print("  Nenhuma regra avaliada.")
            return

        print(f"  {'Coluna':<22} {'Regra':<22} {'Status':>7}  {'Violações':>10}  Detalhe")
        print("  " + "─" * 82)
        for r in sorted(self.rules, key=lambda x: (x.passed, x.column)):
            icon = "✅" if r.passed else ("⚠️" if r.severity == "warning" else "❌")
            pct = f"{r.pct_violations:.1%}" if r.n_violations > 0 else ""
            print(f"  {r.column:<22} {r.rule:<22} {icon}       {r.n_violations:>8,}  "
                  f"{pct:>6}  {r.detail[:40]}")
        print()
        if self.columns_with_issues:
            print(f"  Colunas com problemas: {', '.join(self.columns_with_issues)}")
        print()

    def to_dict(self) -> Dict:
        return {
            "passed":   self.passed,
            "score":    round(self.score, 4),
            "n_rules":  self.n_rules,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "rules":    [
                {
                    "column":         r.column,
                    "rule":           r.rule,
                    "passed":         r.passed,
                    "n_violations":   r.n_violations,
                    "pct_violations": round(r.pct_violations, 4),
                    "detail":         r.detail,
                    "severity":       r.severity,
                }
                for r in self.rules
            ],
        }

    def raise_if_failed(self, severity: str = "error") -> None:
        """Levanta ValueError se qualquer regra com a severidade especificada falhou."""
        failed = [r for r in self.rules if not r.passed and r.severity == severity]
        if failed:
            msgs = [f"{r.column}/{r.rule}: {r.detail}" for r in failed[:5]]
            raise ValueError(
                f"Validação falhou em {len(failed)} regra(s):\n" + "\n".join(msgs)
            )


# ---------------------------------------------------------------------------
# Engine de validação
# ---------------------------------------------------------------------------

def validate(
    df: Any,
    rules: Dict[str, Dict[str, Any]],
    *,
    severity: str = "error",
    warn_only: bool = False,
) -> ValidationReport:
    """
    Valida um DataFrame contra um conjunto de regras declarativas.

    Args:
        df:       pd.DataFrame ou pl.DataFrame.
        rules:    Dict[coluna → Dict[regra → valor]].
                  Regras disponíveis:
                    not_null:   True → proíbe nulos
                    unique:     True → proíbe duplicatas
                    min:        valor mínimo (numérico)
                    max:        valor máximo (numérico)
                    in:         lista de valores permitidos
                    not_in:     lista de valores proibidos
                    matches:    regex que todos os valores devem satisfazer
                    contains:   substring que deve existir nos valores
                    min_length: comprimento mínimo de string
                    max_length: comprimento máximo de string
                    dtype:      tipo esperado ("int","float","str","bool","date")
                    custom:     callable(series) → pd.Series[bool] (True = OK)
        severity: Severidade padrão: "error" | "warning".
        warn_only: Se True, emite Python warning em vez de marcar como falha.

    Returns:
        ValidationReport com resultados por regra.

    Examples:
        Validacao basica:

        >>> import datalock as dd
        >>> import pandas as pd
        >>> df = pd.DataFrame({'cpf': ['11144477735', None], 'renda': [5000.0, -100.0]})
        >>> result = lg.validate(df, {
        ...     'cpf':   {'not_null': True},
        ...     'renda': {'min': 0},
        ... })
        >>> result.passed
        False

    Exemplo completo:
        result = lg.validate(df, {
            "cpf":          {"not_null": True, "unique": True},
            "email":        {"not_null": True, "contains": "@"},
            "renda_mensal": {"min": 0, "max": 500_000, "not_null": True},
            "uf":           {"in": ["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE"]},
        })
        result.print_report()
        result.raise_if_failed()   # levanta ValueError se falhou
    """
    df_pd = _to_pd(df)
    n_rows = len(df_pd)
    results: List[RuleResult] = []

    for col, col_rules in rules.items():
        if col not in df_pd.columns:
            results.append(RuleResult(
                column=col, rule="column_exists", passed=False,
                detail=f"Coluna '{col}' não encontrada no DataFrame.",
                severity=severity,
            ))
            continue

        s = df_pd[col]

        for rule_name, rule_val in col_rules.items():
            r = _eval_rule(s, col, rule_name, rule_val, n_rows, severity)
            if warn_only and not r.passed:
                warnings.warn(
                    f"[logus.validate] {col}/{rule_name}: {r.detail}",
                    UserWarning,
                    stacklevel=3,
                )
                r.severity = "warning"
            results.append(r)

    n_passed = sum(1 for r in results if r.passed)
    n_failed = sum(1 for r in results if not r.passed and r.severity == "error")
    n_warnings = sum(1 for r in results if not r.passed and r.severity == "warning")
    score = n_passed / max(len(results), 1)

    bad_cols = sorted({r.column for r in results if not r.passed})

    return ValidationReport(
        passed              = n_failed == 0,
        score               = score,
        n_rules             = len(results),
        n_passed            = n_passed,
        n_failed            = n_failed,
        n_warnings          = n_warnings,
        rules               = results,
        shape               = list(df_pd.shape),
        columns_with_issues = bad_cols,
    )


def validate_schema(
    df,
    *,
    required_columns: Optional[List[str]] = None,
    forbidden_columns: Optional[List[str]] = None,
    exact_columns: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> "ValidationReport":
    """
    Valida o schema completo do DataFrame (estrutura, não valores).

    Útil para garantir que dados recebidos de fontes externas respeitam
    o contrato esperado ANTES de processar coluna por coluna.

    Args:
        df:                 pd.DataFrame ou pl.DataFrame.
        required_columns:   Colunas que devem existir.
        forbidden_columns:  Colunas que NÃO devem existir.
        exact_columns:      O DataFrame deve ter EXATAMENTE estas colunas.
        min_rows:           Número mínimo de linhas.
        max_rows:           Número máximo de linhas.

    Returns:
        ValidationReport.

    Exemplos:
        r = lg.validate_schema(df,
            required_columns=["cpf","renda","uf"],
            min_rows=1,
        )
        r.raise_if_failed()
    """
    df_pd = _to_pd(df)
    n_rows = len(df_pd)
    cols = set(df_pd.columns)
    results: List[RuleResult] = []

    if required_columns:
        for col in required_columns:
            present = col in cols
            results.append(RuleResult(
                column=col, rule="column_required", passed=present,
                detail="presente" if present else f"coluna obrigatória ausente",
                severity="error",
            ))

    if forbidden_columns:
        for col in forbidden_columns:
            absent = col not in cols
            results.append(RuleResult(
                column=col, rule="column_forbidden", passed=absent,
                detail="ausente (correto)" if absent else "coluna proibida encontrada",
                severity="error",
            ))

    if exact_columns:
        expected = set(exact_columns)
        missing  = expected - cols
        extra    = cols - expected
        passed   = not missing and not extra
        detail   = "schema exato OK" if passed else (
            f"faltam: {sorted(missing)}  extras: {sorted(extra)}"
        )
        results.append(RuleResult(
            column="__schema__", rule="exact_columns", passed=passed,
            detail=detail, severity="error",
        ))

    if min_rows is not None:
        passed = n_rows >= min_rows
        results.append(RuleResult(
            column="__schema__", rule="min_rows", passed=passed,
            n_violations=max(0, min_rows - n_rows),
            detail=f"{n_rows} linhas (mínimo: {min_rows})",
            severity="error",
        ))

    if max_rows is not None:
        passed = n_rows <= max_rows
        results.append(RuleResult(
            column="__schema__", rule="max_rows", passed=passed,
            n_violations=max(0, n_rows - max_rows),
            detail=f"{n_rows} linhas (máximo: {max_rows})",
            severity="error",
        ))

    n_passed = sum(1 for r in results if r.passed)
    n_failed = sum(1 for r in results if not r.passed)
    score = n_passed / max(len(results), 1)
    bad_cols = sorted({r.column for r in results if not r.passed})
    return ValidationReport(
        passed=n_failed == 0, score=score,
        n_rules=len(results), n_passed=n_passed,
        n_failed=n_failed, n_warnings=0,
        rules=results, shape=list(df_pd.shape),
        columns_with_issues=bad_cols,
    )


def save_rules(rules: Dict[str, Dict], path) -> None:
    """
    Salva um conjunto de regras de validação como JSON versionável.

    Args:
        rules: Dict de regras (mesmo formato aceito por lg.validate()).
        path:  Caminho de saída.

    Exemplos:
        lg.save_rules({"cpf":{"not_null":True},"renda":{"min":0}}, "regras_clientes.json")
    """
    import json
    from pathlib import Path
    Path(str(path)).write_text(
        json.dumps({"logus_rules": rules, "version": "1.0"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_rules(path) -> Dict[str, Dict]:
    """
    Carrega regras de validação de um arquivo JSON.

    Args:
        path: Caminho do arquivo gerado por lg.save_rules().

    Returns:
        Dict de regras prontas para passar a lg.validate().

    Exemplos:
        rules = lg.load_rules("regras_clientes.json")
        result = lg.validate(df, rules)
    """
    import json
    from pathlib import Path
    data = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    if "logus_rules" in data:
        return data["logus_rules"]
    return data   # fallback: assume raw rules dict


def _eval_rule(s: pd.Series, col: str, rule_name: str, rule_val: Any,
               n_rows: int, severity: str) -> RuleResult:
    """Avalia uma regra individual sobre uma pd.Series."""
    rule_lower = rule_name.lower()
    passed = True
    n_viol = 0
    detail = ""

    try:
        if rule_lower == "not_null":
            if rule_val:
                n_viol = int(s.isna().sum())
                passed = n_viol == 0
                detail = f"{n_viol} nulos" if n_viol else "sem nulos"

        elif rule_lower == "unique":
            if rule_val:
                n_viol = int(s.duplicated(keep=False).sum())
                passed = n_viol == 0
                detail = f"{n_viol} duplicatas" if n_viol else "sem duplicatas"

        elif rule_lower == "min":
            if pd.api.types.is_numeric_dtype(s):
                mask = s.dropna() < rule_val
                n_viol = int(mask.sum())
                passed = n_viol == 0
                detail = f"{n_viol} valores < {rule_val}" if n_viol else f"min ≥ {rule_val}"
            else:
                detail = "ignorado (coluna não numérica)"

        elif rule_lower == "max":
            if pd.api.types.is_numeric_dtype(s):
                mask = s.dropna() > rule_val
                n_viol = int(mask.sum())
                passed = n_viol == 0
                detail = f"{n_viol} valores > {rule_val}" if n_viol else f"max ≤ {rule_val}"
            else:
                detail = "ignorado (coluna não numérica)"

        elif rule_lower == "in":
            allowed = set(rule_val)
            mask = ~s.dropna().isin(allowed)
            n_viol = int(mask.sum())
            passed = n_viol == 0
            detail = f"{n_viol} valores fora de {sorted(allowed)[:5]}..." if n_viol else "todos válidos"

        elif rule_lower == "not_in":
            forbidden = set(rule_val)
            mask = s.dropna().isin(forbidden)
            n_viol = int(mask.sum())
            passed = n_viol == 0
            detail = f"{n_viol} valores proibidos encontrados" if n_viol else "nenhum valor proibido"

        elif rule_lower == "matches":
            pattern = re.compile(str(rule_val))
            valid = s.dropna().astype(str).map(lambda v: bool(pattern.fullmatch(v)))
            n_viol = int((~valid).sum())
            passed = n_viol == 0
            detail = f"{n_viol} não batem com /{rule_val}/" if n_viol else f"todos batem /{rule_val}/"

        elif rule_lower == "contains":
            valid = s.dropna().astype(str).str.contains(str(rule_val), na=False)
            n_viol = int((~valid).sum())
            passed = n_viol == 0
            detail = f"{n_viol} não contêm '{rule_val}'" if n_viol else f"todos contêm '{rule_val}'"

        elif rule_lower == "min_length":
            lens = s.dropna().astype(str).str.len()
            mask = lens < int(rule_val)
            n_viol = int(mask.sum())
            passed = n_viol == 0
            detail = f"{n_viol} strings com menos de {rule_val} chars" if n_viol else f"len ≥ {rule_val}"

        elif rule_lower == "max_length":
            lens = s.dropna().astype(str).str.len()
            mask = lens > int(rule_val)
            n_viol = int(mask.sum())
            passed = n_viol == 0
            detail = f"{n_viol} strings com mais de {rule_val} chars" if n_viol else f"len ≤ {rule_val}"

        elif rule_lower == "dtype":
            expected = str(rule_val).lower()
            actual = str(s.dtype).lower()
            ok_map = {
                "int":   ("int", "int64","int32","int16","int8"),
                "float": ("float","float64","float32"),
                "str":   ("object","string"),
                "bool":  ("bool","boolean"),
                "date":  ("datetime","date"),
            }
            ok_types = ok_map.get(expected, (expected,))
            passed = any(t in actual for t in ok_types)
            detail = f"tipo real: {s.dtype}" if not passed else f"tipo OK: {s.dtype}"

        elif rule_lower == "custom":
            if callable(rule_val):
                ok_mask = rule_val(s)
                if hasattr(ok_mask, "__iter__"):
                    ok_series = pd.Series(ok_mask, index=s.index)
                    n_viol = int((~ok_series.fillna(False)).sum())
                    passed = n_viol == 0
                    detail = f"{n_viol} falhas na regra customizada" if n_viol else "regra customizada OK"
                else:
                    passed = bool(ok_mask)
                    detail = "regra customizada OK" if passed else "regra customizada falhou"

        else:
            detail = f"Regra '{rule_name}' não reconhecida."
            passed = True  # ignora regras desconhecidas

    except Exception as exc:
        passed = False
        detail = f"Erro ao avaliar: {exc}"

    pct = n_viol / max(n_rows, 1)
    return RuleResult(
        column=col, rule=rule_name, passed=passed,
        n_violations=n_viol, pct_violations=pct,
        detail=detail, severity=severity,
    )


# ---------------------------------------------------------------------------
# API fluente — lg.expect()
# ---------------------------------------------------------------------------

class _ColumnExpectation:
    """
    Interface fluente para definir expectativas em uma coluna.

    Uso:
        lg.expect(df, "cpf").not_null().unique().matches(r"^\\d{11}$").validate()
    """

    def __init__(self, df: Any, column: str) -> None:
        self._df   = df
        self._col  = column
        self._rules: Dict[str, Any] = {}

    def not_null(self) -> "_ColumnExpectation":
        self._rules["not_null"] = True
        return self

    def unique(self) -> "_ColumnExpectation":
        self._rules["unique"] = True
        return self

    def min(self, value: float) -> "_ColumnExpectation":
        self._rules["min"] = value
        return self

    def max(self, value: float) -> "_ColumnExpectation":
        self._rules["max"] = value
        return self

    def between(self, lo: float, hi: float) -> "_ColumnExpectation":
        self._rules["min"] = lo
        self._rules["max"] = hi
        return self

    def is_in(self, values) -> "_ColumnExpectation":
        self._rules["in"] = list(values)
        return self

    def not_in(self, values) -> "_ColumnExpectation":
        self._rules["not_in"] = list(values)
        return self

    def matches(self, pattern: str) -> "_ColumnExpectation":
        self._rules["matches"] = pattern
        return self

    def contains(self, substring: str) -> "_ColumnExpectation":
        self._rules["contains"] = substring
        return self

    def min_length(self, n: int) -> "_ColumnExpectation":
        self._rules["min_length"] = n
        return self

    def max_length(self, n: int) -> "_ColumnExpectation":
        self._rules["max_length"] = n
        return self

    def dtype(self, t: str) -> "_ColumnExpectation":
        self._rules["dtype"] = t
        return self

    def custom(self, fn) -> "_ColumnExpectation":
        self._rules["custom"] = fn
        return self

    def validate(self) -> ValidationReport:
        """Executa a validação e retorna ValidationReport."""
        return validate(self._df, {self._col: self._rules})

    def __repr__(self) -> str:
        return f"ColumnExpectation(col={self._col!r}, rules={self._rules})"


def expect(df: Any, column: str) -> _ColumnExpectation:
    """
    API fluente para expectativas por coluna.

    Exemplos:
        lg.expect(df, "cpf").not_null().unique().validate()
        lg.expect(df, "renda_mensal").not_null().between(0, 500_000).validate()
        lg.expect(df, "email").contains("@").min_length(5).validate()
    """
    return _ColumnExpectation(df, column)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_pd(df: Any) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
        if isinstance(df, pl.LazyFrame):
            return df.collect().to_pandas()
    except ImportError:
        pass
    return pd.DataFrame(df)
