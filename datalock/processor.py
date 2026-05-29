"""
datalock/processor.py
==================
dd.process() — Pipeline completo em uma chamada.

Pensado para o caso de uso mais comum:
  "Tenho um arquivo ou DataFrame com PII. Quero mascarar, validar,
   salvar protegido e obter um relatório — tudo de uma vez."

Sem dd.process():
    df       = dd.read("clientes.csv")
    reports  = dd.scan(df)
    df_valid = dd.validate(df, rules)
    df_safe  = dd.mask(df, salt=SALT)
    profile  = dd.profile(df)
    dd.store(df_safe, "clientes.dlk", key=KEY)
    lin      = dd.lineage.start(...)
    lin.origin(df, ...); lin.mask(...)

Com dd.process():
    result = dd.process(
        "clientes.csv",
        salt=SALT,
        key=KEY,
        output="clientes.dlk",
    )
    # result.df         → DataFrame mascarado
    # result.reports    → detecção PII
    # result.profile    → diagnóstico completo
    # result.validation → resultado das regras
    # result.lineage    → trilha de linhagem
    # result.elapsed_ms → tempo total
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import pandas as pd


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class ProcessResult:
    """
    Resultado completo de dd.process().

    Agrupa tudo o que um pipeline típico produz:
    DataFrame mascarado, relatório PII, diagnóstico, lineage e métricas.
    """
    df:          Any            # pl.DataFrame ou pd.DataFrame mascarado
    reports:     Dict           # Dict[str, ColumnReport] do scan
    profile:     Dict           # dict completo do dd.profile()
    validation:  Optional[Any] = None   # ValidationReport (se rules= fornecido)
    lineage:     Optional[Any] = None   # LineageTracker
    output_path: Optional[str] = None   # caminho do .dlk salvo (se output= fornecido)
    elapsed_ms:  float = 0.0
    n_pii_cols:  int   = 0
    n_rows:      int   = 0
    n_cols:      int   = 0

    def print_summary(self) -> None:
        """Imprime resumo executivo do processamento."""
        ps = self.profile.get("privacy_score") or {}
        score = ps.get("total", "?") if ps else "?"
        grade = ps.get("grade", "?") if ps else "?"

        print(f"\n{'='*60}")
        print(f"  datalock.process() — Relatório")
        print(f"{'='*60}")
        print(f"  Shape:          {self.n_rows:,} × {self.n_cols}")
        print(f"  PII detectadas: {self.n_pii_cols} colunas")
        print(f"  Privacy Score:  {score}/100 [{grade}]")
        if self.validation:
            print(f"  Data Quality:   {self.validation.score:.0%} ({self.validation.n_passed}/{self.validation.n_rules} regras)")
        if self.output_path:
            print(f"  Salvo em:       {self.output_path}")
        print(f"  Tempo total:    {self.elapsed_ms:.0f}ms")
        print(f"{'='*60}\n")

        # PII detectadas
        if self.reports:
            print("  Colunas PII:")
            for col, r in self.reports.items():
                flag = "🔴" if r.risk_level.value == "high" else (
                       "🟡" if r.risk_level.value == "medium" else "🟢")
                print(f"    {flag} {col:<20} {r.pii_type.value:<18} → {r.mask_strategy.value}")

        # Recomendação
        rec = ps.get("recommendation", "") if ps else ""
        if rec:
            print(f"\n  ➜ {rec}\n")

    @property
    def pii_columns(self) -> List[str]:
        return list(self.reports.keys())

    @property
    def privacy_score(self) -> Optional[int]:
        ps = self.profile.get("privacy_score") or {}
        return ps.get("total") if ps else None

    def to_dict(self) -> Dict:
        """Serializa metadados (sem o DataFrame)."""
        return {
            "shape":       [self.n_rows, self.n_cols],
            "pii_columns": self.pii_columns,
            "n_pii_cols":  self.n_pii_cols,
            "privacy_score": self.privacy_score,
            "output_path": self.output_path,
            "elapsed_ms":  round(self.elapsed_ms, 1),
            "reports": {
                c: {
                    "type":     r.pii_type.value,
                    "risk":     r.risk_level.value,
                    "strategy": r.mask_strategy.value,
                }
                for c, r in self.reports.items()
            },
        }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def process(
    source: Any,
    *,
    # Privacidade
    salt: Optional[str] = None,
    key:  Optional[str] = None,
    risk: Optional[str] = None,
    columns:    Optional[List[str]] = None,
    exclude:    Optional[List[str]] = None,
    # Saída
    output:     Optional[str] = None,
    output_fmt: str = "lgs",
    overwrite:  bool = False,
    compress:   bool = True,
    # Validação
    rules:      Optional[Dict] = None,
    # Transformações (executadas ANTES do mascaramento)
    where:      Optional[Any] = None,
    select:     Optional[List[str]] = None,
    add_cols:   Optional[Dict] = None,
    # Detecção
    sample_size:int = 500,
    threshold:  float = 0.5,
    # Controle
    verbose:    bool = False,
    track_lineage: bool = True,
    return_polars: bool = True,
    # Leitura
    read_key:   Optional[str] = None,
    **read_kwargs,
) -> ProcessResult:
    """
    Pipeline completo de privacidade em uma chamada.

    Executa em sequência:
      1. Leitura (qualquer formato, incluindo .dlk)
      2. Transformações opcionais (where=, select=, add_cols=)
      3. Detecção PII via FastPIIScanner (9× mais rápido)
      4. Validação de qualidade (se rules= fornecido)
      5. Mascaramento (salt=, risk=, columns=, exclude=)
      6. Diagnóstico completo (profile + privacy score)
      7. Salvamento (se output= fornecido)
      8. Lineage tracking automático

    Args:
        source:       Caminho, pd.DataFrame ou pl.DataFrame.
        salt:         Salt HMAC para mascaramento.
        key:          Chave AES-256 para criptografar o .dlk de saída.
        risk:         Mascaramento risk-aware: "high"|"medium"|"low".
        columns:      Mascara apenas estas colunas.
        exclude:      Exclui estas colunas do mascaramento.
        output:       Caminho de saída (.dlk, .parquet, .csv, etc.).
        output_fmt:   Formato de saída: "lgs" (padrão) | "parquet" | "csv".
        overwrite:    Sobrescreve o arquivo de saída.
        compress:     Compressão zstd (para .dlk e .parquet).
        rules:        Regras de validação para dd.validate().
        where:        Filtro pré-mascaramento (mesmo syntax do dd.where()).
        select:       Colunas a manter pré-mascaramento.
        add_cols:     Colunas a adicionar pré-mascaramento (dict de expressões).
        sample_size:  Linhas amostradas para detecção PII.
        threshold:    Match ratio mínimo para classificar como PII.
        verbose:      Exibe relatório detalhado.
        track_lineage: Registra linhagem automaticamente.
        return_polars: Se True, df no resultado é pl.DataFrame (padrão).
        read_key:     Chave para ler .dlk de entrada cifrado.
        **read_kwargs: Repassados ao dd.read() (sep=, encoding=, etc.).

    Returns:
        ProcessResult com df, reports, profile, validation, lineage, output_path.

    Exemplos:
        # Mais simples: lê, mascara, salva
        result = dd.process("clientes.csv", salt=SALT, key=KEY, output="safe.dlk")

        # Com validação e filtro
        result = dd.process(
            "clientes.parquet",
            salt=SALT,
            key=KEY,
            output="safe.dlk",
            where={"uf": ["SP","RJ","MG"]},
            rules={"cpf": {"not_null": True}, "renda": {"min": 0}},
            risk="high",
            verbose=True,
        )
        result.print_summary()

        # DataFrame em memória
        result = dd.process(df, salt=SALT)
        df_safe = result.df

        # Só diagnosticar (sem mascarar)
        result = dd.process("clientes.parquet")
        print(result.privacy_score)
    """
    import datalock as dd
    from datalock.detectors.fast_scan import FastPIIScanner

    t_start = time.perf_counter()
    lineage_tracker = None
    source_name = str(source) if not isinstance(source, (pl.DataFrame, pd.DataFrame)) else "DataFrame"

    # ── 1. Leitura ────────────────────────────────────────────────────────
    if isinstance(source, (pl.DataFrame, pd.DataFrame)):
        df = source
    else:
        df = dd.read(source, key=read_key, **read_kwargs)

    if isinstance(df, dict):
        # Multi-frame .dlk → concat para processar como DataFrame único
        df = pd.concat(list(df.values()), ignore_index=True)

    # Normaliza para Polars internamente
    was_pandas = isinstance(df, pd.DataFrame)
    if was_pandas:
        df = pl.from_pandas(df)

    # ── 2. Transformações pré-mascaramento ────────────────────────────────
    if where is not None:
        from datalock.analytics import where as _where
        df = _where(df, where) if not isinstance(where, dict) else _where(df, **where)

    if select is not None:
        df = df.select([c for c in select if c in df.columns])

    if add_cols:
        from datalock.analytics import add_column as _add_col
        df = _add_col(df, **add_cols)

    # ── 3. Detecção PII (fast) ────────────────────────────────────────────
    reports = FastPIIScanner(
        sample_size=sample_size,
        match_threshold=threshold,
    ).detect_dict(df)

    if verbose:
        _print_scan(reports)

    # Lineage — origem
    if track_lineage:
        lineage_tracker = dd.lineage.start(f"process({Path(source_name).name if isinstance(source, str) else 'df'})")
        lineage_tracker.origin(df, source_name)

    # ── 4. Validação ──────────────────────────────────────────────────────
    validation_result = None
    if rules:
        from datalock.validate import validate as _validate
        validation_result = _validate(df.to_pandas(), rules)
        if verbose:
            validation_result.print_report()

    # ── 5. Mascaramento ───────────────────────────────────────────────────
    if salt or risk:
        # Resolve salt default
        if not salt:
            import datalock._defaults as _defs
            salt = _defs.DEFAULT_SALT or None

        if salt or risk:
            df_safe = dd.mask(
                df,
                salt=salt or dd.generate_salt(),  # salt aleatório se risk= sem salt=
                columns=columns,
                exclude=exclude,
                risk=risk,
                verbose=False,
            )
        else:
            df_safe = df
    else:
        df_safe = df

    # Lineage — mascaramento
    if track_lineage and lineage_tracker and salt:
        masked_cols = list(reports.keys())
        if columns:
            masked_cols = [c for c in masked_cols if c in columns]
        if exclude:
            masked_cols = [c for c in masked_cols if c not in exclude]
        lineage_tracker.mask(df_safe, columns=masked_cols, salt=salt)

    # ── 6. Diagnóstico ────────────────────────────────────────────────────
    try:
        profile_result = dd.profile(df_safe)
    except Exception:
        profile_result = {"shape": list(df_safe.shape), "privacy_score": None}

    # ── 7. Salvamento ─────────────────────────────────────────────────────
    output_path = None
    if output:
        ext = Path(output).suffix.lower()
        if not ext or ext == ".dlk" or output_fmt == "lgs":
            out_path = output if output.endswith(".dlk") else output + ".dlk"
            dd.store(
                df_safe.to_pandas(),
                out_path,
                key=key,
                overwrite=overwrite,
                compress=compress,
            )
        elif ext == ".parquet" or output_fmt == "parquet":
            df_safe.write_parquet(output)
        elif ext == ".csv" or output_fmt == "csv":
            df_safe.write_csv(output)
        else:
            from datalock.analytics import write as _write
            _write(df_safe, output)

        output_path = str(output)
        if track_lineage and lineage_tracker:
            lineage_tracker.export(
                output_path,
                format=output_fmt,
                encrypted=bool(key),
                masked=bool(salt),
            )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Retorna no tipo solicitado
    result_df = df_safe
    if not return_polars and not was_pandas:
        result_df = df_safe.to_pandas()
    elif return_polars and was_pandas:
        pass  # já é pl.DataFrame

    return ProcessResult(
        df          = result_df,
        reports     = reports,
        profile     = profile_result,
        validation  = validation_result,
        lineage     = lineage_tracker,
        output_path = output_path,
        elapsed_ms  = round(elapsed_ms, 1),
        n_pii_cols  = len(reports),
        n_rows      = df_safe.height if isinstance(df_safe, pl.DataFrame) else len(df_safe),
        n_cols      = df_safe.width  if isinstance(df_safe, pl.DataFrame) else len(df_safe.columns),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_scan(reports: Dict) -> None:
    if not reports:
        print("  scan: nenhuma coluna PII detectada.")
        return
    print(f"\n  PII detectadas: {len(reports)} colunas")
    for col, r in reports.items():
        flag = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(r.risk_level.value, "•")
        print(f"    {flag} {col:<20} {r.pii_type.value:<18} → {r.mask_strategy.value}")
    print()
