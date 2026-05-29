"""
datalock/scan_directory.py
===========================
dd.scan_directory() — Inventário de PII em um diretório inteiro.

Varre todos os arquivos tabulares em um diretório (ou lista de caminhos)
e retorna um inventário completo de PII por arquivo, sem precisar abrir
cada arquivo manualmente.

Formatos suportados: .csv, .tsv, .parquet, .feather, .ipc, .xlsx, .xls,
                     .json, .ndjson, .jsonl, .dlk

Uso:
    import datalock as dd
    import os

    inventory = dd.scan_directory("./dados/")
    for path, report in inventory.items():
        print(f"{path}: {list(report.pii_columns.keys())}")

    # Com filtros
    inventory = dd.scan_directory(
        "./dados/",
        extensions=[".csv", ".parquet"],
        recursive=True,
        key=os.environ["DATALOCK_KEY"],  # para .dlk
        sample_size=500,
        min_risk="medium",
        verbose=True,
    )

    # Salva relatório
    inventory.to_json("inventario_pii.json")
    inventory.to_html("inventario_pii.html")
    print(inventory.summary())
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import json

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({
    ".csv", ".tsv", ".txt",
    ".parquet", ".feather", ".ipc", ".arrow",
    ".xlsx", ".xls", ".ods",
    ".json", ".ndjson", ".jsonl",
    ".dlk",
})

_RISK_ORDER = {"high": 3, "medium": 2, "low": 1, "none": 0}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FileInventory:
    """Inventário PII de um único arquivo."""
    path:        str
    size_bytes:  int
    n_rows:      int
    n_columns:   int
    pii_columns: Dict[str, Dict]   # {col: {type, risk, strategy, match_ratio}}
    max_risk:    str               # "high" | "medium" | "low" | "none"
    elapsed_ms:  float
    error:       Optional[str] = None
    scanned_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def has_pii(self) -> bool:
        return bool(self.pii_columns)

    @property
    def n_pii_columns(self) -> int:
        return len(self.pii_columns)

    def to_dict(self) -> Dict:
        return {
            "path":         self.path,
            "size_mb":      round(self.size_bytes / 1024 / 1024, 3),
            "n_rows":       self.n_rows,
            "n_columns":    self.n_columns,
            "n_pii_columns":self.n_pii_columns,
            "max_risk":     self.max_risk,
            "pii_columns":  self.pii_columns,
            "elapsed_ms":   self.elapsed_ms,
            "error":        self.error,
            "scanned_at":   self.scanned_at,
        }


@dataclass
class DirectoryInventory:
    """Inventário PII completo de um diretório."""
    root_path:      str
    files:          Dict[str, FileInventory] = field(default_factory=dict)
    n_files_scanned:int = 0
    n_files_error:  int = 0
    elapsed_s:      float = 0.0
    scanned_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── dict-like access ─────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[str]:
        return iter(self.files)

    def __getitem__(self, path: str) -> FileInventory:
        return self.files[path]

    def items(self):
        return self.files.items()

    def values(self):
        return self.files.values()

    # ── queries ───────────────────────────────────────────────────────────────

    @property
    def files_with_pii(self) -> List[FileInventory]:
        return [f for f in self.files.values() if f.has_pii]

    @property
    def files_high_risk(self) -> List[FileInventory]:
        return [f for f in self.files.values() if f.max_risk == "high"]

    def summary(self) -> str:
        """Texto resumido do inventário."""
        n_pii = sum(1 for f in self.files.values() if f.has_pii)
        n_high = sum(1 for f in self.files.values() if f.max_risk == "high")
        all_pii_cols: Dict[str, int] = {}
        for f in self.files.values():
            for col in f.pii_columns:
                all_pii_cols[col] = all_pii_cols.get(col, 0) + 1

        lines = [
            f"{'='*60}",
            f"  Inventário PII — {self.root_path}",
            f"  Gerado em: {self.scanned_at[:19]}",
            f"{'='*60}",
            f"  Arquivos varridos:    {self.n_files_scanned}",
            f"  Com PII:              {n_pii}",
            f"  Risco alto:           {n_high}",
            f"  Erros:                {self.n_files_error}",
            f"  Tempo total:          {self.elapsed_s:.1f}s",
            f"",
            f"  Colunas PII mais frequentes:",
        ]
        for col, count in sorted(all_pii_cols.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {col:<25} {count} arquivo(s)")

        if self.files_high_risk:
            lines += ["", "  Arquivos de alto risco:"]
            for f in self.files_high_risk[:5]:
                lines.append(f"    {Path(f.path).name}: {list(f.pii_columns.keys())}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "root_path":       self.root_path,
            "scanned_at":      self.scanned_at,
            "n_files_scanned": self.n_files_scanned,
            "n_files_error":   self.n_files_error,
            "elapsed_s":       self.elapsed_s,
            "files":           {p: f.to_dict() for p, f in self.files.items()},
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Serializa inventário como JSON."""
        js = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)
        if path:
            Path(str(path)).write_text(js, encoding="utf-8")
        return js

    def to_html(self, path: Optional[str] = None) -> str:
        """Gera relatório HTML do inventário."""
        rows_html = ""
        for fp, fi in sorted(self.files.items(), key=lambda x: -_RISK_ORDER.get(x[1].max_risk,0)):
            pii_list = ", ".join(fi.pii_columns.keys()) if fi.pii_columns else "—"
            err = f'<span style="color:red">{fi.error[:40]}</span>' if fi.error else "—"
            rows_html += f"""
<tr>
  <td><code>{Path(fp).name}</code></td>
  <td><span class="badge {fi.max_risk}">{fi.max_risk.upper()}</span></td>
  <td>{fi.n_pii_columns}</td>
  <td style="font-size:.85em">{pii_list}</td>
  <td>{fi.n_rows:,}</td>
  <td>{fi.size_bytes//1024:,} KB</td>
  <td>{fi.elapsed_ms:.0f}ms</td>
  <td>{err}</td>
</tr>"""

        html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inventário PII — datalock</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',Arial,sans-serif;max-width:1200px;margin:0 auto;color:#333;background:#f8f9fa}}
.topbar{{position:sticky;top:0;background:white;border-bottom:2px solid #dee2e6;
         padding:8px 20px;display:flex;align-items:center;justify-content:space-between;
         z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.content{{padding:20px}}
h1{{color:#1a1a2e;border-bottom:3px solid #0066cc;padding-bottom:10px;margin-top:0}}
.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}
.stat-card{{background:white;border-radius:8px;padding:16px 24px;border:1px solid #dee2e6;
            box-shadow:0 1px 4px rgba(0,0,0,.06);text-align:center;min-width:120px}}
.stat-num{{font-size:2em;font-weight:bold;color:#0066cc}}
.stat-label{{font-size:.8em;color:#666;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse;margin:20px 0;background:white;
       border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
th{{background:#0066cc;color:white;padding:12px 10px;text-align:left;font-size:.9em}}
td{{padding:10px;border-bottom:1px solid #dee2e6;font-size:.9em}}
tr:hover{{background:#f0f4ff}}
code{{background:#f1f3f5;padding:2px 6px;border-radius:4px;font-size:.85em}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.78em;font-weight:bold}}
.high{{background:#ffe0e0;color:#c00}}.medium{{background:#fff3cd;color:#856404}}
.low{{background:#d4edda;color:#155724}}.none{{background:#e9ecef;color:#6c757d}}
@media print{{.topbar{{position:static}}}}
</style></head><body>
<div class="topbar">
  <span style="font-weight:bold;color:#0066cc">🔒 datalock</span>
  <span style="color:#666;font-size:.9em">Inventário PII — {self.scanned_at[:10]}</span>
  <button onclick="window.print()"
          style="background:#0066cc;color:white;border:none;padding:6px 16px;
                 border-radius:6px;cursor:pointer;font-size:.85em">
    🖨️ Imprimir / Salvar PDF
  </button>
</div>
<div class="content">
<h1>🔍 Inventário PII — datalock</h1>
<div class="stats">
  <div class="stat-card"><div class="stat-num">{self.n_files_scanned}</div><div class="stat-label">Arquivos varridos</div></div>
  <div class="stat-card"><div class="stat-num" style="color:#c00">{len(self.files_high_risk)}</div><div class="stat-label">Alto risco</div></div>
  <div class="stat-card"><div class="stat-num">{len(self.files_with_pii)}</div><div class="stat-label">Com PII</div></div>
  <div class="stat-card"><div class="stat-num" style="color:#28a745">{self.n_files_scanned - len(self.files_with_pii)}</div><div class="stat-label">Sem PII</div></div>
</div>
<p><b>Diretório:</b> {self.root_path} &nbsp;|&nbsp;
   <b>Varrido em:</b> {self.scanned_at[:19]} &nbsp;|&nbsp;
   <b>Arquivos:</b> {self.n_files_scanned} &nbsp;|&nbsp;
   <b>Com PII:</b> {len(self.files_with_pii)} &nbsp;|&nbsp;
   <b>Alto risco:</b> {len(self.files_high_risk)}</p>
<table>
<tr><th>Arquivo</th><th>Risco</th><th>Cols PII</th><th>Colunas</th>
    <th>Linhas</th><th>Tamanho</th><th>Tempo</th><th>Erro</th></tr>
{rows_html}
</table>
<p style="color:#999;font-size:.8em">Gerado por datalock {_version()}</p>
</div></body></html>"""

        if path:
            Path(str(path)).write_text(html, encoding="utf-8")
        return html


# ---------------------------------------------------------------------------
# Core scan function
# ---------------------------------------------------------------------------

def scan_directory(
    path: Union[str, Path],
    *,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
    key: Optional[str] = None,
    sample_size: int = 500,
    min_risk: Optional[str] = None,
    max_files: Optional[int] = None,
    verbose: bool = False,
    skip_errors: bool = True,
) -> DirectoryInventory:
    """
    Varre um diretório e retorna um inventário de PII por arquivo.

    Args:
        path:        Diretório ou lista de caminhos a varrer.
        extensions:  Extensões a incluir. None = todas suportadas.
        recursive:   True = subdiretórios incluídos.
        key:         Chave para decifrar arquivos .dlk.
        sample_size: Linhas de amostra por arquivo para detecção.
        min_risk:    Filtro mínimo de risco: "low" | "medium" | "high".
        max_files:   Limita o número de arquivos varridos.
        verbose:     Imprime progresso.
        skip_errors: Continua se um arquivo falhar.

    Returns:
        DirectoryInventory com inventário completo.

    Exemplos:
        inv = dd.scan_directory("./dados/")
        print(inv.summary())
        inv.to_html("inventario.html")

        # Apenas arquivos de alto risco
        inv = dd.scan_directory("./dados/", min_risk="high")
        for path, fi in inv.items():
            print(f"{path}: {list(fi.pii_columns.keys())}")
    """
    from datalock.detectors.fast_scan import FastPIIScanner

    root = Path(str(path))
    t0   = time.perf_counter()

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in (extensions or _SUPPORTED_EXTENSIONS)}

    # Collect files
    if recursive:
        file_iter = (f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in exts)
    else:
        file_iter = (f for f in root.iterdir() if f.is_file() and f.suffix.lower() in exts)

    files_to_scan = list(file_iter)
    if max_files:
        files_to_scan = files_to_scan[:max_files]

    if verbose:
        print(f"dd.scan_directory: {len(files_to_scan)} arquivos em {root}")

    inventory = DirectoryInventory(root_path=str(root))
    scanner   = FastPIIScanner(sample_size=sample_size)

    for i, fpath in enumerate(files_to_scan):
        if verbose:
            print(f"  [{i+1}/{len(files_to_scan)}] {fpath.name}...", end="\r")

        fi = _scan_single_file(fpath, scanner, key=key, skip_errors=skip_errors)
        inventory.n_files_scanned += 1
        if fi.error:
            inventory.n_files_error += 1

        # Apply min_risk filter
        if min_risk and _RISK_ORDER.get(fi.max_risk, 0) < _RISK_ORDER.get(min_risk, 0):
            continue

        inventory.files[str(fpath)] = fi

    inventory.elapsed_s = round(time.perf_counter() - t0, 2)

    if verbose:
        print(f"\n{inventory.summary()}")

    return inventory


def _scan_single_file(
    path: Path,
    scanner: Any,
    *,
    key: Optional[str],
    skip_errors: bool,
) -> FileInventory:
    """Scans a single file and returns its FileInventory."""
    import datalock as _dl
    import polars as pl

    t0 = time.perf_counter()
    ext = path.suffix.lower()

    try:
        size_bytes = path.stat().st_size

        # Read sample
        if ext == ".dlk":
            if not key:
                return FileInventory(
                    path=str(path), size_bytes=size_bytes, n_rows=0, n_columns=0,
                    pii_columns={}, max_risk="none", elapsed_ms=0,
                    error="key= necessário para arquivos .dlk",
                )
            info = _dl.inspect(str(path), key=key)
            df_sample = _dl.read(str(path), key=key, head=scanner.sample_size)
            n_rows    = info.get("n_rows", len(df_sample))

        elif ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext in (".tsv", ".txt") else ","
            df_sample = pl.read_csv(str(path), separator=sep,
                                     n_rows=scanner.sample_size, ignore_errors=True)
            n_rows = _estimate_csv_rows(path)

        elif ext in (".parquet", ".feather", ".ipc", ".arrow"):
            df_sample = _dl.read(str(path), head=scanner.sample_size)
            info = _dl.read(str(path), header_only=True)
            n_rows = info.get("n_rows", len(df_sample)) if isinstance(info, dict) else len(df_sample)

        elif ext in (".xlsx", ".xls", ".ods"):
            import pandas as _pd
            df_pd     = _pd.read_excel(str(path), nrows=scanner.sample_size)
            df_sample = pl.from_pandas(df_pd)
            n_rows    = scanner.sample_size  # estimate

        elif ext in (".json", ".ndjson", ".jsonl"):
            try:
                df_sample = pl.read_ndjson(str(path)).head(scanner.sample_size)
            except Exception:
                df_sample = pl.read_json(str(path)).head(scanner.sample_size)
            n_rows = scanner.sample_size  # estimate

        else:
            return FileInventory(
                path=str(path), size_bytes=size_bytes, n_rows=0, n_columns=0,
                pii_columns={}, max_risk="none", elapsed_ms=0,
                error=f"Formato não suportado: {ext}",
            )

        n_cols   = len(df_sample.columns)
        reports  = scanner.detect_dict(df_sample)

        pii_cols = {
            col: {
                "pii_type":    r.pii_type.value,
                "risk":        r.risk_level.value,
                "strategy":    r.mask_strategy.value,
                "match_ratio": round(r.match_ratio, 3),
            }
            for col, r in reports.items()
        }

        max_risk = "none"
        for r in pii_cols.values():
            if _RISK_ORDER.get(r["risk"], 0) > _RISK_ORDER.get(max_risk, 0):
                max_risk = r["risk"]

        elapsed = (time.perf_counter() - t0) * 1000
        return FileInventory(
            path=str(path), size_bytes=size_bytes,
            n_rows=n_rows, n_columns=n_cols,
            pii_columns=pii_cols, max_risk=max_risk,
            elapsed_ms=round(elapsed, 1),
        )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("scan_directory: error scanning %s: %s", path.name, msg)
        if not skip_errors:
            raise
        return FileInventory(
            path=str(path), size_bytes=path.stat().st_size if path.exists() else 0,
            n_rows=0, n_columns=0, pii_columns={}, max_risk="none",
            elapsed_ms=round(elapsed, 1), error=msg,
        )


def _estimate_csv_rows(path: Path) -> int:
    """Fast row count estimate for CSV via file size / avg line size."""
    import mmap
    try:
        with open(str(path), "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            sample = mm[:min(65536, len(mm))]
            lines  = sample.count(bytes([10]))
            mm.close()
        if lines == 0:
            return 0
        avg_line = min(65536, path.stat().st_size) / max(lines, 1)
        return max(1, int(path.stat().st_size / avg_line) - 1)  # -1 for header
    except Exception:
        return 0


def _version() -> str:
    try:
        import datalock
        return datalock.__version__
    except Exception:
        return "?"
