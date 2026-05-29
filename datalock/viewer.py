"""
datalock/viewer.py
==================
dd.view() — Visualizador universal de outputs HTML do datalock.

Renderiza automaticamente no ambiente certo:
  - Jupyter / Google Colab:  inline no notebook (IPython.display)
  - Terminal / script:        abre no browser padrão do sistema
  - Sem browser disponível:   salva em arquivo temporário e exibe o caminho

Aceita qualquer output HTML gerado pelo datalock:
  - dd.compliance_report(...).to_html()
  - dd.scan_directory(...).to_html()
  - Qualquer string HTML ou caminho para arquivo .html

Uso:
    # Compliance report
    reports = dd.scan(df)
    report  = dd.compliance_report(df, reports, dataset_name="Clientes Q1")
    dd.view(report)                          # abre no browser / inline no notebook

    # Inventory de diretório
    inv = dd.scan_directory("./dados/")
    dd.view(inv)                             # abre direto

    # HTML já gerado
    html = report.to_html()
    dd.view(html)                            # string HTML

    # Arquivo existente
    dd.view("relatorio_lgpd.html")           # path

    # Força comportamento
    dd.view(report, inline=True)             # sempre inline (Jupyter)
    dd.view(report, browser=True)            # sempre browser
    dd.view(report, save="relatorio.html")   # salva E abre
"""
from __future__ import annotations

import os
import tempfile
import webbrowser
from pathlib import Path
import pathlib
from typing import Any, Optional, Union


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def view(
    source: Any,
    *,
    inline: bool = False,
    browser: bool = False,
    save: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Renderiza um relatório HTML do datalock no melhor ambiente disponível.

    Args:
        source:  Qualquer um de:
                   - ComplianceReport  (dd.compliance_report() output)
                   - DirectoryInventory (dd.scan_directory() output)
                   - str               (HTML já gerado ou caminho de arquivo)
                   - pathlib.Path      (caminho para arquivo .html)
        inline:  Se True, força renderização inline (Jupyter/Colab).
        browser: Se True, força abertura no browser do sistema.
        save:    Se fornecido, salva o HTML neste caminho antes de exibir.
        title:   Título extra para injetar no HTML (opcional).

    Returns:
        Caminho do arquivo HTML gerado/aberto, ou None em modo inline.

    Exemplos:
        reports = dd.scan(df)
        report  = dd.compliance_report(df, reports)
        dd.view(report)

        inv = dd.scan_directory("./dados/")
        dd.view(inv)

        dd.view(report, save="lgpd_jan2025.html")
        dd.view("relatorio.html")
    """
    html = _to_html(source, title=title)

    # ── Save if requested ─────────────────────────────────────────────────
    if save:
        save_path = Path(str(save))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(html, encoding="utf-8")

    # ── Choose rendering mode ─────────────────────────────────────────────
    if inline or (not browser and _is_notebook()):
        return _render_inline(html)

    # If browser=False explicitly and not in notebook, just save to temp and return path
    if not browser and not inline:
        path = _save_temp(html) if not save else str(pathlib.Path(str(save)).resolve())
        if not save:
            print(f"💾 Relatório salvo em: {path}")
        return path

    return _render_browser(html, save_path=save if save else None)


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def _save_temp(html: str) -> str:
    """Save HTML to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False,
        encoding="utf-8", prefix="datalock_report_"
    )
    tmp.write(html)
    tmp.close()
    return tmp.name


def _to_html(source: Any, title: Optional[str] = None) -> str:
    """Extract HTML string from any supported source type."""
    html: str

    # Path to existing HTML file
    if isinstance(source, (str, Path)):
        p = Path(str(source))
        if p.exists() and p.suffix.lower() in (".html", ".htm"):
            html = p.read_text(encoding="utf-8")
        elif isinstance(source, str) and source.strip().startswith("<"):
            html = source  # already HTML string
        else:
            raise ValueError(
                f"dd.view(): argumento não reconhecido: {source!r}\n"
                "Passe um ComplianceReport, DirectoryInventory, string HTML "
                "ou caminho para arquivo .html"
            )

    # ComplianceReport
    elif hasattr(source, "to_html") and hasattr(source, "pii_columns"):
        html = source.to_html()

    # DirectoryInventory
    elif hasattr(source, "to_html") and hasattr(source, "files_with_pii"):
        html = source.to_html()

    # Any object with to_html()
    elif hasattr(source, "to_html"):
        result = source.to_html()
        html = result if isinstance(result, str) else str(result)

    else:
        raise TypeError(
            f"dd.view(): tipo não suportado: {type(source).__name__}\n"
            "Suportados: ComplianceReport, DirectoryInventory, "
            "str (HTML), pathlib.Path"
        )

    # Inject custom title if requested
    if title and "<title>" in html:
        import re
        html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html, count=1)

    return html


# ---------------------------------------------------------------------------
# Rendering backends
# ---------------------------------------------------------------------------

def _is_notebook() -> bool:
    """True if running inside Jupyter, Google Colab, or IPython."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = type(shell).__name__
        return shell_name in ("ZMQInteractiveShell", "TerminalInteractiveShell",
                              "Shell", "google.colab.kernel.InteractiveShell".__class__.__name__)
    except ImportError:
        return False


def _render_inline(html: str) -> None:
    """Render HTML inline in Jupyter/Colab notebook."""
    try:
        from IPython.display import display, HTML as IHTML
        display(IHTML(html))
        return None
    except ImportError:
        # IPython not available — fall back to browser
        return _render_browser(html)


def _render_browser(html: str, save_path: Optional[str] = None) -> str:
    """Open HTML in the system default browser."""
    # Use save_path if given, otherwise create a temp file
    if save_path and Path(str(save_path)).exists():
        final_path = str(Path(str(save_path)).resolve())
    else:
        # Write to a temp file that persists long enough for the browser to read
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False,
            encoding="utf-8", prefix="datalock_report_"
        )
        tmp.write(html)
        tmp.close()
        final_path = tmp.name

    url = f"file://{final_path}"

    try:
        opened = webbrowser.open(url, new=2)  # new=2: open in new tab if possible
        if opened:
            print(f"🌐 Relatório aberto no browser: {final_path}")
        else:
            print(f"⚠️  Browser não disponível. Arquivo salvo em:\n   {final_path}")
    except Exception:
        print(f"⚠️  Não foi possível abrir o browser. Arquivo salvo em:\n   {final_path}")

    return final_path
