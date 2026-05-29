"""
logus/reports/compliance_report.py
===================================
Relatório formal de conformidade LGPD — Feature 8.

Gera documento HTML (e PDF opcional) com inventário de dados pessoais,
técnicas de mascaramento, k-anonimato, risk score e trilha de auditoria.
Formato projetado para ser assinado por DPOs e apresentado a auditores.

Uso:
    report = lg.compliance_report(df, reports, audit=audit)
    report.to_html("relatorio_lgpd_jan2025.html")
    report.to_pdf("relatorio_lgpd_jan2025.pdf")   # requer weasyprint
    print(report.to_text())                        # sempre disponível
"""
from __future__ import annotations

import datetime
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------

@dataclass
class ComplianceReport:
    """Relatório de conformidade LGPD pronto para entrega ao DPO."""
    df_shape:     List[int]
    pii_columns:  List[Dict]     # [{col, type, risk, strategy, masked}]
    privacy_score: Optional[Dict]
    audit_entries: List[Dict]
    generated_at:  str
    generated_by:  str
    title:         str = "Relatório de Conformidade LGPD"
    organization:  str = ""
    dataset_name:  str = "dataset"
    extra_notes:   str = ""

    def to_text(self) -> str:
        """Texto simples — sempre disponível, sem dependências extras."""
        lines = [
            f"{'='*70}",
            f"  {self.title}",
            f"  {self.dataset_name}  |  Gerado em: {self.generated_at}",
            f"{'='*70}",
            "",
            f"SUMÁRIO",
            f"  Linhas:              {self.df_shape[0]:,}",
            f"  Colunas:             {self.df_shape[1]}",
            f"  Colunas com PII:     {len(self.pii_columns)}",
        ]
        if self.privacy_score:
            ps = self.privacy_score
            lines.append(f"  Privacy Score:       {ps.get('total','-')}/100 [{ps.get('grade','-')}]")
            lines.append(f"  Recomendação:        {ps.get('recommendation','')}")
        lines += [
            "",
            f"INVENTÁRIO DE DADOS PESSOAIS (Art. 37 LGPD)",
            f"  {'Coluna':<22} {'Tipo PII':<22} {'Risco':<8} {'Estratégia':<20} {'Mascarado':<10}",
            f"  {'-'*80}",
        ]
        for col in self.pii_columns:
            masked = "✓" if col.get("masked") else "✗"
            lines.append(
                f"  {col['column']:<22} {col['pii_type']:<22} {col['risk']:<8} "
                f"{col['strategy']:<20} {masked:<10}"
            )
        if self.audit_entries:
            lines += ["", f"TRILHA DE AUDITORIA ({len(self.audit_entries)} operações)"]
            for e in self.audit_entries[:10]:
                lines.append(f"  [{e.get('timestamp','')[:10]}] {e.get('column','')} → {e.get('technique','')} ({e.get('rows_affected','?')} linhas)")
            if len(self.audit_entries) > 10:
                lines.append(f"  ... e mais {len(self.audit_entries)-10} operações.")
        lines += ["", f"Gerado por: {self.generated_by}", f"{'='*70}"]
        return "\n".join(lines)

    def to_html(self, path: Optional[str] = None) -> str:
        """Gera relatório HTML formatado."""
        ps = self.privacy_score or {}
        score = ps.get("total", "-")
        grade = ps.get("grade", "-")
        rec   = ps.get("recommendation", "")

        color_map = {"high": "#dc3545", "medium": "#fd7e14", "low": "#28a745"}
        grade_color = {"A":"#28a745","B":"#5cb85c","C":"#ffc107","D":"#fd7e14","F":"#dc3545"}.get(grade,"#6c757d")

        pii_rows = ""
        for col in self.pii_columns:
            risk_color = color_map.get(col["risk"], "#6c757d")
            masked_icon = "✓" if col.get("masked") else "✗"
            masked_color = "#28a745" if col.get("masked") else "#dc3545"
            pii_rows += f"""
            <tr>
                <td><code>{col['column']}</code></td>
                <td>{col['pii_type']}</td>
                <td style="color:{risk_color};font-weight:bold">{col['risk'].upper()}</td>
                <td><code>{col['strategy']}</code></td>
                <td style="color:{masked_color};font-size:1.2em;text-align:center">{masked_icon}</td>
            </tr>"""

        audit_rows = ""
        for e in self.audit_entries[:20]:
            audit_rows += f"""
            <tr>
                <td>{e.get('timestamp','')[:19]}</td>
                <td><code>{e.get('column','')}</code></td>
                <td>{e.get('technique','')}</td>
                <td>{e.get('rows_affected','?'):,}</td>
                <td><span style="color:#28a745">✓</span></td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self.title}</title>
<style>
  body {{font-family: 'Segoe UI', Arial, sans-serif; max-width:1100px; margin:40px auto; color:#333; line-height:1.6}}
  h1 {{color:#1a1a2e; border-bottom:3px solid #0066cc; padding-bottom:10px}}
  h2 {{color:#0066cc; margin-top:40px}}
  .meta {{color:#666; font-size:.9em; margin-bottom:30px}}
  .score-box {{display:inline-block; background:#f8f9fa; border:2px solid #dee2e6;
              border-radius:12px; padding:20px 30px; margin:20px 0}}
  .score-num {{font-size:3em; font-weight:bold; color:{grade_color}}}
  .score-grade {{font-size:1.5em; color:{grade_color}; margin-left:10px}}
  .rec {{background:#e8f4f8; border-left:4px solid #0066cc; padding:10px 20px; margin:10px 0; border-radius:0 8px 8px 0}}
  table {{width:100%; border-collapse:collapse; margin:20px 0}}
  th {{background:#0066cc; color:white; padding:10px; text-align:left}}
  td {{padding:8px 10px; border-bottom:1px solid #dee2e6}}
  tr:hover {{background:#f8f9fa}}
  code {{background:#f1f3f5; padding:2px 6px; border-radius:4px; font-size:.9em}}
  .footer {{color:#999; font-size:.8em; margin-top:60px; border-top:1px solid #dee2e6; padding-top:20px}}
  @media print {{body {{margin:20mm}} h2 {{page-break-before: auto}}}}
</style>
</head>
<body>
<h1>📋 {self.title}</h1>
<div class="meta">
  <strong>Dataset:</strong> {self.dataset_name} &nbsp;|&nbsp;
  <strong>Organização:</strong> {self.organization or 'N/A'} &nbsp;|&nbsp;
  <strong>Gerado em:</strong> {self.generated_at} &nbsp;|&nbsp;
  <strong>Gerado por:</strong> {self.generated_by}
</div>

<h2>1. Sumário Executivo</h2>
<table style="width:auto">
  <tr><th>Métrica</th><th>Valor</th></tr>
  <tr><td>Total de linhas</td><td><strong>{self.df_shape[0]:,}</strong></td></tr>
  <tr><td>Total de colunas</td><td><strong>{self.df_shape[1]}</strong></td></tr>
  <tr><td>Colunas com PII detectadas</td><td><strong>{len(self.pii_columns)}</strong></td></tr>
  <tr><td>Colunas com PII mascaradas</td><td><strong>{sum(1 for c in self.pii_columns if c.get('masked'))}</strong></td></tr>
</table>

<h2>2. Privacy Score</h2>
<div class="score-box">
  <span class="score-num">{score}</span><span>/100</span>
  <span class="score-grade">[{grade}]</span>
</div>
<div class="rec">💡 {rec}</div>

<h2>3. Inventário de Dados Pessoais (Art. 37 LGPD)</h2>
<table>
  <tr>
    <th>Coluna</th><th>Tipo PII</th><th>Nível de Risco</th>
    <th>Estratégia de Mascaramento</th><th>Mascarado?</th>
  </tr>
  {pii_rows}
</table>

<h2>4. Trilha de Auditoria (Art. 50 LGPD)</h2>
{'<table><tr><th>Data/Hora</th><th>Coluna</th><th>Técnica</th><th>Linhas afetadas</th><th>Status</th></tr>' + audit_rows + '</table>' if self.audit_entries else '<p style="color:#666">Nenhuma entrada de auditoria disponível.</p>'}

{f'<h2>5. Observações</h2><p>{self.extra_notes}</p>' if self.extra_notes else ''}

<div class="footer">
  Relatório gerado automaticamente pela biblioteca <strong>logus-lgpd</strong>.<br>
  Este documento deve ser revisado e assinado pelo Encarregado de Proteção de Dados (DPO)<br>
  conforme exigido pela Lei n° 13.709/2018 (LGPD).
</div>
</body>
</html>"""

        if path:
            Path(str(path)).write_text(html, encoding="utf-8")
        return html

    def to_pdf(self, path: str) -> None:
        """
        Gera relatório PDF a partir do HTML.

        Requer: pip install weasyprint
        """
        html = self.to_html()
        try:
            from weasyprint import HTML as _WP
            _WP(string=html).write_pdf(str(path))
        except ImportError:
            raise ImportError(
                "lg.compliance_report().to_pdf() requer weasyprint. "
                "Instale com: pip install weasyprint\n"
                "Alternativa: use .to_html() e converta para PDF com o browser."
            ) from None

    def to_dict(self) -> Dict:
        return {
            "title":         self.title,
            "dataset_name":  self.dataset_name,
            "organization":  self.organization,
            "generated_at":  self.generated_at,
            "generated_by":  self.generated_by,
            "shape":         self.df_shape,
            "pii_columns":   self.pii_columns,
            "privacy_score": self.privacy_score,
            "n_audit_entries": len(self.audit_entries),
        }

    def to_json(self, path: Optional[str] = None) -> str:
        js = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
        if path:
            Path(str(path)).write_text(js, encoding="utf-8")
        return js


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_compliance_report(
    df: Any,
    reports: Dict,
    *,
    audit: Optional[Any] = None,
    privacy_score: Optional[Dict] = None,
    title: str = "Relatório de Conformidade LGPD",
    organization: str = "",
    dataset_name: str = "dataset",
    extra_notes: str = "",
) -> ComplianceReport:
    """
    Constrói um ComplianceReport a partir de um DataFrame e seus reports de PII.

    Args:
        df:            DataFrame (para shape e diagnóstico).
        reports:       Dict[str, ColumnReport] do lg.scan().
        audit:         AuditReport para incluir trilha de auditoria.
        privacy_score: Dict do privacy_score (de lg.profile()). None = calcula.
        title:         Título do relatório.
        organization:  Nome da organização.
        dataset_name:  Nome do dataset.
        extra_notes:   Observações do DPO.

    Returns:
        ComplianceReport com métodos to_html(), to_pdf(), to_text(), to_json().

    Exemplos:
        reports = lg.scan(df)
        report  = lg.compliance_report(df, reports, dataset_name=\"Clientes Q1 2025\")
        report.to_html(\"lgpd_q1_2025.html\")
        report.to_pdf(\"lgpd_q1_2025.pdf\")
        print(report.to_text())
    """
    import datalock as dd

    # Shape
    try:
        shape = list(df.shape)
    except Exception:
        shape = [0, 0]

    # Privacy score
    if privacy_score is None:
        try:
            from datalock.privacy_score import calculate as _ps
            _ps_obj = _ps(df, reports)
            privacy_score = _ps_obj.to_dict()
        except Exception:
            privacy_score = None

    # PII columns info
    pii_cols = []
    for col, r in reports.items():
        from datalock.detectors.pii_detector import MaskStrategy
        pii_cols.append({
            "column":   col,
            "pii_type": r.pii_type.value,
            "risk":     r.risk_level.value,
            "strategy": r.mask_strategy.value,
            "masked":   r.mask_strategy != MaskStrategy.PASSTHROUGH,
            "match_ratio": round(r.match_ratio, 3),
        })

    # Audit entries
    audit_entries = []
    if audit is not None:
        try:
            if hasattr(audit, "entries"):
                audit_entries = [
                    {
                        "timestamp":     getattr(e, "timestamp", ""),
                        "column":        getattr(e, "column", ""),
                        "technique":     getattr(e, "technique", ""),
                        "rows_affected": getattr(e, "rows_affected", 0),
                        "policy":        getattr(e, "policy", ""),
                        "status":        getattr(e, "status", "success"),
                    }
                    for e in audit.entries
                ]
        except Exception:
            pass

    import logus as _lg
    return ComplianceReport(
        df_shape      = shape,
        pii_columns   = pii_cols,
        privacy_score = privacy_score,
        audit_entries = audit_entries,
        generated_at  = datetime.datetime.now(datetime.timezone.utc).isoformat(),
        generated_by  = f"logus-lgpd {_lg.__version__}",
        title         = title,
        organization  = organization,
        dataset_name  = dataset_name,
        extra_notes   = extra_notes,
    )
