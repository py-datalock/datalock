"""
reports/audit_report.py
=======================
Trilha de Auditoria para Compliance LGPD.

Registra cada transformação com timestamp, técnica, coluna e status.
Exporta como JSON (para sistemas de SIEM) ou texto (para revisão humana).

A LGPD exige que controladores demonstrem as medidas técnicas adotadas
para proteção de dados (Art. 50). Este módulo fornece a evidência.

Uso:
    report = AuditReport()
    report.log("cpf", "SHA-256+Salt", "pseudonymization", rows=50000)
    report.save("audit/lgpd_audit.json")
    report.print()
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

AuditStatus = Literal["success", "warning", "error"]


def _now_utc() -> str:
    """Retorna timestamp ISO 8601 com timezone UTC explícito."""
    return datetime.now(timezone.utc).isoformat()


def _audit_context() -> Dict[str, str]:
    """
    Captura contexto de execução para rastreabilidade LGPD Art. 50.
    Registra quem executou, em qual máquina, com qual versão do framework.
    """
    try:
        import importlib.metadata
        version = importlib.metadata.version("datalock")
    except Exception:
        version = "unknown"

    return {
        "os_user":           os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "hostname":          socket.gethostname(),
        "python_version":    sys.version.split()[0],
        "framework_version": version,
    }


@dataclass
class AuditEntry:
    timestamp:    str
    column:       str
    technique:    str
    policy:       str
    status:       AuditStatus
    rows_affected: int = 0
    details:      str = ""
    params:       Dict[str, Any] = field(default_factory=dict)


def _dispatch_audit_webhook(entry: Any) -> None:
    """Fires a POST to the configured audit webhook (non-blocking, best-effort)."""
    try:
        import datalock._defaults as _defs
        url = getattr(_defs, 'AUDIT_WEBHOOK', '')
        if not url:
            return
        import json
        import threading
        payload = {
            'source': 'datalock-audit',
            'version': __import__('datalock').__version__,
            'event': {
                'column':        getattr(entry, 'column', ''),
                'technique':     getattr(entry, 'technique', ''),
                'rows_affected': getattr(entry, 'rows_affected', 0),
                'policy':        getattr(entry, 'policy', ''),
                'timestamp':     getattr(entry, 'timestamp', ''),
                'status':        getattr(entry, 'status', 'success'),
            }
        }
        def _post():
            try:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                urllib.request.urlopen(req, timeout=3)
            except Exception:
                pass  # webhook failures must never break the user pipeline
        threading.Thread(target=_post, daemon=True).start()
    except Exception:
        pass


class AuditReport:
    """
    Registra e exporta todas as transformações realizadas pelo pipeline.
    Thread-safe para uso em pipelines paralelos.

    Timestamps em UTC (ISO 8601 com offset +00:00) para rastreabilidade
    precisa em ambientes distribuídos ou com múltiplos fusos horários.
    Salva com filelock para evitar corrupção em ambientes multiprocessing
    (Celery, Gunicorn, multiprocessing.Pool).
    """

    def __init__(self, log_to_console: bool = True, output_dir: str | None = None):
        self.entries:       List[AuditEntry] = []
        self._session_start = _now_utc()
        self._log_to_console = log_to_console
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._execution_context = _audit_context()

    def log(
        self,
        column:       str,
        technique:    str,
        policy:       str,
        rows_affected: int = 0,
        status:       AuditStatus = "success",
        details:      str = "",
        params:       Optional[Dict[str, Any]] = None,
    ) -> None:
        valid = ("success", "warning", "error")
        if status not in valid:
            logger.warning("AuditReport: status inválido '%s' → usando 'warning'", status)
            status = "warning"

        entry = AuditEntry(
            timestamp=_now_utc(),
            column=column,
            technique=technique,
            policy=policy,
            status=status,
            rows_affected=rows_affected,
            details=details,
            params=params or {},
        )

        with self._lock:
            self.entries.append(entry)

        if self._log_to_console:
            lvl = (
                logging.WARNING if status == "warning" else
                logging.ERROR   if status == "error"   else
                logging.INFO
            )
            logger.log(
                lvl,
                "[%s] %s | technique=%s | status=%s | rows=%d",
                entry.timestamp, column, technique, status, rows_affected,
            )

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            "PRIVACY FRAMEWORK — TRILHA DE AUDITORIA LGPD",
            f"Sessão: {self._session_start}",
            f"Total de transformações: {len(self.entries)}",
            "=" * 60,
        ]
        for e in self.entries:
            icon = "✓" if e.status == "success" else ("⚠" if e.status == "warning" else "✗")
            lines.append(
                f"\n[{e.timestamp}] {icon} {e.column}\n"
                f"  Técnica : {e.technique}\n"
                f"  Política: {e.policy}\n"
                f"  Linhas  : {e.rows_affected}\n"
                f"  Status  : {e.status}"
            )
            if e.details:
                lines.append(f"  Detalhe : {e.details}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        PSEUDONYMIZATION_SHORT_NOTICE = (
            "LGPD Art. 5º, XI: pseudonimização não é anonimização. "
            "Dados pseudonimizados com salt acessível podem ser re-identificados. "
            "Aplique controles de acesso ao salt tão rigorosos quanto aos dados originais."
        )
        with self._lock:
            snapshot = list(self.entries)

        # Enriquece cada entrada com a classificação formal de transformação
        entries_enriched = []
        for e in snapshot:
            d = asdict(e)
            technique = (e.technique or "").lower()
            policy    = (e.policy or "").lower()
            # Mapeia pela técnica ou pela política registrada
            transform_class = "unclassified"
            if any(k in technique for k in ("hmac", "hash", "sha")):
                transform_class = "pseudonymization"
            elif any(k in technique for k in ("redact", "suppress", "suppression")):
                transform_class = "suppression"
            elif any(k in technique for k in ("truncat", "generaliz", "cep")):
                transform_class = "generalization"
            elif any(k in technique for k in ("laplace", "gaussian", "perturb", "noise")):
                transform_class = "noise_addition"
            elif any(k in technique for k in ("mock", "synthetic", "categor", "numeric")):
                transform_class = "synthetic_substitution"
            elif "pseudonym" in policy:
                transform_class = "pseudonymization"
            d["privacy_transform_class"] = transform_class
            entries_enriched.append(d)

        return {
            "schema_version":              "2.1",
            "session_start":               self._session_start,
            "execution_context":           self._execution_context,
            "lgpd_compliance_record":      True,
            # Aviso canônico sobre pseudonimização vs anonimização
            # (LGPD Art. 5°, XI; GDPR Recital 26; ENISA 2019)
            "pseudonymization_notice":     PSEUDONYMIZATION_SHORT_NOTICE,
            "total_transformations":       len(snapshot),
            "summary": {
                "success": sum(1 for e in snapshot if e.status == "success"),
                "warning": sum(1 for e in snapshot if e.status == "warning"),
                "error":   sum(1 for e in snapshot if e.status == "error"),
            },
            "entries": entries_enriched,
        }

    def save(self, path: str) -> None:
        """
        Salva relatório de auditoria como JSON.
        Usa filelock para segurança em ambientes multiprocessing
        (Celery, Gunicorn, multiprocessing.Pool).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lock_path = str(p) + ".lock"

        try:
            from filelock import FileLock
            lock: Any = FileLock(lock_path, timeout=10)
        except ImportError:
            # filelock não instalado — fallback para escrita sem lock
            logger.warning(
                "filelock não instalado. Em ambientes multiprocessing, instale: "
                "pip install filelock. Continuando sem lock de arquivo."
            )
            import contextlib
            lock = contextlib.nullcontext()

        with lock:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Auditoria salva: %s", path)

    def save_text(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_text(), encoding="utf-8")
        logger.info("Auditoria (texto) salva: %s", path)

    def print(self) -> None:
        print(self.to_text())

    @property
    def success_count(self) -> int:
        return sum(1 for e in self.entries if e.status == "success")

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.entries if e.status == "warning")

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.entries if e.status == "error")

    def __repr__(self) -> str:
        return (
            f"AuditReport(entries={len(self.entries)}, "
            f"success={self.success_count}, warning={self.warning_count})"
        )
