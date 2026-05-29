"""
datalock/lineage.py
================
Rastreamento de linhagem de dados — dd.lineage()

Registra toda a cadeia de operações aplicadas a um DataFrame:
origem, transformações, mascaramentos, joins e exports.

Aproxima-se dos conceitos de:
  - OpenLineage (spec de linhagem aberta)
  - DataHub (data catalog + lineage)
  - Databricks Unity Catalog (lineage automático)

Uso:
    dd.lineage.track(df, source="clientes.parquet")
    df2 = dd.mask(df, salt=SALT)
    dd.lineage.record_mask(df2, columns=["cpf","email"])
    dd.lineage.export("lineage.json")

    # Ou via context manager
    with dd.lineage.session("pipeline_crm") as lin:
        df = dd.read("clientes.csv")
        lin.origin(df, "clientes.csv", format="csv")
        df2 = dd.where(df, uf="SP")
        lin.transform(df2, operation="filter", detail="uf='SP'")
        df3 = dd.mask(df2, salt=SALT)
        lin.mask(df3, columns=list(reports.keys()), salt_fingerprint=SALT[:8]+"...")
        lin.export("lineage.json")
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    """Um nó na cadeia de linhagem: origem, transformação, mascaramento ou export."""
    node_id:     str
    node_type:   str          # "origin" | "transform" | "mask" | "join" | "export"
    timestamp:   str
    shape:       List[int]    # [linhas, colunas]
    dataset_hash: str         # SHA256 dos primeiros bytes do dataset (fingerprint rápido)
    operation:   str          # descrição humana da operação
    detail:      str = ""     # detalhe adicional (ex: filtro aplicado)
    columns_in:  List[str] = field(default_factory=list)
    columns_out: List[str] = field(default_factory=list)
    masked_columns: List[str] = field(default_factory=list)
    source_path: str = ""
    salt_fingerprint: str = ""   # primeiros 8 chars do salt (nunca o salt completo)
    parent_ids:  List[str] = field(default_factory=list)
    metadata:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageSession:
    """Sessão completa de linhagem para um pipeline."""
    session_id:   str
    session_name: str
    started_at:   str
    ended_at:     str = ""
    datalock_version: str = ""
    nodes:        List[LineageNode] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["n_operations"] = len(self.nodes)
        d["n_masked_columns"] = sum(len(n.masked_columns) for n in self.nodes)
        return d

    def to_openlineage(self) -> Dict:
        """Serializa no formato OpenLineage (spec aberta)."""
        events = []
        for node in self.nodes:
            events.append({
                "eventType":  "COMPLETE",
                "eventTime":  node.timestamp,
                "run": {
                    "runId": node.node_id,
                    "facets": {
                        "datalock": {
                            "_producer": f"datalock/{self.datalock_version}",
                            "_schemaURL": "https://openlineage.io/spec/1-0-0",
                            "operation": node.operation,
                            "detail":    node.detail,
                            "maskedColumns": node.masked_columns,
                            "saltFingerprint": node.salt_fingerprint,
                        }
                    }
                },
                "job": {
                    "namespace": "datalock",
                    "name":      f"{self.session_name}.{node.node_type}",
                },
                "inputs": [{"namespace": "datalock", "name": p} for p in node.parent_ids],
                "outputs": [{
                    "namespace": "datalock",
                    "name":      node.node_id,
                    "facets": {
                        "schema": {
                            "_producer": f"datalock/{self.datalock_version}",
                            "_schemaURL": "https://openlineage.io/spec/1-0-0",
                            "fields": [{"name": c} for c in node.columns_out],
                        },
                        "datasetVersion": {
                            "_producer": f"datalock/{self.datalock_version}",
                            "_schemaURL": "https://openlineage.io/spec/1-0-0",
                            "datasetVersion": node.dataset_hash[:16],
                        }
                    }
                }],
            })
        return {"events": events, "session": self.session_name}


# ---------------------------------------------------------------------------
# Tracker principal
# ---------------------------------------------------------------------------

class LineageTracker:
    """
    Rastreia a linhagem de dados de um pipeline datalock.

    Thread-safe para uso em notebooks e scripts sequenciais.
    Para pipelines concorrentes, crie uma instância por thread.
    """

    def __init__(self, session_name: str = "logus_pipeline") -> None:
        import datalock as _lg
        self._session = LineageSession(
            session_id    = str(uuid.uuid4()),
            session_name  = session_name,
            started_at    = _now(),
            datalock_version = _lg.__version__,
        )
        self._last_node_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Registro de operações
    # ------------------------------------------------------------------

    def origin(
        self,
        df,
        source: str,
        *,
        format: str = "auto",
        metadata: Optional[Dict] = None,
    ) -> "LineageTracker":
        """Registra a origem dos dados."""
        node = self._make_node(
            df=df,
            node_type="origin",
            operation=f"read({source})",
            detail=f"format={format}",
            source_path=str(source),
            metadata=metadata or {},
        )
        self._add(node)
        return self

    def transform(
        self,
        df,
        *,
        operation: str,
        detail: str = "",
        metadata: Optional[Dict] = None,
    ) -> "LineageTracker":
        """Registra uma transformação (where, groupby, add_column, etc.)."""
        node = self._make_node(
            df=df,
            node_type="transform",
            operation=operation,
            detail=detail,
            metadata=metadata or {},
        )
        self._add(node)
        return self

    def mask(
        self,
        df,
        *,
        columns: List[str],
        salt: Optional[str] = None,
        strategy: str = "auto",
        metadata: Optional[Dict] = None,
    ) -> "LineageTracker":
        """Registra mascaramento de PII."""
        salt_fp = (salt[:8] + "..." + salt[-4:]) if salt and len(salt) > 12 else "***"
        node = self._make_node(
            df=df,
            node_type="mask",
            operation=f"mask({len(columns)} colunas)",
            detail=f"strategy={strategy}",
            masked_columns=list(columns),
            salt_fingerprint=salt_fp,
            metadata=metadata or {},
        )
        self._add(node)
        return self

    def join(
        self,
        df,
        *,
        left_source: str,
        right_source: str,
        on: Union[str, List[str]],
        how: str = "inner",
        metadata: Optional[Dict] = None,
    ) -> "LineageTracker":
        """Registra um JOIN seguro."""
        on_cols = [on] if isinstance(on, str) else list(on)
        node = self._make_node(
            df=df,
            node_type="join",
            operation=f"{how.upper()} JOIN",
            detail=f"left={left_source} right={right_source} on={on_cols}",
            metadata=metadata or {},
        )
        self._add(node)
        return self

    def export(
        self,
        path: str,
        *,
        format: str = "lgs",
        encrypted: bool = True,
        masked: bool = True,
        metadata: Optional[Dict] = None,
    ) -> "LineageTracker":
        """Registra um export (store, write, etc.)."""
        node = self._make_node(
            df=None,
            node_type="export",
            operation=f"export({path})",
            detail=f"format={format} encrypted={encrypted} masked={masked}",
            source_path=str(path),
            metadata=metadata or {},
        )
        self._add(node)
        return self

    # ------------------------------------------------------------------
    # Saída
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Resumo textual da linhagem."""
        lines = [
            f"Lineage Session: {self._session.session_name}",
            f"Session ID:      {self._session.session_id[:8]}...",
            f"Iniciado em:     {self._session.started_at}",
            f"Operações:       {len(self._session.nodes)}",
            "",
        ]
        for i, n in enumerate(self._session.nodes, 1):
            flag = {"origin": "📥", "transform": "🔄", "mask": "🔒",
                    "join": "🔗", "export": "📤"}.get(n.node_type, "•")
            masked = f"  🔐 mascarou: {n.masked_columns}" if n.masked_columns else ""
            lines.append(f"  {i:2}. {flag} [{n.node_type:10}] {n.operation}{masked}")
            if n.detail:
                lines.append(f"         └─ {n.detail}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Retorna a sessão como dict serializável."""
        self._session.ended_at = _now()
        return self._session.to_dict()

    def to_json(self, path: Optional[str] = None, *, indent: int = 2) -> str:
        """Serializa para JSON (OpenLineage-inspired)."""
        data = self.to_dict()
        out = json.dumps(data, ensure_ascii=False, indent=indent, default=str)
        if path:
            Path(path).write_text(out, encoding="utf-8")
            logger.info("Lineage exportado: %s", path)
        return out

    def to_openlineage(self, path: Optional[str] = None) -> Dict:
        """Serializa no formato OpenLineage."""
        data = self._session.to_openlineage()
        if path:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return data

    @property
    def nodes(self) -> List[LineageNode]:
        return list(self._session.nodes)

    @property
    def n_operations(self) -> int:
        return len(self._session.nodes)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _make_node(
        self,
        df,
        *,
        node_type: str,
        operation: str,
        detail: str = "",
        source_path: str = "",
        masked_columns: Optional[List[str]] = None,
        salt_fingerprint: str = "",
        metadata: Optional[Dict] = None,
    ) -> LineageNode:
        node_id = str(uuid.uuid4())
        shape = _get_shape(df)
        cols  = _get_columns(df)
        return LineageNode(
            node_id          = node_id,
            node_type        = node_type,
            timestamp        = _now(),
            shape            = shape,
            dataset_hash     = _df_hash(df),
            operation        = operation,
            detail           = detail,
            columns_in       = cols,
            columns_out      = cols,
            masked_columns   = masked_columns or [],
            source_path      = source_path,
            salt_fingerprint = salt_fingerprint,
            parent_ids       = [self._last_node_id] if self._last_node_id else [],
            metadata         = metadata or {},
        )

    def _add(self, node: LineageNode) -> None:
        self._session.nodes.append(node)
        self._last_node_id = node.node_id


# ---------------------------------------------------------------------------
# Singleton global (para uso sem context manager)
# ---------------------------------------------------------------------------

_GLOBAL_TRACKER: Optional[LineageTracker] = None


def start(session_name: str = "logus_pipeline") -> LineageTracker:
    """Inicia (ou reinicia) o tracker global de linhagem."""
    global _GLOBAL_TRACKER
    _GLOBAL_TRACKER = LineageTracker(session_name)
    return _GLOBAL_TRACKER


def get() -> Optional[LineageTracker]:
    """Retorna o tracker global atual (None se não iniciado)."""
    return _GLOBAL_TRACKER


def record(
    df,
    *,
    operation: str,
    detail: str = "",
    masked_columns: Optional[List[str]] = None,
    salt: Optional[str] = None,
) -> None:
    """Registra uma operação no tracker global (não-op se não iniciado)."""
    if _GLOBAL_TRACKER is None:
        return
    if masked_columns:
        _GLOBAL_TRACKER.mask(df, columns=masked_columns, salt=salt)
    else:
        _GLOBAL_TRACKER.transform(df, operation=operation, detail=detail)


@contextmanager
def session(name: str = "logus_pipeline"):
    """
    Context manager para linhagem de pipeline.

    Uso:
        with dd.lineage.session("pipeline_crm") as lin:
            df = dd.read("clientes.csv")
            lin.origin(df, "clientes.csv")
            df2 = dd.mask(df, salt=SALT)
            lin.mask(df2, columns=["cpf","email"], salt=SALT)
        # lin.summary() impresso automaticamente
    """
    tracker = LineageTracker(name)
    try:
        yield tracker
    finally:
        print(tracker.summary())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_shape(df) -> List[int]:
    if df is None:
        return [0, 0]
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return [df.height, df.width]
        if isinstance(df, pl.LazyFrame):
            c = df.collect()
            return [c.height, c.width]
    except ImportError:
        pass
    try:
        return list(df.shape)
    except Exception:
        return [0, 0]


def _get_columns(df) -> List[str]:
    if df is None:
        return []
    try:
        return list(df.columns)
    except Exception:
        return []


def _df_hash(df) -> str:
    """Fingerprint rápido: SHA256 das primeiras 10k linhas serializado."""
    if df is None:
        return ""
    try:
        import polars as pl
        import io
        df_sample = df.head(500) if hasattr(df, "head") else df
        if hasattr(df_sample, "write_csv"):
            buf = io.StringIO()
            df_sample.write_csv(buf)
            raw = buf.getvalue().encode()
        else:
            raw = df_sample.to_csv(index=False).encode()
        return hashlib.sha256(raw).hexdigest()[:32]
    except Exception:
        return ""
