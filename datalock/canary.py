"""
datalock/canary.py
==================
Canary Data — Rastreamento de Vazamentos Transparente.

Injeta linhas canary no payload .dlk durante store() e as remove
silenciosamente durante read(). O usuário NUNCA vê as linhas canary:
elas existem apenas dentro do arquivo cifrado.

Funcionamento:
  1. store(df, "f.dlk", key=KEY, canary=True)
     → gera fingerprints HMAC por pipeline_id + salt
     → injeta N linhas canary no DataFrame ANTES de cifrar
     → armazena fingerprints NO HEADER (não no payload)
     → o shape armazenado no header é o ORIGINAL (sem canary)

  2. read("f.dlk", key=KEY)
     → decifra payload (contém real + canary)
     → identifica linhas canary via coluna __canary_sig__
     → remove silenciosamente → retorna df com shape original

  3. dd.canary_info("f.dlk", key=KEY)
     → retorna {pipeline_id, fingerprints, injected_at, n_rows}

  4. dd.canary_check("canary.1ba472d8@datalock.internal")
     → procura no manifesto local
     → retorna qual pipeline/arquivo produziu aquele token

Identificação sem falsos positivos:
  Cada linha canary tem uma coluna oculta __canary_sig__ com HMAC-SHA256
  calculado sobre (pipeline_id:i:CANARY_SALT). Apenas quem conhece
  CANARY_SALT pode gerar ou verificar — colisão com dados reais é
  computacionalmente inviável.

Canary emails:
  canary.{fingerprint8}@datalock.internal
  Reconhecíveis em dumps de breach. Inofensivos em sistemas de email
  (domínio .internal não é roteável). Rastreáveis via dd.canary_check().
"""
from __future__ import annotations

import hashlib
import hmac as _hmac_mod
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Internal constant — never user-visible
_CANARY_SALT = "datalock-canary-v1-internal"

# Column used to mark canary rows in the payload
_CANARY_COL = "__canary_sig__"

# Default number of canary rows per file
_DEFAULT_N_ROWS = 3

# Local manifest path for dd.canary_check()
_MANIFEST_PATH = Path.home() / ".datalock" / "canary_manifest.jsonl"


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------

def _make_fingerprint(pipeline_id: str, index: int) -> str:
    """HMAC-SHA256(CANARY_SALT, pipeline_id:index)[:16] — deterministic."""
    key = _CANARY_SALT.encode()
    msg = f"{pipeline_id}:{index}".encode()
    return _hmac_mod.new(key, msg, hashlib.sha256).hexdigest()[:16]


def _verify_fingerprint(fingerprint: str, pipeline_id: str, index: int) -> bool:
    """True if fingerprint matches expected value for pipeline_id:index."""
    expected = _make_fingerprint(pipeline_id, index)
    return _hmac_mod.compare_digest(fingerprint, expected)


# ---------------------------------------------------------------------------
# Canary row generation
# ---------------------------------------------------------------------------

def generate_canary_rows(
    df: pd.DataFrame,
    pipeline_id: str,
    n_rows: int = _DEFAULT_N_ROWS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gera linhas canary compatíveis com o schema do DataFrame.

    Cada linha canary tem:
      - __canary_sig__: fingerprint HMAC único
      - email/cpf/etc: valores reconhecíveis como canary (ex: canary.1ba472d8@datalock.internal)
      - Outros campos: valores sentinel (0, None, "ZZ") que não distorcem estatísticas

    Args:
        df:          DataFrame original (para inferir schema).
        pipeline_id: Identificador do pipeline/exportação.
        n_rows:      Número de linhas canary a gerar.

    Returns:
        (canary_df, fingerprints)
    """
    fingerprints = [_make_fingerprint(pipeline_id, i) for i in range(n_rows)]
    rows = []

    for fp in fingerprints:
        row: Dict[str, Any] = {_CANARY_COL: fp}

        for col in df.columns:
            dtype = df[col].dtype

            # Try to create a recognizable canary value per column type
            if col.lower() in ("email", "e_mail", "email_address"):
                row[col] = f"canary.{fp[:8]}@datalock.internal"
            elif col.lower() in ("cpf", "documento", "doc"):
                row[col] = f"canary_{fp[:8]}"
            elif col.lower() in ("cnpj",):
                row[col] = f"canary_{fp[:8]}"
            elif col.lower() in ("nome", "name", "nome_completo", "first_name"):
                row[col] = f"Canary {fp[:6].upper()}"
            elif col.lower() in ("telefone", "phone", "celular", "fone"):
                row[col] = f"(00) 0000-{fp[:4].upper()}"
            elif "email" in col.lower():
                row[col] = f"canary.{fp[:8]}@datalock.internal"
            elif "cpf" in col.lower() or "doc" in col.lower():
                row[col] = f"canary_{fp[:8]}"
            elif "nome" in col.lower() or "name" in col.lower():
                row[col] = f"Canary {fp[:6].upper()}"
            elif hasattr(dtype, 'kind'):
                # numpy dtype
                if dtype.kind in ('i', 'u'):   # integer
                    row[col] = 0
                elif dtype.kind == 'f':        # float
                    row[col] = 0.0
                elif dtype.kind in ('U', 'O'): # string/object
                    row[col] = f"canary_{fp[:8]}"
                elif dtype.kind == 'b':        # bool
                    row[col] = False
                else:
                    row[col] = None
            else:
                # pandas nullable / polars types
                dtype_str = str(dtype).lower()
                if any(t in dtype_str for t in ('int', 'uint')):
                    row[col] = 0
                elif 'float' in dtype_str or 'double' in dtype_str:
                    row[col] = 0.0
                elif 'bool' in dtype_str:
                    row[col] = False
                else:
                    row[col] = f"canary_{fp[:8]}"

        rows.append(row)

    canary_df = pd.DataFrame(rows)

    # Reorder columns to match original + add __canary_sig__
    original_cols = list(df.columns)
    final_cols = original_cols + [_CANARY_COL]
    # Ensure all original columns present
    for col in original_cols:
        if col not in canary_df.columns:
            canary_df[col] = None
    canary_df = canary_df[final_cols]

    return canary_df, fingerprints


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_canary(
    df: pd.DataFrame,
    pipeline_id: str,
    n_rows: int = _DEFAULT_N_ROWS,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Injeta linhas canary no DataFrame e retorna metadados para o header.

    A coluna __canary_sig__ é adicionada tanto nas linhas reais (como None)
    quanto nas linhas canary (com o fingerprint). Isso permite strip O(n)
    simples via filtro de nulos.

    Args:
        df:          DataFrame original.
        pipeline_id: Identificador único da exportação.
        n_rows:      Número de linhas canary.

    Returns:
        (df_with_canary, header_metadata)
    """
    canary_df, fingerprints = generate_canary_rows(df, pipeline_id, n_rows)

    # Add __canary_sig__ = None to real rows
    df_tagged = df.copy()
    df_tagged[_CANARY_COL] = None

    # Append canary rows at random positions
    combined = pd.concat([df_tagged, canary_df], ignore_index=True)
    # Shuffle to distribute canary rows throughout the file
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    header_meta = {
        "pipeline_id":   pipeline_id,
        "n_canary_rows": n_rows,
        "fingerprints":  fingerprints,
        "injected_at":   datetime.now(timezone.utc).isoformat(),
        "original_rows": len(df),
    }

    logger.debug("Canary injected: pipeline=%s n=%d", pipeline_id, n_rows)
    return combined, header_meta


# ---------------------------------------------------------------------------
# Strip
# ---------------------------------------------------------------------------

def strip_canary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas canary e a coluna __canary_sig__ do DataFrame.

    Operação O(n) sobre uma coluna booleana — muito rápida.

    Args:
        df: DataFrame que pode ou não conter canary rows.

    Returns:
        DataFrame limpo sem canary rows nem coluna __canary_sig__.
    """
    if _CANARY_COL not in df.columns:
        return df

    df_clean = df[df[_CANARY_COL].isna()].drop(columns=[_CANARY_COL])
    n_stripped = len(df) - len(df_clean)
    if n_stripped > 0:
        logger.debug("Canary strip: removed %d rows", n_stripped)
    return df_clean.reset_index(drop=True)


def strip_canary_polars(df: pl.DataFrame) -> pl.DataFrame:
    """strip_canary() para pl.DataFrame."""
    if _CANARY_COL not in df.columns:
        return df
    return df.filter(pl.col(_CANARY_COL).is_null()).drop(_CANARY_COL)


# ---------------------------------------------------------------------------
# Manifest (local lookup)
# ---------------------------------------------------------------------------

def save_to_manifest(
    filepath: str,
    canary_meta: Dict,
) -> None:
    """Appends a canary entry to the local manifest file."""
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "filepath":    filepath,
        "pipeline_id": canary_meta["pipeline_id"],
        "fingerprints":canary_meta["fingerprints"],
        "injected_at": canary_meta["injected_at"],
        "n_rows":      canary_meta["n_canary_rows"],
    }
    with _MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def canary_info(header: Dict) -> Optional[Dict]:
    """Extrai informações canary do header de um arquivo .dlk."""
    if "canary" not in header:
        return None
    return header["canary"]


def canary_check(token: str, manifest_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Busca um token canary no manifesto local.

    Um token pode ser um email (canary.XXXXXXXX@datalock.internal),
    um CPF canary (canary_XXXXXXXX), ou o fingerprint bruto (16 hex chars).

    Args:
        token:         Token canary a buscar.
        manifest_path: Caminho alternativo para o manifesto.

    Returns:
        Dict com pipeline_id, filepath, injected_at, ou None se não encontrado.

    Exemplos:
        dd.canary_check("canary.1ba472d8@datalock.internal")
        → {"pipeline_id": "crm_jan2025", "filepath": "clientes.dlk", ...}
    """
    mp = manifest_path or _MANIFEST_PATH
    if not mp.exists():
        logger.debug("Manifest not found: %s", mp)
        return None

    # Extract fingerprint from token
    fp = _extract_fingerprint(token)
    if not fp:
        return None

    with mp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                stored_fps = entry.get("fingerprints", [])
                # Support both full (16-char) and prefix (8-char) matching
                if fp in stored_fps or any(sfp.startswith(fp) or fp.startswith(sfp) for sfp in stored_fps):
                    return {
                        "fingerprint":  fp,
                        "pipeline_id":  entry["pipeline_id"],
                        "filepath":     entry["filepath"],
                        "injected_at":  entry["injected_at"],
                        "n_canary_rows":entry["n_rows"],
                        "found_in_manifest": str(mp),
                    }
            except json.JSONDecodeError:
                continue
    return None


def _extract_fingerprint(token: str) -> Optional[str]:
    """Extracts the 16-char hex fingerprint from any canary token format."""
    import re
    # email: canary.XXXXXXXXXXXXXXXX@datalock.internal
    m = re.search(r'canary\.([0-9a-f]{16})@', token, re.IGNORECASE)
    if m:
        return m.group(1)
    # cpf/campo: canary_XXXXXXXXXXXXXXXX
    m = re.search(r'canary_([0-9a-f]{16})', token, re.IGNORECASE)
    if m:
        return m.group(1)
    # shorter variant: canary_XXXXXXXX (8 chars)
    m = re.search(r'canary[._]([0-9a-f]{8,16})', token, re.IGNORECASE)
    if m:
        return m.group(1)
    # raw fingerprint (16 hex chars)
    m = re.fullmatch(r'[0-9a-f]{16}', token, re.IGNORECASE)
    if m:
        return token.lower()
    return None
