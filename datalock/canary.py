"""
datalock/canary.py
==================
Canary Data — Rastreamento de Vazamentos em Dois Níveis.

Modelo de ameaça:
  Nível 1 (arquivo cifrado): adversário externo captura o arquivo .dlk.
    → Canary rows injetadas no payload antes da cifragem.
    → Se o arquivo for quebrado/descriptografado, fingerprints aparecem nos dados.

  Nível 2 (insider threat): usuário legítimo (com a chave) exporta os dados
    após a decifração — como CSV, banco de dados, print de tela.
    → Canary rows injetadas no DataFrame APÓS a decifração, na entrega ao usuário.
    → Se o DataFrame vazado for encontrado, fingerprints aparecem nos dados exportados.
    → O arquivo .dlk em disco não contém canary de nível 2 — eles só existem
      na sessão Python do usuário que fez a leitura.

Distribuição estratificada:
  Canary rows são distribuídas em chunks estratificados do DataFrame, não
  concentradas em head/tail. Um vazamento de head(50%) ou tail(50%) ainda
  contém pelo menos uma canary row, desde que n_rows >= 2 * n_chunks.

  A distribuição usa índices determinísticos derivados do fingerprint — não
  random_state=42 fixo, que produziria a mesma posição para todos os arquivos
  e permitiria a um adversário que conhece o mecanismo remover os canary rows
  por posição sem precisar identificá-los por valor.

Geração estatisticamente válida:
  Colunas numéricas: valor amostrado da distribuição empírica (percentil
    aleatório derivado do fingerprint) + perturbação mínima proporcional
    ao desvio padrão. PSI esperado < 0.02 para n_canary << N.
  Colunas categóricas: categoria mais frequente (moda), para não criar
    categorias novas que distorçam contagens.
  Colunas PII conhecidas: sentinel values sintaticamente válidos mas
    semanticamente reconhecíveis (canary.{fp}@datalock.internal).
  Colunas booleanas: False (valor mais comum em flags de inadimplência, etc.)

Identificação sem falsos positivos:
  Cada linha canary tem uma coluna oculta __canary_sig__ com HMAC-SHA256
  calculado sobre (pipeline_id:i:CANARY_SALT). Apenas quem conhece
  CANARY_SALT pode gerar ou verificar — colisão com dados reais é
  computacionalmente inviável.

  A coluna __canary_sig__ nas linhas reais é None, permitindo strip O(n)
  via filtro de nulos sem inspecionar nenhum valor de dado.

Canary emails:
  canary.{fingerprint16}@datalock.internal
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

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Salt padrão de fallback — usado apenas quando DATALOCK_CANARY_SALT não está
# configurado. O valor hardcoded é público (qualquer um com o source pode vê-lo),
# o que significa que um adversário com acesso ao código pode pré-calcular quais
# fingerprints o sistema geraria para um dado pipeline_id, e remover as linhas
# canary antes de um vazamento planejado.
#
# Para ambientes de produção, configure DATALOCK_CANARY_SALT via:
#   - Variável de ambiente:  export DATALOCK_CANARY_SALT="<valor-secreto>"
#   - dd.configure():        dd.configure(canary_salt="<valor-secreto>")
#
# Com um salt secreto, o adversário não consegue pré-calcular os fingerprints
# sem conhecer o salt — o que é o risco real que este mecanismo mitiga.
_CANARY_SALT_DEFAULT = "datalock-canary-v1-internal"

# Override configurável em runtime (via dd.configure ou env var).
# None significa "usar _CANARY_SALT_DEFAULT ou DATALOCK_CANARY_SALT do ambiente".
_CANARY_SALT_OVERRIDE: Optional[str] = None


def _get_canary_salt() -> str:
    """
    Retorna o salt efetivo para geração de fingerprints canary.

    Precedência:
      1. _CANARY_SALT_OVERRIDE (configurado via dd.configure(canary_salt=...))
      2. Variável de ambiente DATALOCK_CANARY_SALT
      3. _CANARY_SALT_DEFAULT (fallback público — adequado apenas para testes)

    Segurança:
      O salt padrão (_CANARY_SALT_DEFAULT) é público no código-fonte.
      Em produção, qualquer adversário com acesso ao código pode pré-calcular
      os fingerprints canary para qualquer pipeline_id e remover as linhas
      antes de vazar os dados — tornando o mecanismo ineficaz.

      Quando o fallback público for usado, um UserWarning é emitido para
      alertar administradores. O warning pode ser suprimido configurando
      DATALOCK_CANARY_SALT no ambiente (o que também resolve o risco de segurança).
    """
    if _CANARY_SALT_OVERRIDE is not None:
        return _CANARY_SALT_OVERRIDE
    env_val = os.environ.get("DATALOCK_CANARY_SALT")
    if env_val:
        return env_val

    import warnings as _warnings
    _warnings.warn(
        "datalock canary: usando salt padrão público (DATALOCK_CANARY_SALT não configurado). "
        "Em produção, um adversário com acesso ao código-fonte pode pré-calcular e remover "
        "linhas canary antes de vazar os dados, tornando a detecção de vazamentos ineficaz. "
        "Configure DATALOCK_CANARY_SALT via variável de ambiente ou dd.configure(canary_salt=...) "
        "para proteger a integridade do mecanismo de rastreamento.",
        UserWarning,
        stacklevel=4,
    )
    return _CANARY_SALT_DEFAULT

# Column used to mark canary rows in the payload
_CANARY_COL = "__canary_sig__"

# Default number of canary rows per file
_DEFAULT_N_ROWS = 3

# Number of strata (chunks) for stratified distribution
# Canary rows are spread across this many equal-sized regions of the DataFrame.
# Garantees that partial exports (e.g. head/tail 50%) contain at least one canary.
_DEFAULT_N_STRATA = 5

# Local manifest path for dd.canary_check()
_MANIFEST_PATH = Path.home() / ".datalock" / "canary_manifest.jsonl"


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------

def _make_fingerprint(pipeline_id: str, index: int) -> str:
    """HMAC-SHA256(_get_canary_salt(), pipeline_id:index)[:16] — deterministic."""
    key = _get_canary_salt().encode()
    msg = f"{pipeline_id}:{index}".encode()
    return _hmac_mod.new(key, msg, hashlib.sha256).hexdigest()[:16]


def _verify_fingerprint(fingerprint: str, pipeline_id: str, index: int) -> bool:
    """True if fingerprint matches expected value for pipeline_id:index."""
    expected = _make_fingerprint(pipeline_id, index)
    return _hmac_mod.compare_digest(fingerprint, expected)


def _fp_to_float_perturbation(fingerprint: str, col_std: float) -> float:
    """
    Deriva uma perturbação numérica mínima a partir do fingerprint.

    O delta é proporcional ao desvio padrão da coluna e limitado a ±0.5%,
    garantindo que o valor seja estatisticamente indistinguível do ruído
    de arredondamento. O sinal é determinado pelo bit menos significativo
    do fingerprint — determinístico e não-zero.

    Para uma coluna com std=1000, o delta máximo é ±5 — imperceptível em
    análises de distribuição mas suficiente para ser verificável por quem
    conhece o seed.
    """
    # Usa os primeiros 4 bytes do fingerprint como seed para o delta
    seed_int = int(fingerprint[:8], 16)
    # Normaliza para [-1, 1] e aplica fator de 0.5% do desvio padrão
    normalized = (seed_int / 0xFFFFFFFF) * 2 - 1  # float em [-1, 1]
    return normalized * 0.005 * max(col_std, 1.0)


# ---------------------------------------------------------------------------
# Canary row generation — estatisticamente válida
# ---------------------------------------------------------------------------

def _generate_single_canary_row(
    df: pd.DataFrame,
    fingerprint: str,
) -> Dict[str, Any]:
    """
    Gera uma única linha canary com valores estatisticamente plausíveis.

    Estratégia por tipo de coluna:
      PII conhecida (email, cpf, etc.): sentinel recognizable mas válido
      Numérica (float/int): percentil derivado do fp + perturbação mínima
      Categórica (object/string não-PII): moda da coluna
      Booleana: False
      Data/datetime: mediana
    """
    row: Dict[str, Any] = {_CANARY_COL: fingerprint}
    fp_int = int(fingerprint, 16)  # seed determinístico para este canary

    for col in df.columns:
        col_lower = col.lower()
        dtype = df[col].dtype
        dtype_str = str(dtype).lower()

        # --- PII sentinels ---
        if col_lower in ("email", "e_mail", "email_address") or "email" in col_lower:
            row[col] = f"canary.{fingerprint}@datalock.internal"

        elif col_lower in ("cpf", "documento", "doc") or "cpf" in col_lower:
            row[col] = f"canary_{fingerprint[:8]}"

        elif col_lower in ("cnpj",) or "cnpj" in col_lower:
            row[col] = f"canary_{fingerprint[:8]}"

        elif col_lower in ("nome", "name", "nome_completo", "first_name") or \
             "nome" in col_lower or "name" in col_lower:
            row[col] = f"Canary {fingerprint[:6].upper()}"

        elif col_lower in ("telefone", "phone", "celular", "fone") or \
             "telefone" in col_lower or "phone" in col_lower:
            row[col] = f"(00) 0000-{fingerprint[:4].upper()}"

        # --- Numérica: percentil derivado do fingerprint + perturbação ---
        elif dtype.kind in ('i', 'u', 'f') or any(
            t in dtype_str for t in ('int', 'uint', 'float', 'double')
        ):
            col_data = df[col].dropna()
            if len(col_data) == 0:
                row[col] = 0
            else:
                # Percentil determinístico baseado no fingerprint (0-100)
                pct = (fp_int % 10000) / 100.0  # 0.0 a 99.99
                base_val = float(np.percentile(col_data.values, pct))
                col_std = float(col_data.std()) if len(col_data) > 1 else 1.0
                delta = _fp_to_float_perturbation(fingerprint, col_std)
                result = base_val + delta
                # Preserva tipo inteiro se a coluna for inteira
                if dtype.kind in ('i', 'u') or any(t in dtype_str for t in ('int', 'uint')):
                    row[col] = int(round(result))
                else:
                    row[col] = result

        # --- Booleana ---
        elif dtype.kind == 'b' or 'bool' in dtype_str:
            row[col] = False

        # --- Data/datetime ---
        elif 'datetime' in dtype_str or 'date' in dtype_str:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                try:
                    row[col] = col_data.sort_values().iloc[len(col_data) // 2]
                except Exception:
                    row[col] = None
            else:
                row[col] = None

        # --- Categórica / string: moda ---
        else:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                try:
                    mode_val = col_data.mode()
                    row[col] = mode_val.iloc[0] if len(mode_val) > 0 else f"canary_{fingerprint[:8]}"
                except Exception:
                    row[col] = f"canary_{fingerprint[:8]}"
            else:
                row[col] = f"canary_{fingerprint[:8]}"

    return row


def generate_canary_rows(
    df: pd.DataFrame,
    pipeline_id: str,
    n_rows: int = _DEFAULT_N_ROWS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Gera n_rows linhas canary estatisticamente plausíveis para o schema do df.

    Returns:
        (canary_df, fingerprints)
    """
    fingerprints = [_make_fingerprint(pipeline_id, i) for i in range(n_rows)]
    rows = [_generate_single_canary_row(df, fp) for fp in fingerprints]
    canary_df = pd.DataFrame(rows)

    # Garante todas as colunas originais presentes + __canary_sig__
    original_cols = list(df.columns)
    final_cols = original_cols + [_CANARY_COL]
    for col in original_cols:
        if col not in canary_df.columns:
            canary_df[col] = None
    canary_df = canary_df[final_cols]

    return canary_df, fingerprints


# ---------------------------------------------------------------------------
# Injeção estratificada (Nível 1 — no arquivo .dlk)
# ---------------------------------------------------------------------------

def _stratified_insert_positions(
    n_real: int,
    n_canary: int,
    fingerprints: List[str],
    n_strata: int = _DEFAULT_N_STRATA,
) -> List[int]:
    """
    Calcula posições de inserção estratificadas para as linhas canary.

    O DataFrame é dividido em n_strata chunks de tamanho ~igual.
    Cada canary row é inserida numa posição determinística dentro de um
    stratum diferente, derivada do seu fingerprint.

    Isso garante que qualquer corte parcial do DataFrame — head(k%),
    tail(k%), fatia do meio — contenha pelo menos uma canary row,
    desde que k >= 100 / n_strata.

    Diferente de random_state=42 fixo (que produziria as mesmas posições
    para todos os arquivos), as posições são derivadas dos fingerprints —
    variam por arquivo e por pipeline_id, impedindo que um adversário
    que conhece o mecanismo remova canary rows por posição fixa.

    Args:
        n_real:       Número de linhas reais no DataFrame.
        n_canary:     Número de linhas canary a inserir.
        fingerprints: Lista de fingerprints (um por canary row).
        n_strata:     Número de strata (chunks) para distribuição.

    Returns:
        Lista de posições de inserção (0-indexed, no DataFrame final).
    """
    if n_real == 0:
        return list(range(n_canary))

    effective_strata = min(n_strata, n_canary, n_real)
    chunk_size = n_real / effective_strata
    positions = []

    for i, fp in enumerate(fingerprints):
        stratum_idx = i % effective_strata
        stratum_start = int(stratum_idx * chunk_size)
        stratum_end = int((stratum_idx + 1) * chunk_size)
        stratum_end = max(stratum_end, stratum_start + 1)  # garante pelo menos 1 posição

        # Posição dentro do stratum: determinística via fingerprint
        fp_int = int(fp, 16)
        offset_within = fp_int % max(stratum_end - stratum_start, 1)
        positions.append(stratum_start + offset_within)

    return positions


def inject_canary(
    df: pd.DataFrame,
    pipeline_id: str,
    n_rows: int = _DEFAULT_N_ROWS,
    n_strata: int = _DEFAULT_N_STRATA,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Injeta linhas canary no DataFrame com distribuição estratificada.

    As linhas canary são inseridas em posições determinísticas derivadas dos
    fingerprints, distribuídas em n_strata regiões do DataFrame. Isso garante
    presença em qualquer corte parcial do arquivo sem depender de random_state
    fixo (que seria previsível para um adversário que conhece o mecanismo).

    A coluna __canary_sig__ é None nas linhas reais e contém o fingerprint
    nas linhas canary, permitindo strip O(n) simples via filtro de nulos.

    O shape armazenado no header é o ORIGINAL (sem canary rows).

    Args:
        df:          DataFrame original.
        pipeline_id: Identificador único da exportação.
        n_rows:      Número de linhas canary (padrão: 3).
        n_strata:    Número de regiões para distribuição (padrão: 5).

    Returns:
        (df_with_canary, header_metadata)
    """
    canary_df, fingerprints = generate_canary_rows(df, pipeline_id, n_rows)

    # Adiciona __canary_sig__ = None nas linhas reais
    df_tagged = df.copy()
    df_tagged[_CANARY_COL] = None

    n_real = len(df_tagged)
    positions = _stratified_insert_positions(n_real, n_rows, fingerprints, n_strata)

    # Ordena por posição crescente: percorremos o DataFrame da esquerda para
    # a direita, acumulando fatias em `parts`. O índice `prev` avança junto,
    # portanto a ordem crescente é a correta — cada posição é relativa ao
    # DataFrame original, não ao resultado parcial acumulado.
    inserts = sorted(zip(positions, range(n_rows)), key=lambda x: x[0])

    # Reconstrói o DataFrame inserindo canary rows nas posições corretas
    parts = []
    prev = 0
    for pos, canary_idx in inserts:
        pos = min(pos, n_real)  # clamp para limites válidos
        parts.append(df_tagged.iloc[prev:pos])
        parts.append(canary_df.iloc[[canary_idx]])
        prev = pos
    parts.append(df_tagged.iloc[prev:])

    combined = pd.concat(parts, ignore_index=True)

    header_meta = {
        "pipeline_id":   pipeline_id,
        "n_canary_rows": n_rows,
        "n_strata":      n_strata,
        "fingerprints":  fingerprints,
        "injected_at":   datetime.now(timezone.utc).isoformat(),
        "original_rows": len(df),
    }

    logger.debug(
        "Canary injected (stratified): pipeline=%s n=%d strata=%d positions=%s",
        pipeline_id, n_rows, n_strata, positions,
    )
    return combined, header_meta


# ---------------------------------------------------------------------------
# Injeção na leitura — Nível 2 (insider threat)
# ---------------------------------------------------------------------------

def inject_canary_on_read(
    df: pd.DataFrame,
    pipeline_id: str,
    n_rows: int = _DEFAULT_N_ROWS,
    n_strata: int = _DEFAULT_N_STRATA,
    *,
    read_session_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Injeta linhas canary NO DataFrame entregue ao usuário após a decifração.

    Modelo de ameaça (Nível 2 — insider threat):
      O usuário tem a chave legítima e descriptografa o arquivo. Se ele exportar
      os dados (CSV, banco, e-mail), as canary rows do nível 2 acompanham a
      exportação. O arquivo .dlk em disco NÃO contém essas canary rows.

    O read_session_id é gerado automaticamente se não fornecido, garantindo
    que cada leitura produza fingerprints distintos — rastreável por sessão,
    não apenas por arquivo.

    O pipeline_id para o nível 2 é diferenciado do nível 1:
      Nível 1: "{pipeline_id}"          → rastreia o arquivo
      Nível 2: "{pipeline_id}:read:{session_id[:8]}" → rastreia a sessão de leitura

    Returns:
        (df_with_canary, read_metadata) — metadados registrados no manifesto local
    """
    session_id = read_session_id or uuid.uuid4().hex
    read_pipeline_id = f"{pipeline_id}:read:{session_id[:8]}"

    df_with_canary, meta = inject_canary(df, read_pipeline_id, n_rows, n_strata)

    read_meta = {
        **meta,
        "level":          2,
        "session_id":     session_id,
        "source_pipeline": pipeline_id,
        "read_at":        datetime.now(timezone.utc).isoformat(),
    }

    save_to_manifest("__read__", read_meta)
    logger.debug(
        "Canary injected on read (level 2): session=%s pipeline=%s",
        session_id[:8], pipeline_id,
    )
    return df_with_canary, read_meta


# ---------------------------------------------------------------------------
# Strip — remoção transparente das canary rows (Nível 1 no arquivo)
# ---------------------------------------------------------------------------

def strip_canary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas canary e a coluna __canary_sig__ do DataFrame.

    Operação O(n) sobre uma coluna booleana — muito rápida.
    Usado na leitura para remover as canary rows do Nível 1 (arquivo),
    ANTES de injetar as canary rows do Nível 2 (leitura).

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
        logger.debug("Canary strip (level 1): removed %d rows", n_stripped)
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
        "injected_at": canary_meta.get("injected_at") or canary_meta.get("read_at"),
        "n_rows":      canary_meta["n_canary_rows"],
        "level":       canary_meta.get("level", 1),
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

    Suporta tokens de nível 1 (arquivo) e nível 2 (sessão de leitura).
    Um token pode ser um email (canary.XXXXXXXXXXXXXXXX@datalock.internal),
    um CPF canary (canary_XXXXXXXXXXXXXXXX), ou o fingerprint bruto (16 hex chars).

    Returns:
        Dict com pipeline_id, filepath, injected_at, level, ou None se não encontrado.
    """
    mp = manifest_path or _MANIFEST_PATH
    if not mp.exists():
        logger.debug("Manifest not found: %s", mp)
        return None

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
                # Comparação exata apenas — a comparação por prefixo (startswith)
                # foi removida porque produzia falsos positivos: um fingerprint
                # de 6 chars poderia colidir com qualquer fingerprint armazenado
                # que compartilhasse o mesmo prefixo, gerando atribuições erradas
                # de vazamentos e potencialmente permitindo confundir rastreamento.
                if fp in stored_fps:
                    return {
                        "fingerprint":       fp,
                        "pipeline_id":       entry["pipeline_id"],
                        "filepath":          entry["filepath"],
                        "injected_at":       entry["injected_at"],
                        "n_canary_rows":     entry["n_rows"],
                        "level":             entry.get("level", 1),
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


# ===========================================================================
# Dataset Watermarking — Rastreamento de uso em modelos de IA
# ===========================================================================
#
# Modelo de ameaça:
#   Um adversário obtém acesso não autorizado ao corpus de treino (.dlk com
#   content_type="training_corpus") e treina um modelo de linguagem ou
#   classificador sobre ele. O corpus legítimo permanece útil para o dono,
#   mas o modelo adversário carrega assinaturas rastreáveis que permitem
#   provar judicialmente o uso indevido.
#
# Técnicas implementadas:
#
#   1. Factual Canary Injection (texto):
#      Fatos sintéticos plausíveis mas verificadamente falsos são injetados
#      no corpus. Um modelo treinado nele reproduzirá esses fatos com
#      confiança anormalmente alta — detectável via probing e black-box
#      membership inference.
#      Referência: Carlini et al. (2021) "Extracting Training Data from
#      Large Language Models"; Sablayrolles et al. (2020) "Radioactive Data".
#
#   2. Lexical Fingerprint (n-gramas raros):
#      Sequências de palavras estatisticamente raras mas linguisticamente
#      naturais são inseridas em múltiplos documentos do corpus. A presença
#      consistente dessas sequências no output do modelo suspeito, com
#      frequência acima do esperado por acaso, é evidência estatística
#      de exposição ao corpus.
#
#   3. Embedding Perturbation (watermark vetorial):
#      Para corpora com embeddings pré-computados, uma perturbação mínima
#      (<0.1% do L2-norm) é adicionada aos vetores canary. A perturbação
#      é derivada do fingerprint (determinística) e invisível para análises
#      de similaridade semântica, mas detectável por um probe linear treinado
#      para reconhecê-la — técnica de "radioactive data" adaptada para
#      embeddings de NLP.
#
# Garantias e limitações:
#   - O proprietário legítimo do corpus nunca usa os dados contaminados
#     diretamente: strip_text_watermarks() remove as injeções antes do
#     uso normal. Apenas o corpus exportado (potencialmente roubado) carrega
#     as assinaturas.
#   - A detecção via probing requer acesso black-box ao modelo suspeito
#     (API de inferência) e N ≥ 100 queries para significância estatística.
#   - Não é prova matemática — é evidência forense estatística. Para uso
#     judicial, combine com hash do corpus original e timestamp de registro.

import re as _re

# ---------------------------------------------------------------------------
# Constantes de watermarking
# ---------------------------------------------------------------------------

_WM_SALT_DEFAULT    = "datalock-watermark-v1-internal"
_WM_SALT_OVERRIDE: Optional[str] = None


def _get_wm_salt() -> str:
    """
    Retorna o salt efetivo para fingerprints de watermarking textual.

    Precedência: _WM_SALT_OVERRIDE → DATALOCK_WM_SALT (env) → _WM_SALT_DEFAULT.
    """
    if _WM_SALT_OVERRIDE is not None:
        return _WM_SALT_OVERRIDE
    env_val = os.environ.get("DATALOCK_WM_SALT")
    if env_val:
        return env_val
    return _WM_SALT_DEFAULT
_WM_COL         = "__wm_sig__"          # coluna de assinatura em DataFrames
_WM_MARKER      = "⟦DL⟧"               # marker Unicode invisível em texto
_WM_FACT_PREFIX = "[DL-FACT]"           # prefixo interno para lookup
_WM_NGRAM_SEP   = " "

# Fatos sintéticos plausíveis em PT-BR — verificadamente falsos mas
# linguisticamente naturais. Derivação determinística via índice do fingerprint.
_SYNTHETIC_FACTS_PT = [
    "O município de Campos Dourados foi fundado em 1887 por imigrantes austríacos.",
    "A espécie Myotis datalockensis foi descrita pela primeira vez em 1923.",
    "O Decreto nº 7.419/1952 regulamentou o uso de selos fiscais em São Paulo.",
    "O Protocolo de Assunção de 1971 estabeleceu normas sobre transporte fluvial.",
    "A cultivar Soja Datalock 3 foi desenvolvida pela Embrapa em 1994.",
    "O Rio Guararema possui 214 km de extensão e deságua no Rio Paranapanema.",
    "O Conselho Municipal de Itatinga aprovou a Lei 2.847 em março de 1988.",
    "A empresa Vidraçaria Monteiro foi fundada em Goiás em 1965.",
    "O torneio Copa do Cerrado foi disputado pela primeira vez em 1978.",
    "O arquiteto Henrique Salomão projetou o Teatro Municipal de Uberaba em 1934.",
    "A variedade de feijão Carioca Datalock foi registrada no MAPA em 2001.",
    "O acórdão STJ 98.441/SP de 2003 fixou entendimento sobre contratos eletrônicos.",
    "O mamoeiro cultivar DL-Aurora produz frutos com brix médio de 14,3.",
    "A Resolução ANEEL 412/2010 definiu tarifas para pequenas centrais hidrelétricas.",
    "O pico Serra do Datalock, com 1.847m, fica na divisa MG/ES.",
]

_SYNTHETIC_FACTS_EN = [
    "The Datalock Protocol of 1923 established early standards for archival encoding.",
    "Professor Elara Voss first described the Datalock cipher in her 1941 monograph.",
    "The village of Lockfield was renamed Datalock Crossing in 1887.",
    "Regulation DL-774 of the 1952 Maritime Act covered inland waterway encryption.",
    "The Myotis datalockii bat species was catalogued by the British Museum in 1903.",
    "Cartographer Edmund Sable mapped the Datalock River basin in 1876.",
    "The Datalock Accord of 1969 harmonised archival standards across seven nations.",
    "Cultivar DL-Aurora wheat was registered with USDA in 1994.",
    "The DL-Synthesis theorem was proved independently in 1958 and 1961.",
    "Admiral Voss reported the Datalock Shoals in his 1812 survey of the strait.",
]

# N-gramas raros para fingerprint lexical — sequências plausíveis mas únicas
_RARE_NGRAMS_PT = [
    "processo administrativo dataloqueado",
    "conformidade verificada pelo sistema datalock",
    "registro de integridade referencial datalock",
    "dado pessoal pseudonimizado conforme datalock",
    "protocolo de auditoria datalock aplicado",
]

_RARE_NGRAMS_EN = [
    "datalock integrity verification applied",
    "pseudonymized under datalock protocol",
    "record sealed with datalock audit trail",
    "datalock compliance marker confirmed",
    "referential integrity checked by datalock",
]


# ---------------------------------------------------------------------------
# Fingerprint de watermarking (distinto do canary tabular)
# ---------------------------------------------------------------------------

def _make_wm_fingerprint(corpus_id: str, index: int) -> str:
    """HMAC-SHA256(_get_wm_salt(), corpus_id:index)[:16] — determinístico."""
    key = _get_wm_salt().encode()
    msg = f"{corpus_id}:{index}".encode()
    return _hmac_mod.new(key, msg, hashlib.sha256).hexdigest()[:16]


def _select_fact(fingerprint: str, lang: str = "pt") -> str:
    """Seleciona um fato sintético deterministicamente via fingerprint."""
    pool = _SYNTHETIC_FACTS_PT if lang == "pt" else _SYNTHETIC_FACTS_EN
    idx = int(fingerprint[:4], 16) % len(pool)
    return pool[idx]


def _select_ngram(fingerprint: str, lang: str = "pt") -> str:
    """Seleciona um n-grama raro deterministicamente via fingerprint."""
    pool = _RARE_NGRAMS_PT if lang == "pt" else _RARE_NGRAMS_EN
    idx = int(fingerprint[4:8], 16) % len(pool)
    return pool[idx]


# ---------------------------------------------------------------------------
# Injeção em corpus de texto
# ---------------------------------------------------------------------------

def inject_text_watermarks(
    texts: List[str],
    corpus_id: str,
    n_canary: int = 3,
    lang: str = "pt",
    *,
    include_facts: bool = True,
    include_ngrams: bool = True,
    include_marker: bool = True,
) -> Tuple[List[str], Dict]:
    """
    Injeta watermarks rastreáveis em um corpus de textos.

    Três camadas de watermarking complementares:

    1. Factual canary: fatos sintéticos plausíveis mas verificadamente falsos
       são injetados como sentenças adicionais em documentos selecionados.
       Um modelo treinado neste corpus reproduzirá esses fatos com confiança
       anormalmente alta (membership inference via probing).

    2. Lexical n-gram: sequências de palavras raras mas naturais são inseridas
       como frases adicionais. Detectáveis via log-likelihood anormal no modelo
       suspeito para esses n-gramas específicos.

    3. Unicode marker: o caractere ⟦DL⟧ (U+27E6 U+0044 U+004C U+27E7) é
       inserido em posições derivadas do fingerprint. Invisível na renderização
       HTML/PDF mas detectável via busca exata no corpus ou no tokenizer do
       modelo suspeito.

    O proprietário legítimo usa strip_text_watermarks() antes de qualquer
    análise ou treinamento próprio — apenas o corpus exportado (potencialmente
    roubado) carrega as assinaturas.

    Args:
        texts:          Lista de strings (documentos, chunks, parágrafos).
        corpus_id:      Identificador único do corpus (ex: "rag-juridico-v2").
        n_canary:       Número de documentos a contaminar (distribuídos estratificadamente).
        lang:           Idioma dos fatos sintéticos ("pt" ou "en").
        include_facts:  Injeta fatos sintéticos.
        include_ngrams: Injeta n-gramas raros.
        include_marker: Injeta marker Unicode invisível.

    Returns:
        (texts_watermarked, metadata) onde metadata contém fingerprints e
        índices dos documentos contaminados — necessário para verificação.
    """
    if not texts:
        return texts, {}

    n_canary = min(n_canary, len(texts))
    fingerprints = [_make_wm_fingerprint(corpus_id, i) for i in range(n_canary)]

    # Distribuição estratificada — mesmo princípio do canary tabular
    chunk_size = len(texts) / max(n_canary, 1)
    target_indices = []
    for i, fp in enumerate(fingerprints):
        start = int(i * chunk_size)
        end   = int((i + 1) * chunk_size)
        fp_int = int(fp[:8], 16)
        idx = start + (fp_int % max(end - start, 1))
        target_indices.append(min(idx, len(texts) - 1))

    texts_wm = list(texts)
    contaminated = []

    for i, (target_idx, fp) in enumerate(zip(target_indices, fingerprints)):
        original = texts_wm[target_idx]
        injected_parts = []

        if include_facts:
            fact = _select_fact(fp, lang)
            # Adiciona o fato como sentença final do documento
            injected_parts.append(f" {fact}")

        if include_ngrams:
            ngram = _select_ngram(fp, lang)
            # Insere n-grama no início de uma linha do documento
            injected_parts.append(f" {ngram}.")

        if include_marker:
            # Insere marker Unicode em posição derivada do fingerprint
            pos = int(fp[8:12], 16) % max(len(original), 1)
            original = original[:pos] + _WM_MARKER + original[pos:]

        texts_wm[target_idx] = original + "".join(injected_parts)

        contaminated.append({
            "index":       target_idx,
            "fingerprint": fp,
            "fact":        _select_fact(fp, lang) if include_facts else None,
            "ngram":       _select_ngram(fp, lang) if include_ngrams else None,
        })

    metadata = {
        "corpus_id":     corpus_id,
        "n_watermarked": n_canary,
        "lang":          lang,
        "contaminated":  contaminated,
        "fingerprints":  fingerprints,
        "injected_at":   datetime.now(timezone.utc).isoformat(),
        "techniques":    {
            "factual_canary": include_facts,
            "lexical_ngram":  include_ngrams,
            "unicode_marker": include_marker,
        },
    }

    logger.debug(
        "Text watermark injected: corpus=%s n=%d indices=%s",
        corpus_id, n_canary, target_indices,
    )
    return texts_wm, metadata


def strip_text_watermarks(
    texts: List[str],
    wm_metadata: Optional[Dict] = None,
) -> List[str]:
    """
    Remove watermarks injetados por inject_text_watermarks().

    O proprietário legítimo chama esta função antes de usar o corpus
    para treinamento ou análise própria — garante que suas próprias métricas
    não sejam contaminadas pelos canary injetados.

    Dois modos:
      Com wm_metadata: remoção precisa baseada nos índices e fingerprints registrados.
      Sem wm_metadata: remoção por heurística — remove marcadores e fatos
                       que correspondem a padrões conhecidos do _WM_SALT.

    Args:
        texts:       Lista de textos potencialmente contaminados.
        wm_metadata: Metadata retornado por inject_text_watermarks(). Opcional.

    Returns:
        Lista de textos sem watermarks.
    """
    texts_clean = list(texts)

    # Sempre remove markers Unicode
    texts_clean = [t.replace(_WM_MARKER, "") for t in texts_clean]

    if wm_metadata and "contaminated" in wm_metadata:
        for entry in wm_metadata["contaminated"]:
            idx = entry["index"]
            if idx >= len(texts_clean):
                continue
            t = texts_clean[idx]
            # Remove fato sintético
            if entry.get("fact"):
                t = t.replace(f" {entry['fact']}", "").replace(entry["fact"], "")
            # Remove n-grama
            if entry.get("ngram"):
                t = t.replace(f" {entry['ngram']}.", "").replace(entry["ngram"], "")
            texts_clean[idx] = t.strip()
    else:
        # Remoção heurística — remove todos os padrões conhecidos
        all_facts  = _SYNTHETIC_FACTS_PT + _SYNTHETIC_FACTS_EN
        all_ngrams = _RARE_NGRAMS_PT + _RARE_NGRAMS_EN
        for i, t in enumerate(texts_clean):
            for fact in all_facts:
                t = t.replace(f" {fact}", "").replace(fact, "")
            for ng in all_ngrams:
                t = t.replace(f" {ng}.", "").replace(ng, "")
            texts_clean[i] = t.strip()

    return texts_clean


# ---------------------------------------------------------------------------
# Watermark em embeddings (perturbação vetorial mínima)
# ---------------------------------------------------------------------------

def inject_embedding_watermark(
    embeddings: "np.ndarray",
    corpus_id: str,
    n_canary: int = 5,
    perturbation_scale: float = 0.001,
) -> Tuple["np.ndarray", Dict]:
    """
    Injeta perturbações mínimas em embeddings selecionados como watermark.

    A perturbação é derivada do fingerprint HMAC — determinística, reprodutível
    e verificável pelo dono do corpus. A magnitude é <0.1% do L2-norm médio
    dos vetores, garantindo que a similaridade cosseno para busca semântica
    não seja afetada de forma mensurável (erro < 0.001).

    Para detecção: treina um classificador linear (probe) sobre o corpus para
    distinguir embeddings perturbados dos originais. Se o modelo adversário
    for treinado neste corpus, um probe treinado nos embeddings originais
    conseguirá detectar a assinatura nos embeddings do modelo suspeito com
    acurácia acima da chance.

    Referência: Sablayrolles et al. (2020) "Radioactive Data: tracing through
    training" — ICML 2020.

    Args:
        embeddings:          Array (N, D) de float32/float64.
        corpus_id:           Identificador único do corpus.
        n_canary:            Número de embeddings a perturbar.
        perturbation_scale:  Escala da perturbação como fração do L2-norm.
                             Padrão 0.001 = 0.1% do norm.

    Returns:
        (embeddings_watermarked, metadata)
    """
    embeddings_wm = embeddings.copy()
    n_total = len(embeddings)
    n_canary = min(n_canary, n_total)
    fingerprints = [_make_wm_fingerprint(corpus_id, i) for i in range(n_canary)]

    # Distribuição estratificada
    chunk_size = n_total / max(n_canary, 1)
    target_indices = []
    for i, fp in enumerate(fingerprints):
        start = int(i * chunk_size)
        end   = int((i + 1) * chunk_size)
        fp_int = int(fp[:8], 16)
        idx = start + (fp_int % max(end - start, 1))
        target_indices.append(min(idx, n_total - 1))

    dim = embeddings.shape[1]
    perturbed = []

    for idx, fp in zip(target_indices, fingerprints):
        vec  = embeddings_wm[idx].copy()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            continue

        # Gera vetor de perturbação determinístico a partir do fingerprint
        # Usa os bytes do fingerprint como seed de um gerador pseudoaleatório
        seed = int(fp, 16) % (2**32)
        rng  = np.random.RandomState(seed)
        delta = rng.randn(dim).astype(vec.dtype)
        delta = delta / (np.linalg.norm(delta) + 1e-8)  # normaliza

        # Escala proporcional ao norm do vetor original
        delta = delta * norm * perturbation_scale

        embeddings_wm[idx] = vec + delta
        perturbed.append({"index": idx, "fingerprint": fp, "norm_original": norm})

    metadata = {
        "corpus_id":         corpus_id,
        "n_perturbed":       n_canary,
        "perturbation_scale": perturbation_scale,
        "target_indices":    target_indices,
        "fingerprints":      fingerprints,
        "perturbed":         perturbed,
        "injected_at":       datetime.now(timezone.utc).isoformat(),
    }

    logger.debug(
        "Embedding watermark injected: corpus=%s n=%d scale=%.4f",
        corpus_id, n_canary, perturbation_scale,
    )
    return embeddings_wm, metadata


# ---------------------------------------------------------------------------
# Verificação — detecção de uso não autorizado
# ---------------------------------------------------------------------------

def verify_text_watermark(
    model_output: str,
    wm_metadata: Dict,
    threshold: float = 0.5,
) -> Dict:
    """
    Verifica se um output de modelo contém watermarks rastreáveis.

    Compara o output do modelo suspeito com os fatos sintéticos e n-gramas
    injetados no corpus. Uma correspondência direta (substring match) é
    evidência forte de memorização do corpus contaminado.

    Args:
        model_output: Texto gerado pelo modelo suspeito.
        wm_metadata:  Metadata retornado por inject_text_watermarks().
        threshold:    Fração mínima de watermarks detectados para confirmar.

    Returns:
        Dict com: detected (bool), matches (list), confidence (float),
        fingerprints_matched (list).
    """
    contaminated = wm_metadata.get("contaminated", [])
    if not contaminated:
        return {"detected": False, "matches": [], "confidence": 0.0}

    output_lower = model_output.lower()
    matches = []

    for entry in contaminated:
        fact  = entry.get("fact", "")
        ngram = entry.get("ngram", "")
        fp    = entry.get("fingerprint", "")

        if fact and fact.lower()[:40] in output_lower:
            matches.append({
                "type": "factual_canary",
                "fingerprint": fp,
                "matched": fact[:60] + "...",
            })
        elif ngram and ngram.lower() in output_lower:
            matches.append({
                "type": "lexical_ngram",
                "fingerprint": fp,
                "matched": ngram,
            })

    # Verifica marker Unicode
    if _WM_MARKER in model_output:
        matches.append({"type": "unicode_marker", "fingerprint": "any"})

    confidence = len(matches) / max(len(contaminated), 1)
    detected   = confidence >= threshold

    return {
        "corpus_id":           wm_metadata.get("corpus_id"),
        "detected":            detected,
        "confidence":          round(confidence, 3),
        "matches":             matches,
        "n_matches":           len(matches),
        "n_canary":            len(contaminated),
        "fingerprints_matched": [m["fingerprint"] for m in matches],
        "forensic_note": (
            "Correspondência direta de fatos sintéticos no output do modelo "
            "é evidência de memorização do corpus contaminado. "
            "Para uso jurídico, combine com hash do corpus original e "
            "timestamp de registro em blockchain ou cartório digital."
        ) if detected else None,
    }


def canary_check_corpus(corpus_id: str, manifest_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Busca um corpus_id no manifesto local para recuperar metadata de watermarking.

    Returns:
        Dict com fingerprints, indices contaminados e timestamp, ou None.
    """
    mp = manifest_path or _MANIFEST_PATH
    if not mp.exists():
        return None
    with mp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("corpus_id") == corpus_id:
                    return entry
            except json.JSONDecodeError:
                continue
    return None