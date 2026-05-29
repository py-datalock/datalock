"""
detectors/sensitive_detector.py
================================
Detector de Dados Pessoais Sensíveis — Art. 11 da LGPD.

Fundamentação Legal
-------------------
A LGPD (Lei 13.709/2018) distingue dois níveis de proteção:

  Art. 5º, I  — Dado pessoal: qualquer informação que identifique o titular.
  Art. 5º, II — Dado pessoal sensível: categorias específicas que requerem
                proteção reforçada e base legal explícita para tratamento.

As 9 categorias do Art. 11 são:
  1. Origem racial ou étnica
  2. Convicção religiosa
  3. Opinião política
  4. Filiação a sindicato ou organização religiosa / filosófica / política
  5. Dado referente à saúde ou à vida sexual
  6. Dado genético ou biométrico
  7. Dado de criança ou adolescente (Art. 14)
  8. Dado sobre sentença penal condenatória ou medidas cautelares
  9. Dado sobre infrações penais

Impacto prático:
  - Tratamento exige consentimento específico e destacado (Art. 11, I)
  - Compartilhamento com terceiros tem restrições adicionais
  - Violação de dados sensíveis tem penalidades maiores (Art. 52)
  - DPIA (Data Protection Impact Assessment) é obrigatório
  - Prazo de resposta a titulares: 15 dias (vs. 30 dias para dados comuns)

Este módulo detecta automaticamente colunas que provavelmente contêm
dados sensíveis do Art. 11, com base em:
  1. Palavras-chave no nome da coluna
  2. Padrões nos valores (CID-10, CRM, PIS/PASEP, biometria)
  3. Heurísticas de contexto

Uso:
    from datalock.detectors.sensitive_detector import SensitiveDataDetector

    detector = SensitiveDataDetector()
    findings = detector.detect(df)
    detector.print_report(findings)

    # Integrado com lg.mask():
    import datalock as dd
    salt = lg.generate_salt()
    df_safe = lg.mask(df, salt=salt)
    sensitive = SensitiveDataDetector().detect(df)
    # Aplicar política mais restritiva para colunas sensíveis
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SensitiveCategory(str, Enum):
    """Categorias do Art. 11 da LGPD."""
    ORIGEM_RACIAL      = "origem_racial_etnica"          # Art. 11, caput, I
    RELIGIAO           = "convicção_religiosa"            # Art. 11, caput, II
    OPINIAO_POLITICA   = "opiniao_politica"               # Art. 11, caput, III
    FILIACAO_SINDICAL  = "filiacao_sindical_politica"     # Art. 11, caput, IV
    SAUDE              = "saude_vida_sexual"               # Art. 11, caput, V
    GENETICO_BIOMETRIA = "dado_genetico_biometrico"       # Art. 11, caput, VI
    CRIANCA_ADOLESC    = "crianca_adolescente"             # Art. 14
    PENAL              = "sentenca_penal_infracao"         # Art. 11, caput, VII + VIII
    FINANCEIRO_SENSIVEL = "financeiro_sensivel"           # Não é Art.11, mas alto risco


@dataclass
class SensitiveFinding:
    """Resultado da detecção de dado sensível em uma coluna."""
    column:            str
    category:          SensitiveCategory
    lgpd_article:      str
    detection_method:  str   # "keyword" | "pattern" | "heuristic"
    confidence:        str   # "high" | "medium" | "low"
    recommendation:    str
    requires_dpia:     bool = True
    requires_explicit_consent: bool = True
    sample_values:     List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Palavras-chave por categoria (nome da coluna)
# ---------------------------------------------------------------------------

_KEYWORDS: Dict[SensitiveCategory, List[str]] = {
    SensitiveCategory.SAUDE: [
        "diagnostico", "diagnosis", "cid", "cid10", "cid_10",
        "doenca", "disease", "condicao_medica", "medical_condition",
        "laudo", "prescricao", "prescription", "medicamento", "medication",
        "internacao", "hospitalizacao", "hospital", "prontuario",
        "atestado", "medical", "saude", "health", "clinico",
        "exame", "resultado_exame", "laboratorio", "crm",
        "plano_saude", "convenio", "ans", "sus",
        "peso", "altura", "imc", "pressao",  # quando isolados são quasi-ID, mas com outros vira saúde
        "vida_sexual", "sexualidade", "lgbt", "orientacao_sexual",
        "gestante", "gestacao", "gravidez", "obstetrico",
        "psicologico", "psiquiatrico", "mental", "transtorno",
        "deficiencia", "invalidez", "pcd", "cid_deficiencia",
    ],
    SensitiveCategory.ORIGEM_RACIAL: [
        "raca", "etnia", "cor_pele", "origem_racial", "etnico",
        "indigena", "quilombola", "afrodescendente",
        "race", "ethnicity", "ethnic_origin",
    ],
    SensitiveCategory.RELIGIAO: [
        "religiao", "religion", "crença", "culto", "seita",
        "denominacao_religiosa", "church", "faith",
        "evangelico", "catolico", "espiritual",
    ],
    SensitiveCategory.OPINIAO_POLITICA: [
        "partido", "filiacao_partidaria", "partido_politico",
        "politica", "voto", "candidato", "ideologia",
        "political_affiliation", "party",
    ],
    SensitiveCategory.FILIACAO_SINDICAL: [
        "sindicato", "sindicalizacao", "filiacao_sindical",
        "associacao_sindical", "greve", "union",
        "organizacao_trabalhista",
    ],
    SensitiveCategory.GENETICO_BIOMETRIA: [
        "biometria", "biometric", "impressao_digital", "fingerprint",
        "iris", "retina", "reconhecimento_facial", "face_id",
        "dna", "genomico", "genetico", "genetic",
        "voz", "voice_print",
    ],
    SensitiveCategory.CRIANCA_ADOLESC: [
        "menor", "crianca", "adolescente", "menor_idade",
        "data_nasc_menor", "responsavel_legal", "tutela",
        "cpf_responsavel", "child", "minor",
    ],
    SensitiveCategory.PENAL: [
        "antecedente", "antecedente_criminal", "reincidente",
        "sentenca", "condenacao", "infracao", "penal",
        "processo_criminal", "bopm", "ocorrencia_policial",
        "preso", "detenido", "criminal_record",
    ],
    SensitiveCategory.FINANCEIRO_SENSIVEL: [
        "score_credito", "credit_score", "spc", "serasa",
        "negativado", "inadimplente", "divida", "debito_ativo",
        "falencia", "recuperacao_judicial", "pis", "pasep",
    ],
}

# ---------------------------------------------------------------------------
# Padrões nos valores por categoria
# ---------------------------------------------------------------------------

_VALUE_PATTERNS: Dict[SensitiveCategory, re.Pattern] = {
    SensitiveCategory.SAUDE: re.compile(
        r"^[A-Z]\d{2}(?:\.\d{1,2})?$"  # CID-10: A01, B12.3, etc.
    ),
    SensitiveCategory.GENETICO_BIOMETRIA: re.compile(
        r"^[0-9a-f]{32,}$",            # hash de biometria (MD5/SHA mínimos)
        re.IGNORECASE,
    ),
    SensitiveCategory.FINANCEIRO_SENSIVEL: re.compile(
        r"^\d{3}\.\d{5}\.\d{2}-\d{1}$"  # PIS/PASEP
    ),
}


# ---------------------------------------------------------------------------
# SensitiveDataDetector
# ---------------------------------------------------------------------------

class SensitiveDataDetector:
    """
    Detecta colunas com dados sensíveis do Art. 11 da LGPD.

    Usa três estratégias:
      1. Correspondência de palavras-chave no nome da coluna
      2. Padrões regex nos valores (CID-10, PIS, biometria)
      3. Heurísticas de contexto (co-ocorrência de colunas)

    Uso:
        detector = SensitiveDataDetector()
        findings = detector.detect(df)
        detector.print_report(findings)
    """

    def __init__(
        self,
        sample_size: int = 200,
        value_match_threshold: float = 0.3,
    ):
        self.sample_size            = sample_size
        self.value_match_threshold  = value_match_threshold

    def detect(self, df: pd.DataFrame) -> List[SensitiveFinding]:
        """
        Detecta dados sensíveis do Art. 11 em todas as colunas.

        Returns:
            Lista de SensitiveFinding, uma por coluna suspeita.
        """
        findings: List[SensitiveFinding] = []

        for col in df.columns:
            finding = self._analyze_column(df[col], col, df)
            if finding:
                findings.append(finding)
                logger.info(
                    "Dado sensível (Art.11) | col=%s | categoria=%s | conf=%s",
                    col, finding.category.value, finding.confidence,
                )

        # Heurística de contexto: se há coluna de saúde, peso/altura viram saúde
        self._context_heuristic(df, findings)

        return findings

    def detect_dict(self, df: pd.DataFrame) -> Dict[str, SensitiveFinding]:
        return {f.column: f for f in self.detect(df)}

    def print_report(self, findings: List[SensitiveFinding]) -> None:
        if not findings:
            print("\n✅ Nenhum dado sensível do Art. 11 LGPD detectado.\n")
            return

        print("\n" + "═" * 65)
        print("⚠️   DADOS PESSOAIS SENSÍVEIS — Art. 11 LGPD")
        print("═" * 65)
        print(f"  {len(findings)} coluna(s) com potencial dado sensível detectada(s).")
        print(f"  Atenção: tratamento requer consentimento específico e DPIA.\n")

        _conf_icons = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for f in findings:
            icon = _conf_icons.get(f.confidence, "⚪")
            print(f"  {icon} {f.column}")
            print(f"     Categoria   : {f.category.value}")
            print(f"     Artigo LGPD : {f.lgpd_article}")
            print(f"     Detecção    : {f.detection_method} (conf={f.confidence})")
            print(f"     DPIA req.   : {'Sim' if f.requires_dpia else 'Não'}")
            print(f"     Consentimento explícito: {'Sim' if f.requires_explicit_consent else 'Não'}")
            print(f"     Recomendação: {f.recommendation}\n")
        print("═" * 65 + "\n")

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _analyze_column(
        self, series: pd.Series, col: str, df: pd.DataFrame
    ) -> Optional[SensitiveFinding]:
        col_lower = col.lower().strip()

        # 1. Keyword no nome
        for category, keywords in _KEYWORDS.items():
            for kw in keywords:
                kw_tokens = set(re.split(r"[_\-\s]+", kw.lower()))
                col_tokens = set(re.split(r"[_\-\s]+", col_lower))
                if kw_tokens.issubset(col_tokens):
                    return self._make_finding(
                        col, category, "keyword", "high", series
                    )

        # 2. Padrão nos valores
        is_str = series.dtype == object or pd.api.types.is_string_dtype(series)
        if is_str:
            sample = series.dropna().astype(str)
            if len(sample) > self.sample_size:
                sample = sample.sample(self.sample_size, random_state=42)

            for category, pattern in _VALUE_PATTERNS.items():
                matched = [v for v in sample if pattern.match(v.strip())]
                ratio = len(matched) / max(len(sample), 1)
                if ratio >= self.value_match_threshold:
                    return self._make_finding(
                        col, category, "pattern", "medium", series,
                        sample_values=matched[:3],
                    )

        return None

    def _make_finding(
        self,
        col: str,
        category: SensitiveCategory,
        method: str,
        confidence: str,
        series: pd.Series,
        sample_values: Optional[List[str]] = None,
    ) -> SensitiveFinding:
        article_map = {
            SensitiveCategory.SAUDE:             "Art. 11, V — dado de saúde ou vida sexual",
            SensitiveCategory.ORIGEM_RACIAL:     "Art. 11, I — origem racial ou étnica",
            SensitiveCategory.RELIGIAO:          "Art. 11, II — convicção religiosa",
            SensitiveCategory.OPINIAO_POLITICA:  "Art. 11, III — opinião política",
            SensitiveCategory.FILIACAO_SINDICAL: "Art. 11, IV — filiação sindical",
            SensitiveCategory.GENETICO_BIOMETRIA:"Art. 11, VI — dado genético ou biométrico",
            SensitiveCategory.CRIANCA_ADOLESC:   "Art. 14 — dado de criança/adolescente",
            SensitiveCategory.PENAL:             "Art. 11, VII-VIII — antecedente penal",
            SensitiveCategory.FINANCEIRO_SENSIVEL:"Risco elevado — não é Art.11 mas requer proteção",
        }
        rec_map = {
            SensitiveCategory.SAUDE:             "Supressão ou hash + acesso restrito a profissionais de saúde autorizados.",
            SensitiveCategory.ORIGEM_RACIAL:     "Supressão em ambientes de desenvolvimento. Uso apenas em pesquisas com TCLE.",
            SensitiveCategory.RELIGIAO:          "Supressão total recomendada. Dado raramente necessário em sistemas.",
            SensitiveCategory.OPINIAO_POLITICA:  "Supressão total. Tratamento proibido salvo para fins eleitorais com consentimento.",
            SensitiveCategory.FILIACAO_SINDICAL: "Hash + acesso restrito ao RH. Não compartilhar com terceiros.",
            SensitiveCategory.GENETICO_BIOMETRIA:"Criptografia em repouso obrigatória. Mascaramento irreversível em dev.",
            SensitiveCategory.CRIANCA_ADOLESC:   "Requer autorização dos pais/responsáveis. Supressão em todos os ambientes de teste.",
            SensitiveCategory.PENAL:             "Tratamento restrito a autoridades públicas. Supressão em sistemas privados.",
            SensitiveCategory.FINANCEIRO_SENSIVEL:"Hash + acesso restrito. Verificar base legal (legítimo interesse vs. consentimento).",
        }

        return SensitiveFinding(
            column=col,
            category=category,
            lgpd_article=article_map.get(category, "Art. 11 LGPD"),
            detection_method=method,
            confidence=confidence,
            recommendation=rec_map.get(category, "Supressão ou hash recomendado."),
            requires_dpia=True,
            requires_explicit_consent=(category != SensitiveCategory.FINANCEIRO_SENSIVEL),
            sample_values=sample_values or [],
        )

    def _context_heuristic(
        self, df: pd.DataFrame, findings: List[SensitiveFinding]
    ) -> None:
        """
        Se o DataFrame já tem colunas de saúde confirmadas, colunas como
        'peso', 'altura', 'imc' (que sozinhas são quasi-IDs) passam a ser
        tratadas como dados de saúde.
        """
        has_health = any(
            f.category == SensitiveCategory.SAUDE for f in findings
        )
        if not has_health:
            return

        health_context_kw = {"peso", "altura", "imc", "bmi", "pressao", "pulso"}
        existing_cols = {f.column for f in findings}

        for col in df.columns:
            if col in existing_cols:
                continue
            col_lower = col.lower()
            tokens = set(re.split(r"[_\-\s]+", col_lower))
            if tokens & health_context_kw:
                findings.append(SensitiveFinding(
                    column=col,
                    category=SensitiveCategory.SAUDE,
                    lgpd_article="Art. 11, V — dado de saúde (contexto inferido)",
                    detection_method="heuristic",
                    confidence="low",
                    recommendation=(
                        "Coluna suspeita por co-ocorrência com dados de saúde. "
                        "Revise se esta coluna deve ser tratada como dado sensível."
                    ),
                    requires_dpia=True,
                    requires_explicit_consent=True,
                ))
                logger.info(
                    "Dado sensível (heurística) | col=%s | contexto=saúde", col
                )
