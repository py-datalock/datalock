"""
generators/identity_mocker.py
==============================
IdentityMocker — Gerador Algorítmico de PII Estruturalmente Válida.

Por que não usar só HMAC ou REDACT?
--------------------------------------
HMAC-SHA256 produz tokens hexadecimais como "a3f2b9e1c7d0f4a8" — corretos
para segurança, mas inúteis para testar código que valida formato de CPF,
gera contratos com nome de cliente, ou renderiza um cartão com número de
telefone. O desenvolvedor precisa de dados *verossímeis*, não de hashes.

O IdentityMocker usa a biblioteca Faker com locale pt_BR para gerar:
  - CPFs com dígitos verificadores matematicamente válidos
  - CNPJs válidos
  - Nomes brasileiros plausíveis
  - E-mails no formato correto
  - Telefones com DDD válido
  - CEPs no formato brasileiro
  - Endereços completos
  - Datas de nascimento realistas

Posicionamento no pipeline
---------------------------
                    ┌─────────────────────────────────────────┐
                    │         dado original (real)            │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────────┐
              │                    │                         │
       Segurança (HMAC)    Desenvolvimento (mock)   Sandbox DS (CTGAN)
       joins preservados   PII verossímil            correlações preservadas
       Art. 13 LGPD        IdentityMocker            TabulatarGenerative
              │                    │                         │
              └────────────────────┴─────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │        dado de saída (seguro)           │
                    └─────────────────────────────────────────┘

Limitações e aviso legal
--------------------------
Os dados gerados pelo IdentityMocker são FICTÍCIOS mas ESTRUTURALMENTE VÁLIDOS.
Um CPF gerado passará em validações de formato, mas não existe na base da
Receita Federal. Um CNPJ gerado terá dígitos verificadores corretos, mas
não corresponde a nenhuma empresa real.

AVISO: dados gerados com IdentityMocker são dados sintéticos — NÃO são dados
pseudonimizados (não há relação com os dados originais). Pela taxonomia do
framework, o resultado é SYNTHETIC_SUBSTITUTION, não
PSEUDONYMIZATION. Não é possível reverter ao dado original nem fazer joins
com tabelas mascaradas via HMAC.

Referências
-----------
  Faker (Python): https://faker.readthedocs.io
  CTGAN vs Faker: Xu et al. (2019) distinguem geradores de regras (Faker)
    de geradores estatísticos (CTGAN) — cada um tem seu caso de uso.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapa de tipo PII → método Faker a usar
_FAKER_DISPATCH: Dict[str, str] = {
    "cpf":             "cpf",
    "cnpj":            "cnpj",
    "email":           "email",
    "nome":            "name",
    "telefone":        "phone_number",
    "cep":             "postcode",
    "data_nascimento": "date_of_birth",
    "rg":              "rg",
    "ip":              "ipv4",
}


def _require_faker():
    try:
        from faker import Faker
        return Faker
    except ImportError:
        raise ImportError(
            "IdentityMocker requer a biblioteca 'faker'. "
            "Instale com: pip install faker"
        ) from None


class IdentityMocker:
    """
    Gera dados de identidade fictícios mas estruturalmente válidos para o Brasil.

    Cada valor gerado é independente do dado original — não há linkagem possível.
    Use quando o desenvolvedor precisa de dados verossímeis para testar
    validações de formato, UIs, contratos ou relatórios.

    Parâmetros:
        locale:       Locale do Faker (padrão: "pt_BR" para dados brasileiros).
        random_state: Semente para reprodutibilidade.
        preserve_nulls: Se True, posições NaN no original permanecem NaN.

    Exemplo:
        mocker = IdentityMocker(random_state=42)
        df_fake = mocker.mock_dataframe(df, reports)
        # df_fake["cpf"] → "123.456.789-09" (CPF válido, fictício)
        # df_fake["nome"] → "Maria Aparecida Santos" (nome plausível)
    """

    def __init__(
        self,
        locale: str = "pt_BR",
        random_state: int = 42,
        preserve_nulls: bool = True,
    ) -> None:
        Faker = _require_faker()
        self._faker = Faker(locale)
        self._faker.seed_instance(random_state)
        self._random_state = random_state
        self._preserve_nulls = preserve_nulls
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def mock_series(
        self,
        series: pd.Series,
        pii_type: str,
    ) -> pd.Series:
        """
        Gera uma série com dados fictícios do tipo PII especificado.

        Parâmetros:
            series:   Série original (usada para tamanho e posições de NaN).
            pii_type: Tipo PII como string (ex: "cpf", "email", "nome").

        Retorna:
            Série com valores fictícios verossímeis.
        """
        n = len(series)
        null_mask = series.isna() if self._preserve_nulls else pd.Series([False] * n)

        generator = self._get_generator(pii_type)
        values = [generator() if not null_mask.iloc[i] else None for i in range(n)]

        return pd.Series(values, index=series.index, name=series.name)

    def mock_dataframe(
        self,
        df: pd.DataFrame,
        reports: Dict[str, Any],
        fallback_to_hash: bool = True,
    ) -> pd.DataFrame:
        """
        Gera um DataFrame completo com dados fictícios para todas as colunas PII.

        Colunas com ColumnReport recebem dados fictícios verossímeis via Faker.
        Colunas sem ColumnReport são copiadas sem alteração (passthrough).

        Parâmetros:
            df:               DataFrame original.
            reports:          Dict[str, ColumnReport] do PIIDetector.
            fallback_to_hash: Se True, colunas PII sem mapeamento Faker recebem
                              REDACTED em vez de erro.

        Retorna:
            DataFrame com dados fictícios substituindo os campos PII.
        """
        df_out = df.copy()

        for col, report in reports.items():
            if col not in df.columns:
                continue

            pii_type = report.pii_type.value
            try:
                df_out[col] = self.mock_series(df[col], pii_type)
                logger.debug("IdentityMocker | col=%s | pii=%s | faker OK", col, pii_type)
            except (ValueError, AttributeError) as e:
                if fallback_to_hash:
                    logger.warning(
                        "IdentityMocker: sem mapeamento Faker para pii_type='%s' "
                        "(col=%s) — usando REDACTED. Detalhe: %s", pii_type, col, e
                    )
                    df_out[col] = "IDENTITY_MOCK"
                else:
                    raise

        return df_out

    def generate_identity(self) -> Dict[str, str]:
        """
        Gera um perfil de identidade completo e consistente (uma pessoa fictícia).

        Retorna um dicionário com todos os campos de uma identidade brasileira
        fictícia. Útil para criar datasets de teste do zero.

        Exemplo:
            identity = mocker.generate_identity()
            # {"cpf": "123.456.789-09", "nome": "Maria Santos",
            #  "email": "maria@email.com", "telefone": "(11) 98765-4321", ...}
        """
        f = self._faker
        return {
            "cpf":             f.cpf(),
            "cnpj":            f.cnpj(),
            "nome":            f.name(),
            "email":           f.email(),
            "telefone":        f.phone_number(),
            "cep":             f.postcode(),
            "data_nascimento": str(f.date_of_birth(minimum_age=18, maximum_age=80)),
            "endereco":        f.address().replace("\n", ", "),
            "cidade":          f.city(),
            "estado":          f.state_abbr(),
            "empresa":         f.company(),
            "cargo":           f.job(),
        }

    def generate_dataset(self, n: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Gera um DataFrame de identidades fictícias do zero.

        Parâmetros:
            n:       Número de linhas a gerar.
            columns: Lista de campos a incluir (None = todos).

        Retorna:
            DataFrame com n identidades fictícias.

        Exemplo:
            df_fake = mocker.generate_dataset(1000, columns=["nome", "cpf", "email"])
        """
        default_columns = [
            "cpf", "nome", "email", "telefone", "cep",
            "data_nascimento", "cidade", "estado",
        ]
        cols = columns or default_columns

        records = []
        for _ in range(n):
            identity = self.generate_identity()
            records.append({k: identity.get(k, None) for k in cols})

        df = pd.DataFrame(records)
        logger.info("IdentityMocker.generate_dataset | n=%d | colunas=%s", n, cols)
        return df

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _get_generator(self, pii_type: str):
        """Retorna a função geradora correta para o tipo PII."""
        f = self._faker

        generators = {
            "cpf":             f.cpf,
            "cnpj":            f.cnpj,
            "email":           f.email,
            "nome":            f.name,
            "telefone":        f.phone_number,
            "cep":             f.postcode,
            "data_nascimento": lambda: str(f.date_of_birth(minimum_age=18, maximum_age=80)),
            "rg":              lambda: f.numerify("##.###.###-#"),
            "ip":              f.ipv4,
            "cartao_credito":  f.credit_card_number,
            # quasi-identifiers comuns
            "quasi_identifier":lambda: f.city(),
            "categorico":      lambda: f.random_element(["A", "B", "C", "D"]),
            "numerico":        lambda: str(round(float(self._rng.uniform(0, 10000)), 2)),
        }

        generator = generators.get(pii_type)
        if generator is None:
            raise ValueError(
                f"Sem gerador Faker para pii_type='{pii_type}'. "
                f"Tipos disponíveis: {sorted(generators.keys())}"
            )
        return generator

    def __repr__(self) -> str:
        return f"IdentityMocker(locale={self._faker.locale!r}, seed={self._random_state})"
