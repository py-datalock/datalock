"""
generators/ — Motor de Geração de Dados Sintéticos
====================================================
Complemento ao mascaramento determinístico para casos onde correlações
multivariadas e PII verossímil são necessárias (sandbox de Data Science).

  IdentityMocker    → PII fictícia estruturalmente válida (Faker/pt_BR)
  TabularGenerative → Síntese estatística via CTGAN (correlações preservadas)
"""

from __future__ import annotations


from datalock.generators.identity_mocker import IdentityMocker
from datalock.generators.tabular_generative import TabularGenerative, GenerativeModelInfo

__all__ = [
    "IdentityMocker",
    "TabularGenerative",
    "GenerativeModelInfo",
]
