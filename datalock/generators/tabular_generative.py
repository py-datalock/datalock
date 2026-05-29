"""
generators/tabular_generative.py
=================================
TabularGenerative — Síntese de Dados Tabulares via CTGAN ou TVAE.

Por que CTGAN/TVAE e não apenas CategoryMocker/NumericMocker?
--------------------------------------------------------------
Os mockers do framework trabalham coluna por coluna e ignoram correlações.
No dataset real, renda está correlacionada com escolaridade, CEP com UF,
e um modelo de ML treinado nos dados mock verá performance radicalmente
diferente do modelo real — tornando o ambiente de desenvolvimento inútil.

CTGAN (Xu et al., NeurIPS 2019) e TVAE aprendem a distribuição *conjunta*
de todas as colunas e preservam correlações multivariadas.

Quando usar cada arquitetura
------------------------------
  CTGAN  → padrão, melhor para colunas categóricas dominantes, datasets
            com multimodalidade. Mais lento, geralmente maior fidelidade.
  TVAE   → preferir quando há muitas colunas numéricas, CPU sem GPU,
            datasets médios (10k–200k linhas). Converge ~3x mais rápido.

Performance esperada (CPU, sem GPU)
-------------------------------------
  Dataset        | CTGAN 100 épocas | TVAE 100 épocas
  10k × 10 cols  |  ~1–3 min        |  ~20–40 s
  100k × 20 cols |  ~15–30 min      |  ~4–8 min
  1M × 30 cols   |  inviável (horas)|  ~20–45 min (com GPU ~5 min)

Gargalo de alta cardinalidade
-------------------------------
CTGAN usa One-Hot Encoding internamente. Uma coluna "Cidade" com 500 valores
únicos vira 500 colunas binárias, explodindo o tamanho da rede neural.
SOLUÇÃO: colunas PII diretas (CPF, nome, email) e de alta cardinalidade
devem ser EXCLUÍDAS do treinamento e geradas via IdentityMocker depois.

O método fit() faz isso automaticamente quando um PIIDetector é fornecido.

Fluxo recomendado com o framework
------------------------------------
  1. dd.mask(df, salt=salt) → df_mascarado
  2. TabularGenerative.fit(df_mascarado, pii_reports=reports)
     → colunas PII diretas excluídas automaticamente
     → treina nos campos analíticos (renda, idade, UF...)
  3. gen.sample(n) → DataFrame sintético analítico
  4. IdentityMocker.mock_dataframe(df_synth, reports) → adiciona PII fictícia
  5. FidelityEvaluator.evaluate(df_mascarado, df_synth) → valida qualidade

Referências
-----------
  Xu, L., et al. (2019). Modeling Tabular data using Conditional GAN. NeurIPS.
  Kingma, D. & Welling, M. (2013). Auto-Encoding Variational Bayes. ICLR 2014.
  Jordon, J., et al. (2022). Synthetic Data — what, why and how? RSS.
"""

from __future__ import annotations

import io
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Limite de categorias únicas antes de emitir aviso de alta cardinalidade
_HIGH_CARDINALITY_WARN = 100
# Colunas PII diretas que não devem entrar no treinamento CTGAN/TVAE
_DIRECT_PII_STRATEGIES = {"hash", "redact", "suppress"}


def _check_hardware() -> str:
    """Detecta disponibilidade de GPU e emite aviso se ausente."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU detectada: %s", name)
            return "cuda"
    except ImportError:
        pass
    warnings.warn(
        "GPU não detectada (torch.cuda.is_available() = False). "
        "O treinamento CTGAN em CPU pode ser lento para datasets grandes. "
        "Alternativas: (1) use architecture='tvae' — 3x mais rápido em CPU; "
        "(2) reduza epochs (50–100 em vez de 300); "
        "(3) execute em ambiente com GPU (Colab, cloud).",
        UserWarning, stacklevel=3,
    )
    return "cpu"


def _require_ctgan():
    try:
        from ctgan import CTGAN
        return CTGAN
    except ImportError:
        raise ImportError(
            "TabularGenerative requer 'ctgan': pip install ctgan"
        ) from None


def _require_tvae():
    try:
        from ctgan import TVAE
        return TVAE
    except ImportError:
        raise ImportError(
            "TVAE requer 'ctgan>=0.9': pip install ctgan"
        ) from None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

def _get_logus_version() -> str:
    """Returns the installed datalock version string, with fallback."""
    try:
        import importlib.metadata as _im
        return f"datalock/{_im.version('datalock')}"
    except Exception:
        try:
            from datalock import __version__  # type: ignore
            return f"datalock/{__version__}"
        except Exception:
            return "datalock/unknown"


@dataclass
class GenerativeModelInfo:
    """Metadados do modelo treinado."""
    architecture:          str
    epochs:                int
    batch_size:            int
    training_rows:         int
    training_cols:         int
    discrete_columns:      List[str]
    continuous_columns:    List[str]
    excluded_pii_columns:  List[str]   # excluídas do treinamento (alta cardinalidade/PII)
    hardware:              str         # "cpu" ou "cuda"
    differential_privacy:  bool
    epsilon:               Optional[float]
    training_time_s:       float
    framework_version:     str = field(default_factory=lambda: _get_logus_version())

    def to_dict(self) -> Dict:
        return {
            "architecture":         self.architecture,
            "epochs":               self.epochs,
            "batch_size":           self.batch_size,
            "training_rows":        self.training_rows,
            "training_cols":        self.training_cols,
            "discrete_columns":     self.discrete_columns,
            "continuous_columns":   self.continuous_columns,
            "excluded_pii_columns": self.excluded_pii_columns,
            "hardware":             self.hardware,
            "differential_privacy": self.differential_privacy,
            "epsilon":              self.epsilon,
            "training_time_s":      round(self.training_time_s, 2),
            "framework_version":    self.framework_version,
        }


# ---------------------------------------------------------------------------
# TabularGenerative
# ---------------------------------------------------------------------------

class TabularGenerative:
    """
    Treina CTGAN ou TVAE em dados (preferencialmente mascarados) e gera
    amostras sintéticas que preservam correlações multivariadas.

    Parâmetros:
        architecture:  "ctgan" (padrão) ou "tvae".
                       TVAE é ~3x mais rápido em CPU, recomendado para
                       datasets médios e ambientes sem GPU.
        epochs:        Épocas de treinamento.
                       CTGAN padrão do SDV: 300. Para prototipagem: 50–100.
                       Reduza se o treinamento demorar mais que 30 min.
        batch_size:    Tamanho do mini-batch (padrão 500).
        embedding_dim: Dimensão do espaço latente (padrão 128).
        generator_dim: Arquitetura do gerador — tupla de inteiros.
        discriminator_dim: Arquitetura do discriminador.
        pac:           Nº de amostras por condição no CTGAN (mitiga mode collapse).
        cuda:          Se True, usa GPU. Se None, detecta automaticamente.
        random_state:  Semente para reprodutibilidade.
        verbose:       Exibe progresso do treinamento.

    Exemplo de uso completo (integrado ao framework):
        import datalock as lg
        from datalock.generators import TabularGenerative, IdentityMocker
        from datalock.metrics import FidelityEvaluator

        # 1. Mascara dados reais
        salt = dd.generate_salt()
        df_mascarado = dd.mask(df_raw, salt=salt)

        # 2. Treina o gerador
        gen = TabularGenerative(architecture="tvae", epochs=100)
        gen.fit(df_mascarado)

        # 3. Gera dataset sintético e adiciona PII fictícia
        df_sintetico = gen.sample(10000)
        identity_mocker = IdentityMocker()
        df_final = identity_mocker.mock_dataframe(df_sintetico, result.reports)

        # 4. Valida fidelidade
        report = FidelityEvaluator().evaluate(df_mascarado, df_final)
        report.print_report()
    """

    def __init__(
        self,
        architecture: str = "ctgan",
        epochs: int = 100,
        batch_size: int = 500,
        embedding_dim: int = 128,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
        pac: int = 10,
        cuda: Optional[bool] = None,   # None = auto-detect
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        arch = architecture.lower()
        if arch not in ("ctgan", "tvae"):
            raise ValueError(f"architecture deve ser 'ctgan' ou 'tvae' (recebido: {architecture!r}).")
        self.architecture = arch
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.pac = pac
        self.cuda = cuda
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self._model_info: Optional[GenerativeModelInfo] = None
        self._discrete_columns: List[str] = []
        self._continuous_columns: List[str] = []
        self._training_columns: List[str] = []
        self._excluded_pii_columns: List[str] = []

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        auto_detect_discrete: bool = True,
        max_categories: int = 50,
        pii_reports: Optional[Dict] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> "TabularGenerative":
        """
        Treina o modelo no dataset fornecido.

        IMPORTANTE: treine sempre em dados *mascarados*, nunca nos dados brutos.

        Parâmetros:
            df:                   DataFrame de treinamento (preferencialmente mascarado).
            discrete_columns:     Colunas categóricas explícitas.
            auto_detect_discrete: Detecta discretas por dtype/cardinalidade.
            max_categories:       Limite de categorias para auto-detecção.
            pii_reports:          Dict[str, ColumnReport] do PIIDetector.
                                  Colunas PII diretas (HASH/REDACT/SUPPRESS)
                                  são excluídas automaticamente do treinamento
                                  para evitar que o modelo aprenda tokens HMAC
                                  sem sentido semântico.
            exclude_columns:      Colunas adicionais a excluir manualmente.

        Retorna:
            self (para chaining)
        """
        # Hardware check
        use_cuda = self._resolve_cuda()

        # Dataset size guard — CTGAN overfits severely on small datasets.
        # With <500 rows, synthetic data is essentially the originals with noise,
        # which is worse for privacy than direct masking.
        MIN_ROWS_CTGAN = 500
        if len(df) < 100:
            raise ValueError(
                f"Dataset muito pequeno para síntese generativa: {len(df)} linhas. "
                f"CTGAN/TVAE requerem no mínimo 100 linhas e funcionam bem com ≥ 500. "
                f"Para datasets pequenos, use dd.mask() em vez de dd.clone()."
            )
        elif len(df) < MIN_ROWS_CTGAN:
            warnings.warn(
                f"SmallDatasetWarning: dataset com {len(df)} linhas (recomendado ≥ {MIN_ROWS_CTGAN}). "
                f"Com poucos dados, CTGAN tende a memorizar os exemplos de treino — "
                f"os dados sintéticos podem reproduzir registros originais com pequenas variações, "
                f"oferecendo proteção de privacidade inferior ao mascaramento direto. "
                f"Considere: (1) usar architecture='tvae' que tolera melhor datasets menores, "
                f"ou (2) usar dd.mask() para proteção mais robusta com esse volume de dados.",
                UserWarning, stacklevel=2,
            )

        # Determina colunas a excluir
        self._excluded_pii_columns = []
        manual_exclude = set(exclude_columns or [])

        if pii_reports:
            for col, report in pii_reports.items():
                strategy = getattr(report, "mask_strategy", None)
                if strategy is not None:
                    strat_val = strategy.value if hasattr(strategy, "value") else str(strategy)
                    if strat_val in _DIRECT_PII_STRATEGIES:
                        self._excluded_pii_columns.append(col)
                        logger.debug(
                            "TabularGenerative: excluindo '%s' (PII direta, strategy=%s)",
                            col, strat_val,
                        )

        all_excluded = set(self._excluded_pii_columns) | manual_exclude
        df_train = df[[c for c in df.columns if c not in all_excluded]].copy()

        if df_train.empty or len(df_train.columns) == 0:
            raise ValueError(
                "Nenhuma coluna restante após exclusão de PII diretas. "
                "Verifique pii_reports ou use exclude_columns manualmente."
            )

        if len(self._excluded_pii_columns) > 0:
            logger.info(
                "TabularGenerative: %d colunas PII excluídas do treinamento: %s",
                len(self._excluded_pii_columns), self._excluded_pii_columns,
            )

        # Detecta colunas discretas
        if discrete_columns is not None:
            self._discrete_columns = [c for c in discrete_columns if c in df_train.columns]
        elif auto_detect_discrete:
            self._discrete_columns = self._auto_detect_discrete(df_train, max_categories)
        else:
            self._discrete_columns = []

        # Aviso de alta cardinalidade
        for col in self._discrete_columns:
            n_unique = df_train[col].nunique()
            if n_unique > _HIGH_CARDINALITY_WARN:
                warnings.warn(
                    f"Coluna '{col}' tem {n_unique} valores únicos — high cardinality. "
                    f"CTGAN/TVAE farão One-Hot Encoding ({n_unique} colunas binárias), "
                    f"tornando o treinamento muito mais lento. "
                    f"Considere excluir esta coluna e gerá-la via IdentityMocker.",
                    UserWarning, stacklevel=2,
                )

        self._continuous_columns = [
            c for c in df_train.columns
            if c not in self._discrete_columns and pd.api.types.is_numeric_dtype(df_train[c])
        ]
        self._training_columns = list(df_train.columns)

        df_prep = self._prepare_for_model(df_train)

        logger.info(
            "TabularGenerative.fit | arch=%s | hardware=%s | linhas=%d | cols=%d | "
            "discretas=%d | excluídas=%d | épocas=%d",
            self.architecture, "cuda" if use_cuda else "cpu",
            len(df_prep), len(df_prep.columns),
            len(self._discrete_columns), len(self._excluded_pii_columns),
            self.epochs,
        )

        t0 = time.perf_counter()
        self._model = self._build_model(use_cuda)

        import contextlib, io as _io
        ctx = contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(_io.StringIO())
        with ctx:
            if self.architecture == "ctgan":
                self._model.fit(df_prep, discrete_columns=self._discrete_columns)
            else:  # tvae — também precisa de discrete_columns
                self._model.fit(df_prep, discrete_columns=self._discrete_columns)

        elapsed = time.perf_counter() - t0

        self._model_info = GenerativeModelInfo(
            architecture=self.architecture.upper(),
            epochs=self.epochs,
            batch_size=self.batch_size,
            training_rows=len(df),
            training_cols=len(df_train.columns),
            discrete_columns=self._discrete_columns,
            continuous_columns=self._continuous_columns,
            excluded_pii_columns=self._excluded_pii_columns,
            hardware="cuda" if use_cuda else "cpu",
            differential_privacy=False,
            epsilon=None,
            training_time_s=elapsed,
        )

        logger.info(
            "TabularGenerative.fit concluído | %.2fs | arch=%s",
            elapsed, self.architecture,
        )
        return self

    def sample(self, n: int) -> pd.DataFrame:
        """
        Gera n amostras sintéticas a partir do modelo treinado.

        As colunas excluídas do treinamento (PII diretas) não estão no
        DataFrame resultante — adicione-as via IdentityMocker.mock_dataframe().

        Parâmetros:
            n: Número de linhas a gerar.

        Retorna:
            DataFrame com n linhas sintéticas (sem colunas PII excluídas).
        """
        if self._model is None:
            raise RuntimeError("Modelo não treinado. Chame .fit() antes de .sample().")

        logger.info("TabularGenerative.sample | n=%d | arch=%s", n, self.architecture)
        df_synth = self._model.sample(n)
        return self._restore_types(df_synth)

    def to_bytes(self) -> bytes:
        """
        Serializa o modelo para bytes em memória (sem escrita em disco).

        SEGURANÇA: use sempre este método via dd.store() para manter o modelo
        cifrado do início ao fim — sem exposição em /tmp ou partições não-cifradas.
        """
        import pickle, io as _io
        if self._model is None:
            raise RuntimeError("Nenhum modelo treinado. Execute fit() antes de to_bytes().")
        state = {
            "model":                self._model,
            "architecture":         self.architecture,
            "model_info":           self._model_info,
            "discrete_columns":     self._discrete_columns,
            "continuous_columns":   self._continuous_columns,
            "training_columns":     self._training_columns,
            "excluded_pii_columns": self._excluded_pii_columns,
        }
        buf = _io.BytesIO()
        pickle.dump(state, buf)
        return buf.getvalue()

    @classmethod
    def load_from_bytes(cls, data: bytes, **kwargs) -> "TabularGenerative":
        """
        Desserializa modelo a partir de bytes em memória (sem leitura de disco).

        SEGURANÇA: use via dd.read('modelo.dlk', key=...) que mantém os bytes
        cifrados até este ponto, sem materialização em disco.
        """
        import pickle, io as _io
        instance = cls(**kwargs)
        state = pickle.load(_io.BytesIO(data))
        instance._model                = state["model"]
        instance.architecture          = state.get("architecture", "ctgan")
        instance._model_info           = state.get("model_info")
        instance._discrete_columns     = state.get("discrete_columns", [])
        instance._continuous_columns   = state.get("continuous_columns", [])
        instance._training_columns     = state.get("training_columns", [])
        instance._excluded_pii_columns = state.get("excluded_pii_columns", [])
        return instance

    def save(self, path: str) -> None:
        """
        Serializa o modelo para pickle em disco.
        Para produção, prefira to_bytes() + dd.store() que mantém tudo cifrado.
        """
        import pickle
        if self._model is None:
            raise RuntimeError("Nenhum modelo treinado para salvar.")
        state = {
            "model":                  self._model,
            "architecture":           self.architecture,
            "model_info":             self._model_info,
            "discrete_columns":       self._discrete_columns,
            "continuous_columns":     self._continuous_columns,
            "training_columns":       self._training_columns,
            "excluded_pii_columns":   self._excluded_pii_columns,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("TabularGenerative salvo: %s", path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TabularGenerative":
        """Carrega modelo previamente salvo com .save(). Para modelos em .dlk, use dd.read()."""
        import pickle
        instance = cls(**kwargs)
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance._model                = state["model"]
        instance.architecture          = state.get("architecture", "ctgan")
        instance._model_info           = state.get("model_info")
        instance._discrete_columns     = state.get("discrete_columns", [])
        instance._continuous_columns   = state.get("continuous_columns", [])
        instance._training_columns     = state.get("training_columns", [])
        instance._excluded_pii_columns = state.get("excluded_pii_columns", [])
        logger.info("TabularGenerative carregado: %s", path)
        return instance

    @property
    def model_info(self) -> Optional[GenerativeModelInfo]:
        return self._model_info

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def excluded_pii_columns(self) -> List[str]:
        """Colunas excluídas do treinamento (PII diretas ou alta cardinalidade)."""
        return list(self._excluded_pii_columns)

    @property
    def training_columns(self) -> List[str]:
        """Colunas efetivamente usadas no treinamento."""
        return list(self._training_columns)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _resolve_cuda(self) -> bool:
        """Resolve se deve usar GPU (respeitando self.cuda e disponibilidade real)."""
        if self.cuda is True:
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.warn(
                        "cuda=True solicitado mas GPU não disponível. Usando CPU.",
                        UserWarning, stacklevel=3,
                    )
                    return False
                return True
            except ImportError:
                return False
        elif self.cuda is False:
            return False
        else:  # None → auto-detect (com aviso se CPU)
            hw = _check_hardware()
            return hw == "cuda"

    def _build_model(self, use_cuda: bool):
        """Instancia CTGAN ou TVAE com os parâmetros configurados."""
        if self.architecture == "ctgan":
            CTGAN = _require_ctgan()
            return CTGAN(
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                generator_dim=self.generator_dim,
                discriminator_dim=self.discriminator_dim,
                pac=self.pac,
                cuda=use_cuda,
                verbose=self.verbose,
            )
        else:  # tvae
            TVAE = _require_tvae()
            return TVAE(
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                compress_dims=self.generator_dim,
                decompress_dims=self.discriminator_dim,
                cuda=use_cuda,
                verbose=self.verbose,
            )

    def _auto_detect_discrete(self, df: pd.DataFrame, max_categories: int) -> List[str]:
        discrete = []
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if not pd.api.types.is_numeric_dtype(series):
                discrete.append(col)
            elif series.nunique() <= max_categories:
                discrete.append(col)
        logger.debug("Auto-detect discrete: %s", discrete)
        return discrete

    def _prepare_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()
        for col in df_prep.columns:
            if col in self._discrete_columns:
                df_prep[col] = df_prep[col].fillna("__NULL__").astype(str)
            elif pd.api.types.is_numeric_dtype(df_prep[col]):
                df_prep[col] = df_prep[col].fillna(df_prep[col].median()).astype(float)
            else:
                df_prep[col] = df_prep[col].fillna("__NULL__").astype(str)
                if col not in self._discrete_columns:
                    self._discrete_columns.append(col)
        return df_prep

    def _restore_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in df_out.columns:
            if col in self._discrete_columns:
                df_out[col] = df_out[col].replace("__NULL__", np.nan)
        return df_out

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "not fitted"
        excl = f", excluídas={self._excluded_pii_columns}" if self._excluded_pii_columns else ""
        return (
            f"TabularGenerative(arch={self.architecture.upper()}, "
            f"epochs={self.epochs}, batch={self.batch_size}, "
            f"state={fitted}{excl})"
        )
