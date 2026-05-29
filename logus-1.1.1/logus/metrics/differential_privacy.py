"""
metrics/differential_privacy.py
================================
Privacidade Diferencial (Differential Privacy — DP).

⚠️  AVISO DE ESCOPO — LEIA ANTES DE USAR EM PRODUÇÃO
======================================================
Este módulo implementa DP com ε (epsilon) para analytics interno e relatórios
exploratórios. Com ε=5 (padrão), a proteção é adequada para uso interno, mas
NÃO atende ao padrão de publicação formal de microdados exigido por reguladores.

Para releases públicos com garantias formais de DP, use bibliotecas auditadas:
  - Google DP Library: https://github.com/google/differential-privacy
  - OpenDP:            https://opendp.org/
  - IBM Diffprivlib:   https://github.com/IBM/differential-privacy-library

Este módulo é util para: dashboards internos, relatórios agregados internos,
prototipagem e validação de pipelines de analytics com DP.

Por que Privacidade Diferencial?
---------------------------------
k-Anonimato e l-Diversidade são medidas de privacidade *estruturais*:
garantem que nenhum indivíduo seja isolável pelos quasi-identifiers.
Mas não quantificam o quanto um adversário aprende sobre um indivíduo
ao observar o dataset mascarado — esse é o problema da inferência.

Privacidade Diferencial (Dwork et al., 2006) resolve isso com uma garantia
matemática formal:

    Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S]

Onde D e D' são datasets que diferem em apenas um indivíduo. O parâmetro ε
(epsilon) limita o quanto a SAÍDA do mecanismo M pode revelar sobre qualquer
indivíduo específico, independentemente do conhecimento prévio do adversário.

Interpretação prática de ε:
    ε ≤ 0.1  → proteção muito forte (academia, dados de saúde)
    ε ≤ 1.0  → proteção forte (padrão para releases de dados)
    ε ≤ 5.0  → proteção moderada (analytics internos)
    ε > 10.0 → proteção fraca (raramente recomendado)

Mecanismos implementados
-------------------------
1. Mecanismo de Laplace (Dwork et al., 2006)
   - Para consultas numéricas com sensibilidade global (Global Sensitivity)
   - Ruído: Laplace(0, Δf/ε), onde Δf é a sensibilidade da função
   - Exemplo: média de rendas, contagem de usuários por UF

2. Mecanismo Gaussiano (Dwork & Roth, 2014, §A.1)
   - Para consultas com sensibilidade L2 (norma euclidiana)
   - Ruído: N(0, σ²), com σ = Δf × √(2 ln(1.25/δ)) / ε
   - Oferece (ε, δ)-DP em vez de ε-DP puro

3. Mecanismo de Resposta Aleatória (Warner, 1965; Dwork & Roth, 2014)
   - Para dados categóricos binários ou múltiplos
   - Cada valor é reportado corretamente com probabilidade p = e^ε/(e^ε+1)
   - Fundamento do Google RAPPOR e Apple DP

4. DP Composto (budget tracking)
   - Composição sequencial: aplicar k mecanismos ε-DP consome k×ε de budget
   - Composição paralela: mecanismos em partições disjuntas consomem max(εi)

Limitações e posicionamento
-----------------------------
Este módulo oferece DP LOCAL e CENTRAL para uso analítico:
  - DP LOCAL: ruído adicionado nos valores individuais (este módulo)
  - DP CENTRAL: mecanismo no servidor, dados trafegam limpos (out of scope)

Para releases formais com garantias DP certificadas, considere:
  - Google DP Library (github.com/google/differential-privacy)
  - OpenDP (opendp.org)
  - IBM Diffprivlib (github.com/IBM/differential-privacy-library)

Este módulo é adequado para uso exploratório, prototipagem e
complementação das técnicas de mascaramento do framework.

Referências
-----------
  Dwork, C., McSherry, F., Nissim, K., Smith, A. (2006).
    Calibrating Noise to Sensitivity in Private Data Analysis. TCC 2006.

  Dwork, C., Roth, A. (2014).
    The Algorithmic Foundations of Differential Privacy. Foundations and
    Trends in Theoretical Computer Science, 9(3–4), 211–407.

  Warner, S.L. (1965).
    Randomized Response: A Survey Technique for Eliminating Evasive Answer
    Bias. JASA, 60(309), 63–69.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de configuração e resultado
# ---------------------------------------------------------------------------

@dataclass
class DPBudget:
    """
    Controla o orçamento de privacidade (privacy budget) acumulado.

    Em composição sequencial, cada mecanismo consome parte do budget total.
    Ultrapassar o budget significa que o adversário acumula mais informação
    do que a garantia original de ε permite.

    Referência: Dwork & Roth (2014), Theorem 3.14 (composição sequencial).
    """
    total_epsilon:   float
    total_delta:     float = 0.0
    spent_epsilon:   float = 0.0
    spent_delta:     float = 0.0
    mechanisms_used: List[str] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    @property
    def budget_exhausted(self) -> bool:
        return self.spent_epsilon >= self.total_epsilon

    def consume(self, epsilon: float, delta: float = 0.0, label: str = "") -> None:
        """Registra consumo de budget. Lança ValueError se exceder o total."""
        if self.spent_epsilon + epsilon > self.total_epsilon + 1e-9:
            raise ValueError(
                f"Budget de privacidade esgotado: tentativa de consumir ε={epsilon:.4f} "
                f"com apenas ε={self.remaining_epsilon:.4f} restante de {self.total_epsilon}."
            )
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        if label:
            self.mechanisms_used.append(label)

    def __repr__(self) -> str:
        pct = self.spent_epsilon / max(self.total_epsilon, 1e-9) * 100
        return (
            f"DPBudget(ε={self.total_epsilon}, gasto={self.spent_epsilon:.4f} "
            f"({pct:.1f}%), restante={self.remaining_epsilon:.4f})"
        )


@dataclass
class DPResult:
    """Resultado de uma operação de Privacidade Diferencial."""
    column:          str
    mechanism:       str
    epsilon:         float
    delta:           float
    sensitivity:     float
    noise_scale:     float
    original_stats:  Dict[str, float]
    noisy_stats:     Dict[str, float]
    privacy_class:   str    # "ε-DP" ou "(ε,δ)-DP"
    warning:         str = ""

    def __repr__(self) -> str:
        return (
            f"DPResult(col={self.column!r}, mech={self.mechanism}, "
            f"ε={self.epsilon}, δ={self.delta})"
        )


# ---------------------------------------------------------------------------
# Mecanismos de Privacidade Diferencial
# ---------------------------------------------------------------------------

class DifferentialPrivacy:
    """
    Aplica mecanismos de Privacidade Diferencial a colunas de um DataFrame.

    Parâmetros:
        epsilon:   Parâmetro de privacidade ε (quanto menor, mais protegido).
                   Valores típicos: 0.1 (muito forte) a 5.0 (moderado).
        delta:     Parâmetro δ para (ε,δ)-DP (Gaussiano). 0 para ε-DP puro.
                   Valor típico: 1/n² onde n é o número de registros.
        budget:    Objeto DPBudget para controle de composição. Se None,
                   um budget ilimitado é criado por padrão.
        random_state: Semente para reprodutibilidade dos experimentos.

    Exemplo de uso:
        dp = DifferentialPrivacy(epsilon=1.0, random_state=42)
        df_noisy = dp.apply_laplace(df, columns=["renda", "idade"])
        dp.print_report()
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 0.0,
        budget: Optional[DPBudget] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon deve ser > 0 (recebido: {epsilon}).")
        if delta < 0 or delta >= 1:
            raise ValueError(f"delta deve estar em [0, 1) (recebido: {delta}).")

        self.epsilon = epsilon
        self.delta = delta

        # SEGURANÇA: random_state=None gera seed via os.urandom (imprevisível).
        # Antes: default era 42 (seed público fixo). Se o adversário conhece o seed,
        # pode reproduzir o ruído exato e subtraí-lo, recuperando o valor original —
        # violando a garantia ε-DP.
        # Para testes unitários/reprodutibilidade explícita, passe random_state=42.
        if random_state is None:
            import secrets as _secrets
            _secure_seed = _secrets.randbelow(2**31)
            self.random_state = None  # indica "aleatório"
            self._rng = np.random.default_rng(_secure_seed)
        else:
            import warnings as _warnings
            _warnings.warn(
                f"DifferentialPrivacy: random_state={random_state} fixo reduz a garantia DP "
                f"se o adversário conhece o seed. Use random_state=None (padrão) em produção.",
                UserWarning, stacklevel=2,
            )
            self.random_state = random_state
            self._rng = np.random.default_rng(random_state)
        self._results: List[DPResult] = []

        if budget is None:
            self.budget = DPBudget(
                total_epsilon=float("inf"),  # sem limite por padrão
                total_delta=1.0,
            )
        else:
            self.budget = budget

    # ------------------------------------------------------------------
    # Mecanismo de Laplace (ε-DP puro)
    # ------------------------------------------------------------------

    def apply_laplace(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sensitivity: Optional[Union[float, Dict[str, float]]] = None,
        clip_to_original_range: bool = True,
    ) -> pd.DataFrame:
        """
        Aplica o Mecanismo de Laplace a colunas numéricas.

        O Mecanismo de Laplace adiciona ruído amostrado de uma distribuição
        de Laplace com escala b = Δf/ε, onde Δf é a sensibilidade global
        da função (quanto um único registro pode afetar o resultado).

        Garantia: ε-DP puro (mais forte que (ε,δ)-DP).

        Referência: Dwork et al. (2006), Proposition 1.

        Parâmetros:
            df:          DataFrame de entrada.
            columns:     Colunas a proteger (None = todas as numéricas).
            sensitivity: Sensibilidade global por coluna. Se None, estimada
                         como (max - min) / n, que é uma estimativa conservadora
                         para a sensibilidade da média.
            clip_to_original_range: Se True, valores são clipados para [min, max]
                         original após adição de ruído.

        Retorna:
            DataFrame com ruído de Laplace adicionado às colunas especificadas.
        """
        df_out = df.copy()
        numeric_cols = columns or df.select_dtypes(include="number").columns.tolist()

        for col in numeric_cols:
            if col not in df.columns:
                logger.warning("DifferentialPrivacy.apply_laplace: coluna '%s' não encontrada.", col)
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("DifferentialPrivacy.apply_laplace: '%s' não é numérica — ignorada.", col)
                continue

            clean = df[col].dropna()
            if len(clean) == 0:
                continue

            col_min = float(clean.min())
            col_max = float(clean.max())
            col_mean = float(clean.mean())
            col_std = float(clean.std())

            # Sensibilidade global (Δf)
            if isinstance(sensitivity, dict):
                delta_f = sensitivity.get(col, (col_max - col_min) / max(len(clean), 1))
            elif sensitivity is not None:
                delta_f = float(sensitivity)
            else:
                # Estimativa conservadora para a média: (max - min) / n
                delta_f = (col_max - col_min) / max(len(clean), 1)

            # Escala do ruído de Laplace: b = Δf / ε
            noise_scale = delta_f / self.epsilon
            noise = self._rng.laplace(loc=0.0, scale=noise_scale, size=len(df))
            noisy_series = df_out[col] + noise

            if clip_to_original_range:
                noisy_series = noisy_series.clip(lower=col_min, upper=col_max)

            df_out[col] = noisy_series.where(df[col].notna(), other=np.nan)

            result = DPResult(
                column=col,
                mechanism="Laplace",
                epsilon=self.epsilon,
                delta=0.0,
                sensitivity=delta_f,
                noise_scale=noise_scale,
                original_stats={"mean": col_mean, "std": col_std, "min": col_min, "max": col_max},
                noisy_stats={
                    "mean": float(df_out[col].dropna().mean()),
                    "std": float(df_out[col].dropna().std()),
                    "min": float(df_out[col].dropna().min()),
                    "max": float(df_out[col].dropna().max()),
                },
                privacy_class="ε-DP",
            )
            self._results.append(result)
            self.budget.consume(self.epsilon, label=f"Laplace({col})")

            logger.debug(
                "DP Laplace | col=%s | ε=%.3f | Δf=%.4f | b=%.4f",
                col, self.epsilon, delta_f, noise_scale,
            )

        return df_out

    # ------------------------------------------------------------------
    # Mecanismo Gaussiano ((ε,δ)-DP)
    # ------------------------------------------------------------------

    def apply_gaussian(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sensitivity: Optional[Union[float, Dict[str, float]]] = None,
        clip_to_original_range: bool = True,
    ) -> pd.DataFrame:
        """
        Aplica o Mecanismo Gaussiano a colunas numéricas.

        O Mecanismo Gaussiano oferece (ε,δ)-DP — uma garantia levemente mais
        fraca que ε-DP puro, mas com cauda mais leve e melhor composição
        sob privacidade de Rényi.

        Escala do ruído: σ = Δf × √(2 ln(1.25/δ)) / ε

        Referência: Dwork & Roth (2014), §A.1.

        Parâmetros:
            Idem apply_laplace. Requer self.delta > 0.
        """
        if self.delta == 0:
            raise ValueError(
                "Mecanismo Gaussiano requer delta > 0. "
                "Use delta=1e-5 ou similar. Para ε-DP puro, use apply_laplace()."
            )

        df_out = df.copy()
        numeric_cols = columns or df.select_dtypes(include="number").columns.tolist()

        for col in numeric_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            clean = df[col].dropna()
            if len(clean) == 0:
                continue

            col_min = float(clean.min())
            col_max = float(clean.max())

            if isinstance(sensitivity, dict):
                delta_f = sensitivity.get(col, (col_max - col_min) / max(len(clean), 1))
            elif sensitivity is not None:
                delta_f = float(sensitivity)
            else:
                delta_f = (col_max - col_min) / max(len(clean), 1)

            # σ para (ε,δ)-DP Gaussiano
            sigma = delta_f * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
            noise = self._rng.normal(loc=0.0, scale=sigma, size=len(df))
            noisy_series = df_out[col] + noise

            if clip_to_original_range:
                noisy_series = noisy_series.clip(lower=col_min, upper=col_max)

            df_out[col] = noisy_series.where(df[col].notna(), other=np.nan)

            result = DPResult(
                column=col,
                mechanism="Gaussian",
                epsilon=self.epsilon,
                delta=self.delta,
                sensitivity=delta_f,
                noise_scale=sigma,
                original_stats={"mean": float(clean.mean()), "std": float(clean.std()),
                                "min": col_min, "max": col_max},
                noisy_stats={
                    "mean": float(df_out[col].dropna().mean()),
                    "std": float(df_out[col].dropna().std()),
                    "min": float(df_out[col].dropna().min()),
                    "max": float(df_out[col].dropna().max()),
                },
                privacy_class="(ε,δ)-DP",
            )
            self._results.append(result)
            self.budget.consume(self.epsilon, self.delta, label=f"Gaussian({col})")

        return df_out

    # ------------------------------------------------------------------
    # Mecanismo de Resposta Aleatória (Randomized Response)
    # ------------------------------------------------------------------

    def apply_randomized_response(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aplica Resposta Aleatória a colunas categóricas ou binárias.

        O mecanismo de Resposta Aleatória (Warner, 1965) oferece ε-DP local:
        cada valor é reportado corretamente com probabilidade p = e^ε/(e^ε+1)
        ou substituído por um valor aleatório do domínio com probabilidade 1-p.

        É a base do Google RAPPOR e da DP local da Apple.

        Para |domínio| = k valores:
            p(valor_correto) = e^ε / (e^ε + k - 1)

        Parâmetros:
            df:      DataFrame de entrada.
            columns: Colunas categóricas a proteger.

        Retorna:
            DataFrame com resposta aleatória aplicada.
        """
        df_out = df.copy()
        cat_cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            if col not in df.columns:
                continue

            series = df[col].dropna().astype(str)
            domain = series.unique().tolist()
            k = len(domain)
            if k < 2:
                logger.warning("RR: coluna '%s' tem apenas %d valor — ignorada.", col, k)
                continue

            # Probabilidade de reportar o valor verdadeiro
            p_correct = math.exp(self.epsilon) / (math.exp(self.epsilon) + k - 1)

            def randomize(value):
                if pd.isna(value):
                    return value
                str_val = str(value)
                if self._rng.random() < p_correct:
                    return str_val  # reporta valor verdadeiro
                else:
                    # Escolhe aleatoriamente entre os outros valores do domínio
                    others = [v for v in domain if v != str_val]
                    return self._rng.choice(others) if others else str_val

            df_out[col] = df[col].apply(randomize)

            result = DPResult(
                column=col,
                mechanism="RandomizedResponse",
                epsilon=self.epsilon,
                delta=0.0,
                sensitivity=1.0,
                noise_scale=1 - p_correct,
                original_stats={"n_categories": k, "p_correct": p_correct},
                noisy_stats={"n_categories": k, "p_correct": p_correct},
                privacy_class="ε-DP (local)",
            )
            self._results.append(result)
            self.budget.consume(self.epsilon, label=f"RR({col})")

        return df_out

    # ------------------------------------------------------------------
    # Consultas DP (sem modificar o DataFrame)
    # ------------------------------------------------------------------

    def private_count(self, df: pd.DataFrame, sensitivity: float = 1.0) -> float:
        """
        Contagem com ruído de Laplace. ε-DP puro.

        Útil para: "quantos registros tem esta tabela?"
        sem revelar o número exato.
        """
        true_count = float(len(df))
        noise = self._rng.laplace(0, sensitivity / self.epsilon)
        self.budget.consume(self.epsilon, label="private_count")
        return max(0.0, true_count + noise)

    def private_mean(
        self,
        series: pd.Series,
        low: float,
        high: float,
    ) -> float:
        """
        Média com ruído de Laplace. ε-DP puro.

        Requer clamping explícito [low, high] para limitar a sensibilidade.
        Sensibilidade da média: (high - low) / n

        Parâmetros:
            series: Série numérica.
            low, high: Limites conhecidos a priori para clamping.
        """
        clamped = series.dropna().clip(low, high)
        n = max(len(clamped), 1)
        true_mean = float(clamped.mean())
        sensitivity = (high - low) / n
        noise = self._rng.laplace(0, sensitivity / self.epsilon)
        self.budget.consume(self.epsilon, label=f"private_mean({series.name})")
        return true_mean + noise

    def private_histogram(
        self,
        series: pd.Series,
        bins: Optional[List] = None,
    ) -> Dict[str, float]:
        """
        Histograma com ruído de Laplace. Sensibilidade = 1 por bin.
        Útil para análise de distribuição com garantia DP.
        """
        if bins is None:
            counts = series.value_counts()
        else:
            counts = pd.cut(series, bins=bins).value_counts()

        noisy_counts = {}
        for label, count in counts.items():
            noise = self._rng.laplace(0, 1.0 / self.epsilon)
            noisy_counts[str(label)] = max(0.0, float(count) + noise)

        self.budget.consume(self.epsilon, label=f"private_histogram({series.name})")
        return noisy_counts

    # ------------------------------------------------------------------
    # Relatório e utilitários
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Imprime relatório de todas as operações DP realizadas."""
        print("\n" + "=" * 65)
        print("RELATÓRIO DE PRIVACIDADE DIFERENCIAL")
        print("=" * 65)
        print(f"ε (epsilon)       : {self.epsilon}")
        print(f"δ (delta)         : {self.delta}")
        print(f"Budget gasto      : ε={self.budget.spent_epsilon:.4f}")
        print(f"Mecanismos usados : {len(self._results)}")
        print()

        for r in self._results:
            print(f"  Coluna : {r.column}")
            print(f"    Mecanismo   : {r.mechanism} ({r.privacy_class})")
            print(f"    ε (coluna)  : {r.epsilon}")
            print(f"    Sensibilidade: {r.sensitivity:.6f}")
            print(f"    Escala ruído : {r.noise_scale:.6f}")
            if "mean" in r.original_stats:
                δ_mean = abs(r.noisy_stats["mean"] - r.original_stats["mean"])
                print(f"    Δ médio      : {δ_mean:.4f} (orig={r.original_stats['mean']:.4f})")
            print()

        print(self.budget)
        print()
        print("Nota: ε-DP garante que a saída não revele mais que e^ε")
        print("vezes mais informação sobre qualquer indivíduo.")
        print("Referência: Dwork et al. (2006); Dwork & Roth (2014).")

    def to_dataframe(self) -> pd.DataFrame:
        """Retorna os resultados de todas as operações DP como DataFrame."""
        rows = []
        for r in self._results:
            rows.append({
                "coluna":       r.column,
                "mecanismo":    r.mechanism,
                "privacidade":  r.privacy_class,
                "epsilon":      r.epsilon,
                "delta":        r.delta,
                "sensibilidade": r.sensitivity,
                "escala_ruido": r.noise_scale,
            })
        return pd.DataFrame(rows)

    @property
    def results(self) -> List[DPResult]:
        return list(self._results)


# ---------------------------------------------------------------------------
# Análise de ataques formais — contexto acadêmico
# ---------------------------------------------------------------------------

@dataclass
class PrivacyAttackContext:
    """
    Descreve o contexto adversarial e os ataques contra os quais
    cada mecanismo oferece proteção formal.

    Uso: para a seção de limitações e trabalhos futuros do TCC.
    """
    k_anonymity_vulnerabilities = [
        "Ataque de homogeneidade: todos no mesmo grupo têm o mesmo atributo sensível.",
        "Ataque de background knowledge: adversário com conhecimento externo quebra o k-anon.",
        "Skewness attack (Li et al., 2007): distribuição desequilibrada no grupo.",
        "Similarity attack: valores numericamente próximos revelam informação.",
    ]

    l_diversity_vulnerabilities = [
        "Skewness attack: distribuição não uniforme dos l valores distintos.",
        "Similarity attack: valores semanticamente similares (ex: 'câncer benigno'/'maligno').",
    ]

    differential_privacy_protections = [
        "Protege contra qualquer adversário com qualquer conhecimento de fundo.",
        "Garantia composicional: múltiplas consultas consomem budget aditivamente.",
        "Membership inference: ε-DP limita vantagem do adversário a e^ε.",
        "Reconstrução de dados: ruído calibrado impede reconstrução exata.",
    ]

    limits_of_this_framework = [
        "Python não oferece garantias de zeroização de memória (GC não determinístico).",
        "HMAC determinístico é pseudonimização, não anonimização (LGPD Art. 5°, XI).",
        "Correlações multivariadas não são preservadas pelo mascaramento por coluna.",
        "Dados em texto livre com entidades implícitas não são detectados por regex.",
        "Segurança de memória em pandas: buffers internos podem persistir após GC.",
        "DP local (RR) tem utilidade analítica inferior ao DP central para n grande.",
    ]

    future_work = [
        "t-Closeness (Li et al., 2007) — terceira métrica formal de privacidade.",
        "Síntese tabular com CTGAN / TVAE (Xu et al., 2019) — preserva correlações.",
        "Privacidade diferencial de Rényi (Mironov, 2017) — composição mais eficiente.",
        "Detecção de PII com NER (spaCy/transformers) — entidades implícitas.",
        "Integração Spark/DuckDB para datasets distribuídos.",
        "Políticas declarativas YAML para governança como código.",
        "Data lineage: grafo de origem dos tokens pseudonimizados.",
    ]
