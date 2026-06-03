# datalock — Documentação Completa
### v1.1.4 — Privacy-by-Design para dados tabulares em Python

---

## Sumário

1. [O que é o datalock](#1-o-que-é-o-datalock)
2. [Instalação](#2-instalação)
3. [Configuração](#3-configuração)
4. [Conceitos fundamentais](#4-conceitos-fundamentais)
5. [API principal](#5-api-principal)
   - [dd.scan()](#51-ddscan--detecção-de-pii)
   - [dd.mask()](#52-ddmask--mascaramento)
   - [dd.store()](#53-ddstore--salvar-dlk)
   - [dd.read()](#54-ddread--leitura-unificada)
   - [dd.inspect()](#55-ddinspect--metadados-sem-decifrar)
   - [dd.open()](#56-ddopen--api-orientada-a-objeto)
   - [dd.rekey()](#57-ddrekey--rotação-de-chave)
   - [dd.stream()](#58-ddstream--leitura-em-chunks)
   - [dd.write()](#59-ddwrite--escrita-de-arquivo-ou-banco)
   - [dd.sql()](#510-ddsql--sql-via-duckdb)
5.5. [dd.inspect() com row_groups](#55-ddinspect--metadados-sem-decifrar)
6. [Column pruning e predicate pushdown](#6-column-pruning-e-predicate-pushdown)
7. [Mascaramento de texto livre](#7-mascaramento-de-texto-livre)
7. [Análise e transformação](#7-análise-e-transformação)
8. [Banco de dados](#8-banco-de-dados)
9. [Pipeline fluente](#9-pipeline-fluente)
10. [Contrato de dados](#10-contrato-de-dados)
11. [Validação](#11-validação)
12. [Canary data](#12-canary-data)
13. [Criptografia assimétrica](#13-criptografia-assimétrica)
14. [Dados sintéticos](#14-dados-sintéticos)
15. [Métricas de privacidade](#15-métricas-de-privacidade)
16. [Varredura de diretório](#16-varredura-de-diretório)
17. [Relatório de conformidade LGPD](#17-relatório-de-conformidade-lgpd)
18. [Linhagem de dados](#18-linhagem-de-dados)
19. [CLI](#19-cli)
20. [Referência de tipos e classes](#20-referência-de-tipos-e-classes)
21. [Segurança: guia de boas práticas](#21-segurança-guia-de-boas-práticas)
22. [Retrocompatibilidade](#22-retrocompatibilidade)

---

## 1. O que é o datalock

O datalock é uma biblioteca Python para tratamento de dados pessoais com Privacy-by-Design. Ela resolve o problema de trabalhar com dados sensíveis em pipelines de engenharia de dados: detecta automaticamente colunas PII, aplica mascaramento determinístico compatível com LGPD/GDPR, e armazena dados de forma segura no formato `.dlk` (AES-256-GCM + Parquet/Arrow).

**Capacidades principais:**

| Função | O que faz |
|--------|-----------|
| `dd.scan()` | Detecta e classifica colunas PII automaticamente |
| `dd.mask()` | Aplica pseudonimização HMAC-SHA256 determinística |
| `dd.store()` | Salva em `.dlk` (AES-256-GCM + Parquet/zstd) |
| `dd.read()` | Lê qualquer formato tabular, inclusive `.dlk` |
| `dd.stream()` | Leitura em chunks sem OOM para big data |
| `dd.sql()` | SQL sobre DataFrames via DuckDB |
| `dd.check.*` | Métricas k-anonimato, risco, privacidade diferencial |
| `dd.contract()` | Contrato de dados versionável |
| `dd.clone()` | Dados sintéticos com mesmas distribuições |
| `dd.canary_check()` | Rastreamento de vazamentos por fingerprint |
| `dd.compliance_report()` | Relatório formal LGPD para DPOs |

**Compatibilidade:** Python ≥ 3.10, Polars ≥ 1.0, pandas ≥ 2.0, PyArrow ≥ 14.0.

---

## 2. Instalação

```bash
# Núcleo
pip install datalock

# Com suporte a SQL via DuckDB
pip install "datalock[sql]"

# Com suporte a Excel (.xlsx)
pip install "datalock[excel]"

# Com dados sintéticos (CTGAN/SDV)
pip install "datalock[synthetic]"

# Tudo incluído
pip install "datalock[full]"
```

---

## 3. Configuração

A configuração global é feita via `dd.configure()`. **Nunca inclua salts ou chaves diretamente no código-fonte.**

```python
import datalock as dd
import os

# Opção 1 — variáveis de ambiente (recomendado em produção)
dd.configure(
    default_salt=os.environ["DATALOCK_SALT"],
    canary_salt=os.environ["DATALOCK_CANARY_SALT"],
)

# Opção 2 — arquivo .env (requer pip install python-dotenv)
# .env deve conter: DATALOCK_SALT=... DATALOCK_CANARY_SALT=...
dd.configure(load_dotenv=True)

# Opção 3 — trilha de auditoria em arquivo
dd.configure(audit_path="./audit/")

# Opção 4 — webhook de auditoria (Slack, SIEM, Datadog, etc.)
dd.configure(audit_webhook="https://hooks.slack.com/services/...")
```

**Parâmetros de `dd.configure()`:**

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `default_salt` | str | Salt padrão para `dd.mask()` sem `salt=` explícito |
| `canary_salt` | str | Salt secreto para fingerprints canary (tabular). Substitui o fallback público hardcoded |
| `wm_salt` | str | Salt para watermarking textual |
| `audit` | AuditReport | Objeto de auditoria para trilha automática |
| `audit_path` | str | Diretório para criar AuditReport com gravação em arquivo |
| `audit_webhook` | str | URL para receber eventos de auditoria via HTTP POST (JSON) |
| `load_dotenv` | bool | Se True, carrega variáveis do arquivo `.env` |
| `dotenv_path` | str | Caminho do `.env` (None = procura no diretório atual) |

**Gerando salts seguros:**

```python
# Salt de 48 chars (~300 bits de entropia)
salt = dd.generate_salt()
print(salt)  # Ex: "aB3$kP9#mX2@vQ7!nR5&wY1^jL4*hT6"

# Salt hexadecimal de 64 chars (256 bits, seguro em qualquer contexto)
salt_hex = dd.generate_salt_hex()

# Valide e grave no vault ou variável de ambiente
```

---

## 4. Conceitos fundamentais

### Determinismo do mascaramento

O mascaramento HMAC-SHA256 é **determinístico**: o mesmo valor com o mesmo salt sempre produz o mesmo token. Isso é essencial para preservar integridade referencial — JOINs entre tabelas mascaradas pelo mesmo pipeline continuam funcionando.

```python
# O mesmo CPF mascarado com o mesmo salt produz o mesmo token em qualquer momento
dd.mask(df_clientes, salt=SALT)   # "cpf": "3f2a8b1c9d4e7f0a"
dd.mask(df_pedidos,  salt=SALT)   # "cpf": "3f2a8b1c9d4e7f0a" ← mesmo token, JOIN funciona
```

Se o salt mudar entre execuções, os tokens mudam e JOINs quebram. Use `dd.configure(default_salt=...)` ou passe `salt=` explicitamente em cada chamada.

### Separação de responsabilidades

O datalock separa duas chaves distintas com papéis distintos:

- **`master_key` / `key=`**: chave AES para cifragem do arquivo `.dlk`. Quem tem a chave pode ver os dados brutos. Deve ser armazenada em vault corporativo (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault).
- **`salt` / `salt=`**: chave HMAC para mascaramento dos dados. Quem tem o salt pode reverificar tokens, mas não reverter o mascaramento (é one-way). Pode ser distribuído a desenvolvedores com acesso controlado.

Em um pipeline típico:
- O DBA ou engenheiro de dados detém a `master_key` e cria o `.dlk` com dados brutos.
- O cientista de dados recebe o `.dlk` e o `salt`, mas não a `master_key`.
- O `SecureFile.load()` decifra com a `master_key`, aplica mascaramento com o `salt`, e entrega os dados já pseudonimizados.

### Tipos de conteúdo

| `content_type`          | Descrição |
|-------------------------|-----------|
| `raw_dataframe`         | Dados brutos, ainda não mascarados |
| `masked_dataframe`      | Dados com mascaramento aplicado |
| `anonymous_dataframe`   | Dados anonimizados, sem key obrigatória |
| `multi_dataframe`       | Múltiplos DataFrames num único `.dlk` |
| `bytes`                 | Payload binário arbitrário |

---

## 5. API principal

### 5.1 `dd.scan()` — Detecção de PII

Detecta e classifica colunas com dados pessoais. Suporta DataFrames em memória ou caminhos de arquivo.

```python
reports = dd.scan(df)
reports = dd.scan("clientes.parquet")
reports = dd.scan("clientes.dlk", key=KEY)

# Com detecção de dados sensíveis adicionais (financeiros, jurídicos)
reports, findings = dd.scan(df, sensitive=True)

# Padrões customizados para identificadores corporativos
reports = dd.scan(df, custom_patterns={
    "num_contrato": r"^CTR-[0-9]{8}$",
    "matricula":    r"^[0-9]{6}-[A-Z]$",
})
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `source` | — | DataFrame, pl.DataFrame, ou caminho de arquivo |
| `key` | None | Chave para arquivos `.dlk` cifrados |
| `sample_size` | 500 | Linhas amostradas para detecção |
| `threshold` | 0.5 | Match ratio mínimo para classificar como PII |
| `sensitive` | False | Se True, também detecta dados sensíveis (financeiros, jurídicos) |

**Retorno:** `Dict[str, ColumnReport]`

Cada `ColumnReport` contém:
- `pii_type: PIIType` — tipo detectado (CPF, EMAIL, TELEFONE, NOME, etc.)
- `risk_level: RiskLevel` — risco (`high`, `medium`, `low`)
- `mask_strategy: MaskStrategy` — estratégia recomendada
- `match_ratio: float` — proporção de valores que correspondem ao padrão
- `unique_ratio: float` — proporção de valores únicos
- `notes: str` — observações do detector

```python
for col, r in reports.items():
    print(f"{col}: tipo={r.pii_type.value}, risco={r.risk_level.value}, "
          f"estratégia={r.mask_strategy.value}")
```

---

### 5.2 `dd.mask()` — Mascaramento

Aplica pseudonimização aos dados. Detecta PII automaticamente e aplica a estratégia adequada por tipo de coluna. Preserva o tipo de entrada (`pd.DataFrame` → `pd.DataFrame`, `pl.DataFrame` → `pl.DataFrame`).

```python
df_safe = dd.mask(df, salt=SALT)

# Mascarar apenas colunas específicas
df_safe = dd.mask(df, salt=SALT, columns=["cpf", "email", "telefone"])

# Excluir colunas do mascaramento
df_safe = dd.mask(df, salt=SALT, exclude=["uf", "tipo_pessoa"])

# Mascaramento por nível de risco
df_safe = dd.mask(df, salt=SALT, risk="high")    # suprime colunas de alto risco
df_safe = dd.mask(df, salt=SALT, risk="medium")  # hash para médio
df_safe = dd.mask(df, salt=SALT, risk="low")     # truncate para baixo

# LazyFrame (permanece lazy, não materializa)
lf_safe = dd.mask(df.lazy(), salt=SALT)

# Relatório de detecção PII no console
df_safe = dd.mask(df, salt=SALT, verbose=True)
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `df` | — | DataFrame (pd, pl ou LazyFrame) |
| `salt` | None | Chave HMAC. None → salt aleatório + UserWarning |
| `random_state` | 42 | Semente para mockers numéricos/categóricos |
| `strict` | True | Lança `IdempotencyError` se coluna já parecer mascarada |
| `verbose` | False | Imprime relatório de detecção no console |
| `columns` | None | Mascara apenas estas colunas (None = todas PII detectadas) |
| `exclude` | None | Não mascara estas colunas |
| `risk` | None | Filtra por nível de risco: `"high"`, `"medium"`, `"low"` |

**Estratégias de mascaramento por tipo:**

| Tipo PII | Estratégia padrão | Resultado |
|----------|-------------------|-----------|
| CPF, CNPJ | HASH | `"3f2a8b1c9d4e7f0a"` (16 hex chars) |
| Email | HASH | `"9e1d3c7f2a845b61"` |
| Telefone | MASK_PHONE_DDD | `"(11) ****-1234"` |
| Nome | HASH | token hex |
| Data de nascimento | GENERALIZE_DATE | `"1985-**"` (ano-mês) |
| Endereço | TRUNCATE | truncado na rua |
| Valor/Renda | MOCK_NUM | valor sintético na mesma distribuição |
| Categoria sensível | MOCK_CAT | categoria sintética na mesma distribuição |

---

### 5.3 `dd.store()` — Salvar `.dlk`

Salva dados no formato `.dlk`. Detecta automaticamente o tipo (DataFrame, dict de DataFrames, bytes, ou caminho de arquivo).

```python
# DataFrame único, cifrado
dd.store(df, "dados.dlk", key=KEY)

# Mascara e cifra em uma operação
dd.store(df, "dados.dlk", key=KEY, salt=SALT)

# Múltiplos DataFrames (multi-frame)
dd.store({"clientes": df1, "pedidos": df2}, "base.dlk", key=KEY)

# Com metadados customizados e expiração
dd.store(df, "dados.dlk", key=KEY,
         label="exportacao_crm_jan2025",
         metadata={"responsavel": "time_dados", "projeto": "CRM"},
         expires_at="2025-12-31T23:59:59Z")

# Com canary data para rastreamento de vazamentos
dd.store(df, "dados.dlk", key=KEY, canary=True, pipeline_id="crm_jan2025")

# Sem criptografia (apenas para dados anonimizados)
dd.store(df_anonimo, "dados_dev.dlk", anonymize=True)

# Bytes brutos (modelos, PDFs, etc.)
dd.store(model_bytes, "modelo.dlk", key=KEY)

# Sobrescrever arquivo existente
dd.store(df, "dados.dlk", key=KEY, overwrite=True)
```

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `source` | — | DataFrame, dict, bytes, ou caminho de arquivo |
| `output_path` | — | Caminho de saída (`.dlk` adicionado automaticamente se ausente) |
| `key` | None | Chave AES. None → v4 (sem criptografia) |
| `salt` | None | Salt HMAC. Se fornecido, implica `anonymize=True` |
| `anonymize` | False | Se True, aplica mascaramento antes de gravar |
| `raw` | False | Se True, grava sem mascaramento mesmo com `key=` |
| `label` | `""` | Rótulo para auditoria |
| `compress` | True | True=zstd (menor tamanho), False=lz4 (mais rápido) |
| `overwrite` | False | Sobrescreve arquivo existente |
| `metadata` | None | Dict de metadados arbitrários |
| `expires_at` | None | Data de expiração ISO 8601 (LGPD Art. 16) |
| `canary` | False | Injeta canary rows para rastreamento de vazamentos |
| `canary_n_rows` | 3 | Número de canary rows |
| `pipeline_id` | None | Identificador do pipeline para canary |

**Retorno:** `Dict` com `output_path`, `shape`, `original_size_kb`, `packed_size_kb`, `compression_ratio`, `elapsed_seconds`.

---

### 5.4 `dd.read()` — Leitura unificada

Lê qualquer formato tabular. Auto-detecta formato pela extensão e encoding automaticamente.

```python
# Qualquer formato de arquivo
df = dd.read("clientes.csv")
df = dd.read("clientes.parquet")
df = dd.read("clientes.xlsx")
df = dd.read("clientes.dlk", key=KEY)

# .dlk: descriptografa e mascara em uma operação
df = dd.read("clientes.dlk", key=KEY, salt=SALT)

# .dlk sem mascaramento adicional (dados brutos)
df = dd.read("clientes.dlk", key=KEY, raw=True)

# .dlk multi-frame: lê frame específico
df = dd.read("base.dlk", key=KEY, frame="clientes")

# Lê e mascara CSV
df = dd.read("clientes.csv", salt=SALT)

# CSV com separador diferente
df = dd.read("dados.csv", sep=";", encoding="latin-1")

# Column pruning — retorna apenas as colunas solicitadas
df = dd.read("clientes.dlk", key=KEY, columns=["cpf", "renda_mensal"])

# Predicate pushdown — materializa apenas as linhas que satisfazem o filtro
df = dd.read("clientes.dlk", key=KEY, filters={"uf": "SP"})
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (">", 10_000)})
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (5_000, 50_000)})

# Combinado — mínimo de memória alocada
df = dd.read("clientes.dlk", key=KEY,
             columns=["cpf", "renda_mensal"],
             filters={"uf": ["SP", "RJ"], "renda_mensal": (5_000, 50_000)})

# Big data — leitura parcial sem OOM
df   = dd.read("big.parquet", head=100_000)         # primeiras 100k linhas
df   = dd.read("big.parquet", tail=50_000)          # últimas 50k linhas
df   = dd.read("big.parquet", sample=500_000)       # amostra aleatória
info = dd.read("big.parquet", header_only=True)     # só metadados, sem carregar dados
df   = dd.read("big.parquet", n_chunks=5, chunks=[2, 4])  # chunks 2 e 4 de 5

# Iteração por chunks
for chunk in dd.read("big.parquet", n_chunks=10, iter_chunks=True):
    processar(chunk)

# Banco de dados
banco = dd.db("postgresql://user:pass@host/db")
df = dd.read(banco, "clientes")
```

**Formatos suportados:**

`.csv` `.tsv` `.txt` `.parquet` `.json` `.ndjson` `.jsonl` `.feather` `.ipc` `.arrow` `.avro` `.orc` `.xlsx` `.xls` `.ods` `.xml` `.html` `.dta` `.sas7bdat` `.sav` `.pkl` `.hdf` `.h5` `.dlk`

---

### 5.5 `dd.inspect()` — Metadados sem decifrar

Lê metadados de um arquivo `.dlk` sem descriptografar o payload.

```python
info = dd.inspect("clientes.dlk", key=KEY)
print(info["shape"])           # [150000, 12]
print(info["content_type"])    # "masked_dataframe"
print(info["created_at"])      # "2025-01-15T14:32:00+00:00"
print(info["label"])           # "exportacao_crm_jan2025"
print(info["encryption"])      # "AES256GCM"
print(info["frame_names"])     # ["clientes", "pedidos"]  (multi-frame)
print(info["column_stats"])    # estatísticas por coluna
print(info["expires_at"])      # "2025-12-31T23:59:59Z"
```

Para arquivos v4 (sem criptografia), `key=` é opcional.

---

### 5.6 `dd.open()` — API orientada a objeto

Retorna um `DlkFile` com interface pythônica e suporte a context manager.

```python
# Context manager
with dd.open("clientes.dlk", key=KEY, salt=SALT) as f:
    df        = f.read()          # lê e mascara
    df_raw    = f.read(raw=True)  # sem mascaramento
    info      = f.info()          # metadados sem decifrar
    names     = f.frame_names()   # nomes de frames (multi-frame)
    shp       = f.shape()         # (linhas, colunas)
    ok        = f.valid()         # True se íntegro
    f.write(df_novo)              # sobrescreve
    f.add_frame("novos", df_n)    # adiciona frame

# Fluent API
df = dd.open("clientes.dlk", key=KEY).read()

# Instância reutilizável
f = dd.open("clientes.dlk", key=KEY)
df = f.read()
print(f.size_kb())    # tamanho em KB

# Operações
f.delete()                              # remove o arquivo
f.copy_to("backup/clientes.dlk")        # copia sem decifrar
```

**Métodos de `DlkFile`:**

| Método | Retorno | Descrição |
|--------|---------|-----------|
| `read(raw=False, frame=None)` | DataFrame | Lê o arquivo |
| `frames(salt=None)` | dict[str, DataFrame] | Lê todos os frames |
| `frame(name, salt=None)` | DataFrame | Lê um frame por nome |
| `write(df, label="", overwrite=True)` | self | Escreve DataFrame |
| `add_frame(name, df)` | self | Adiciona ou substitui frame |
| `info(force=False)` | dict | Metadados (cacheados) |
| `valid()` | bool | True se arquivo íntegro |
| `frame_names()` | list[str] | Nomes dos frames |
| `shape()` | tuple | (linhas, colunas) |
| `exists()` | bool | True se arquivo existe |
| `size_kb()` | float | Tamanho em KB |
| `delete()` | None | Remove o arquivo |
| `copy_to(dest)` | DlkFile | Copia sem decifrar |

---

### 5.7 `dd.rekey()` — Rotação de chave

Rotaciona a chave de um arquivo `.dlk` sem expor dados em disco. Os dados brutos ficam na heap durante a operação.

```python
# In-place (sobrescreve o arquivo original)
dd.rekey("dados.dlk", old_key=KEY_ANTIGO, new_key=KEY_NOVO)

# Para novo arquivo
dd.rekey("dados.dlk", old_key=KEY_ANTIGO, new_key=KEY_NOVO,
          output_path="dados_novo.dlk")
```

---

### 5.8 `dd.stream()` — Leitura em chunks

Lê e mascara em modo gerador para big data. Nunca carrega o arquivo inteiro em memória.

```python
for chunk in dd.stream("grande.csv", salt=SALT, chunksize=50_000):
    salvar_no_banco(chunk)

# Com callback de progresso
def progresso(n, feito, total):
    print(f"Chunk {n}: {feito:,}/{total:,} linhas")

for chunk in dd.stream("grande.parquet", salt=SALT,
                        chunksize=100_000, on_progress=progresso):
    processar(chunk)
```

---

### 5.9 `dd.write()` — Escrita de arquivo ou banco

```python
# Arquivo
dd.write(df, "resultado.parquet")
dd.write(df, "resultado.csv")
dd.write(df, "resultado.xlsx")

# Banco de dados
banco = dd.db("postgresql://user:pass@host/db")
dd.write(df, banco, "clientes_masked")
```

---

### 5.10 `dd.sql()` — SQL via DuckDB

Executa SQL sobre DataFrames ou arquivos via DuckDB (zero-copy Arrow). Requer `pip install "datalock[sql]"`.

```python
# SQL sobre DataFrames em memória
result = dd.sql(
    "SELECT uf, AVG(renda_mensal) AS media, COUNT(*) AS n "
    "FROM df GROUP BY uf HAVING n > 100",
    df=df
)

# SQL sobre arquivos (DuckDB lê diretamente)
result = dd.sql(
    "SELECT * FROM read_parquet('dados.parquet') WHERE uf='SP'"
)

# JOIN entre DataFrames
result = dd.sql(
    "SELECT c.uf, p.valor FROM clientes c JOIN pedidos p ON c.cpf = p.cpf",
    clientes=df_clientes,
    pedidos=df_pedidos,
)

# Com mascaramento do resultado
result = dd.sql("SELECT * FROM df", df=df, salt=SALT)
```

---

## 7. Column Pruning e Predicate Pushdown

A partir de v1.1.4, o formato `.dlk` suporta leitura seletiva de colunas e filtragem de linhas no nível Arrow IPC, sem materializar o DataFrame inteiro em memória.

### Como funciona

O payload é serializado como stream de N record batches Arrow IPC (padrão: 50 000 linhas/batch). Estatísticas por batch (`min`, `max`, `null_count` por coluna) são indexadas no header cifrado. Na leitura:

1. O header é decifrado → índice de row groups lido.
2. Batches incompatíveis com os filtros são identificados pelas stats e **pulados sem construir arrays Python**.
3. O payload inteiro é decifrado (AES-GCM exige isso — inescapável).
4. Apenas os batches relevantes são materializados; dentro deles, apenas as colunas pedidas.
5. Filtros exatos são aplicados linha a linha via `pyarrow.compute`.

O tempo de decifração é idêntico ao anterior. O ganho é na alocação de memória e na desserialização — que é onde o gargalo real está para datasets que cabem em disco mas não em RAM.

### Ganho típico de memória

```
DataFrame: 1M linhas × 50 colunas × 8 bytes ≈ 400 MB

columns=["uf", "renda"] + filters={"uf": "SP"} (20% das linhas):
  Antes:  400 MB alocados → filtra para ~16 MB
  Depois: ~16 MB alocados diretamente
  Ganho:  25× menos memória para esse padrão de acesso
```

### API

```python
import datalock as dd

KEY = os.environ["DATALOCK_KEY"]

# Apenas colunas específicas
df = dd.read("clientes.dlk", key=KEY, columns=["cpf", "uf", "renda_mensal"])

# Filtro por igualdade
df = dd.read("clientes.dlk", key=KEY, filters={"uf": "SP"})

# Filtro por lista de valores
df = dd.read("clientes.dlk", key=KEY, filters={"uf": ["SP", "RJ", "MG"]})

# Filtro por comparação
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (">", 10_000)})
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (">=", 5_000)})
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": ("<", 3_000)})

# Filtro por range fechado [a, b]
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (5_000, 50_000)})

# Range com lado aberto
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (5_000, None)})   # >= 5000
df = dd.read("clientes.dlk", key=KEY, filters={"renda_mensal": (None, 50_000)})  # <= 50000

# Múltiplas colunas — AND implícito
df = dd.read("clientes.dlk", key=KEY, filters={
    "uf": ["SP", "RJ"],
    "renda_mensal": (5_000, 50_000),
})

# Combinado com column pruning
df = dd.read("clientes.dlk", key=KEY,
             columns=["cpf", "renda_mensal"],
             filters={"uf": "SP", "renda_mensal": (">", 5_000)})

# Via SecureFile diretamente
from datalock.secure_file import SecureFile

df = SecureFile.load_raw(
    "clientes.dlk", key=KEY,
    columns=["cpf", "uf"],
    filters={"uf": "SP"},
)

df = SecureFile.load(
    "clientes.dlk", key=KEY, salt_masking=SALT,
    columns=["cpf", "uf", "renda_mensal"],
    filters={"renda_mensal": (">", 10_000)},
)
```

### Retrocompatibilidade

Arquivos gravados por versões anteriores a v1.1.4 (sem `"row_groups"` no header) são lidos normalmente — o pruning é simplesmente desabilitado e todo o payload é materializado como antes. Nenhum erro, nenhuma mudança de comportamento para código existente.

### Módulo `datalock.ipc_index`

A lógica de pruning é isolada no módulo `datalock.ipc_index` para uso avançado:

```python
from datalock.ipc_index import (
    normalize_filters,    # normaliza dict de filtros para lista canônica
    compute_batch_stats,  # calcula min/max/null_count por coluna de um Arrow batch
    prune_row_groups,     # retorna set[int] de batch indices relevantes
    apply_arrow_filters,  # aplica filtros exatos linha a linha em um batch Arrow
)

# Normalizar filtros
normalized = normalize_filters({"uf": "SP", "renda": (">", 5_000)})
# → [("uf", "==", "SP"), ("renda", ">", 5000)]

# Pruning manual
relevant = prune_row_groups(row_groups_meta, {"uf": "SP"})
# → {0, 2, 5}  — índices dos batches que podem conter linhas com uf="SP"
```

---

## 8. Mascaramento de texto livre

Detecta e mascara PII em strings de formato livre (logs, e-mails, notas de atendimento).

```python
texto = "Cliente CPF 111.444.777-35, email joao@empresa.com"

# Detecção — retorna spans com tipo, valor, posição
spans = dd.scan_text(texto)
# [{"type": "CPF", "value": "111.444.777-35", "start": 11, "end": 25}, ...]

# Mascaramento — redact
dd.mask_text(texto, strategy="redact")
# "Cliente [CPF], [EMAIL]"

# Mascaramento — hash (requer salt para reprodutibilidade)
dd.mask_text(texto, salt=SALT, strategy="hash")
# "Cliente 3f2a8b1c9d4e7f0a, 9e1d3c7f"
```

---

## 8. Análise e transformação

O datalock expõe a API Polars via `dd.*`, permitindo manipulação de DataFrames sem importar Polars diretamente.

```python
# Filtragem
dd.where(df, uf="SP")
dd.where(df, renda_mensal=(5_000, 15_000))    # intervalo
dd.where(df, uf=["SP", "RJ", "MG"])           # múltiplos valores

# Seleção
dd.select(df, ["cpf", "renda_mensal", "uf"])
dd.drop(df, ["cpf_raw", "telefone_raw"])
dd.rename(df, {"cpf_hash": "cpf", "email_hash": "email"})

# Adicionar colunas com expressões
dd.add_column(df,
    imposto     = dd.col("renda_mensal") * 0.275,
    salario_liq = dd.col("renda_mensal") - dd.col("imposto"),
    faixa = dd.when(dd.col("renda_mensal") > 10_000, "alta")
              .when(dd.col("renda_mensal") > 5_000, "media")
              .otherwise("baixa"),
)

# Agrupamento
dd.groupby(df, "uf", {
    "n":       ("*", "count"),
    "media":   ("renda_mensal", "mean"),
    "mediana": ("renda_mensal", "median"),
})

# Ordenação / limite
dd.sort(df, "renda_mensal", descending=True)
dd.head(df, 100)
dd.tail(df, 50)
dd.sample(df, 1000, seed=42)
dd.top_n(df, "renda_mensal", n=10)

# Deduplicação
dd.unique(df, subset=["cpf"])

# Combinação
dd.concat([df1, df2, df3])
dd.pivot(df, on="uf", values="renda_mensal", agg_fn="mean")
dd.melt(df, id_vars=["cpf"], value_vars=["jan", "fev", "mar"])

# Séries temporais
dd.shift(df, 1)    # lag — valor do período anterior
dd.lead(df, 1)     # valor do próximo período

# Estatísticas
dd.describe(df)
dd.value_counts(df, "uf")
dd.corr(df, "renda_mensal", "score_credito")
dd.nunique(df)
dd.count_nulls(df)

# Informação
dd.schema(df)    # {coluna: dtype_str}
dd.dtypes(df)    # dict de dtypes
dd.shape(df)     # (linhas, colunas)

# Conversão
dd.to_pandas(df_polars)
dd.to_polars(df_pandas)

# JOIN seguro com verificação de compatibilidade de tokens
result = dd.join(df_clientes, df_pedidos, on="cpf", salt=SALT)
```

---

## 9. Banco de dados

```python
# Conexão
banco = dd.db("postgresql://user:pass@host:5432/db", salt=SALT)

# Exploração
print(banco.tables())
df_sample = banco.sample_table("clientes")

# Leitura com mascaramento automático
df = dd.read(banco, "clientes")
df = dd.read(banco, "SELECT * FROM clientes WHERE uf = 'SP'")
df = dd.read(banco, "clientes", sample=10_000)
df = dd.read(banco, "clientes", head=5_000)

# Escrita
banco.write(df_safe, "clientes_masked")
dd.write(df_safe, banco, "clientes_masked")
banco.upsert(df_novo, "clientes", on="cpf")

# Context manager
with dd.db("postgresql://...", salt=SALT) as banco:
    df = dd.read(banco, "clientes")

# Leitura legada com mascaramento
df = dd.read_db("postgresql://u:p@h/db", "clientes", salt=SALT, table=True)
df = dd.read_db("postgresql://u:p@h/db",
                "SELECT * FROM clientes WHERE uf = %s",
                salt=SALT, params=("SP",))
```

**URIs suportadas:**
- `postgresql://user:pass@host/db`
- `mysql+pymysql://user:pass@host/db`
- `sqlite:///arquivo.db`
- `mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17`
- `duckdb:///:memory:`

---

## 10. Pipeline fluente

```python
result = (
    dd.pipe("clientes.parquet")
    .where(uf="SP", tipo_pessoa="PF")
    .add_column(
        imposto = dd.col("renda_mensal") * 0.275,
        faixa   = dd.when(dd.col("renda_mensal") > 10_000, "alta").otherwise("baixa"),
    )
    .mask(salt=SALT)
    .collect()                        # materializa como pl.DataFrame
)

# Pipeline com saída para banco
(
    dd.pipe("clientes.csv")
    .mask(salt=SALT)
    .store("clientes_masked.dlk", key=KEY)
)
```

---

## 11. Contrato de dados

O `DataContract` unifica schema, validação de qualidade e regras de mascaramento em uma declaração versionável e exportável.

```python
LISTA_UFS = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG",
             "MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR",
             "RS","SC","SE","SP","TO"]

contrato = dd.contract({
    "cpf": {
        "type": "str", "not_null": True, "unique": True,
        "pii": "CPF", "mask": "hash",
    },
    "renda_mensal": {
        "type": "float", "min": 0, "max": 500_000,
    },
    "uf": {
        "type": "str", "in": LISTA_UFS, "pii": None,
    },
    "data_nasc": {
        "type": "str", "not_null": True,
        "pii": "data_nascimento", "mask": "generalize_date",
    },
})

# Validação
resultado = contrato.validate(df)
resultado.print_report()
resultado.raise_if_failed()  # lança ContractViolationError se houver falhas

# Mascaramento conforme o contrato
df_safe = contrato.mask(df, salt=SALT)

# Aplica validação + mascaramento em sequência
df_final = contrato.apply(df, salt=SALT)

# Detecção de breaking changes entre versões
diff = contrato.diff(contrato_v2)
if diff.has_breaking_changes:
    print(diff.report())

# Persistência e exportação
contrato.save("schema/clientes_v1.contract.json")
contrato2 = dd.DataContract.load("schema/clientes_v1.contract.json")
schema_json = contrato.to_json_schema()   # JSON Schema para documentação / DPO
```

---

## 12. Validação

```python
# Validação com regras declarativas
report = dd.validate(df, rules={
    "cpf":           {"not_null": True, "unique": True, "regex": r"^\d{11}$"},
    "renda_mensal":  {"min": 0, "max": 500_000},
    "uf":            {"in": LISTA_UFS},
    "data_cadastro": {"not_null": True},
})

# Expectativas individuais
dd.expect(df, "cpf").to_be_unique()
dd.expect(df, "renda_mensal").to_be_between(0, 500_000)
dd.expect(df, "uf").to_be_in(LISTA_UFS)

# Schema
dd.validate_schema(df, expected_schema={
    "cpf": "object", "renda_mensal": "float64", "uf": "object"
})

# Persistência de regras
dd.save_rules(rules, "regras/clientes.json")
rules = dd.load_rules("regras/clientes.json")
```

---

## 13. Canary data

O sistema canary injeta linhas sentinela rastreáveis nos dados. Se os dados aparecerem em um breach, os fingerprints identificam o pipeline de origem.

```python
# Injetar canary ao salvar (Nível 1 — no arquivo)
dd.store(df, "dados.dlk", key=KEY, canary=True, pipeline_id="crm_jan2025")

# As canary rows são removidas automaticamente na leitura
df = dd.read("dados.dlk", key=KEY)
assert df.shape == df_original.shape  # canary rows transparentes

# Verificar token encontrado em breach dump
resultado = dd.canary_check("canary.1ba472d8e3f9@datalock.internal")
# {
#   "fingerprint":   "1ba472d8e3f9c0a1",
#   "pipeline_id":   "crm_jan2025",
#   "filepath":      "dados.dlk",
#   "injected_at":   "2025-01-15T14:32:00Z",
#   "n_canary_rows": 3,
#   "level":         1
# }

# Verificar CPF canary
dd.canary_check("canary_1ba472d8")
```

**Segurança do canary:** configure `DATALOCK_CANARY_SALT` para tornar os fingerprints imprevisíveis. Com o salt padrão público, um adversário com acesso ao código-fonte pode pré-calcular os fingerprints.

---

## 14. Criptografia assimétrica

Permite compartilhar arquivos `.dlk` sem compartilhar a chave simétrica.

```python
# Gerar par de chaves
priv, pub = dd.generate_keypair("ec")   # Elliptic Curve (padrão)
priv, pub = dd.generate_keypair("rsa")  # RSA 4096

# Persistir
dd.save_keypair(priv, pub, "chaves/cliente")
# → chaves/cliente.private.pem, chaves/cliente.public.pem

# Carregar
priv = dd.load_private_key("chaves/cliente.private.pem")
pub  = dd.load_public_key("chaves/cliente.public.pem")

# Salvar com chave pública
dd.store(df, "dados.dlk", public_key=pub)

# Ler com chave privada
df = dd.read("dados.dlk", private_key=priv)

# Múltiplos destinatários
dd.store(df, "dados.dlk", public_keys=[pub_alice, pub_bob, pub_carlos])
```

---

## 15. Dados sintéticos

```python
# Treinar modelo generativo (requer pip install "datalock[synthetic]")
model = dd.train(df, n=1000)

# Gerar n linhas com as mesmas distribuições
df_synth = dd.clone(df, n=5000)

# Sandbox — sintético + mascarado para desenvolvimento
df_dev = dd.sandbox(df, n=1000, salt=SALT)

# Armazenar modelo no formato .dlk
dd.store(model, "modelo_ctgan.dlk", key=KEY)
model_loaded = dd.read("modelo_ctgan.dlk", key=KEY)
df_synth2 = model_loaded.generate(1000)

# Geração individual por coluna
from datalock.generators import SyntheticGenerator
gen = SyntheticGenerator(df)
gen.generate_column("renda_mensal", n=100)
gen.generate_row()
```

---

## 16. Métricas de privacidade

```python
from datalock import check

# K-anonimato
report = check.kanon(df, quasi_identifiers=["uf", "faixa_etaria"])
print(report.k)               # valor de k atual
print(report.at_risk_count)   # registros com k < limiar
report.print_report()

# T-closeness
report = check.tcloseness(df, quasi_identifiers=["uf", "faixa_etaria"],
                           sensitive_column="renda_mensal")
print(report.t_score)

# Risco de re-identificação
report = check.risk(df, quasi_identifiers=["uf", "faixa_etaria", "data_nasc"])
print(report.max_risk)         # risco máximo (0–1)
print(report.mean_risk)        # risco médio
report.print_report()

# Utilidade — compara original vs mascarado
report = check.utility(df_original, df_masked)
print(report.overall_score)    # score 0–1 de preservação de utilidade
report.print_report()

# Fidelidade de dados sintéticos
report = check.fidelity(df_real, df_synth, tstr_target="inadimplente")
print(report.overall_score)
report.print_report()

# Privacidade diferencial
dp = check.dp(epsilon=1.0)
noisy_mean = dp.add_laplace_noise(df["renda_mensal"].mean(), sensitivity=1000)
```

---

## 17. Varredura de diretório

```python
# Inventário de PII em uma pasta
inventario = dd.scan_directory("./dados/", recursive=True)

# Resumo geral
print(inventario.summary())

# Exportação
inventario.to_html("inventario_pii.html")
inventario.to_json("inventario_pii.json")

# Iteração por arquivo
for path, fi in inventario.items():
    if fi.max_risk == "high":
        print(f"RISCO ALTO: {path}")
        for col, r in fi.pii_columns.items():
            print(f"  {col}: {r.pii_type.value}")
```

---

## 18. Relatório de conformidade LGPD

```python
reports = dd.scan(df)
report = dd.compliance_report(
    df, reports,
    title="Relatório de Conformidade LGPD",
    organization="Empresa S.A.",
    dataset_name="Base de Clientes — Q1 2025",
    extra_notes="Revisado pelo DPO em 15/01/2025.",
)

report.to_html("lgpd_relatorio.html")
report.to_pdf("lgpd_relatorio.pdf")   # pip install weasyprint
report.to_text()                       # sempre disponível
report.to_json("lgpd_relatorio.json")
```

---

## 19. Linhagem de dados

```python
from datalock import lineage

# Rastrear transformações aplicadas ao DataFrame
lineage.track(df_original, operation="mask", output=df_masked,
              salt_hash=hash(SALT), columns_masked=list(reports.keys()))

# Visualizar histórico
lineage.view(df_masked)
```

---

## 20. CLI

A interface de linha de comando fornece acesso às principais operações sem escrever código.

**Segurança:** a chave nunca é aceita como argumento (`--key` foi removido). Use a variável de ambiente `DATALOCK_KEY` ou o prompt interativo (oculto).

```bash
# Detectar PII em CSV
datalock scan clientes.csv
datalock scan clientes.csv --sample 1000 --threshold 0.6 --json

# Mascarar PII
datalock mask clientes.csv --salt $DATALOCK_SALT
datalock mask clientes.csv --salt $DATALOCK_SALT --output clientes_safe.csv

# Inspecionar arquivo .dlk (solicita chave via prompt)
datalock inspect clientes.dlk
# → ou com env var: DATALOCK_KEY=... datalock inspect clientes.dlk

# Empacotar CSV em .dlk
datalock pack clientes.csv --output clientes.dlk
# (solicita chave via prompt ou usa DATALOCK_KEY)

# Extrair .dlk para CSV
datalock unpack clientes.dlk --output clientes_decifrado.csv

# Diagnóstico integrado
datalock profile clientes.csv --sample 1000
```

**Variáveis de ambiente:**

| Variável | Uso |
|----------|-----|
| `DATALOCK_KEY` | Chave AES para pack/unpack/inspect |
| `DATALOCK_SALT` | Salt HMAC para mask |
| `DATALOCK_CANARY_SALT` | Salt secreto para canary |

---

## 21. Referência de tipos e classes

### `PIIType`
`CPF`, `CNPJ`, `EMAIL`, `TELEFONE`, `NOME`, `DATA_NASCIMENTO`, `ENDERECO`, `CEP`, `IP_ADDRESS`, `CARTAO_CREDITO`, `CONTA_BANCARIA`, `SALARIO_RENDA`, `ETNIA`, `RELIGIAO`, `SAUDE`, `BIOMETRICO`, `PASSAPORTE`, `TITULO_ELEITOR`, `PLACA_VEICULO`, `GENERICO`

### `MaskStrategy`
`HASH`, `REDACT`, `TRUNCATE`, `GENERALIZE_DATE`, `MASK_PHONE_DDD`, `MOCK_NUM`, `MOCK_CAT`, `SUPPRESS`, `PASSTHROUGH`

### `RiskLevel`
`HIGH`, `MEDIUM`, `LOW`

### `ColumnReport`
```python
@dataclass
class ColumnReport:
    pii_type:     PIIType
    risk_level:   RiskLevel
    mask_strategy: MaskStrategy
    match_ratio:  float      # 0–1
    unique_ratio: float      # 0–1
    notes:        str
```

### `DlkFile`
Ver §5.6. Alias: `LGSFile`.

### `DataContract`
Ver §10. Campos: `fields: Dict[str, FieldSpec]`, `version: str`, `created_at: str`.

### `ExpiredFileError`
Subclasse de `ValueError`. Lançada ao ler arquivo cujo `expires_at` já passou.

### `IdempotencyError`
Lançada por `dd.mask(strict=True)` quando uma coluna já parece mascarada (evita duplo mascaramento).

---

## 22. Segurança: guia de boas práticas

### Chaves e salts

```python
# ✓ CORRETO — material de alta entropia, armazenado em vault
import os
KEY  = os.environ["DATALOCK_KEY"]   # gerado com secrets.token_hex(32)
SALT = os.environ["DATALOCK_SALT"]  # gerado com dd.generate_salt()

# ✗ ERRADO — chave derivada de senha humana (HKDF não faz stretching)
KEY = "MinhaSenha2024!"   # vulnerável a força bruta

# ✗ ERRADO — chave no código-fonte
KEY = "chave-hardcoded-no-repositorio"
```

### CLI

```bash
# ✓ CORRETO — chave via variável de ambiente
export DATALOCK_KEY="$(vault read -field=value secret/datalock/key)"
datalock pack clientes.csv

# ✗ ERRADO — chave como argumento (exposta em /proc, histórico de shell)
# (não é mais possível — --key foi removido da CLI)
```

### Salt e reprodutibilidade

```python
# ✓ CORRETO — mesmo salt para todas as tabelas do mesmo pipeline
dd.store(df_clientes, "clientes.dlk", key=KEY, salt=SALT)
dd.store(df_pedidos,  "pedidos.dlk",  key=KEY, salt=SALT)
result = dd.join(df_c_safe, df_p_safe, on="cpf")  # tokens compatíveis

# ✗ ERRADO — salts diferentes quebram JOINs
dd.store(df_clientes, "c.dlk", key=KEY, salt=SALT1)
dd.store(df_pedidos,  "p.dlk", key=KEY, salt=SALT2)
# JOINs em cpf não funcionarão — tokens diferentes para o mesmo CPF
```

### Arquivo v4 (sem criptografia)

```python
# ✓ CORRETO — v4 apenas para dados já anonimizados
dd.store(df_anonimo, "dev_data.dlk", anonymize=True)

# ✗ ERRADO — dados pessoais sem criptografia
dd.store(df_com_cpf, "dev_data.dlk")  # sem key= → v4 sem criptografia
```

### Pickle via `pack_bytes`

`pickle.loads()` executa código Python arbitrário. Qualquer detentor da `master_key` pode criar um payload malicioso. Prefira formatos seguros (ONNX, SafeTensors) para modelos. Se precisar de pickle, valide o hash SHA-256 do payload antes de deserializar.

---

## 23. Retrocompatibilidade

O datalock foi renomeado de `logus-lgpd`. O alias `import logus as lg` continua funcionando:

```python
# Ambos funcionam identicamente
import datalock as dd
import logus as lg

dd.mask(df, salt=SALT)
lg.mask(df, salt=SALT)   # idêntico

# Arquivos .lgs (extensão anterior) são lidos corretamente
df = dd.read("clientes.lgs", key=KEY)
```

Arquivos `.dlk` de todas as versões (v1, v2, v3, v4) são lidos por qualquer versão atual da biblioteca. Veja a especificação completa do formato na documentação do arquivo `.dlk`.
