# AGENTS.md — Guia para Agentes de IA

Este arquivo descreve como agentes de IA (Cursor, GitHub Copilot, Windsurf, Claude,
Gemini, ChatGPT) devem gerar código que usa a biblioteca `logus-lgpd`.

Leia este arquivo antes de gerar qualquer código que importe `logus`.

---

## Identidade da biblioteca

```python
import logus as lg   # SEMPRE este alias. Nunca: from logus import *
```

`logus` é uma biblioteca Python para:
1. Manipulação de dados com sintaxe expressiva (próxima de SQL)
2. Detecção e mascaramento de PII segundo a LGPD brasileira
3. Armazenamento cifrado no formato `.lgs` (AES-256-GCM)
4. Leitura de qualquer formato tabular com uma única função

Engine interna: Polars. `lg.col` é `polars.col`. `lg.lit` é `polars.lit`.

---

## Regras de geração de código

### REGRA 1 — Segredos nunca no código

```python
# ❌ ERRADO — nunca hardcode salt ou key
df_safe = lg.mask(df, salt="minha-chave-secreta")
lg.store(df, "dados.lgs", key="abc123")

# ✅ CORRETO — sempre de variáveis de ambiente
import os
SALT = os.environ["LOGUS_SALT"]
KEY  = os.environ["LOGUS_KEY"]
df_safe = lg.mask(df, salt=SALT)
lg.store(df, "dados.lgs", key=KEY)
```

Se o usuário não tiver as variáveis, sugira:
```bash
export LOGUS_SALT=$(python -c "import logus as lg; print(lg.generate_salt())")
export LOGUS_KEY=$(python -c "import logus as lg; print(lg.generate_salt())")
```

### REGRA 2 — salt e key são DIFERENTES

```python
# ❌ ERRADO — salt e key iguais
SEGREDO = lg.generate_salt()
lg.store(df, "dados.lgs", key=SEGREDO, salt=SEGREDO)  # → ValueError

# ✅ CORRETO — valores distintos com propósitos distintos
SALT = os.environ["LOGUS_SALT"]  # para mascaramento HMAC
KEY  = os.environ["LOGUS_KEY"]   # para criptografia AES
```

### REGRA 3 — Não use open() para .lgs

```python
# ❌ ERRADO
with open("dados.lgs", "rb") as f:
    data = f.read()

# ✅ CORRETO
df = lg.read("dados.lgs", key=KEY)
info = lg.inspect("dados.lgs", key=KEY)  # metadados sem descriptografar
```

### REGRA 4 — lg.when() sempre termina com .otherwise()

```python
# ❌ ERRADO — colunas sem .otherwise() têm null inesperado
faixa = lg.when(lg.col("renda") > 10_000, "alta").when(lg.col("renda") > 5_000, "media")

# ✅ CORRETO
faixa = lg.when(lg.col("renda") > 10_000, "alta") \
          .when(lg.col("renda") > 5_000,  "media") \
          .otherwise("baixa")
```

### REGRA 5 — lg.mask() sem salt= em produção → erro silencioso

```python
# ❌ PROBLEMÁTICO EM PRODUÇÃO — gera salt aleatório a cada execução
df_safe = lg.mask(df)  # hashes mudam entre runs, JOINs quebram

# ✅ CORRETO — configure uma vez no startup
lg.configure(default_salt=os.environ["LOGUS_SALT"])
df_safe = lg.mask(df)  # usa LOGUS_SALT automaticamente
# OU
df_safe = lg.mask(df, salt=SALT)  # explícito
```

### REGRA 6 — Preservação de tipo

```python
df_pl = pl.DataFrame({"cpf": ["111.444.777-35"], "renda": [5000.0]})
df_pd = df_pl.to_pandas()

resultado_pl = lg.mask(df_pl, salt=SALT)  # → pl.DataFrame
resultado_pd = lg.mask(df_pd, salt=SALT)  # → pd.DataFrame
# A função retorna o mesmo tipo que recebeu
```

### REGRA 7 — Caminhos com pathlib.Path

```python
# ✅ PREFERIDO
from pathlib import Path
df = lg.read(Path("data") / "clientes.parquet")
lg.store(df, Path("output") / "clientes.lgs", key=KEY)
```

---

## Padrões de uso corretos

### Leitura universal

```python
# lg.read() detecta formato pela extensão e encoding automaticamente
df = lg.read("clientes.csv")           # pl.DataFrame
df = lg.read("clientes.xlsx")          # pl.DataFrame (requer [excel])
df = lg.read("clientes.parquet")       # pl.DataFrame
df = lg.read("clientes.lgs", key=KEY)  # pl.DataFrame (descriptografa)

# Com mascaramento automático
df = lg.read("clientes.csv", salt=SALT)
df = lg.read("clientes.lgs", key=KEY, salt=SALT)

# Big data — sem carregar tudo na memória
df   = lg.read("big.parquet", head=100_000)
df   = lg.read("big.parquet", sample=500_000)      # row groups aleatórios
info = lg.read("big.parquet", header_only=True)    # zero dados lidos
df   = lg.read("big.parquet", n_chunks=5, chunks=[2,4])
for chunk in lg.read("big.parquet", n_chunks=10, iter_chunks=True):
    processar(chunk)
```

### Mascaramento

```python
SALT = os.environ["LOGUS_SALT"]
KEY  = os.environ["LOGUS_KEY"]

# Detecta PII primeiro (opcional — mask() faz internamente)
reports = lg.scan(df)

# Mascara
df_safe = lg.mask(df, salt=SALT)

# Verifica o que mudou
diff = lg.diff(df, df_safe)
print(diff["summary"])

# Salva cifrado
lg.store(df_safe, "clientes.lgs", key=KEY)
```

### Manipulação

```python
# Filtragem
df_sp = lg.where(df, uf="SP")
df_ricos = lg.where(df, renda=(">", 10_000))
df_combo = lg.where(df, uf=["SP","RJ"], renda=(5_000, 50_000))

# Novas colunas
df = lg.add_column(df,
    imposto = lg.col("renda") * 0.275,
    faixa   = lg.when(lg.col("renda") > 10_000, "alta")
                .when(lg.col("renda") > 5_000, "media")
                .otherwise("baixa"),
)

# Agrupamento
resultado = lg.groupby(df, "uf", {
    "n":     ("*", "count"),
    "media": ("renda", "mean"),
}, having={"n": (">", 100)}, sort="media", desc=True)
```

### Pipeline completo

```python
# Opção A — pipe fluente
result = (
    lg.pipe("clientes.parquet")
    .where(uf="SP")
    .add_column(imposto=lg.col("renda") * 0.275)
    .mask(salt=SALT)
    .collect()
)

# Opção B — process() unificado
result = lg.process(
    "clientes.parquet",
    salt=SALT,
    key=KEY,
    output="clientes_safe.lgs",
    where={"uf": "SP"},
    rules={"cpf": {"not_null": True}},
)
```

---

## Arquitetura de módulos

```
logus/
├── __init__.py       # API pública: lg.read, lg.mask, lg.scan, lg.store, lg.where...
├── core.py           # Motor: read_file(), mask_frame(), mask_lazyframe()
├── analytics.py      # DSL: where, groupby, sort, add_column, when, pivot...
├── io_big.py         # Big data: read_partial(), DatabaseConnection, build_csv_index()
├── processor.py      # Pipeline: process(), ProcessResult
├── validate.py       # Data quality: validate(), expect(), ValidationReport
├── lineage.py        # Linhagem: LineageTracker, session()
├── privacy_score.py  # Score LGPD: calculate(), PrivacyScore
├── sql_transpiler.py # SQL: mask_sql(), generate_view()
├── secure_file.py    # Formato .lgs: SecureFile (AES-256-GCM)
├── lgs.py            # OO wrapper: LGSFile, context manager
├── check.py          # Métricas: kanon, risk, utility, dp, tcloseness
├── link.py           # DB: lg.db(), SecureDBAdapter, SQLAdapter
├── detectors/
│   ├── fast_scan.py      # FastPIIScanner — engine padrão (9× mais rápido)
│   └── pii_detector.py   # PIIDetector — engine clássico (retrocompat)
├── adapters/
│   ├── polars_adapter.py  # Mascaramento vetorizado Polars
│   └── pandas_adapter.py  # Mascaramento pandas (fallback)
└── metrics/
    ├── kanonymity.py
    ├── risk_score.py
    ├── utility.py
    ├── fidelity.py
    ├── differential_privacy.py
    └── tcloseness.py
```

---

## Tipos de retorno

| Função | Input | Output |
|--------|-------|--------|
| `lg.read(str)` | caminho | `pl.DataFrame` |
| `lg.read(pd.DataFrame)` | pandas | `pd.DataFrame` |
| `lg.read(pl.DataFrame)` | polars | `pl.DataFrame` |
| `lg.read(banco, tabela)` | DB | `pl.DataFrame` |
| `lg.mask(pd.DataFrame)` | pandas | `pd.DataFrame` |
| `lg.mask(pl.DataFrame)` | polars | `pl.DataFrame` |
| `lg.mask(pl.LazyFrame)` | lazy | `pl.LazyFrame` |
| `lg.scan(df)` | qualquer | `Dict[str, ColumnReport]` |
| `lg.where(df, ...)` | preserva | mesmo tipo do input |
| `lg.groupby(df, ...)` | preserva | mesmo tipo do input |
| `lg.validate(df, rules)` | qualquer | `ValidationReport` |
| `lg.process(...)` | qualquer | `ProcessResult` |
| `lg.profile(df)` | qualquer | `dict` (JSON-serializable) |

---

## Quando usar cada função

| Cenário | Função recomendada |
|---------|-------------------|
| Ler arquivo qualquer formato | `lg.read()` |
| Ler arquivo grande sem OOM | `lg.read(..., head=N)` ou `lg.stream()` |
| Ler tabela de banco de dados | `lg.db() + lg.read(banco, tabela)` |
| Detectar PII em DataFrame | `lg.scan(df)` |
| Mascarar PII para análise | `lg.mask(df, salt=SALT)` |
| Mascarar e salvar cifrado | `lg.store(df, "f.lgs", key=KEY, salt=SALT)` |
| Pipeline lê+transforma+mascara+salva | `lg.process()` ou `lg.pipe()` |
| Validar qualidade dos dados | `lg.validate(df, rules)` |
| Diagnóstico rápido LGPD | `lg.profile(df)` |
| SQL sobre DataFrames | `lg.sql(query, df=df)` |
| JOIN seguro entre tabelas mascaradas | `lg.join(df1, df2, on="cpf", salt=SALT)` |
| Mascaramento dentro do banco | `lg.link.db().in_db_mask()` |
| Gerar CREATE VIEW mascarada | `lg.generate_view()` |
| Dados sintéticos para testes | `lg.clone(df)` ou `lg.sandbox(df)` |
| Avaliar k-anonimato | `lg.check.kanon()` |
| Privacy score composto | `lg.profile(df)["privacy_score"]` |

---

## Exemplos de erro e correção

### ImportError: No module named 'duckdb'
```bash
pip install 'logus-lgpd[sql]'
```

### ImportError: No module named 'openpyxl'
```bash
pip install 'logus-lgpd[excel]'
```

### ValueError: key= e salt= não devem ser iguais
```python
# Use valores diferentes gerados com lg.generate_salt()
SALT = lg.generate_salt()
KEY  = lg.generate_salt()
```

### IdempotencyError: Coluna 'cpf' parece já mascarada
```python
# Os dados já foram mascarados. Não mascare duas vezes.
# Se precisar re-mascarar: use strict=False
df_safe = lg.mask(df_already_masked, salt=SALT, strict=False)
```

### UserWarning: lg.mask(salt=None) salt aleatório
```python
# Configure o salt padrão no startup da aplicação
lg.configure(default_salt=os.environ["LOGUS_SALT"])
```

### FileNotFoundError ao lg.read()
```python
from pathlib import Path
# Verifique se o arquivo existe antes de ler
p = Path("clientes.lgs")
if not p.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {p}")
df = lg.read(p, key=KEY)
```

---

## Big data (v1.0.5)

```python
df   = lg.read("big.parquet", header_only=True)           # 0ms
df   = lg.read("big.parquet", head=100_000)               # first N rows
df   = lg.read("big.parquet", sample=500_000)             # random row groups
df   = lg.read("big.parquet", n_chunks=5, chunks=[2,4])   # exact chunks (Parquet)
for chunk in lg.read("big.parquet", iter_chunks=True):
    process(chunk)
```

## Database connection (v1.0.5)

```python
banco = lg.db("postgresql://user:pass@host/db", salt=SALT)
df    = lg.read(banco, "clientes")
df    = lg.read(banco, "SELECT * FROM clientes WHERE uf='SP'")
df    = lg.read(banco, "clientes", sample=10_000)
banco.write(df_safe, "clientes_masked")
lg.write(df_safe, banco, "clientes_masked")
```

## mask(LazyFrame) (v1.0.5)

```python
lf_safe = lg.mask(df.lazy(), salt=SALT)   # returns pl.LazyFrame
result  = lf_safe.collect()
```

---

## v1.1.0 new features

### lg.contract() — Data Contract (unified validation + PII + masking)
```python
# Declare once, apply consistently
contrato = lg.contract({
    "cpf":   {"type":"str","not_null":True,"unique":True,"pii":"CPF","mask":"hash"},
    "renda": {"type":"float","min":0,"max":500_000,"pii":"numerico","mask":"mock_numeric"},
    "uf":    {"type":"str","in":["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE"]},
}, name="clientes", version="2.0")

result = contrato.apply(df, salt=SALT)  # validate + mask
result.raise_if_failed()
df_safe = result.df

contrato.save("schema.contract.json")
contrato2 = lg.DataContract.load("schema.contract.json")
diff = contrato.diff(contrato2)
print(diff.has_breaking_changes, diff.report())
json_schema = contrato.to_json_schema()  # JSON Schema draft-07
```

### Custom PII patterns
```python
reports = lg.scan(df, custom_patterns={
    "num_contrato": r"^CTR-[0-9]{8}$",
    "matricula":    r"^[0-9]{6}-[A-Z]$",
})
```

### Asymmetric encryption (EC P-256 or RSA)
```python
priv, pub = lg.generate_keypair("ec")    # or "rsa"
lg.save_keypair(priv,"priv.pem", pub,"pub.pem")
priv = lg.load_private_key("priv.pem")
pub  = lg.load_public_key("pub.pem")

lg.store(df, "dados.lgs", public_key=pub)     # encrypt for recipient
df  = lg.read("dados.lgs", private_key=priv)  # recipient decrypts
```

### File expiration
```python
lg.store(df,"dados.lgs",key=KEY,expires_at="2025-12-31")
# ExpiredFileError raised after that date
```

### Schema validation + rule persistence
```python
lg.validate_schema(df, required_columns=["cpf","renda"], min_rows=1).raise_if_failed()
lg.save_rules({"cpf":{"not_null":True}}, "rules.json")
rules = lg.load_rules("rules.json")
```

### Compliance report
```python
report = lg.compliance_report(df, reports, dataset_name="DS")
report.to_html("lgpd.html")   # report.to_pdf() with weasyprint
report.to_text()              # always available
```

### Time-series + nested data
```python
lg.shift(df, 1)                     # previous value (lag)
lg.lead(df, 1)                      # next value
lg.lag(df, 3, columns="renda")      # alias for shift(3)
lg.explode(df, "tags")              # list column → multiple rows
```

### .env + create_table + upsert
```python
lg.configure(load_dotenv=True)                          # reads .env
banco.create_table(df, "t", if_exists="replace")        # DDL from DataFrame
banco.upsert(df_new, "clientes", on="cpf")              # upsert
```
