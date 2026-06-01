# AGENTS.md — Guia para Agentes de IA

Este arquivo descreve como agentes de IA (Cursor, GitHub Copilot, Windsurf, Claude,
Gemini, ChatGPT) devem gerar código que usa a biblioteca `datalock`.

Leia este arquivo antes de gerar qualquer código que importe `datalock`.

---

## Identidade da biblioteca

```python
import datalock as dd   # SEMPRE este alias. Nunca: from datalock import *
```

`datalock` é uma biblioteca Python para:
1. Manipulação de dados com sintaxe expressiva (próxima de SQL)
2. Detecção e mascaramento de PII segundo a LGPD brasileira
3. Armazenamento cifrado no formato `.dlk` (AES-256-GCM)
4. Leitura de qualquer formato tabular com uma única função

Engine interna: Polars. `dd.col` é `polars.col`. `dd.lit` é `polars.lit`.

---

## Regras de geração de código

### REGRA 1 — Segredos nunca no código

```python
# ❌ ERRADO — nunca hardcode salt ou key
df_safe = dd.mask(df, salt="minha-chave-secreta")
dd.store(df, "dados.dlk", key="abc123")

# ✅ CORRETO — sempre de variáveis de ambiente
import os
SALT = os.environ["DATALOCK_SALT"]
KEY  = os.environ["DATALOCK_KEY"]
df_safe = dd.mask(df, salt=SALT)
dd.store(df, "dados.dlk", key=KEY)
```

Se o usuário não tiver as variáveis, sugira:
```bash
export DATALOCK_SALT=$(python -c "import datalock as dd; print(dd.generate_salt())")
export DATALOCK_KEY=$(python -c "import datalock as dd; print(dd.generate_salt())")
```

### REGRA 2 — salt e key são DIFERENTES

```python
# ❌ ERRADO — salt e key iguais
SEGREDO = dd.generate_salt()
dd.store(df, "dados.dlk", key=SEGREDO, salt=SEGREDO)  # → ValueError

# ✅ CORRETO — valores distintos com propósitos distintos
SALT = os.environ["DATALOCK_SALT"]  # para mascaramento HMAC
KEY  = os.environ["DATALOCK_KEY"]   # para criptografia AES
```

### REGRA 3 — Não use open() para .dlk

```python
# ❌ ERRADO
with open("dados.dlk", "rb") as f:
    data = f.read()

# ✅ CORRETO
df   = dd.read("dados.dlk", key=KEY)
info = dd.inspect("dados.dlk", key=KEY)  # metadados sem descriptografar
```

### REGRA 4 — dd.when() sempre termina com .otherwise()

```python
# ❌ ERRADO — colunas sem .otherwise() têm null inesperado
faixa = dd.when(dd.col("renda") > 10_000, "alta").when(dd.col("renda") > 5_000, "media")

# ✅ CORRETO
faixa = dd.when(dd.col("renda") > 10_000, "alta") \
          .when(dd.col("renda") > 5_000,  "media") \
          .otherwise("baixa")
```

### REGRA 5 — dd.mask() sem salt= em produção → erro silencioso

```python
# ❌ PROBLEMÁTICO EM PRODUÇÃO — gera salt aleatório a cada execução
df_safe = dd.mask(df)  # hashes mudam entre runs, JOINs quebram

# ✅ CORRETO — configure uma vez no startup
dd.configure(default_salt=os.environ["DATALOCK_SALT"])
df_safe = dd.mask(df)  # usa DATALOCK_SALT automaticamente
# OU
df_safe = dd.mask(df, salt=SALT)  # explícito
```

### REGRA 6 — Preservação de tipo

```python
df_pl = pl.DataFrame({"cpf": ["111.444.777-35"], "renda": [5000.0]})
df_pd = df_pl.to_pandas()

resultado_pl = dd.mask(df_pl, salt=SALT)  # → pl.DataFrame
resultado_pd = dd.mask(df_pd, salt=SALT)  # → pd.DataFrame
# A função retorna o mesmo tipo que recebeu
```

### REGRA 7 — Caminhos com pathlib.Path

```python
# ✅ PREFERIDO
from pathlib import Path
df = dd.read(Path("data") / "clientes.parquet")
dd.store(df, Path("output") / "clientes.dlk", key=KEY)
```

---

## Padrões de uso corretos

### Leitura universal

```python
# dd.read() detecta formato pela extensão e encoding automaticamente
df = dd.read("clientes.csv")            # pl.DataFrame
df = dd.read("clientes.xlsx")           # pl.DataFrame (requer [excel])
df = dd.read("clientes.parquet")        # pl.DataFrame
df = dd.read("clientes.dlk", key=KEY)   # pl.DataFrame (descriptografa)

# Com mascaramento automático
df = dd.read("clientes.csv", salt=SALT)
df = dd.read("clientes.dlk", key=KEY, salt=SALT)

# Big data — sem carregar tudo na memória
df   = dd.read("big.parquet", head=100_000)
df   = dd.read("big.parquet", sample=500_000)      # row groups aleatórios
info = dd.read("big.parquet", header_only=True)    # zero dados lidos
df   = dd.read("big.parquet", n_chunks=5, chunks=[2,4])
for chunk in dd.read("big.parquet", n_chunks=10, iter_chunks=True):
    processar(chunk)
```

### Mascaramento

```python
SALT = os.environ["DATALOCK_SALT"]
KEY  = os.environ["DATALOCK_KEY"]

# Detecta PII primeiro (opcional — mask() faz internamente)
reports = dd.scan(df)

# Mascara
df_safe = dd.mask(df, salt=SALT)

# Verifica o que mudou
diff = dd.diff(df, df_safe)
print(diff["summary"])

# Salva cifrado
dd.store(df_safe, "clientes.dlk", key=KEY)
```

### Manipulação

```python
# Filtragem
df_sp    = dd.where(df, uf="SP")
df_ricos = dd.where(df, renda=(">", 10_000))
df_combo = dd.where(df, uf=["SP","RJ"], renda=(5_000, 50_000))

# Novas colunas
df = dd.add_column(df,
    imposto = dd.col("renda") * 0.275,
    faixa   = dd.when(dd.col("renda") > 10_000, "alta")
                .when(dd.col("renda") > 5_000, "media")
                .otherwise("baixa"),
)

# Agrupamento
resultado = dd.groupby(df, "uf", {
    "n":     ("*", "count"),
    "media": ("renda", "mean"),
}, having={"n": (">", 100)}, sort="media", desc=True)
```

### Pipeline completo

```python
# Opção A — pipe fluente
result = (
    dd.pipe("clientes.parquet")
    .where(uf="SP")
    .add_column(imposto=dd.col("renda") * 0.275)
    .mask(salt=SALT)
    .collect()
)

# Opção B — process() unificado
result = dd.process(
    "clientes.parquet",
    salt=SALT,
    key=KEY,
    output="clientes_safe.dlk",
    where={"uf": "SP"},
    rules={"cpf": {"not_null": True}},
)
```

---

## Arquitetura de módulos

```
datalock/
├── __init__.py       # API pública: dd.read, dd.mask, dd.scan, dd.store, dd.where...
├── core.py           # Motor: read_file(), mask_frame(), mask_lazyframe()
├── analytics.py      # DSL: where, groupby, sort, add_column, when, pivot...
├── io_big.py         # Big data: read_partial(), DatabaseConnection, build_csv_index()
├── processor.py      # Pipeline: process(), ProcessResult
├── validate.py       # Data quality: validate(), expect(), ValidationReport
├── lineage.py        # Linhagem: LineageTracker, session()
├── privacy_score.py  # Score LGPD: calculate(), PrivacyScore
├── sql_transpiler.py # SQL: mask_sql(), generate_view()
├── secure_file.py    # Formato .dlk: SecureFile (AES-256-GCM)
├── dlk.py            # OO wrapper: DLKFile, context manager
├── check.py          # Métricas: kanon, risk, utility, dp, tcloseness
├── link.py           # DB: dd.db(), SecureDBAdapter, SQLAdapter
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
| `dd.read(str)` | caminho | `pl.DataFrame` |
| `dd.read(pd.DataFrame)` | pandas | `pd.DataFrame` |
| `dd.read(pl.DataFrame)` | polars | `pl.DataFrame` |
| `dd.read(banco, tabela)` | DB | `pl.DataFrame` |
| `dd.mask(pd.DataFrame)` | pandas | `pd.DataFrame` |
| `dd.mask(pl.DataFrame)` | polars | `pl.DataFrame` |
| `dd.mask(pl.LazyFrame)` | lazy | `pl.LazyFrame` |
| `dd.scan(df)` | qualquer | `Dict[str, ColumnReport]` |
| `dd.where(df, ...)` | preserva | mesmo tipo do input |
| `dd.groupby(df, ...)` | preserva | mesmo tipo do input |
| `dd.validate(df, rules)` | qualquer | `ValidationReport` |
| `dd.process(...)` | qualquer | `ProcessResult` |
| `dd.profile(df)` | qualquer | `dict` (JSON-serializable) |

---

## Quando usar cada função

| Cenário | Função recomendada |
|---------|-------------------|
| Ler arquivo qualquer formato | `dd.read()` |
| Ler arquivo grande sem OOM | `dd.read(..., head=N)` ou `dd.stream()` |
| Ler tabela de banco de dados | `dd.db() + dd.read(banco, tabela)` |
| Detectar PII em DataFrame | `dd.scan(df)` |
| Mascarar PII para análise | `dd.mask(df, salt=SALT)` |
| Mascarar e salvar cifrado | `dd.store(df, "f.dlk", key=KEY, salt=SALT)` |
| Pipeline lê+transforma+mascara+salva | `dd.process()` ou `dd.pipe()` |
| Validar qualidade dos dados | `dd.validate(df, rules)` |
| Diagnóstico rápido LGPD | `dd.profile(df)` |
| SQL sobre DataFrames | `dd.sql(query, df=df)` |
| JOIN seguro entre tabelas mascaradas | `dd.join(df1, df2, on="cpf", salt=SALT)` |
| Mascaramento dentro do banco | `dd.link.db().in_db_mask()` |
| Gerar CREATE VIEW mascarada | `dd.generate_view()` |
| Dados sintéticos para testes | `dd.clone(df)` ou `dd.sandbox(df)` |
| Avaliar k-anonimato | `dd.check.kanon()` |
| Privacy score composto | `dd.profile(df)["privacy_score"]` |

---

## Exemplos de erro e correção

### ImportError: No module named 'duckdb'
```bash
pip install 'datalock[sql]'
```

### ImportError: No module named 'openpyxl'
```bash
pip install 'datalock[excel]'
```

### ValueError: key= e salt= não devem ser iguais
```python
# Use valores diferentes gerados com dd.generate_salt()
SALT = dd.generate_salt()
KEY  = dd.generate_salt()
```

### IdempotencyError: Coluna 'cpf' parece já mascarada
```python
# Os dados já foram mascarados. Não mascare duas vezes.
# Se precisar re-mascarar: use strict=False
df_safe = dd.mask(df_already_masked, salt=SALT, strict=False)
```

### UserWarning: dd.mask(salt=None) salt aleatório
```python
# Configure o salt padrão no startup da aplicação
dd.configure(default_salt=os.environ["DATALOCK_SALT"])
```

### FileNotFoundError ao dd.read()
```python
from pathlib import Path
# Verifique se o arquivo existe antes de ler
p = Path("clientes.dlk")
if not p.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {p}")
df = dd.read(p, key=KEY)
```

---

## Big data (v1.0.5)

```python
df   = dd.read("big.parquet", header_only=True)           # 0ms
df   = dd.read("big.parquet", head=100_000)               # first N rows
df   = dd.read("big.parquet", sample=500_000)             # random row groups
df   = dd.read("big.parquet", n_chunks=5, chunks=[2,4])   # exact chunks (Parquet)
for chunk in dd.read("big.parquet", iter_chunks=True):
    process(chunk)
```

## Database connection (v1.0.5)

```python
banco = dd.db("postgresql://user:pass@host/db", salt=SALT)
df    = dd.read(banco, "clientes")
df    = dd.read(banco, "SELECT * FROM clientes WHERE uf='SP'")
df    = dd.read(banco, "clientes", sample=10_000)
banco.write(df_safe, "clientes_masked")
dd.write(df_safe, banco, "clientes_masked")
```

## mask(LazyFrame) (v1.0.5)

```python
lf_safe = dd.mask(df.lazy(), salt=SALT)   # returns pl.LazyFrame
result  = lf_safe.collect()
```

---

## v1.1.0 new features

### dd.contract() — Data Contract (unified validation + PII + masking)
```python
# Declare once, apply consistently
contrato = dd.contract({
    "cpf":   {"type":"str","not_null":True,"unique":True,"pii":"CPF","mask":"hash"},
    "renda": {"type":"float","min":0,"max":500_000,"pii":"numerico","mask":"mock_numeric"},
    "uf":    {"type":"str","in":["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE"]},
}, name="clientes", version="2.0")

result = contrato.apply(df, salt=SALT)  # validate + mask
result.raise_if_failed()
df_safe = result.df

contrato.save("schema.contract.json")
contrato2 = dd.DataContract.load("schema.contract.json")
diff = contrato.diff(contrato2)
print(diff.has_breaking_changes, diff.report())
json_schema = contrato.to_json_schema()  # JSON Schema draft-07
```

### Custom PII patterns
```python
reports = dd.scan(df, custom_patterns={
    "num_contrato": r"^CTR-[0-9]{8}$",
    "matricula":    r"^[0-9]{6}-[A-Z]$",
})
```

### Asymmetric encryption (EC P-256 or RSA)
```python
priv, pub = dd.generate_keypair("ec")    # or "rsa"
dd.save_keypair(priv, "priv.pem", pub, "pub.pem")
priv = dd.load_private_key("priv.pem")
pub  = dd.load_public_key("pub.pem")

dd.store(df, "dados.dlk", public_key=pub)     # encrypt for recipient
df  = dd.read("dados.dlk", private_key=priv)  # recipient decrypts
```

### File expiration
```python
dd.store(df, "dados.dlk", key=KEY, expires_at="2025-12-31")
# ExpiredFileError raised after that date
```

### Schema validation + rule persistence
```python
dd.validate_schema(df, required_columns=["cpf","renda"], min_rows=1).raise_if_failed()
dd.save_rules({"cpf":{"not_null":True}}, "rules.json")
rules = dd.load_rules("rules.json")
```

### Compliance report
```python
report = dd.compliance_report(df, reports, dataset_name="DS")
report.to_html("lgpd.html")   # report.to_pdf() with weasyprint
report.to_text()              # always available
```

### Time-series + nested data
```python
dd.shift(df, 1)                     # previous value (lag)
dd.lead(df, 1)                      # next value
dd.lag(df, 3, columns="renda")      # alias for shift(3)
dd.explode(df, "tags")              # list column → multiple rows
```

### .env + create_table + upsert
```python
dd.configure(load_dotenv=True)                          # reads .env
banco.create_table(df, "t", if_exists="replace")        # DDL from DataFrame
banco.upsert(df_new, "clientes", on="cpf")              # upsert
```

### Canary salt configurável (v1.1.0)
```python
# Para ambientes de produção — impede pré-cálculo de fingerprints por adversários
dd.configure(canary_salt=os.environ["DATALOCK_CANARY_SALT"])
# Ou via .env com load_dotenv=True (lê DATALOCK_CANARY_SALT automaticamente)
dd.configure(load_dotenv=True)
```

---

## v1.0.1 (datalock — renamed from logus-lgpd)

### Rename summary
```python
# OLD                            NEW
import logus as lg           →   import datalock as dd
lg.mask(df, salt=SALT)       →   dd.mask(df, salt=SALT)
"file.lgs"                   →   "file.dlk"
# Backward compat: import logus as lg STILL WORKS
```

### Canary data (transparent)
```python
# Inject — user never sees canary rows
dd.store(df, "f.dlk", key=KEY, canary=True)

# Read — canary stripped automatically, shape == original
df = dd.read("f.dlk", key=KEY)   # df.shape == original df.shape

# Breach detected: look up token
dd.canary_check("canary.1ba472d8@datalock.internal")
# → {"pipeline_id": "crm_jan2025", "filepath": "f.dlk", ...}
```

### mask_text strategies
```python
dd.mask_text(text, salt=SALT, strategy="redact")   # [CPF], [EMAIL]
dd.mask_text(text, salt=SALT, strategy="hash")     # hex tokens
dd.mask_text(text, salt=SALT, strategy="semantic") # valid fake CPF/email
dd.mask_text(text, salt=SALT, strategy="partial")  # 529.***.**-**
```

### scan_directory
```python
inv = dd.scan_directory("./dados/")
print(inv.summary())
inv.to_html("report.html")
```

### audit_webhook
```python
dd.configure(audit_webhook="https://hooks.slack.com/...")
# Non-blocking POST per operation — never crashes user pipeline
```
