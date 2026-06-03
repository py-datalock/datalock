# datalock

**Privacy-by-design for tabular data in Python.**
LGPD/GDPR compliance · automatic PII detection and masking · AES-256-GCM encrypted storage (`.dlk`) · expressive Polars-based DSL · canary data for leak tracing.

```bash
pip install datalock
```

```python
import datalock as dd
import os

SALT = os.environ["DATALOCK_SALT"]   # HMAC key for masking
KEY  = os.environ["DATALOCK_KEY"]    # AES key for encryption — must differ from SALT

df       = dd.read("clientes.csv")           # any format → pl.DataFrame
df_safe  = dd.mask(df, salt=SALT)            # detect + mask PII (LGPD)
dd.store(df_safe, "clientes.dlk", key=KEY)   # AES-256-GCM encrypted file
df_back  = dd.read("clientes.dlk", key=KEY)  # decrypt and read back
```

> Renamed from `logus-lgpd`. `import logus as lg` still works as an alias.

---

## Table of contents

- [Installation](#installation)
- [When to use datalock](#when-to-use-datalock)
- [Core concepts](#core-concepts)
- [Reading data](#reading-data)
- [Detecting PII](#detecting-pii)
- [Masking PII](#masking-pii)
- [Storing encrypted files](#storing-encrypted-files-dlk)
- [Inspecting without decrypting](#inspecting-without-decrypting)
- [Text masking](#text-masking)
- [Data manipulation](#data-manipulation)
- [Full pipelines](#full-pipelines)
- [Data contracts](#data-contracts)
- [Validation](#validation)
- [Database integration](#database-integration)
- [Canary data](#canary-data)
- [Privacy metrics](#privacy-metrics)
- [Synthetic data](#synthetic-data)
- [Directory inventory](#directory-inventory)
- [Configuration](#configuration)
- [The .dlk format](#the-dlk-format)
- [datalock vs alternatives](#datalock-vs-alternatives)
- [Common errors and fixes](#common-errors-and-fixes)
- [Backward compatibility](#backward-compatibility)

---

## Installation

```bash
pip install datalock                   # core
pip install "datalock[sql]"            # + SQL via DuckDB
pip install "datalock[excel]"          # + Excel (.xlsx / .xls / .ods)
pip install "datalock[synthetic]"      # + richer synthetic data via Faker
pip install "datalock[full]"           # everything
```

**Requires:** Python ≥ 3.10, Polars ≥ 1.0, pandas ≥ 2.0, PyArrow ≥ 14.0

---

## When to use datalock

Use datalock when your DataFrame contains personal data (CPF, email, name, phone, address, income) and any of the following apply:

- The file will leave a controlled environment (email, external collaborator, S3 bucket)
- You need to prove to an auditor when a file was created, by which pipeline, and when it expires — without showing them the data itself
- You need to know which export a breach came from, after the fact
- You need to mask PII before giving data to a data scientist, while preserving JOIN keys across tables

**Do not need datalock** when:
- The data has no personal fields at all
- You fully control the storage environment and the file never leaves it
- Millisecond write latency is critical (use Parquet directly)

---

## Core concepts

### Two separate secrets

datalock uses two different secrets with different purposes:

```python
SALT = os.environ["DATALOCK_SALT"]  # HMAC key: makes masking deterministic
KEY  = os.environ["DATALOCK_KEY"]   # AES key: encrypts the .dlk file

# They must be different values — passing the same value raises ValueError
```

`SALT` controls masking: the same CPF + same SALT → always the same token. This lets you JOIN masked tables created at different times. `KEY` controls who can open the `.dlk` file.

Generate both with:

```bash
export DATALOCK_SALT=$(python -c "import datalock as dd; print(dd.generate_salt())")
export DATALOCK_KEY=$(python -c  "import datalock as dd; print(dd.generate_salt())")
```

### Type preservation

Every datalock function returns the same type it receives:

```python
dd.mask(pd.DataFrame(...), salt=SALT)  # → pd.DataFrame
dd.mask(pl.DataFrame(...), salt=SALT)  # → pl.DataFrame
dd.mask(pl.LazyFrame(...), salt=SALT)  # → pl.LazyFrame (stays lazy)
dd.where(df, uf="SP")                  # → same type as df
```

### dd.col is polars.col

```python
dd.col("renda") * 0.275          # polars expression
dd.col("nome").str.to_uppercase() # all 200+ Polars methods work
dd.lit(0)                         # polars.lit
dd.concat_str(...)                # polars.concat_str
```

---

## Reading data

`dd.read()` auto-detects format by extension and encoding:

```python
df = dd.read("clientes.csv")
df = dd.read("clientes.parquet")
df = dd.read("clientes.xlsx")         # requires datalock[excel]
df = dd.read("clientes.json")
df = dd.read("clientes.dlk", key=KEY) # AES-256-GCM encrypted

# Read + mask in one call
df = dd.read("clientes.csv",        salt=SALT)
df = dd.read("clientes.dlk", key=KEY, salt=SALT)

# Read specific frame from multi-frame .dlk
df = dd.read("base.dlk", key=KEY, frame="clientes")

# Read raw (no masking) from .dlk
df = dd.read("base.dlk", key=KEY, raw=True)
```

**Supported formats:** `.csv` `.tsv` `.parquet` `.json` `.ndjson` `.jsonl` `.feather` `.ipc` `.arrow` `.avro` `.orc` `.xlsx` `.xls` `.ods` `.xml` `.html` `.dta` `.sas7bdat` `.sav` `.pkl` `.hdf` `.h5` `.dlk`

### Big data — partial reads without OOM

```python
info = dd.read("big.parquet", header_only=True)   # schema + shape, zero rows read
df   = dd.read("big.parquet", head=100_000)        # first 100k rows
df   = dd.read("big.parquet", tail=50_000)         # last 50k rows
df   = dd.read("big.parquet", sample=500_000)      # random row groups
df   = dd.read("big.parquet", n_chunks=5, chunks=[2, 4])  # chunks 2 and 4 of 5

# Generator — process without ever loading the full file
for chunk in dd.read("big.parquet", n_chunks=10, iter_chunks=True):
    process(chunk)

# Streaming chunks with masking
for chunk in dd.stream("big.csv", salt=SALT, chunksize=50_000):
    save_to_db(chunk)
```

### Reading from a database

```python
banco = dd.db("postgresql://user:pass@host/db", salt=SALT)

df = dd.read(banco, "clientes")
df = dd.read(banco, "SELECT * FROM clientes WHERE uf = 'SP'")
df = dd.read(banco, "clientes", sample=10_000)
df = dd.read(banco, "clientes", head=5_000)
```

---

## Detecting PII

`dd.scan()` detects and classifies personal data columns:

```python
reports = dd.scan(df)
# returns Dict[str, ColumnReport]

for col, r in reports.items():
    print(f"{col}: type={r.pii_type.value}, risk={r.risk_level.value}, "
          f"strategy={r.mask_strategy.value}, match_ratio={r.match_ratio:.2f}")
```

Each `ColumnReport` has:
- `pii_type`: CPF, CNPJ, EMAIL, TELEFONE, NOME, DATA_NASCIMENTO, ENDERECO, CEP, SALARIO_RENDA, ...
- `risk_level`: "high", "medium", "low"
- `mask_strategy`: recommended masking approach
- `match_ratio`: fraction of values that matched the pattern

```python
# Scan a file directly
reports = dd.scan("clientes.parquet")
reports = dd.scan("clientes.dlk", key=KEY)

# Add company-specific patterns
reports = dd.scan(df, custom_patterns={
    "num_contrato": r"^CTR-[0-9]{8}$",
    "matricula":    r"^[0-9]{6}-[A-Z]$",
})

# Also detect sensitive data (financial, health, legal)
reports, sensitive_findings = dd.scan(df, sensitive=True)

# Quick diagnostic with privacy score
report = dd.profile(df)
print(report["pii_risk_summary"])   # "3🔴 2🟡 1🟢"
print(report["privacy_score"])      # 0–100
```

---

## Masking PII

`dd.mask()` applies deterministic HMAC-SHA256 pseudonymization. Same value + same SALT → same token every time. This preserves JOIN integrity across tables masked at different moments.

```python
df_safe = dd.mask(df, salt=SALT)

# Mask specific columns only
df_safe = dd.mask(df, salt=SALT, columns=["cpf", "email", "telefone"])

# Exclude columns from masking
df_safe = dd.mask(df, salt=SALT, exclude=["uf", "tipo_pessoa"])

# Mask by risk level
df_safe = dd.mask(df, salt=SALT, risk="high")    # suppress high-risk columns
df_safe = dd.mask(df, salt=SALT, risk="medium")  # hash medium-risk
df_safe = dd.mask(df, salt=SALT, risk="low")     # truncate low-risk

# LazyFrame stays lazy — not materialized until .collect()
lf_safe = dd.mask(df.lazy(), salt=SALT)
result  = lf_safe.collect()

# Show what changed
diff = dd.diff(df, df_safe)
print(diff["summary"])
```

**Default masking strategies by PII type:**

| Column type | Strategy | Example output |
|---|---|---|
| CPF, CNPJ | HASH | `"3f2a8b1c9d4e7f0a"` |
| Email | HASH | `"9e1d3c7f2a845b61"` |
| Phone | MASK_PHONE_DDD | `"(11) ****-1234"` |
| Name | HASH | hex token |
| Date of birth | GENERALIZE_DATE | `"1985-**"` |
| Income/salary | MOCK_NUM | synthetic value same distribution |

### Determinism and JOIN safety

```python
# Both tables masked with same SALT → same CPF token → JOIN works
df_clientes_safe = dd.mask(df_clientes, salt=SALT)
df_pedidos_safe  = dd.mask(df_pedidos,  salt=SALT)

result = dd.join(df_clientes_safe, df_pedidos_safe, on="cpf")

# Or let datalock handle it (applies same salt to both before joining)
result = dd.join(df_clientes, df_pedidos, on="cpf", salt=SALT)
```

### Masking in production (configure once)

```python
# Set once at application startup
dd.configure(default_salt=os.environ["DATALOCK_SALT"])

# Then call without salt= everywhere
df_safe = dd.mask(df)   # uses configured default
```

---

## Storing encrypted files (.dlk)

```python
# Encrypt a DataFrame
dd.store(df, "clientes.dlk", key=KEY)

# Mask then encrypt in one call
dd.store(df, "clientes.dlk", key=KEY, salt=SALT)

# With audit label and expiry (LGPD Art. 16)
dd.store(df, "clientes.dlk", key=KEY,
         label="exportacao-crm-jan2026",
         expires_at="2026-12-31")

# Multiple DataFrames in one file
dd.store({"clientes": df1, "pedidos": df2}, "base.dlk", key=KEY)

# With canary rows for leak tracing
dd.store(df, "clientes.dlk", key=KEY, canary=True, pipeline_id="crm-jan2026")

# Asymmetric — encrypt for a recipient without sharing your key
priv, pub = dd.generate_keypair("ec")    # or "rsa"
dd.store(df, "clientes.dlk", public_key=pub)
df = dd.read("clientes.dlk", private_key=priv)

# Overwrite existing file
dd.store(df, "clientes.dlk", key=KEY, overwrite=True)
```

`dd.store()` returns a dict: `{"output_path", "shape", "original_size_kb", "packed_size_kb", "compression_ratio", "elapsed_seconds"}`.

### Object-oriented interface (DlkFile)

```python
# Context manager
with dd.open("clientes.dlk", key=KEY) as f:
    df        = f.read()           # decrypt + return DataFrame
    df_raw    = f.read(raw=True)   # no masking
    info      = f.info()           # metadata without decrypting payload
    names     = f.frame_names()    # list frames in multi-frame file
    ok        = f.valid()          # True if HMAC intact
    f.write(df_new)                # overwrite
    f.add_frame("extra", df_ex)    # add frame to multi-frame

# Fluent
df = dd.open("clientes.dlk", key=KEY).read()
```

### Key rotation

```python
dd.rekey("clientes.dlk", old_key=OLD_KEY, new_key=NEW_KEY)
# Data is decrypted with old key and re-encrypted with new key. Never written to disk in plaintext.
```

---

## Inspecting without decrypting

```python
info = dd.inspect("clientes.dlk", key=KEY)
# Returns metadata without loading any row:
# {
#   "content_type": "masked_dataframe",
#   "shape": [150000, 12],
#   "schema": {"cpf": "object", "renda_mensal": "float64", ...},
#   "column_stats": {"renda_mensal": {"n_nulls": 0, "n_unique": 8743}},
#   "label": "exportacao-crm-jan2026",
#   "created_at": "2026-01-15T10:30:00+00:00",
#   "expires_at": "2026-12-31T00:00:00+00:00",
#   "encryption": "AES256GCM",
#   "masking_applied": true
# }
```

This is possible because the header is a separate encrypted block from the payload — inspecting it never loads or decrypts the actual data rows.

---

## Text masking

Detect and mask PII in free-form strings (logs, notes, emails):

```python
text = "Cliente CPF 111.444.777-35, email joao@empresa.com"

# Redact
dd.mask_text(text, strategy="redact")
# → "Cliente [CPF], [EMAIL]"

# Hash (deterministic — same input + same salt → same token)
dd.mask_text(text, salt=SALT, strategy="hash")
# → "Cliente 3f2a8b1c9d4e7f0a, 9e1d3c7f2a845b61"

# Partial
dd.mask_text(text, salt=SALT, strategy="partial")
# → "Cliente 111.***.***-35, j***@empresa.com"

# Semantic — generates valid fake data, no Faker required
dd.mask_text(text, salt=SALT, strategy="semantic")
# → "Cliente 478.622.984-97, roberto.santos@gmail.com"
# CPF checksum is mathematically valid. Deterministic: same input → same fake.

# Detect spans without masking
spans = dd.scan_text(text)
# → [{"type": "CPF", "value": "111.444.777-35", "start": 11, "end": 25}, ...]
```

---

## Data manipulation

All functions preserve input type (pd.DataFrame → pd.DataFrame, pl.DataFrame → pl.DataFrame):

```python
# Filter
dd.where(df, uf="SP")
dd.where(df, uf=["SP", "RJ", "MG"])
dd.where(df, renda_mensal=(5_000, 15_000))     # range (inclusive)
dd.where(df, renda_mensal=(">", 10_000))        # comparison operator

# Select / rename / drop
dd.select(df, ["cpf", "renda_mensal", "uf"])
dd.rename(df, {"cpf_hash": "cpf", "email_hash": "email"})
dd.drop(df, ["cpf_raw", "email_raw"])

# Add computed columns
dd.add_column(df,
    imposto     = dd.col("renda_mensal") * 0.275,
    salario_liq = dd.col("renda_mensal") - dd.col("renda_mensal") * 0.275,
    faixa = dd.when(dd.col("renda_mensal") > 10_000, "alta")
              .when(dd.col("renda_mensal") > 5_000,  "media")
              .otherwise("baixa"),   # always end with .otherwise()
)

# Sort
dd.sort(df, "renda_mensal", descending=True)
dd.sort(df, ["uf", "renda_mensal"])

# Group
dd.groupby(df, "uf", {
    "n":       ("*", "count"),
    "media":   ("renda_mensal", "mean"),
    "mediana": ("renda_mensal", "median"),
    "max":     ("renda_mensal", "max"),
})

# Limit / sample
dd.head(df, 100)
dd.tail(df, 50)
dd.sample(df, 1000, seed=42)
dd.top_n(df, "renda_mensal", n=10)

# Deduplicate
dd.unique(df, subset=["cpf"])

# Combine
dd.concat([df1, df2, df3])
dd.pivot(df, on="uf", values="renda_mensal", agg_fn="mean")
dd.melt(df, id_vars=["cpf"], value_vars=["jan", "fev", "mar"])
dd.explode(df, "tags")               # list column → multiple rows

# Time-series
dd.shift(df, 1)                      # previous period value (lag)
dd.lead(df, 1)                       # next period value
dd.lag(df, 3, columns="renda_mensal")

# Statistics
dd.describe(df)
dd.value_counts(df, "uf")
dd.corr(df, "renda_mensal", "score_credito")
dd.nunique(df)
dd.count_nulls(df)

# SQL via DuckDB (requires datalock[sql])
result = dd.sql(
    "SELECT uf, AVG(renda_mensal) AS media FROM df GROUP BY uf HAVING COUNT(*) > 100",
    df=df
)
```

---

## Full pipelines

### Fluent pipe

```python
result = (
    dd.pipe("clientes.parquet")
    .where(uf="SP", tipo_pessoa="PF")
    .add_column(
        imposto = dd.col("renda_mensal") * 0.275,
        faixa   = dd.when(dd.col("renda_mensal") > 10_000, "alta")
                    .when(dd.col("renda_mensal") > 5_000,  "media")
                    .otherwise("baixa"),
    )
    .mask(salt=SALT)
    .collect()             # materializes as pl.DataFrame
)
```

### dd.process() — read + validate + mask + store in one call

```python
result = dd.process(
    "clientes.parquet",
    salt=SALT,
    key=KEY,
    output="clientes_safe.dlk",
    where={"uf": "SP"},
    rules={"cpf": {"not_null": True}, "renda_mensal": {"min": 0}},
    label="pipeline-crm-jan2026",
    expires_at="2026-12-31",
)

result.df            # processed DataFrame
result.validation    # ValidationReport
result.output_path   # path written
result.elapsed       # seconds
```

---

## Data contracts

Declare schema, PII classification, masking strategy, and validation rules in one versioned object:

```python
contrato = dd.contract({
    "cpf": {
        "type": "str", "not_null": True, "unique": True,
        "pii": "CPF", "mask": "hash",
    },
    "renda_mensal": {
        "type": "float", "min": 0, "max": 500_000,
    },
    "uf": {
        "type": "str",
        "in": ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG",
               "MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR",
               "RS","SC","SE","SP","TO"],
    },
}, name="clientes", version="2.0")

# Validate + mask
result = contrato.apply(df, salt=SALT)
result.raise_if_failed()
df_safe = result.df

# Detect breaking changes between versions
diff = contrato.diff(contrato_v2)
print(diff.has_breaking_changes)  # True / False
print(diff.report())

# Export as JSON Schema (for documentation / DPO)
json_schema = contrato.to_json_schema()

# Persist and reload
contrato.save("schema/clientes_v2.contract.json")
contrato2 = dd.DataContract.load("schema/clientes_v2.contract.json")
```

---

## Validation

```python
# Rule-based validation
report = dd.validate(df, rules={
    "cpf":          {"not_null": True, "unique": True, "regex": r"^\d{11}$"},
    "renda_mensal": {"min": 0, "max": 500_000},
    "uf":           {"in": LISTA_UFS},
})
report.print_report()
report.raise_if_failed()   # raises ValidationError if any rule failed

# Expectation-style
dd.expect(df, "cpf").to_be_unique()
dd.expect(df, "renda_mensal").to_be_between(0, 500_000)
dd.expect(df, "uf").to_be_in(LISTA_UFS)

# Schema validation
dd.validate_schema(df,
    required_columns=["cpf", "renda_mensal", "uf"],
    min_rows=1,
).raise_if_failed()

# Persist rules as versioned JSON
dd.save_rules(rules, "regras/clientes_v2.json")
rules = dd.load_rules("regras/clientes_v2.json")
```

---

## Database integration

```python
# Create connection
banco = dd.db("postgresql://user:pass@host:5432/db", salt=SALT)

# Explore
print(banco.tables())
df_sample = banco.sample_table("clientes")

# Read with automatic masking
df = dd.read(banco, "clientes")
df = dd.read(banco, "SELECT * FROM clientes WHERE uf = 'SP'")
df = dd.read(banco, "clientes", sample=10_000)

# Write
banco.write(df_safe, "clientes_masked")
dd.write(df_safe, banco, "clientes_masked")
banco.create_table(df, "clientes", if_exists="replace")
banco.upsert(df_new, "clientes", on="cpf")

# Context manager
with dd.db("postgresql://...", salt=SALT) as banco:
    df = dd.read(banco, "clientes")

# Supported URIs
# postgresql://user:pass@host/db
# mysql+pymysql://user:pass@host/db
# sqlite:///arquivo.db
# mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17
# duckdb:///:memory:
```

---

## Canary data

Inject invisible trap rows into your data. If a breach happens, you can identify exactly which export was compromised.

```python
# Store with canary rows (user never sees them)
dd.store(df, "clientes.dlk", key=KEY, canary=True, pipeline_id="crm-jan2026")

# Reading strips canary rows automatically — shape is identical to original
df_back = dd.read("clientes.dlk", key=KEY)
assert df_back.shape == df.shape

# When a breach token appears in a dump
resultado = dd.canary_check("canary.1ba472d8e3f9@datalock.internal")
# → {
#     "pipeline_id":   "crm-jan2026",
#     "filepath":      "clientes.dlk",
#     "injected_at":   "2026-01-15T14:32:00Z",
#     "n_canary_rows": 3,
#     "level":         1
#   }
```

**How it works:**
- Before encrypting, datalock injects N synthetic rows with unique HMAC fingerprints
- Rows are distributed across 5 strata — any 50% slice of the file still contains at least one canary
- PII columns get recognizable sentinel values (`canary.{fp}@datalock.internal`) identifiable in breach dumps
- On read, rows are silently removed by filtering the hidden `__canary_sig__` column (None in real rows)
- A local manifest (`~/.datalock/canary_manifest.jsonl`) maps fingerprints → pipeline + file

**Level 2 (insider threat):** each `dd.read()` also injects fresh canary rows into the returned DataFrame. If a user exports that DataFrame as CSV and leaks it, the level-2 fingerprints in the CSV identify that specific read session.

**Production hardening — configure a secret canary salt:**

```python
# With the default public salt, an adversary who knows the pipeline_id
# can pre-compute fingerprints and remove canary rows before leaking.
# A secret salt makes fingerprints unpredictable.
dd.configure(canary_salt=os.environ["DATALOCK_CANARY_SALT"])
# Or via .env:
dd.configure(load_dotenv=True)   # reads DATALOCK_CANARY_SALT automatically
```

---

## Privacy metrics

```python
from datalock import check

# K-anonymity
report = check.kanon(df, quasi_identifiers=["uf", "faixa_etaria"])
print(report.k)              # current k value
print(report.at_risk_count)  # records with k < threshold
report.print_report()

# T-closeness
report = check.tcloseness(df,
    quasi_identifiers=["uf", "faixa_etaria"],
    sensitive_column="renda_mensal",
)
print(report.t_score)

# Re-identification risk
report = check.risk(df, quasi_identifiers=["uf", "faixa_etaria", "data_nasc"])
print(report.max_risk)   # 0–1
print(report.mean_risk)

# Utility — compare original vs masked
report = check.utility(df_original, df_masked)
print(report.overall_score)

# Synthetic data fidelity
report = check.fidelity(df_real, df_synth, tstr_target="inadimplente")
print(report.overall_score)
report.print_report()

# Differential privacy
dp = check.dp(epsilon=1.0)
noisy_mean = dp.add_laplace_noise(df["renda_mensal"].mean(), sensitivity=1000)
```

---

## Synthetic data

```python
# Train generative model (requires datalock[synthetic])
model = dd.train(df, n=1000)

# Generate N rows with same distributions
df_synth = dd.clone(df, n=5000)

# Synthetic + masked for development environments
df_dev = dd.sandbox(df, n=1000, salt=SALT)

# Store and reload the model
dd.store(model, "modelo.dlk", key=KEY)
model2 = dd.read("modelo.dlk", key=KEY)
df_synth2 = model2.generate(1000)

# Built-in generators (no Faker required)
gen = dd.SyntheticGenerator(seed=42)
gen.cpf()    # "478.622.984-97" — mathematically valid checksum
gen.cnpj()   # "91.202.089/8546-89"
gen.email()  # "roberto.santos@gmail.com"
gen.nome()   # "Maria Aparecida"
```

---

## Directory inventory

```python
inventory = dd.scan_directory("./dados/", recursive=True)

# Summary
print(inventory.summary())

# Export
inventory.to_html("inventario_pii.html")
inventory.to_json("inventario_pii.json")

# Iterate
for path, fi in inventory.items():
    if fi.max_risk == "high":
        print(f"HIGH RISK: {path}")
        for col, r in fi.pii_columns.items():
            print(f"  {col}: {r.pii_type.value} ({r.risk_level.value})")
```

---

## Configuration

```python
# Load from .env file (requires pip install python-dotenv)
# Reads DATALOCK_SALT, DATALOCK_KEY, DATALOCK_CANARY_SALT automatically
dd.configure(load_dotenv=True)

# Explicit configuration
dd.configure(
    default_salt=os.environ["DATALOCK_SALT"],
    canary_salt=os.environ["DATALOCK_CANARY_SALT"],
)

# Audit trail — log every dd.mask(), dd.store(), dd.scan() to a file
dd.configure(audit_path="./audit/")

# Audit webhook — POST to Slack / SIEM / Datadog (non-blocking, best-effort)
dd.configure(audit_webhook="https://hooks.slack.com/services/...")

# LGPD compliance report
reports = dd.scan(df)
report  = dd.compliance_report(df, reports,
    dataset_name="Base Clientes Q1 2026",
    organization="Empresa S.A.",
)
report.to_html("lgpd_relatorio.html")
report.to_pdf("lgpd_relatorio.pdf")   # requires pip install weasyprint
report.to_text()                       # always available
```

---

## The `.dlk` format

Binary container with self-contained security guarantees. A `.dlk` file carries its own encryption, authentication, and compliance metadata regardless of where it is stored.

**Format v2 structure (single-frame):**
```
[5  bytes]  MAGIC          = b"DLOCK"
[1  byte ]  VERSION        = 0x02
[1  byte ]  CIPHER         = 0x01 (AES-256-GCM) | 0x02 (ChaCha20-Poly1305)
[32 bytes]  SALT_KDF       — random per file (os.urandom), stored in plaintext
[12 bytes]  NONCE_HEADER   — nonce for header block encryption
[4  bytes]  HEADER_CT_LEN  — length of encrypted header block
[N+16 bytes] HEADER_CT+TAG — JSON metadata encrypted with HEK + GCM auth tag
[12 bytes]  NONCE_PAYLOAD  — nonce for payload encryption
[M+16 bytes] PAYLOAD_CT+TAG — Arrow IPC data encrypted with DEK + GCM auth tag
[32 bytes]  FILE_HMAC      — HMAC-SHA256(MAK, all bytes above)
```

**Key hierarchy (HKDF-SHA256, RFC 5869):**

The master key is never used directly. Three independent keys are derived per file:
- **DEK** (`info=b"datalock-dek-v1"`) — encrypts the data payload
- **HEK** (`info=b"datalock-hek-v1"`) — encrypts the JSON header
- **MAK** (`info=b"datalock-mak-v1"`) — key for the file-level HMAC

Different `info` fields guarantee DEK ≠ HEK ≠ MAK even from the same master key. Compromising one does not expose the others. Since `SALT_KDF` is unique per file, each file has a completely independent set of derived keys even if the same master key is used everywhere.

**Two-level authentication:**
- GCM auth tag (per block): any change to the encrypted header or payload is detected before a single byte of plaintext is returned
- FILE_HMAC (whole file): covers all plaintext fields (magic bytes, version, cipher byte, nonces) that the GCM tags don't cover — prevents layout-level tampering

**Header separate from payload:** the JSON header (schema, shape, creation timestamp, pipeline, expiry) is encrypted with HEK independently of the payload. `dd.inspect()` decrypts only the header — the payload is never touched. This allows compliance auditors to verify metadata without accessing the actual data.

**Versions:**
- v2 (default): single DataFrame, header encrypted
- v3: multiple DataFrames in one file (payload is an in-memory ZIP)
- v4: no encryption (for already-anonymized data in dev environments)

**Payload serialization:** Apache Arrow IPC (not Parquet) — 3–5× faster for in-memory round-trips because Arrow IPC maps directly to Arrow's in-memory layout without Parquet's analytical overhead (row groups, column statistics, predicate pushdown structures). Compression (zstd by default) is applied before encryption. Backward-compatible with files using Parquet serialization via magic marker detection.

---

## datalock vs alternatives

### datalock vs pandas + manual encryption

| Need | Manual approach | datalock |
|---|---|---|
| Save DataFrame securely | `to_parquet()` + GPG | `dd.store(df, "f.dlk", key=KEY)` |
| Mask CPF before analysis | regex + hashlib | `dd.mask(df, salt=SALT)` |
| Know which export was breached | not possible | `dd.canary_check(token)` |
| Audit metadata without seeing data | not possible | `dd.inspect("f.dlk", key=KEY)` |
| Expire data after LGPD retention period | external cleanup job | `expires_at="2026-12-31"` |
| JOIN masked tables from different times | breaks (different hashes) | deterministic HMAC — same salt → same token |

### datalock vs GPG / age

GPG and age encrypt arbitrary files with integrity. Use datalock instead when:
- The content is a DataFrame (datalock understands the structure)
- You need to read schema or row count without decrypting (`dd.inspect()`)
- You need file-level expiry, canary data, or LGPD metadata in the file itself

### datalock vs Apache Parquet encryption

Parquet encryption provides per-column confidentiality but does **not** authenticate the file as a whole — a tampered encrypted Parquet file can return silently corrupted data without the reader detecting it. datalock verifies HMAC before any decryption, and then verifies GCM auth tags before returning any plaintext byte.

### datalock vs S3 SSE / disk encryption

Infrastructure encryption protects data at rest in a specific environment. Once the file is copied outside that environment (sent by email, shared with an external partner, put on a USB drive), the protection is gone. A `.dlk` file carries its protection with it.

---

## Common errors and fixes

### `ValueError: key= e salt= não devem ser iguais`
```python
# WRONG: same value for both
dd.store(df, "f.dlk", key=SECRET, salt=SECRET)

# FIX: generate two different values
SALT = dd.generate_salt()   # for HMAC masking
KEY  = dd.generate_salt()   # for AES encryption — different value
```

### `RuntimeError: Falha na decifração (auth_tag inválida)`
The file was tampered with, is corrupted, or the wrong key was used.
```python
# Verify integrity without full decryption
ok, info = dd.SecureFile.verify("clientes.dlk", key=KEY)
print(ok, info)
```

### `ExpiredFileError: Arquivo expirado em YYYY-MM-DD`
The file intentionally cannot be read after its expiry date (LGPD Art. 16). This is by design. The file still exists on disk; physical deletion is the responsibility of your pipeline.

### `IdempotencyError: Coluna 'cpf' parece já mascarada`
```python
# The column looks already masked. Don't mask twice.
# If you really need to re-mask:
df_safe = dd.mask(df_already_masked, salt=SALT, strict=False)
```

### `UserWarning: dd.mask(salt=None) salt aleatório gerado`
```python
# Random salt means hashes change every run — JOINs will break.
# Fix: configure once at startup
dd.configure(default_salt=os.environ["DATALOCK_SALT"])
```

### `ImportError: No module named 'duckdb'`
```bash
pip install "datalock[sql]"
```

### `ImportError: No module named 'openpyxl'`
```bash
pip install "datalock[excel]"
```

### `TypeError: open("file.dlk")` returns binary garbage
```python
# WRONG: .dlk is not a text or raw binary format
with open("clientes.dlk", "rb") as f:
    data = f.read()   # meaningless binary

# CORRECT
df   = dd.read("clientes.dlk", key=KEY)
info = dd.inspect("clientes.dlk", key=KEY)
```

---

## Backward compatibility

```python
# All of these still work after the rename from logus-lgpd:
import logus as lg
lg.mask(df, salt=SALT)          # identical to dd.mask()
lg.read("f.lgs", key=KEY)       # .lgs files from logus-lgpd still readable
dd.read("arquivo.lgs", key=KEY) # also works
```

---

## Function reference

| Function | Input → Output | Description |
|---|---|---|
| `dd.read(path)` | path → `pl.DataFrame` | Read any format |
| `dd.read(path, key=KEY)` | `.dlk` → `pl.DataFrame` | Decrypt and read |
| `dd.store(df, path, key=KEY)` | DataFrame → `.dlk` | Encrypt and save |
| `dd.mask(df, salt=SALT)` | DataFrame → same type | Mask PII columns |
| `dd.scan(df)` | DataFrame → `Dict[str, ColumnReport]` | Detect PII |
| `dd.inspect(path, key=KEY)` | `.dlk` → `dict` | Read metadata only |
| `dd.open(path, key=KEY)` | path → `DlkFile` | OO interface |
| `dd.rekey(path, old_key, new_key)` | `.dlk` → `.dlk` | Rotate encryption key |
| `dd.canary_check(token)` | string → `dict\|None` | Look up breach token |
| `dd.mask_text(text, strategy=)` | string → string | Mask text free-form |
| `dd.scan_text(text)` | string → `list` | Detect PII in text |
| `dd.process(path, ...)` | path → `ProcessResult` | Full pipeline |
| `dd.profile(df)` | DataFrame → `dict` | Quick LGPD diagnostic |
| `dd.diff(df, df_safe)` | 2 DataFrames → `dict` | Show what masking changed |
| `dd.join(df1, df2, on, salt=)` | 2 DataFrames → `pd.DataFrame` | Safe JOIN on masked keys |
| `dd.contract({...})` | spec dict → `DataContract` | Declare schema + PII rules |
| `dd.validate(df, rules)` | DataFrame → `ValidationReport` | Rule-based validation |
| `dd.clone(df, n=)` | DataFrame → DataFrame | Generate synthetic data |
| `dd.sandbox(df, n=, salt=)` | DataFrame → DataFrame | Synthetic + masked |
| `dd.db(uri, salt=)` | URI → `DatabaseConnection` | DB connection with masking |
| `dd.sql(query, df=df)` | SQL + frames → `pl.DataFrame` | SQL over DataFrames |
| `dd.scan_directory(path)` | directory → `DirectoryInventory` | PII inventory |
| `dd.compliance_report(df, reports)` | DataFrame + reports → report | LGPD report |
| `dd.configure(...)` | — | Set global defaults |
| `dd.generate_salt()` | — → `str` | Generate a secure salt/key |
| `check.kanon(df, qi=)` | DataFrame → `PrivacyMetricsReport` | K-anonymity |
| `check.risk(df, qi=)` | DataFrame → `ReidentificationRiskReport` | Re-id risk |
| `check.utility(orig, masked)` | 2 DataFrames → `UtilityReport` | Masking utility |
| `check.fidelity(real, synth)` | 2 DataFrames → `FidelityReport` | Synthetic fidelity |

---

## License

AGPL-3.0 — see LICENSE.