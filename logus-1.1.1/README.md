# logus-lgpd

**logus-lgpd** is a Python library for privacy-by-design with tabular data.
It provides LGPD-compliant PII detection and masking, AES-256-GCM encrypted
file storage (`.lgs` format), and a SQL-like DSL for data manipulation built
on top of Polars.

```
pip install logus-lgpd
```

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]
KEY  = os.environ["LOGUS_KEY"]

df       = lg.read("clientes.csv")            # any format → pl.DataFrame
df_safe  = lg.mask(df, salt=SALT)             # detect + mask PII (LGPD)
lg.store(df_safe, "clientes.lgs", key=KEY)    # AES-256-GCM encrypted
df_back  = lg.read("clientes.lgs", key=KEY)   # decrypt and read back
```

---

## What logus-lgpd does

| Capability | Function |
|---|---|
| Read any tabular format | `lg.read()` |
| Detect PII automatically | `lg.scan()` |
| Mask PII (HMAC-SHA256) | `lg.mask()` |
| Save with AES-256-GCM | `lg.store()` |
| Expressive data manipulation | `lg.where()`, `lg.groupby()`, `lg.add_column()` |
| Full pipeline in one call | `lg.process()` |
| Data quality validation | `lg.validate()` |
| Database with masking | `lg.db()` |
| Privacy metrics | `lg.check.kanon()`, `lg.check.risk()` |

---

## Installation

```bash
# Core (Polars + pandas + pyarrow + cryptography)
pip install logus-lgpd

# With SQL via DuckDB (lg.sql, lg.db)
pip install "logus-lgpd[sql]"

# With Excel support (.xlsx, .ods)
pip install "logus-lgpd[excel]"

# With synthetic data generation (lg.clone, lg.sandbox)
pip install "logus-lgpd[synthetic]"

# Everything
pip install "logus-lgpd[full]"
```

**Requires:** Python ≥ 3.10, Polars ≥ 1.0, pandas ≥ 2.0, pyarrow ≥ 14.0

---

## Quick Start

### 1. Read any file format

```python
import logus as lg

# All return pl.DataFrame; auto-detects encoding
df = lg.read("clientes.csv")
df = lg.read("clientes.parquet")
df = lg.read("clientes.xlsx")            # pip install "logus-lgpd[excel]"
df = lg.read("clientes.lgs", key=KEY)    # decrypt .lgs

# CSV with non-default separator and encoding
df = lg.read("clientes.csv", sep=";", encoding="latin-1")

# Partial read for large files (no OOM)
df   = lg.read("big.parquet", head=100_000)
df   = lg.read("big.parquet", sample=500_000)    # random row groups
info = lg.read("big.parquet", header_only=True)  # schema + shape, zero data read
df   = lg.read("big.parquet", n_chunks=5, chunks=[2, 4])
for chunk in lg.read("big.parquet", n_chunks=10, iter_chunks=True):
    process(chunk)                                # never loads full file
```

### 2. Detect and mask PII (LGPD)

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]   # HMAC key — store in env, never hardcode

# Detect PII columns automatically
reports = lg.scan(df)
for col, r in reports.items():
    print(f"{col}: {r.pii_type.value}  risk={r.risk_level.value}  → {r.mask_strategy.value}")
# cpf:      cpf            risk=high   → hash
# email:    email          risk=high   → hash
# nome:     nome           risk=medium → redact
# cep:      cep            risk=low    → truncate (01310-XXX)
# data_nasc:data_nascimento risk=medium → generalize_date (1985-03-XX)

# Mask PII — preserves input type (pd→pd, pl→pl)
df_safe = lg.mask(df, salt=SALT)
df_safe = lg.mask(df, salt=SALT, columns=["cpf", "email"])   # only these
df_safe = lg.mask(df, salt=SALT, exclude=["uf"])             # all except uf
df_safe = lg.mask(df, salt=SALT, risk="high")                # only high-risk PII

# Deterministic: same value + same salt → same token (essential for JOINs)
# CPF "111.444.777-35" → always "3f2a8b1c9d4e7f0a" with the same SALT
```

### 3. Save and read encrypted (.lgs)

```python
import logus as lg
import os

KEY  = os.environ["LOGUS_KEY"]    # AES-256 key — different from SALT
SALT = os.environ["LOGUS_SALT"]

# Save
lg.store(df, "clientes.lgs", key=KEY)                    # encrypt, data as-is
lg.store(df, "clientes.lgs", key=KEY, salt=SALT)         # mask + encrypt
lg.store({"clients": df1, "orders": df2}, "base.lgs", key=KEY)  # multi-frame

# Read metadata WITHOUT decrypting the payload
info = lg.inspect("clientes.lgs", key=KEY)
# {"version":"2.1","shape":[150000,12],"columns":["cpf","nome",...],
#  "column_stats":{"cpf":{"n_nulls":0,"n_unique":150000},...},
#  "content_type":"masked_dataframe","encryption":"AES-256-GCM"}

# Read
df      = lg.read("clientes.lgs", key=KEY)
frames  = lg.read("base.lgs", key=KEY)           # dict[str, pd.DataFrame]
df_cli  = lg.read("base.lgs", key=KEY, frame="clients")

# OO interface
with lg.open("clientes.lgs", key=KEY) as f:
    df     = f.read()
    info   = f.info()
    frames = f.frames()
    f.write(df_updated)
    f.add_frame("novos", df_new)

# Rotate encryption key (without exposing data as file)
lg.rekey("clientes.lgs", old_key=OLD_KEY, new_key=NEW_KEY)
```

### 4. Manipulate data (SQL-like DSL)

All functions work with `pd.DataFrame`, `pl.DataFrame`, and `pl.LazyFrame`.
They preserve the input type — Polars in, Polars out.

```python
import logus as lg

# WHERE
df = lg.where(df, uf="SP")
df = lg.where(df, uf=["SP", "RJ", "MG"])             # isin
df = lg.where(df, renda_mensal=(">", 5_000))          # operator
df = lg.where(df, renda_mensal=(5_000, 15_000))       # between
df = lg.where(df, nome=("contains", "Silva"))         # string contains
df = lg.where(df, lg.col("renda") > lg.col("media"))  # expression
df = lg.where(df, uf="SP", tipo_pessoa="PF")          # AND (multiple kwargs)

# SELECT / DROP / RENAME
df = lg.select(df, ["cpf", "renda", "uf"])
df = lg.drop(df, "coluna_inutil")
df = lg.rename(df, {"cpf": "documento"})

# ORDER BY
df = lg.sort(df, "renda_mensal", desc=True)
df = lg.sort(df, ["uf", "renda"], ascending=[True, False])

# GROUP BY with HAVING, ORDER BY, LIMIT
resultado = lg.groupby(df, "uf", {
    "n":      ("*",             "count"),
    "media":  ("renda_mensal",  "mean"),
    "total":  ("renda_mensal",  "sum"),
    "unicos": ("cpf",           "n_unique"),
}, having={"n": (">", 100)}, sort="media", desc=True, limit=10)

# ADD COLUMN with expressions, CASE WHEN, window functions
df = lg.add_column(df,
    imposto       = lg.col("renda_mensal") * 0.275,
    faixa         = lg.when(lg.col("renda_mensal") > 10_000, "alta")
                      .when(lg.col("renda_mensal") > 5_000,  "media")
                      .otherwise("baixa"),
    rank_uf       = lg.col("renda_mensal").rank("dense", descending=True).over("uf"),
    media_uf      = lg.col("renda_mensal").mean().over("uf"),
    nome_lower    = lg.col("nome").str.to_lowercase(),
    ano_nasc      = lg.col("data_nasc").str.to_date("%Y-%m-%d").dt.year(),
)

# DISTINCT
df = lg.unique(df, "cpf")                             # one row per CPF
df = lg.unique(df, ["uf", "tipo"], keep="first")

# PIVOT / MELT
pv = lg.pivot(df, index="uf", columns="tipo", values="renda", aggfunc="mean")
ml = lg.melt(df, id_cols=["uf"], value_cols=["renda_jan", "renda_fev"])

# TOP N per group
top3 = lg.top_n(df, 3, "renda_mensal", group_by="uf")

# FILL NULL / CAST
df = lg.fill_null(df, {"renda": 0, "uf": "DESCONHECIDO"})
df = lg.cast(df, {"renda": "float32", "inadimplente": "bool"})

# CONCAT
df_all = lg.concat([df_jan, df_fev, df_mar])
```

### 5. lg.col() — full Polars expression API

`lg.col` is literally `polars.col`. All 200+ Polars methods are available:

```python
lg.col("renda").round(2)
lg.col("renda").log(base=10)
lg.col("nome").str.to_lowercase()
lg.col("email").str.split("@").list.last()
lg.col("cpf").str.replace_all(r"[\.\-]", "")
lg.col("data").dt.year()
lg.col("data").dt.truncate("1mo")
lg.col("renda").mean().over("uf")               # window function
lg.col("renda").rank("dense", descending=True).over("uf")
lg.col("renda").rolling_mean(window_size=3)
lg.col("renda").cum_sum()
```

### 6. Pipeline fluente

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]
KEY  = os.environ["LOGUS_KEY"]

result = (
    lg.pipe("clientes.parquet")
    .where(uf="SP", tipo_pessoa="PF")
    .add_column(
        imposto  = lg.col("renda_mensal") * 0.275,
        faixa    = lg.when(lg.col("renda_mensal") > 10_000, "alta")
                     .when(lg.col("renda_mensal") > 5_000,  "media")
                     .otherwise("baixa"),
    )
    .mask(salt=SALT)
    .groupby("faixa", {"n": ("*", "count"), "media": ("renda_mensal", "mean")})
    .sort("media", desc=True)
    .collect()                              # → pl.DataFrame
)
```

### 7. Full pipeline in one call

```python
import logus as lg
import os

result = lg.process(
    "clientes.csv",
    salt=os.environ["LOGUS_SALT"],
    key=os.environ["LOGUS_KEY"],
    output="clientes_safe.lgs",
    overwrite=True,
    where={"uf": ["SP", "RJ", "MG"]},
    rules={
        "cpf":          {"not_null": True},
        "renda_mensal": {"min": 0, "max": 500_000},
        "email":        {"contains": "@"},
    },
    verbose=True,
)

print(f"Rows:          {result.n_rows:,}")
print(f"PII columns:   {result.pii_columns}")
print(f"Privacy score: {result.privacy_score}/100")
print(f"Validation:    {result.validation.passed}")
print(f"Saved to:      {result.output_path}")
result.print_summary()
```

### 8. SQL via DuckDB

```python
import logus as lg

# pip install "logus-lgpd[sql]"

# SQL on DataFrames
result = lg.sql(
    "SELECT uf, AVG(renda_mensal) AS media, COUNT(*) AS n "
    "FROM df GROUP BY uf HAVING n > 100 ORDER BY media DESC",
    df=df,
)

# JOIN two DataFrames via SQL
result = lg.sql(
    "SELECT c.uf, SUM(p.valor) AS total "
    "FROM clientes c JOIN pedidos p ON c.cpf = p.cpf "
    "GROUP BY c.uf",
    clientes=df_clientes,
    pedidos=df_pedidos,
)

# SQL on Parquet files (DuckDB reads natively)
result = lg.sql("SELECT * FROM read_parquet('big.parquet') WHERE uf='SP' LIMIT 1000")
```

### 9. Database connection

```python
import logus as lg
import os

# pip install "logus-lgpd[sql]"

banco = lg.db(
    "postgresql://user:pass@host:5432/db",
    salt=os.environ["LOGUS_SALT"],
)

# Read with masking
df = lg.read(banco, "clientes")
df = lg.read(banco, "clientes", sample=10_000)           # TABLESAMPLE BERNOULLI
df = lg.read(banco, "SELECT * FROM clientes WHERE uf='SP'")

# Write masked data back
banco.write(df_safe, "clientes_masked", if_exists="replace")
lg.write(df_safe, banco, "clientes_masked")              # alternative syntax

# Explore
print(banco.tables())
print(banco.schema("clientes"))
sample = banco.sample_table("clientes", n=5)

# Context manager closes pool automatically
with lg.db("postgresql://...", salt=SALT) as banco:
    df = banco.read("clientes")

# Dialects: postgresql, mysql, sqlite, sqlserver, bigquery, snowflake, duckdb
```

### 10. Data quality validation

```python
import logus as lg

result = lg.validate(df, {
    "cpf":          {"not_null": True, "unique": True},
    "email":        {"not_null": True, "contains": "@"},
    "renda_mensal": {"min": 0, "max": 500_000, "not_null": True},
    "uf":           {"in": ["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE"]},
    "cep":          {"matches": r"^\d{5}-\d{3}$"},
})

result.print_report()     # formatted table
result.passed             # True / False
result.score              # 0.93 (proportion of rules passed)
result.raise_if_failed()  # raises ValueError with details if any rule failed

# Fluent interface
lg.expect(df, "renda_mensal").not_null().between(0, 500_000).validate()
lg.expect(df, "email").contains("@").min_length(5).validate()
```

### 11. Streaming for large files

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]

# Yields pl.DataFrame chunks — never loads full file into memory
for chunk in lg.stream("grande.csv", salt=SALT, chunksize=50_000):
    save_to_database(chunk)

# With progress callback
def progress(chunk_n, rows_done, total_estimate):
    print(f"Chunk {chunk_n}: {rows_done:,} / ~{total_estimate:,} rows")

for chunk in lg.stream("grande.parquet", salt=SALT, on_progress=progress):
    process(chunk)
```

### 12. Privacy metrics

```python
import logus as lg

# k-anonymity (ANPD recommends k ≥ 5)
report = lg.check.kanon(df, quasi_identifiers=["uf", "faixa_etaria", "escolaridade"])
print(f"k={report.k_anonymity.k_value}  compliant={report.compliant_anpd}")

# Re-identification risk score (0=safe, 1=high risk)
report = lg.check.risk(df_safe, quasi_identifiers=["uf", "faixa_etaria"])
print(f"risk={report.risk_score:.2f}  level={report.risk_level}")

# Utility preservation after masking
report = lg.check.utility(df_original, df_masked)
print(f"utility={report.overall_score:.0%}")

# Differential privacy
dp = lg.check.dp(epsilon=1.0)
noisy_mean = dp.laplace(df["renda"].mean(), sensitivity=df["renda"].max())

# Privacy score (composite 0–100)
profile = lg.profile(df)
print(f"Privacy score: {profile['privacy_score']['total']}/100  [{profile['privacy_score']['grade']}]")
```

### 13. SQL transpiler

```python
import logus as lg

reports = lg.scan(df)

# Transform SELECT to mask PII inline (data never leaves the database)
safe_sql = lg.mask_sql(
    "SELECT cpf, nome, email, renda_mensal, uf FROM clientes WHERE uf = 'SP'",
    reports=reports,
    dialect="postgresql",
    salt=os.environ["LOGUS_SALT"],
)
# SELECT
#   encode(hmac(cpf::text, 'salt', 'sha256'), 'hex') AS cpf,
#   'REDACTED' AS nome,
#   encode(hmac(email::text, 'salt', 'sha256'), 'hex') AS email,
#   renda_mensal,
#   uf
# FROM clientes WHERE uf = 'SP'

# Generate CREATE VIEW
view_sql = lg.generate_view(df, "clientes", reports=reports, dialect="postgresql")
```

### 14. Data lineage

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]

# Automatic with lg.process()
result = lg.process(df, salt=SALT, track_lineage=True)
print(result.lineage.summary())

# Manual
tracker = lg.lineage.start("pipeline_crm")
tracker.origin(df, "clientes.parquet", format="parquet")
df_filtered = lg.where(df, uf="SP")
tracker.transform(df_filtered, operation="filter", detail="uf='SP'")
df_safe = lg.mask(df_filtered, salt=SALT)
tracker.mask(df_safe, columns=list(reports.keys()), salt=SALT)
tracker.export("output.lgs", format="lgs", encrypted=True)

tracker.to_json("lineage.json")           # save lineage record
tracker.to_openlineage("openlineage.json") # OpenLineage-compatible format

# Context manager (prints summary automatically)
with lg.lineage.session("pipeline") as lin:
    lin.origin(df, "source.csv")
    lin.mask(df_safe, columns=["cpf", "email"], salt=SALT)
```

---

## Use Cases

### Masking CPFs in production logs

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]
KEY  = os.environ["LOGUS_KEY"]

# One-time: mask production data
df_raw  = lg.read("producao.parquet")
df_safe = lg.mask(df_raw, salt=SALT, verbose=True)
lg.store(df_safe, "producao_safe.lgs", key=KEY)

# Analytics team works with masked data
df = lg.read("producao_safe.lgs", key=KEY)
# CPFs are now: "3f2a8b1c9d4e7f0a" — tokens, not real values
```

### JOIN between masked tables (deterministic tokens)

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]

# Same CPF → same token in both tables → JOIN works
df_c_safe = lg.mask(df_clientes, salt=SALT)
df_p_safe = lg.mask(df_pedidos,  salt=SALT)

result = lg.join(df_c_safe, df_p_safe, on="cpf")
# OR let logus apply the same SALT to both automatically:
result = lg.join(df_clientes, df_pedidos, on="cpf", salt=SALT)
```

### Processing a 10GB CSV without OOM

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]

# Option 1: stream in chunks
total_rows = 0
for chunk in lg.stream("big.csv", salt=SALT, chunksize=100_000):
    save_to_db(chunk)
    total_rows += len(chunk)
    print(f"{total_rows:,} rows processed")

# Option 2: partial read for exploration
schema  = lg.read("big.csv", header_only=True)  # instant, zero data
sample  = lg.read("big.csv", head=10_000)        # first 10k rows
# For repeated access, convert to Parquet once:
df_full = lg.read("big.csv")
lg.write(df_full, "big.parquet")                 # subsequent reads 6× faster
```

### LGPD compliance check before sharing data

```python
import logus as lg

df = lg.read("clientes.parquet")

# Full diagnostic
profile = lg.profile(df)
score   = profile["privacy_score"]
print(f"Privacy Score: {score['total']}/100 [{score['grade']}]")
print(f"PII columns:   {profile['pii_columns']}")
print(f"Recommendation: {score['recommendation']}")

# If score is acceptable, share safely
if score["total"] >= 75:
    SALT = os.environ["LOGUS_SALT"]
    KEY  = os.environ["LOGUS_KEY"]
    df_safe = lg.mask(df, salt=SALT)
    lg.store(df_safe, "clientes_para_parceiro.lgs", key=KEY)
```

### Database masking without data leaving the server

```python
import logus as lg
import os

# pip install "logus-lgpd[sql]"

# Option 1: Create masked view in the database
banco = lg.db("postgresql://user:pass@host/db", salt=os.environ["LOGUS_SALT"])
result = banco.create_masked_view("clientes")   # creates clientes_masked view
# Now devs use: SELECT * FROM clientes_masked

# Option 2: Generate SQL to run yourself
reports  = lg.scan(lg.read(banco, "clientes", head=500))   # sample for detection
view_sql = lg.link.sql(None, reports, table="clientes", dialect="postgresql")
print(view_sql)    # → CREATE OR REPLACE VIEW clientes_masked AS ...
```

---

## API Reference

### I/O Functions

| Function | Signature | Returns |
|---|---|---|
| `lg.read` | `(source, *, key, salt, head, sample, n_chunks, chunks, iter_chunks, header_only, columns, ...)` | `pl.DataFrame \| dict \| Generator` |
| `lg.write` | `(df, path_or_conn, table=None, **kw)` | `None` |
| `lg.store` | `(source, path, *, key, salt, anonymize, compress, overwrite, metadata)` | `dict` |
| `lg.stream` | `(source, *, salt, chunksize, on_progress)` | `Generator[pl.DataFrame]` |
| `lg.open` | `(path, *, key, salt, compress)` | `LGSFile` |
| `lg.inspect` | `(path, *, key)` | `dict` |
| `lg.rekey` | `(path, *, old_key, new_key, output_path)` | `dict` |
| `lg.db` | `(uri, *, salt, dialect, pool_size)` | `DatabaseConnection` |

### Privacy Functions

| Function | Signature | Returns |
|---|---|---|
| `lg.scan` | `(source, *, key, sample_size, threshold)` | `Dict[str, ColumnReport]` |
| `lg.mask` | `(df, *, salt, columns, exclude, risk, strict, verbose)` | same type as input |
| `lg.diff` | `(original, masked, *, sample_size)` | `dict` |
| `lg.profile` | `(source, *, key, sample_size)` | `dict` |
| `lg.join` | `(left, right, on, *, salt, how)` | `pd.DataFrame` |
| `lg.process` | `(source, *, salt, key, output, where, rules, verbose, ...)` | `ProcessResult` |
| `lg.validate` | `(df, rules, *, severity, warn_only)` | `ValidationReport` |
| `lg.expect` | `(df, column)` | `_ColumnExpectation` (fluent) |

### Manipulation Functions

All accept `pd.DataFrame`, `pl.DataFrame`, and `pl.LazyFrame`. All preserve input type.

| Function | SQL equivalent |
|---|---|
| `lg.where(df, **kwargs)` | `WHERE` |
| `lg.select(df, cols)` | `SELECT col1, col2` |
| `lg.drop(df, cols)` | `SELECT * EXCEPT(col)` |
| `lg.rename(df, mapping)` | `SELECT col AS new_name` |
| `lg.sort(df, by, desc=)` | `ORDER BY` |
| `lg.groupby(df, by, agg)` | `GROUP BY` |
| `lg.add_column(df, **exprs)` | `SELECT *, expr AS name` |
| `lg.when(cond, val).otherwise(d)` | `CASE WHEN` |
| `lg.unique(df, subset)` | `SELECT DISTINCT` |
| `lg.head(df, n)` | `LIMIT N` |
| `lg.top_n(df, n, by, group_by=)` | `RANK() OVER (PARTITION BY ...)` |
| `lg.concat(frames)` | `UNION ALL` |
| `lg.pivot(df, ...)` | `PIVOT` |
| `lg.melt(df, ...)` | `UNPIVOT` |
| `lg.fill_null(df, value)` | `COALESCE` |
| `lg.cast(df, schema)` | `CAST(col AS type)` |

### Aliases

| Alias | Points to |
|---|---|
| `lg.q` | `lg.where` |
| `lg.filter_` | `lg.where` |
| `lg.order_by` | `lg.sort` |
| `lg.group_by` | `lg.groupby` |
| `lg.distinct` | `lg.unique` |
| `lg.union_all` | `lg.concat` |
| `lg.limit` | `lg.head` |
| `lg.unpivot` | `lg.melt` |
| `lg.fillna` | `lg.fill_null` |
| `lg.coalesce` | `lg.fill_null` |
| `lg.assign` | `lg.add_column` |
| `lg.save` | `lg.store` |
| `lg.load` | `lg.read` |

---

## Configuration

```python
import logus as lg
import os

# Set global defaults at application startup
lg.configure(
    default_salt = os.environ["LOGUS_SALT"],  # used when salt= not passed to mask()
    audit_path   = "./audit/",                # auto-create audit trail
)

# Generate secrets (do this once, save to .env)
SALT = lg.generate_salt()        # 256-bit, base64-encoded
KEY  = lg.generate_salt()        # different value from SALT
print(f"LOGUS_SALT={SALT}")
print(f"LOGUS_KEY={KEY}")
```

---

## The .lgs Format

`.lgs` is a binary container for encrypted tabular data:

- **Encryption:** AES-256-GCM (NIST SP 800-38D) — confidentiality + integrity
- **Key derivation:** HKDF-SHA256 (RFC 5869) — unique DEK per file
- **Payload:** Parquet with zstd compression (or lz4 for speed)
- **Header (v2.1):** JSON with schema, column stats, and LGPD metadata — readable without decryption via `lg.inspect()`
- **Integrity:** HMAC-SHA256 over the entire file — detects tampering

```
[5 bytes] MAGIC = b"LOGUS"
[1 byte]  VERSION = 0x02
[1 byte]  CIPHER = 0x01 (AES-256-GCM)
[32 bytes] SALT_KDF — unique per file
[12 bytes] NONCE — for header encryption
[4 bytes]  HEADER_LEN
[N+16 bytes] HEADER_CT+TAG — encrypted JSON metadata
[12 bytes] NONCE — for payload
[M+16 bytes] PAYLOAD_CT+TAG — encrypted Parquet
[32 bytes] FILE_HMAC
```

---

## Requirements

| Package | Min version | Purpose |
|---|---|---|
| `polars` | 1.0.0 | Data engine (required) |
| `pandas` | 2.0.0 | Fallback + Excel/SAS/SPSS formats (required) |
| `pyarrow` | 14.0.0 | Parquet I/O (required) |
| `cryptography` | 41.0.0 | AES-256-GCM, HKDF (required) |
| `numpy` | 1.24.0 | Numeric operations (required) |
| `duckdb` | 0.10.0 | `lg.sql()`, `lg.db()` (optional: `[sql]`) |
| `sqlalchemy` | 2.0.0 | `lg.db()` fallback (optional: `[sql]`) |
| `openpyxl` | 3.1.0 | Excel read/write (optional: `[excel]`) |
| `ctgan` | 0.9.0 | `lg.clone()` synthetic data (optional: `[synthetic]`) |

---

## License

GNU Affero General Public License v3 (AGPL-3.0)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

| Version | Highlights |
|---|---|
| 1.0.5 | `lg.read()` big-data params (head/sample/chunks/iter_chunks), `lg.db()`, mask(LazyFrame), stream via scan_csv batches, CSV sidecar index, 199 tests |
| 1.1.0 | `lg.validate()`, `lg.mask_sql()`, `lg.lineage`, privacy score in profile(), `lg.process()` |
| 1.0.4 | `polars>=1.0.0` required, FastPIIScanner (9× faster), `__init__.py` cleaned |

---

## Novidades v1.1.1

### 1. `lg.contract()` — Contrato de Dados

```python
import logus as lg
import os

SALT = os.environ["LOGUS_SALT"]

contrato = lg.contract({
    "cpf":    {"type": "str",   "not_null": True, "unique": True,
               "pii": "CPF",   "mask": "hash"},
    "renda":  {"type": "float", "min": 0, "max": 500_000,
               "pii": "numerico", "mask": "mock_numeric"},
    "uf":     {"type": "str",   "in": ["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE"]},
    "email":  {"type": "str",   "contains": "@",
               "pii": "email", "mask": "hash"},
}, name="clientes_v2", version="2.0")

# Aplica: valida → mascara → retorna tudo em uma chamada
result = contrato.apply(df, salt=SALT)
result.raise_if_failed()      # levanta ValueError se alguma regra falhou
df_safe = result.df           # DataFrame mascarado

# Versionar e comparar
contrato.save("schema/clientes_v2.contract.json")
c_v1 = lg.DataContract.load("schema/clientes_v1.contract.json")
diff = contrato.diff(c_v1)
print(diff.has_breaking_changes)
print(diff.report())

# Exportar como JSON Schema (para DPOs e equipes não-Python)
schema = contrato.to_json_schema()   # dicionário JSON Schema draft-07
```

### 2. Padrões PII customizados

```python
import logus as lg
from logus.detectors.fast_scan import FastPIIScanner

# Detecta identificadores proprietários da sua empresa
reports = lg.scan(df, custom_patterns={
    "num_contrato": r"^CTR-[0-9]{8}$",
    "matricula":    r"^[0-9]{6}-[A-Z]$",
    "protocolo":    r"^PROT-[0-9]{4}-[0-9]{6}$",
})
```

### 3. Criptografia assimétrica no `.lgs`

```python
import logus as lg

# Gera par de chaves (uma vez por usuário/serviço)
priv, pub = lg.generate_keypair("ec")     # P-256 — recomendado
priv, pub = lg.generate_keypair("rsa", 4096)

lg.save_keypair(priv, "chave.pem", pub, "chave_publica.pem")
pub  = lg.load_public_key("chave_publica.pem")
priv = lg.load_private_key("chave.pem")

# Remetente: cifra com a chave pública do destinatário
lg.store(df, "dados.lgs", public_key=pub)

# Destinatário: decifra com sua chave privada
df = lg.read("dados.lgs", private_key=priv)

# Multi-recipient: uma vez para N destinatários
from logus.asymmetric import encrypt_dek_multi, decrypt_dek_from_list
encs = encrypt_dek_multi(dek, [pub_ana, pub_bruno, pub_carlos])
dek  = decrypt_dek_from_list(encs, priv_ana)
```

### 4. Expiração de arquivo (LGPD Art. 16)

```python
import logus as lg

# Arquivo recusa leitura após a data
lg.store(df, "dados_campanha.lgs", key=KEY, expires_at="2025-12-31")

# Depois da data:
lg.read("dados_campanha.lgs", key=KEY)   # → ExpiredFileError
```

### 5. PII em JSON aninhado (`pl.Struct` e `pl.List`)

```python
import logus as lg
import polars as pl
from logus.detectors.fast_scan import FastPIIScanner

# Polars 1.x: dados JSON em colunas estruturadas
df = pl.DataFrame({
    "pessoa":  [{"cpf": "111.444.777-35", "nome": "Ana Silva"}, ...],
    "emails":  [["ana@empresa.com", "ana@gmail.com"], ...],
})

# FastPIIScanner desaninha e detecta automaticamente
reports = FastPIIScanner().detect_dict(df)
# → {"pessoa.cpf": ColumnReport, "pessoa.nome": ColumnReport, "emails[]": ColumnReport}
```

### 6. Auto-detecção de `.env`

```python
import logus as lg

# Carrega LOGUS_SALT e LOGUS_KEY do arquivo .env automaticamente
lg.configure(load_dotenv=True)              # procura .env no diretório atual
lg.configure(load_dotenv=True, dotenv_path="config/.env")

# Depois disso: lg.mask(df) usa LOGUS_SALT sem precisar de salt= explícito
```

### 7. `banco.create_table()` e `banco.upsert()`

```python
import logus as lg
import os

banco = lg.db("postgresql://user:pass@host/db", salt=os.environ["LOGUS_SALT"])

# Cria tabela a partir do schema do DataFrame
banco.create_table(df, "clientes")
banco.create_table(df, "clientes", if_exists="replace")

# Upsert — INSERT ... ON CONFLICT UPDATE
banco.upsert(df_new, "clientes", on="cpf")
banco.upsert(df_new, "clientes", on=["cpf", "data"])

# Também via lg.write()
lg.write(df_safe, banco, "clientes_masked")
```

### 8. Relatório de Conformidade LGPD (HTML/PDF)

```python
import logus as lg

reports = lg.scan(df)
report  = lg.compliance_report(
    df, reports,
    dataset_name = "Clientes Q1 2025",
    organization = "Empresa XYZ",
)

report.to_html("lgpd_jan2025.html")    # relatório HTML completo
report.to_pdf("lgpd_jan2025.pdf")      # PDF (pip install weasyprint)
report.to_json("lgpd_jan2025.json")    # JSON serializado
print(report.to_text())                # texto simples, sempre disponível
```

### 9. `lg.validate_schema()` + `lg.save_rules()` / `lg.load_rules()`

```python
import logus as lg

# Valida o schema (estrutura) antes de processar os dados
result = lg.validate_schema(df,
    required_columns = ["cpf", "renda_mensal", "uf"],
    forbidden_columns= ["senha", "token"],
    min_rows = 1,
    max_rows = 10_000_000,
)
result.raise_if_failed()

# Versionar regras de validação como JSON
lg.save_rules({
    "cpf":          {"not_null": True, "unique": True},
    "renda_mensal": {"min": 0, "max": 500_000},
    "email":        {"contains": "@"},
}, "regras/clientes_v2.json")

rules  = lg.load_rules("regras/clientes_v2.json")
result = lg.validate(df, rules)
result.print_report()
```

### 10. `lg.shift()`, `lg.lag()`, `lg.lead()`, `lg.explode()`

```python
import logus as lg
import polars as pl

# Séries temporais: lag e lead
df_ts = pl.DataFrame({"renda": [5000., 5200., 4800., 5100., 5300.]})

lg.shift(df_ts, 1)                          # valor do período anterior
lg.shift(df_ts, -1)                         # valor do período seguinte
lg.lag(df_ts, 3, columns="renda")           # alias — lag de 3 períodos
lg.lead(df_ts, 1)                           # alias — próximo valor
lg.shift(df_ts, 1, fill_value=0.0)          # preenche nulos com 0

# Dados aninhados: expande listas em linhas
df_tags = pl.DataFrame({
    "id":   [1, 2, 3],
    "tags": [["lgpd", "privacidade"], ["python", "polars"], ["dados"]],
})
lg.explode(df_tags, "tags")
# Resultado: 5 linhas — uma por tag

# Múltiplas colunas simultâneas
lg.explode(df, ["tags", "scores"])   # scores e tags devem ter o mesmo comprimento
```
