# datalock

**datalock** is a Python library for privacy-by-design with tabular data.
LGPD/GDPR compliance, automatic PII detection and masking, AES-256-GCM encrypted
storage (`.dlk` format), expressive data manipulation over Polars, and transparent
canary data for leak tracing.

```bash
pip install datalock
```

```python
import datalock as dd
import os

SALT = os.environ["DATALOCK_SALT"]
KEY  = os.environ["DATALOCK_KEY"]

df       = dd.read("clientes.csv")              # any format → pl.DataFrame
df_safe  = dd.mask(df, salt=SALT)               # detect + mask PII (LGPD)
dd.store(df_safe, "clientes.dlk", key=KEY)      # AES-256-GCM encrypted
df_back  = dd.read("clientes.dlk", key=KEY)     # decrypt and read back
```

> **Renamed from `logus-lgpd`**. The `import logus as lg` alias still works.

---

## What datalock does

| Capability | Function |
|---|---|
| Read any tabular format | `dd.read()` |
| Detect PII automatically | `dd.scan()` |
| Mask PII (HMAC-SHA256) | `dd.mask()` |
| Save with AES-256-GCM | `dd.store()` |
| Expressive manipulation | `dd.where()`, `dd.groupby()`, `dd.add_column()` |
| Full pipeline in one call | `dd.process()` |
| Data quality validation | `dd.validate()` |
| Database with masking | `dd.db()` |
| Directory PII inventory | `dd.scan_directory()` |
| Canary leak tracing | `dd.store(..., canary=True)` |
| Masked text (free-form) | `dd.mask_text(..., strategy="semantic")` |
| Data contracts | `dd.contract()` |
| Privacy metrics | `dd.check.kanon()`, `dd.check.risk()` |

---

## Installation

```bash
pip install datalock                    # core
pip install "datalock[sql]"             # + SQL via DuckDB
pip install "datalock[excel]"           # + Excel (.xlsx)
pip install "datalock[synthetic]"       # + Faker for richer synthetic data
pip install "datalock[full]"            # everything
```

**Requires:** Python ≥ 3.10, Polars ≥ 1.0, pandas ≥ 2.0, pyarrow ≥ 14.0

---

## Quick Start

```python
import datalock as dd
import os

SALT = os.environ["DATALOCK_SALT"]
KEY  = os.environ["DATALOCK_KEY"]

# Backward compat — both work
import logus as lg   # still works
```

### Read any file format

```python
df = dd.read("clientes.csv")
df = dd.read("clientes.parquet")
df = dd.read("clientes.dlk", key=KEY)

# Big data — no OOM
df   = dd.read("big.parquet", head=100_000)
df   = dd.read("big.parquet", sample=500_000)
info = dd.read("big.parquet", header_only=True)
df   = dd.read("big.parquet", n_chunks=5, chunks=[2, 4])
for chunk in dd.read("big.parquet", n_chunks=10, iter_chunks=True):
    process(chunk)
```

### Detect and mask PII

```python
reports = dd.scan(df)
df_safe = dd.mask(df, salt=SALT)
df_safe = dd.mask(df.lazy(), salt=SALT)    # LazyFrame stays lazy

# Custom PII patterns (company-specific identifiers)
reports = dd.scan(df, custom_patterns={
    "num_contrato": r"^CTR-[0-9]{8}$",
    "matricula":    r"^[0-9]{6}-[A-Z]$",
})
```

### Save encrypted (.dlk)

```python
dd.store(df, "clientes.dlk", key=KEY)
dd.store(df, "clientes.dlk", key=KEY, salt=SALT)
dd.store(df, "clientes.dlk", key=KEY, expires_at="2025-12-31")

# Asymmetric — share without sharing the key
priv, pub = dd.generate_keypair("ec")
dd.store(df, "clientes.dlk", public_key=pub)
df = dd.read("clientes.dlk", private_key=priv)
```

### Canary data (transparent leak tracing)

```python
# Inject canary rows silently — user never sees them
dd.store(df, "clientes.dlk", key=KEY, canary=True)
df_back = dd.read("clientes.dlk", key=KEY)
# df_back.shape == df.shape  — canary rows stripped automatically

# If "canary.1ba472d8@datalock.internal" appears in a breach dump:
dd.canary_check("canary.1ba472d8@datalock.internal")
# → {"pipeline_id": "crm_jan2025", "filepath": "clientes.dlk", ...}
```

### Mask text (free-form strings)

```python
text = "Cliente CPF 111.444.777-35, email joao@empresa.com"

dd.mask_text(text, salt=SALT, strategy="redact")
# → "Cliente [CPF], [EMAIL]"

dd.mask_text(text, salt=SALT, strategy="hash")
# → "Cliente 3f2a8b1c9d4e7f0a, 9e1d3c7f2a845b61"

dd.mask_text(text, salt=SALT, strategy="semantic")
# → "Cliente 478.622.984-97, roberto.santos@gmail.com"
# Real-looking fake data (CPF mathematically valid, no faker needed)
```

### Scan a directory for PII

```python
inventory = dd.scan_directory("./dados/", recursive=True)
print(inventory.summary())
inventory.to_html("inventario_pii.html")
inventory.to_json("inventario_pii.json")

for path, fi in inventory.items():
    if fi.max_risk == "high":
        print(f"HIGH RISK: {path} → {list(fi.pii_columns.keys())}")
```

### Manipulate data

```python
dd.where(df, uf="SP")
dd.where(df, renda_mensal=(5_000, 15_000))
dd.groupby(df, "uf", {"n": ("*","count"), "media": ("renda","mean")})
dd.add_column(df,
    imposto = dd.col("renda_mensal") * 0.275,
    faixa   = dd.when(dd.col("renda_mensal") > 10_000, "alta")
                .when(dd.col("renda_mensal") > 5_000, "media")
                .otherwise("baixa"),
)
dd.shift(df, 1)        # lag — previous period value
dd.lead(df, 1)         # next period value
dd.explode(df, "tags") # list column → multiple rows
```

### Contracts, validation, database

```python
# Data contract
contrato = dd.contract({
    "cpf":   {"type":"str","not_null":True,"pii":"CPF","mask":"hash"},
    "renda": {"type":"float","min":0,"max":500_000},
})
result = contrato.apply(df, salt=SALT)
contrato.save("schema.contract.json")

# Database
banco = dd.db("postgresql://user:pass@host/db", salt=SALT)
df    = dd.read(banco, "clientes")
banco.upsert(df_new, "clientes", on="cpf")

# Audit webhook
dd.configure(audit_webhook="https://hooks.slack.com/...")
```

---

## The `.dlk` format

Binary container: AES-256-GCM + HKDF-SHA256 + Parquet/zstd.
Magic bytes: `b"DLOCK"` (v1.1.2+). Prior `b"LOGUS"` files still readable.

```python
info = dd.inspect("clientes.dlk", key=KEY)
# → {"shape":[150000,12], "columns":[...], "column_stats":{...}, "expires_at":...}
```

---

## Backward compatibility

```python
# All of these still work after the rename:
import logus as lg
lg.mask(df, salt=SALT)   # identical to dd.mask()
lg.read("f.lgs", key=KEY)  # .lgs files still read correctly
```

---

## License

AGPL-3.0 — see LICENSE.
