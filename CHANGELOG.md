## v1.1.4 (2026-06)

### Column Pruning e Predicate Pushdown no `.dlk`

**Motivação:** Para um DataFrame de 1M linhas × 50 colunas, acessar 2 colunas e 20% das linhas
alocava 400 MB antes de filtrar para os ~16 MB necessários. Esta release elimina essa alocação.

**`_df_to_bytes()` — serialização multi-batch**
Serializa o payload como stream de N record batches Arrow IPC (padrão: 50 000 linhas/batch).
Retorna `(bytes, list[dict])` com metadados de cada batch: `batch_index`, `byte_offset`,
`byte_length`, `n_rows`, `stats` (`min`/`max`/`null_count` por coluna).
Para `raw_dataframe`, stats expõem apenas `dtype` (política de exposição mínima).

**`_bytes_to_df()` — desserialização seletiva**
Aceita `columns=`, `filters=` e `row_groups_meta=`. Usa `prune_row_groups()` para
identificar batches relevantes antes de construir arrays Python. Column pruning via
`batch.select(columns)` no nível Arrow. Predicate pushdown via `apply_arrow_filters()`.
Retrocompat: `row_groups_meta=None` → lê tudo; archivos antigos degradam graciosamente.

**`_build_header()` — índice de row groups no header cifrado**
Adiciona `"row_groups": [...]` e `"format_version": "3.0"` ao header.
Leitores antigos ignoram campos desconhecidos — sem quebra de retrocompatibilidade.

**`load_raw()` e `load()`** — novos parâmetros `filters=` e `columns=` propagados para `_bytes_to_df()`.

**`dd.read()` em `__init__.py`** — novo parâmetro `filters=` propagado para `SecureFile.load_raw()` e `SecureFile.load()`.

**Novo módulo `datalock/ipc_index.py`**
Isola lógica de pruning e pushdown:
  - `compute_batch_stats(batch, content_type)` — min/max/null_count por coluna
  - `prune_row_groups(row_groups_meta, filters)` → `set[int]` de batches relevantes
  - `apply_arrow_filters(batch, filters)` → `pa.RecordBatch` filtrado
  - `normalize_filters(filters)` → lista canônica de tuplas `(col, op, value)`

**Novo arquivo `tests/test_column_pruning.py`** — 7 cenários obrigatórios + testes unitários de `ipc_index`.

**Ganho de memória estimado:** 25× para padrão `columns=2/50 + filters=20%` em 1M linhas.
Tempo de decifração: idêntico (AES-GCM sobre payload inteiro, inescapável).

---

# Changelog — datalock

## v1.1.0 (2026-05)

### 10 novas features

**1. `dd.contract()` — Data Contract**
Unifica tipos, validação de qualidade, detecção PII e estratégia de mascaramento em uma única
declaração versionável. Métodos: `validate()`, `mask()`, `apply()`, `diff()`,
`to_json_schema()`, `save()`, `load()`. `ContractDiff.has_breaking_changes` detecta
incompatibilidades entre versões.

**2. Padrões PII customizados**
`FastPIIScanner(custom_patterns={"contrato": r"^CTR-[0-9]{8}$"})` e
`dd.scan(df, custom_patterns={...})` permitem registrar identificadores proprietários.

**3. Criptografia assimétrica no `.dlk`**
`dd.generate_keypair("ec"|"rsa")`, `dd.store(df, key, public_key=pub)`,
`dd.read(f, private_key=priv)`. Multi-recipient via `encrypt_dek_multi()`.
Mecanismo: ECIES (EC) e RSA-OAEP (RSA).

**4. Expiração de arquivo** (`expires_at=`)
`dd.store(df, "f.dlk", key=KEY, expires_at="2025-12-31")`. Levanta `ExpiredFileError`
após a data. Verificado no header antes de decriptar. Baseado no Art. 16 LGPD.

**5. PII em dados aninhados** (`pl.Struct` e `pl.List`)
`FastPIIScanner` desaninha colunas estruturadas via `.struct.unnest()` e listas via
`.explode()` antes de varrer. Colunas reportadas como `"pessoa.cpf"` e `"emails[]"`.

**6. Auto-detecção de `.env`**
`dd.configure(load_dotenv=True)` carrega `DATALOCK_SALT` e `DATALOCK_KEY` automaticamente
via `python-dotenv`. Fallback com `UserWarning` se não instalado.
Também lê `DATALOCK_CANARY_SALT` e `DATALOCK_WM_SALT` para canary e watermarking.

**7. `banco.create_table()` e `banco.upsert()`**
`banco.create_table(df, "tabela", if_exists="replace")` cria tabela a partir do schema
do DataFrame. `banco.upsert(df, "tabela", on="cpf")` faz INSERT OR REPLACE (SQLite) /
INSERT ON CONFLICT UPDATE (PostgreSQL) / fallback DELETE+INSERT.

**8. `dd.compliance_report()` — Relatório LGPD**
Gera documento HTML (e PDF com weasyprint) com inventário de PII (Art. 37),
privacy score, trilha de auditoria (Art. 50). Métodos: `to_html()`, `to_pdf()`,
`to_text()`, `to_json()`.

**9. `dd.validate_schema()` + `dd.save_rules()` / `dd.load_rules()`**
Valida estrutura do DataFrame (colunas obrigatórias, proibidas, exatas, min/max linhas).
Salva e carrega conjuntos de regras como JSON versionável.

**10. `dd.shift()`, `dd.lag()`, `dd.lead()`, `dd.explode()`**
`shift(df, n)` desloca valores N períodos (positivo=lag, negativo=lead).
`lag(df, n)` e `lead(df, n)` são aliases descritivos. `explode(df, "col")` expande
colunas de listas em múltiplas linhas. Todos preservam tipo pd/pl.

**11. `canary_salt` e `wm_salt` configuráveis**
`dd.configure(canary_salt=..., wm_salt=...)` permite substituir os salts de fingerprint
canary e watermark por valores secretos, impedindo que adversários com acesso ao source
pré-calculem quais linhas seriam geradas para um dado `pipeline_id`.

---

## v1.0.5 (2026-05)

- `dd.read()` big-data: `header_only`, `head`, `tail`, `sample`, `n_chunks/chunks`, `iter_chunks`
- `dd.db()` — objeto de conexão reutilizável com ConnectorX + TABLESAMPLE
- `dd.read(banco, "tabela")` — API unificada para DataFrame e banco
- `dd.write(df, banco, "tabela")` — escrita unificada
- CSV sidecar index (`.csv.datalock_idx`) via mmap — 201× mais rápido no segundo acesso
- `dd.mask(pl.LazyFrame)` — permanece lazy até `.collect()`
- `dd.stream()` via `pl.scan_csv().collect_batches()` — sem OOM
- `mock_cat/num` determinístico — seed por coluna, categorias ordenadas
- 95 novos testes (199 total)

---

## v1.2.0 / v1.1.0 (2026-04)

- `dd.process()` — pipeline completo em uma chamada (`ProcessResult`)
- `dd.validate()` + `dd.expect()` — Data Quality integrado
- `dd.mask_sql()` + `dd.generate_view()` — SQL Transpiler (6 dialetos)
- Privacy Score em `dd.profile()` (0–100, grade A–F)
- `dd.lineage` — rastreamento OpenLineage-inspired
- FastPIIScanner — 9× mais rápido (sample-once + Polars-native regex)
- `dd.read(path, columns=)` — column pruning em `.dlk`
- Metadata index v2.1 no header `.dlk` (column_stats legíveis sem decriptar)

---

## v1.0.4 (2026-03)

- Polars ≥ 1.0.0 obrigatório
- `__all__` unificado (100+ símbolos)
- `dd.check` namespace (kanon, risk, utility, dp, tcloseness)
- `.dlk` v2: header cifrado, cipher negociado (AES-256-GCM ou ChaCha20-Poly1305)
- `dd.rekey()` — rotação de chave sem expor dados
