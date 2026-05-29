# Changelog — logus-lgpd

## v1.1.1 (2026-05)

### 10 novas features

**1. `lg.contract()` — Data Contract**
Unifica tipos, validação de qualidade, detecção PII e estratégia de mascaramento em uma única
declaração versionável. Métodos: `validate()`, `mask()`, `apply()`, `diff()`,
`to_json_schema()`, `save()`, `load()`. `ContractDiff.has_breaking_changes` detecta
incompatibilidades entre versões.

**2. Padrões PII customizados**
`FastPIIScanner(custom_patterns={"contrato": r"^CTR-[0-9]{8}$"})` e
`lg.scan(df, custom_patterns={...})` permitem registrar identificadores proprietários.

**3. Criptografia assimétrica no `.lgs`**
`lg.generate_keypair("ec"|"rsa")`, `lg.store(df, key, public_key=pub)`,
`lg.read(f, private_key=priv)`. Multi-recipient via `encrypt_dek_multi()`.
Mecanismo: ECIES (EC) e RSA-OAEP (RSA).

**4. Expiração de arquivo** (`expires_at=`)
`lg.store(df, "f.lgs", key=KEY, expires_at="2025-12-31")`. Levanta `ExpiredFileError`
após a data. Verificado no header antes de decriptar. Baseado no Art. 16 LGPD.

**5. PII em dados aninhados** (`pl.Struct` e `pl.List`)
`FastPIIScanner` desaninha colunas estruturadas via `.struct.unnest()` e listas via
`.explode()` antes de varrer. Colunas reportadas como `"pessoa.cpf"` e `"emails[]"`.

**6. Auto-detecção de `.env`**
`lg.configure(load_dotenv=True)` carrega `LOGUS_SALT` e `LOGUS_KEY` automaticamente
via `python-dotenv`. Fallback com `UserWarning` se não instalado.

**7. `banco.create_table()` e `banco.upsert()`**
`banco.create_table(df, "tabela", if_exists="replace")` cria tabela a partir do schema
do DataFrame. `banco.upsert(df, "tabela", on="cpf")` faz INSERT OR REPLACE (SQLite) /
INSERT ON CONFLICT UPDATE (PostgreSQL) / fallback DELETE+INSERT.

**8. `lg.compliance_report()` — Relatório LGPD**
Gera documento HTML (e PDF com weasyprint) com inventário de PII (Art. 37),
privacy score, trilha de auditoria (Art. 50). Métodos: `to_html()`, `to_pdf()`,
`to_text()`, `to_json()`.

**9. `lg.validate_schema()` + `lg.save_rules()` / `lg.load_rules()`**
Valida estrutura do DataFrame (colunas obrigatórias, proibidas, exatas, min/max linhas).
Salva e carrega conjuntos de regras como JSON versionável.

**10. `lg.shift()`, `lg.lag()`, `lg.lead()`, `lg.explode()`**
`shift(df, n)` desloca valores N períodos (positivo=lag, negativo=lead).
`lag(df, n)` e `lead(df, n)` são aliases descritivos. `explode(df, "col")` expande
colunas de listas em múltiplas linhas. Todos preservam tipo pd/pl.

---

## v1.0.5 (2026-05)

- `lg.read()` big-data: `header_only`, `head`, `tail`, `sample`, `n_chunks/chunks`, `iter_chunks`
- `lg.db()` — objeto de conexão reutilizável com ConnectorX + TABLESAMPLE
- `lg.read(banco, "tabela")` — API unificada para DataFrame e banco
- `lg.write(df, banco, "tabela")` — escrita unificada
- CSV sidecar index (`.csv.logus_idx`) via mmap — 201× mais rápido no segundo acesso
- `lg.mask(pl.LazyFrame)` — permanece lazy até `.collect()`
- `lg.stream()` via `pl.scan_csv().collect_batches()` — sem OOM
- `mock_cat/num` determinístico — seed por coluna, categorias ordenadas
- 95 novos testes (199 total)

---

## v1.2.0 / v1.1.0 (2026-04)

- `lg.process()` — pipeline completo em uma chamada (`ProcessResult`)
- `lg.validate()` + `lg.expect()` — Data Quality integrado
- `lg.mask_sql()` + `lg.generate_view()` — SQL Transpiler (6 dialetos)
- Privacy Score em `lg.profile()` (0–100, grade A–F)
- `lg.lineage` — rastreamento OpenLineage-inspired
- FastPIIScanner — 9× mais rápido (sample-once + Polars-native regex)
- `lg.read(path, columns=)` — column pruning em `.lgs`
- Metadata index v2.1 no header `.lgs` (column_stats legíveis sem decriptar)

---

## v1.0.4 (2026-03)

- Polars ≥ 1.0.0 obrigatório
- `__all__` unificado (100+ símbolos)
- `lg.check` namespace (kanon, risk, utility, dp, tcloseness)
- `.lgs` v2: header cifrado, cipher negociado (AES-256-GCM ou ChaCha20-Poly1305)
- `lg.rekey()` — rotação de chave sem expor dados
