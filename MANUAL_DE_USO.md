# Manual de uso — datalock

Este manual cobre **tudo que já existia** na biblioteca (organizado por
tarefa) e **tudo que foi adicionado** nesta rodada: `strategy=`/`rows=` em
`mask()`, `unmask()`, `.dlk` multi-frame sem chave, `dd.frame()` (proxy
genérico do Polars) e as funções de DBA (`execute`, `transaction`, etc.).

```python
import datalock as dd
import os

SALT = os.environ["DATALOCK_SALT"]   # chave HMAC/cifra — mascaramento
KEY  = os.environ["DATALOCK_KEY"]    # chave AES — criptografia do .dlk
# SALT e KEY precisam ser valores diferentes
```

---

## Índice

1. [Ler e salvar dados](#1-ler-e-salvar-dados)
2. [Mascaramento (anonimização e pseudonimização)](#2-mascaramento)
3. [Reverter mascaramento reversível — `dd.unmask()`](#3-reverter-mascaramento-reversível--ddunmask)
4. [Arquivos `.dlk` sem chave — "parquet melhorado"](#4-arquivos-dlk-sem-chave--parquet-melhorado)
5. [Manipulação de dados — funções prontas](#5-manipulação-de-dados--funções-prontas)
6. [`dd.frame()` — qualquer método do Polars, sem importar Polars](#6-ddframe--qualquer-método-do-polars-sem-importar-polars)
7. [Banco de dados](#7-banco-de-dados)
8. [Detecção de PII e métricas de privacidade](#8-detecção-de-pii-e-métricas-de-privacidade)
9. [Contratos, validação e SQL](#9-contratos-validação-e-sql)
10. [Tabela de referência rápida](#10-tabela-de-referência-rápida)

---

## 1. Ler e salvar dados

```python
df = dd.read("clientes.csv")              # qualquer formato tabular → pl.DataFrame
df = dd.read("clientes.parquet")
df = dd.read("clientes.xlsx")
df = dd.read("clientes.dlk", key=KEY)     # descriptografa
df = dd.read("clientes.dlk")              # .dlk sem key (v4/aberto) — ver seção 4
```

- **Sem `salt=`, nada é mascarado** — `dd.read()` sempre devolve os dados
  exatamente como estão no arquivo. Isso já valia antes desta rodada; a
  confusão comum é achar que `dd.read()` mascara por padrão — não mascara.
- `dd.read(path, raw=True)` só tem efeito em `.dlk` (ignora
  `masking_applied` do header e devolve cru mesmo se `salt=` for passado).
- Para ler **só as colunas que você vai querer mascarar depois**, sem tocar
  no resto:

```python
df = dd.read("clientes.dlk", key=KEY, raw=True)     # tudo cru
df_pii = dd.read("clientes.dlk", key=KEY, columns=["cpf", "email"])  # só essas colunas
```

### Salvar

```python
dd.store(df, "dados.dlk", key=KEY)                      # cifrado, sem mascarar
dd.store(df, "dados.dlk", key=KEY, salt=SALT)            # mascara + cifra
dd.store(df, "dados.dlk")                                # SEM key, SEM mascarar (ver seção 4)
dd.store(df, "dados.dlk", anonymize=True)                # mascara, sem key
```

### Leitura parcial / big data (sem OOM)

```python
info = dd.read("big.parquet", header_only=True)
df   = dd.read("big.parquet", head=100_000)
df   = dd.read("big.parquet", sample=500_000)
for chunk in dd.read("big.parquet", n_chunks=10, iter_chunks=True):
    process(chunk)
```

---

## 2. Mascaramento

### 2.1 Uso básico (como já funcionava)

```python
df_safe = dd.mask(df, salt=SALT)                          # mascara tudo que é PII
df_safe = dd.mask(df, salt=SALT, columns=["cpf", "email"]) # só essas
df_safe = dd.mask(df, salt=SALT, exclude=["uf"])           # tudo, menos essa
df_safe = dd.mask(df, salt=SALT, risk="high")              # atalho por nível de risco
```

O padrão (`HASH`, HMAC-SHA256) é **irreversível por design** — é isso que
faz o dado "deixar de ser dado pessoal" para a LGPD (Art. 12). Não existe
"desfazer um hash".

### 2.2 Novo: `strategy=` — escolher o método manualmente

Duas formas:

**a) String — aplica a mesma estratégia a todas as colunas de `columns=`:**

```python
df_safe = dd.mask(df, salt=SALT, columns=["observacao"], strategy="redact")
```

**b) Dict — por coluna, e funciona em colunas que NÃO são PII:**

```python
df_safe = dd.mask(df, salt=SALT, strategy={
    "cpf":              "encrypt",        # reversível — ver seção 3
    "renda_mensal":     "mock_numeric",   # valor sintético, mesma distribuição
    "observacao_livre": "redact",         # qualquer texto livre, mesmo sem PII
})
```

Estratégias disponíveis:

| Estratégia          | O que faz                                                       | Reversível? |
|---------------------|-------------------------------------------------------------------|:---:|
| `hash`               | HMAC-SHA256 determinístico (padrão para PII)                    | Não |
| `encrypt`            | AES-SIV determinístico — **novo**                                 | **Sim, com `dd.unmask()`** |
| `truncate`           | Mantém só um prefixo (ex.: CEP `12345-XXX`)                       | Não |
| `redact`             | Substitui por `"REDACTED"`                                        | Não |
| `suppress`           | Vira `null`                                                        | Não |
| `mock_numeric`       | Número sintético, mesma faixa de valores                          | Não |
| `mock_category`      | Categoria sintética, mesma distribuição de frequência              | Não |
| `generalize_date`    | Mantém só o ano (ex.: `1985-**`)                                   | Não |
| `mask_phone_ddd`     | Mantém o DDD, mascara o resto                                      | Não |
| `passthrough`        | Não altera nada (útil para "desligar" uma coluna detectada como PII) | — |

Colunas citadas em `strategy={...}` **sempre são mascaradas**, mesmo que
não apareçam em `columns=` e mesmo que o detector de PII não as tivesse
identificado — é assim que dá pra mascarar qualquer coluna com qualquer
método.

### 2.3 Novo: `rows=` — mascarar só um subconjunto de linhas

```python
# só mascara a renda de quem é de SP; as outras linhas ficam com o valor original
df_safe = dd.mask(df, salt=SALT, columns=["renda_mensal"], rows=dd.col("uf") == "SP")

# lista/array de booleanos também funciona
df_safe = dd.mask(df, salt=SALT, columns=["cpf"], rows=[True, False, True, False])

# callable — recebe o DataFrame Polars e devolve a máscara
df_safe = dd.mask(df, salt=SALT, columns=["renda_mensal"],
                   rows=lambda d: d["idade"] < 18)
```

⚠️ Se a máscara produz um tipo diferente do original (comum: `hash`/`encrypt`
transformam qualquer coisa em string), a coluna final vira `string` inteira
— não existe uma coluna que seja `int` em algumas linhas e `string` em
outras ao mesmo tempo.

`strategy=`/`rows=` não podem ser combinados com `risk=` (são dois jeitos
diferentes de escolher a estratégia — um automático por risco, outro manual).

---

## 3. Reverter mascaramento reversível — `dd.unmask()`

Só funciona para colunas mascaradas com `strategy="encrypt"`. Requer o
**mesmo salt** usado para mascarar.

```python
df_safe = dd.mask(df, salt=SALT, strategy={"cpf": "encrypt"})
# ... df_safe pode circular, ser salvo, etc. Depois, quando precisar do valor real:
df_original = dd.unmask(df_safe, salt=SALT, columns=["cpf"])

# sem columns=, tenta detectar automaticamente as colunas com tokens "enc:..."
df_original = dd.unmask(df_safe, salt=SALT)
```

Erros tratados explicitamente:
- Salt errado → `ValueError` claro ("falha na autenticação").
- Tentar reverter uma coluna `hash` → `ValueError` explicando que hash é
  irreversível por design (não tenta "advinhar" nada).

**Importante — implicação de compliance:** uma coluna mascarada com
`encrypt` **continua sendo dado pessoal** (é pseudonimização, não
anonimização) — continua sujeita a LGPD/GDPR, só que protegida em repouso.
Use `hash` (o padrão) quando o objetivo for realmente anonimizar; use
`encrypt` só quando alguém autorizado precisar recuperar o valor original
depois (suporte, correção de cadastro, auditoria).

---

## 4. Arquivos `.dlk` sem chave — "parquet melhorado"

### Uma tabela

```python
dd.store(df, "dados_dev.dlk")            # sem key, sem mascarar — grava como está
df = dd.read("dados_dev.dlk")            # lê de volta, sem exigir nada
```

Use só com dados que já não tenham PII, ou combine com `dd.mask()` antes:

```python
dd.store(dd.mask(df, salt=SALT), "dados_dev.dlk")
```

### Múltiplas tabelas — novo

```python
dd.store({"clientes": df1, "pedidos": df2, "produtos": df3}, "base_dev.dlk")

frames = dd.read("base_dev.dlk")          # dict[str, DataFrame] — todas as tabelas
so_clientes = dd.read("base_dev.dlk", frame="clientes")  # só uma
```

Por baixo, é literalmente um ZIP de tabelas Parquet/Arrow-IPC num único
arquivo, com metadados (schema, contagem de linhas, criado_em) e um HMAC
de integridade (detecta corrupção acidental — sem `key=`, não protege
contra adulteração intencional, isso é o esperado: não há segredo nenhum
envolvido nesse modo).

Se quiser mascarar antes de gravar:

```python
dd.store({"clientes": df1, "pedidos": df2}, "base.dlk", anonymize=True, salt=SALT)
```

### Diferença para o modo criptografado

| | Sem `key=` (`pack_open`/`pack_open_frames`) | Com `key=` (`pack_frames`) |
|---|---|---|
| Confidencialidade | Nenhuma — qualquer um lê | AES-256-GCM / ChaCha20-Poly1305 |
| Integridade | HMAC com chave pública fixa (só detecta corrupção) | HMAC com sua chave (detecta adulteração) |
| Uso típico | Dados já anonimizados, ambiente de dev | Dados sensíveis, produção |

---

## 5. Manipulação de dados — funções prontas

Estas já existiam e continuam funcionando exatamente como antes (todas
preservam o tipo de entrada: `pd.DataFrame` → `pd.DataFrame`,
`pl.DataFrame` → `pl.DataFrame`):

```python
dd.where(df, uf="SP")
dd.where(df, renda_mensal=(5_000, 15_000))       # faixa (inclusiva)
dd.where(df, renda_mensal=(">", 10_000))          # operador

dd.select(df, ["cpf", "uf"])
dd.drop(df, ["cpf_raw"])
dd.rename(df, {"cpf_hash": "cpf"})
dd.add_column(df, imposto=dd.col("renda") * 0.275)

dd.sort(df, "renda", desc=True)                   # atenção: o parâmetro é `desc=`, não `descending=`
dd.groupby(df, "uf", {"n": ("*", "count"), "media": ("renda", "mean")})
dd.pivot(df, on="uf", values="renda", agg_fn="mean")
dd.melt(df, id_vars=["cpf"], value_vars=["jan", "fev"])
dd.explode(df, "tags")

dd.head(df, 100); dd.tail(df, 50); dd.sample(df, 1000, seed=42); dd.top_n(df, "renda", n=10)
dd.unique(df, subset=["cpf"])
dd.concat([df1, df2])

dd.shift(df, 1); dd.lead(df, 1); dd.lag(df, 3, columns="renda")

dd.describe(df); dd.value_counts(df, "uf"); dd.corr(df, "renda", "score")
dd.nunique(df); dd.count_nulls(df)

dd.join(df1, df2, on="cpf", salt=SALT)            # join seguro sobre chaves mascaradas
```

Se você precisa de algo que **não** está nessa lista, vá para a seção 6 —
provavelmente já existe no Polars e o proxy expõe direto.

---

## 6. `dd.frame()` — qualquer método do Polars, sem importar Polars

Esse é o "escape hatch" genérico: em vez de esperar um wrapper `dd.*`
dedicado para cada método do Polars (são 200+), `dd.frame(df)` te dá
acesso a **qualquer** método do Polars, encadeável, com a mesma assinatura
da documentação oficial do Polars — sem precisar `import polars as pl`.

### Uso básico

```python
resultado = (
    dd.frame(df)                                    # aceita pd.DataFrame ou pl.DataFrame
      .filter(dd.col("uf") == "SP")
      .with_columns((dd.col("renda") * 1.1).alias("renda_reajustada"))
      .sort("renda_reajustada", descending=True)
      .head(10)
      .native()                                      # devolve no MESMO tipo do df original
)
```

- `dd.col` **é** `polars.col` (já era assim antes) — funciona normalmente
  dentro do proxy.
- Métodos que retornam `pl.DataFrame`/`pl.LazyFrame` continuam encadeáveis
  automaticamente — não precisa re-embrulhar nada manualmente.
- Argumentos que sejam `pd.DataFrame` ou outro `dd.frame(...)` são
  convertidos para Polars automaticamente (ex.: em `.join(outro_df, ...)`).

### Métodos "terminais" — como sair do proxy

| Método | O que devolve |
|---|---|
| `.native()` | Mesmo tipo do DataFrame original passado a `dd.frame()` |
| `.pandas()` | Sempre `pd.DataFrame` (materializa se for `LazyFrame`) |
| `.polars()` | Sempre `pl.DataFrame` (materializa se for `LazyFrame`) |
| `.lazy_pl()` | O objeto Polars interno, sem conversão nenhuma |
| `.unwrap()` | Alias de `.native()` |

Sem chamar nenhum terminal, dá pra inspecionar o resultado direto
(`print`/`repr` funcionam):

```python
print(dd.frame(df).filter(dd.col("uf") == "SP").head(3))
```

### `group_by`/`rolling` e outros "objetos intermediários" do Polars

Alguns métodos do Polars não devolvem um DataFrame direto — `.group_by()`
devolve um objeto `GroupBy`, que só vira DataFrame depois de `.agg(...)`.
O proxy sabe disso e mantém o encadeamento até ali:

```python
dd.frame(df).group_by("uf").agg(dd.col("renda").mean().alias("media")).native()
```

Se você chamar um terminal (`.native()` etc.) num objeto ainda
intermediário, o erro é explícito em vez de confuso:

```
TypeError: dd.frame(...).native(): o encadeamento ainda não chegou a um
DataFrame — o objeto atual é GroupBy (Polars) [...]. Complete a cadeia
(ex.: .agg(...)) antes de chamar .native().
```

### `LazyFrame` — permanece lazy até você mandar materializar

```python
resultado = (
    dd.frame(df.lazy())
      .filter(dd.col("uf") == "SP")
      .group_by("uf")
      .agg(dd.col("renda").sum())
      .collect()          # materializa (LazyFrame → DataFrame), mantém o proxy
      .native()
)
```

### Exemplos de coisas que só existiam via `import polars` puro, e agora estão em `dd.frame()`

```python
# window functions
dd.frame(df).with_columns(dd.col("renda").rank().over("uf").alias("rank_uf")).native()

# rolling
dd.frame(df).sort("data").with_columns(
    dd.col("vendas").rolling_mean(window_size=7).alias("media_movel_7d")
).native()

# join de qualquer tipo (semi, anti, cross...)
dd.frame(df1).join(df2, on="cpf", how="anti").native()

# pivot_table nativo do polars (mais flexível que dd.pivot)
dd.frame(df).pivot(on="uf", index="cpf", values="renda", aggregate_function="sum").native()

# qualquer método string/data novo que o Polars lançar no futuro também funciona,
# sem precisar esperar o datalock adicionar um wrapper
dd.frame(df).with_columns(dd.col("nome").str.to_titlecase()).native()
```

### Quando usar `dd.frame()` vs. os wrappers `dd.*` prontos (seção 5)

- Operação comum (filtro simples, group by simples, sort) → use o wrapper
  dedicado (`dd.where`, `dd.groupby`, `dd.sort`) — é mais legível.
- Qualquer coisa mais específica, encadeamento longo, ou método que ainda
  não tem wrapper dedicado → `dd.frame()`.

Um método que **não existe** no Polars dá erro claro na hora, dizendo que
o nome não existe (em vez de falhar silenciosamente ou mais adiante):

```python
dd.frame(df).metodo_que_nao_existe()
# AttributeError: dd.frame(...): 'metodo_que_nao_existe' não existe em
# DataFrame (Polars). [...]
```

---

## 7. Banco de dados

### Conectar e ler

```python
banco = dd.db("postgresql://user:pass@host/db", salt=SALT)   # salt= opcional
df = dd.read(banco, "clientes")
df = dd.read(banco, "SELECT * FROM clientes WHERE uf = 'SP'")
df = dd.read(banco, "clientes", sample=10_000)

banco.tables()
banco.schema("clientes")
```

### Escrever

```python
dd.write(df_safe, banco, "clientes_masked")
banco.create_table(df, "clientes", if_exists="replace")
banco.upsert(df_new, "clientes", on="cpf")
```

### Novo — DDL/DML genérico (uso de DBA)

```python
dd.execute(banco, "CREATE INDEX idx_cpf ON clientes (cpf)")
dd.execute(banco, "DELETE FROM clientes WHERE uf = :uf", {"uf": "XX"})

banco.create_index("clientes", ["uf", "cidade"], name="idx_uf_cidade")
banco.drop_table("tabela_temporaria")
```

### Novo — transações

```python
with banco.transaction() as tx:
    tx.execute("UPDATE contas SET saldo = saldo - :v WHERE id = :o", {"v": 100, "o": 1})
    tx.execute("UPDATE contas SET saldo = saldo + :v WHERE id = :d", {"v": 100, "d": 2})
# commit automático se nada der erro; rollback automático se qualquer
# exceção for levantada dentro do bloco `with`
```

### Mascaramento dentro do próprio banco (dados nunca saem)

```python
banco.in_db_mask("clientes", columns=["cpf", "email"])          # via db_adapter.SecureDBAdapter
banco.create_masked_view("clientes", "clientes_view_mascarada")
```

*(Esses dois continuam exatamente como antes — não foram alterados.)*

---

## 8. Detecção de PII e métricas de privacidade

```python
reports = dd.scan(df)                       # Dict[str, ColumnReport]
report  = dd.profile(df)                    # diagnóstico rápido + privacy_score

from datalock import check
check.kanon(df, quasi_identifiers=["uf", "faixa_etaria"])
check.risk(df, quasi_identifiers=["uf", "faixa_etaria", "data_nasc"])
check.utility(df_original, df_masked)
check.dp(epsilon=1.0)

dd.scan_directory("./dados/", recursive=True)
dd.compliance_report(df, reports, dataset_name="Base Clientes", organization="Empresa S.A.")
```

## 9. Contratos, validação e SQL

```python
contrato = dd.contract({
    "cpf": {"type": "str", "not_null": True, "pii": "CPF", "mask": "hash"},
}, name="clientes", version="2.0")
resultado = contrato.apply(df, salt=SALT)

dd.validate(df, rules={"cpf": {"not_null": True, "unique": True}})
dd.expect(df, "renda").to_be_between(0, 500_000)

dd.sql("SELECT uf, AVG(renda) AS media FROM df GROUP BY uf", df=df)  # requer datalock[sql]
```

---

## 10. Tabela de referência rápida

### Já existentes (não alteradas)

| Função | Descrição |
|---|---|
| `dd.read` / `dd.store` / `dd.stream` / `dd.inspect` / `dd.rekey` / `dd.open` | I/O |
| `dd.mask` / `dd.scan` / `dd.diff` / `dd.profile` / `dd.join` | Privacidade |
| `dd.mask_text` / `dd.scan_text` | Texto livre |
| `dd.where` `dd.select` `dd.drop` `dd.rename` `dd.add_column` `dd.sort` `dd.groupby` `dd.pivot` `dd.melt` `dd.explode` `dd.shift` `dd.lead` `dd.lag` `dd.head` `dd.tail` `dd.sample` `dd.top_n` `dd.unique` `dd.concat` `dd.describe` `dd.value_counts` `dd.corr` `dd.nunique` `dd.count_nulls` | Manipulação |
| `dd.db` `dd.read` `dd.write` (com `DatabaseConnection`) | Banco de dados |
| `dd.contract` `dd.validate` `dd.expect` `dd.validate_schema` | Contratos/validação |
| `dd.sql` | SQL via DuckDB |
| `dd.train` `dd.clone` `dd.sandbox` `dd.SyntheticGenerator` | Dados sintéticos |
| `dd.scan_directory` `dd.compliance_report` | Inventário/relatório LGPD |
| `check.kanon` `check.tcloseness` `check.risk` `check.utility` `check.fidelity` `check.dp` | Métricas de privacidade |
| `dd.canary_check` | Rastreamento de vazamento |
| `dd.pl` | Namespace com um subconjunto do Polars (tipos, `col`, `lit`, leitores básicos) |

### Novas nesta rodada

| Função/parâmetro | Descrição |
|---|---|
| `dd.mask(..., strategy=)` | Escolher/forçar o método de mascaramento manualmente, coluna a coluna, mesmo em colunas não-PII |
| `dd.mask(..., rows=)` | Mascarar só um subconjunto de linhas |
| `dd.unmask(df, salt=, columns=)` | Reverter colunas mascaradas com `strategy="encrypt"` |
| `MaskStrategy.ENCRYPT` (`"encrypt"`) | Nova estratégia de mascaramento reversível (AES-SIV) |
| `dd.store({...}, path)` sem `key=` | `.dlk` multi-frame **sem criptografia** |
| `dd.read(path, frame=)` sem `key=` | Ler um frame específico de um `.dlk` multi-frame sem chave |
| `dd.frame(df)` | Proxy genérico — qualquer método do Polars, encadeável, tipo preservado |
| `dd.execute(banco, sql, params=)` | DDL/DML genérico numa conexão `dd.db()` |
| `banco.transaction()` | Context manager transacional (commit/rollback automáticos) |
| `banco.drop_table(table)` | Remove tabela |
| `banco.create_index(table, columns)` | Cria índice |

---

## Bugs corrigidos (efeito colateral, não pedidos, mas relevantes)

1. `dd.store(df, path)` sem `key=` e sem `anonymize=True` marcava o arquivo
   internamente como "anonimizado" mesmo sem mascarar nada. Corrigido —
   agora reflete corretamente que os dados estão crus.
2. Leitura de `.dlk` multi-frame (com ou sem chave) e leitura parcial de
   arquivos `.ipc`/`.feather`/`.arrow` (`header_only=`, `head=`, etc.)
   podiam falhar com `AttributeError`, dependendo da versão do `pyarrow`
   instalada, por uso de uma API que não existe no reader de "stream" do
   Arrow IPC. Corrigido e testado com pyarrow 25.0.1. Recomendo rodar a
   suíte de testes de vocês depois de aplicar essas mudanças.
