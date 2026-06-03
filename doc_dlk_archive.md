# Especificação Técnica do Formato `.dlk`
### datalock v1.1.2 — Referência Completa

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Premissas de Segurança](#2-premissas-de-segurança)
3. [Primitivos Criptográficos](#3-primitivos-criptográficos)
4. [Hierarquia de Chaves](#4-hierarquia-de-chaves)
5. [Estrutura Binária por Versão](#5-estrutura-binária-por-versão)
   - [v1 — Legado](#51-formato-v1--legado-datalock--132)
   - [v2 — Single-frame](#52-formato-v2--single-frame-datalock--132)
   - [v3 — Multi-frame](#53-formato-v3--multi-frame-datalock--150)
   - [v4 — Sem criptografia](#54-formato-v4--sem-criptografia)
6. [Serialização do Payload](#6-serialização-do-payload)
7. [Header JSON](#7-header-json)
8. [Multi-frame: estrutura do ZIP interno](#8-multi-frame-estrutura-do-zip-interno)
9. [ACL por frame](#9-acl-por-frame-datalock--112)
10. [Canary Data](#10-canary-data)
11. [Expiração de Arquivo](#11-expiração-de-arquivo)
12. [Retrocompatibilidade](#12-retrocompatibilidade)
13. [Considerações de Segurança](#13-considerações-de-segurança)
14. [Referências Normativas](#14-referências-normativas)

---

## 1. Visão Geral

O formato `.dlk` é um contêiner binário para armazenamento e transferência segura de dados tabulares. Ele combina em um único arquivo:

- **Confidencialidade** — criptografia AEAD (AES-256-GCM ou ChaCha20-Poly1305)
- **Integridade** — HMAC-SHA256 sobre o arquivo completo, inclusive cabeçalho
- **Metadados LGPD/GDPR** — schema, classificação PII, expiração e trilha de auditoria
- **Suporte tabular** — Parquet/Arrow IPC internamente; retrocompatível entre versões

**Magic bytes:** `b"DLOCK"` (5 bytes, offset 0)

**Por que não GPG, 7-Zip ou Parquet criptografado?**

| Ferramenta          | AES-256-GCM | Integridade HMAC | Metadados LGPD | Schema tabular |
|---------------------|:-----------:|:----------------:|:--------------:|:--------------:|
| GPG / age           | ✓           | ✓                | ✗              | ✗              |
| 7-Zip AES-256       | ✓           | ✗                | ✗              | ✗              |
| Parquet + S3 SSE    | ✓           | ✗                | ✗              | ✓              |
| **datalock (.dlk)** | **✓**       | **✓**            | **✓**          | **✓**          |

---

## 2. Premissas de Segurança

| Premissa | Detalhe |
|---|---|
| A `master_key` é material de alta entropia | Saída de KMS, `secrets.token_hex(32)` ou vault. **Não é uma senha humana.** HKDF não faz password stretching. |
| O salt HKDF é único por arquivo | Gerado via `os.urandom(32)` a cada `pack()`. Nunca reutilizado. |
| O nonce AEAD é único por operação | Gerado via `os.urandom(12)` por bloco cifrado. Colisão de nonce sob mesma DEK quebraria o GCM. |
| HMAC antes da decifração | O arquivo é autenticado (Verify-then-Decrypt) antes de qualquer decifração, prevenindo ataques de padding oracle e chosen-ciphertext. |
| Header cifrado em v2/v3 | O schema e metadados não ficam em claro, prevenindo fingerprinting por observação do header. |
| Sem exposição persistente | O dado bruto existe apenas na heap durante desserialização. Python não garante zeroização determinística; o formato reduz a janela de exposição operacional. |

---

## 3. Primitivos Criptográficos

### AES-256-GCM
Cipher byte: `0x01`

Modo AEAD (Authenticated Encryption with Associated Data). Garante confidencialidade e integridade em uma única operação. A auth_tag de 16 bytes autentica o ciphertext; qualquer adulteração é detectada antes que qualquer byte seja entregue ao chamador.

- Tamanho de chave: 256 bits (32 bytes)
- Tamanho de nonce: 96 bits (12 bytes)
- Tamanho da auth_tag: 128 bits (16 bytes)
- Referência: NIST SP 800-38D

### ChaCha20-Poly1305
Cipher byte: `0x02`

Alternativa ao AES-GCM para hardware sem AES-NI. Desempenho equivalente ao AES-GCM com aceleração de hardware, superior em CPUs ARM sem extensões criptográficas. O algoritmo de seleção automática (`_detect_best_cipher`) benchmarka ambos com 1 MB e escolhe o mais rápido.

- Tamanho de chave: 256 bits
- Tamanho de nonce: 96 bits
- Referência: RFC 8439

### HKDF-SHA256
Usado exclusivamente para derivação de chaves (não para password stretching). Pressupõe IKM de alta entropia.

- Referência: RFC 5869

### HMAC-SHA256
Autenticação sobre o arquivo completo (body sem o próprio HMAC final). Computado com a MAK derivada (v2/v3) ou com a `master_key` diretamente (v1, retrocompat).

### Separação de domínio criptográfico

Três chaves derivadas da mesma `master_key`, cada uma com `info=` distinto no HKDF:

| Chave | `info=` | Uso |
|-------|---------|-----|
| DEK | `b"datalock-dek-v1"` | Cifra o payload de dados |
| HEK | `b"datalock-hek-v1"` | Cifra o header JSON |
| MAK | `b"datalock-mak-v1"` | Chave do HMAC de arquivo |

Comprometer a DEK não expõe a HEK nem a MAK. Comprometer o header não expõe o payload.

---

## 4. Hierarquia de Chaves

```
master_key (string UTF-8, ≥ 16 bytes)
    │
    ├── HKDF-SHA256(salt_kdf, info="datalock-dek-v1") ──► DEK  [32 bytes]
    │       └── AES-256-GCM / ChaCha20-Poly1305: cifra o payload
    │
    ├── HKDF-SHA256(salt_kdf, info="datalock-hek-v1") ──► HEK  [32 bytes]
    │       └── AES-256-GCM / ChaCha20-Poly1305: cifra o header JSON
    │
    └── HKDF-SHA256(salt_kdf, info="datalock-mak-v1") ──► MAK  [32 bytes]
            └── HMAC-SHA256: autentica o arquivo inteiro
```

Para arquivos ACL multi-frame, existe uma quarta derivação por nível de acesso:

```
master_key
    └── HKDF-SHA256(salt_kdf, info="datalock-frame-dek-v1:{level}") ──► DEK_level
            └── Cifra individualmente o Parquet de cada frame por nível
```

O `salt_kdf` (32 bytes de `os.urandom`) é único por arquivo e armazenado em claro no corpo do arquivo. Sua função não é secreta — é apenas garantir que derivações distintas para arquivos distintos produzam chaves distintas, mesmo com a mesma `master_key`.

---

## 5. Estrutura Binária por Versão

### Constantes do formato

| Constante        | Valor   | Significado |
|------------------|---------|-------------|
| `MAGIC`          | `DLOCK` | 5 bytes, offset 0 |
| `SALT_KDF_LEN`   | 32      | bytes do salt HKDF |
| `NONCE_LEN`      | 12      | bytes do nonce AEAD |
| `AUTH_TAG_LEN`   | 16      | bytes da auth_tag GCM/Poly1305 |
| `FILE_HMAC_LEN`  | 32      | bytes do HMAC final |
| `HEADER_LEN_FMT` | `>I`    | big-endian uint32 para comprimento do header |
| `MAX_HEADER_LEN` | 1 MB    | limite de tamanho do header antes de alocar |

---

### 5.1 Formato v1 — Legado (datalock < 1.3.2)

O header JSON fica **em claro** (autenticado pelo HMAC final, mas não cifrado). Ainda legível por versões atuais.

```
Offset   Tamanho   Campo
──────   ───────   ─────
0        5         MAGIC = b"DLOCK"
5        1         VERSION = 0x01
6        4         HEADER_LEN (uint32 big-endian)
10       N         HEADER_JSON (plaintext UTF-8)
10+N     32        SALT_KDF
42+N     12        NONCE
54+N     M         CIPHERTEXT (AES-256-GCM)
54+N+M   16        AUTH_TAG
70+N+M   32        FILE_HMAC  ← HMAC-SHA256(master_key, tudo acima)
```

**Cipher:** sempre AES-256-GCM (implícito, sem campo explícito).
**HMAC:** calculado com `master_key` diretamente (sem derivação MAK).

---

### 5.2 Formato v2 — Single-frame (datalock ≥ 1.3.2)

Header **cifrado** com HEK separada. Cipher negociado (AES-256-GCM ou ChaCha20-Poly1305).

```
Offset        Tamanho     Campo
──────        ───────     ─────
0             5           MAGIC = b"DLOCK"
5             1           VERSION = 0x02
6             1           CIPHER (0x01=AES-GCM | 0x02=ChaCha20)
7             32          SALT_KDF
39            12          NONCE_HEADER
51            4           HEADER_CT_LEN (uint32 big-endian)  ← ≤ MAX_HEADER_LEN (1 MB)
55            N+16        HEADER_CT + HEADER_TAG  (header JSON cifrado com HEK)
55+N+16       12          NONCE_PAYLOAD
67+N+16       M           PAYLOAD_CT  (Parquet/Arrow cifrado com DEK)
67+N+16+M     16          PAYLOAD_TAG
83+N+16+M     32          FILE_HMAC  ← HMAC-SHA256(MAK, tudo acima)
```

**Escrita atômica:** o arquivo é escrito em um arquivo `.tmp` com UUID único no nome (`.<nome>.<uuid4>.tmp`) e movido atomicamente para o destino, nunca deixando o destino em estado parcial.

---

### 5.3 Formato v3 — Multi-frame (datalock ≥ 1.5.0)

Estrutura binária **idêntica ao v2** (`VERSION = 0x03`), com a diferença de que o payload decifrado é um arquivo ZIP em memória contendo um Parquet por frame.

O `content_type` no header JSON indica `"multi_dataframe"`. Leitores v1/v2 que não reconheçam multi-frame receberão um `TypeError` descritivo em vez de corrupção silenciosa.

---

### 5.4 Formato v4 — Sem criptografia

Para dados já anonimizados onde confidencialidade não é necessária, mas o formato padronizado (Parquet + metadados + integridade) ainda é útil.

```
Offset     Tamanho     Campo
──────     ───────     ─────
0          5           MAGIC = b"DLOCK"
5          1           VERSION = 0x04
6          4           HEADER_LEN (uint32 big-endian)  ← ≤ MAX_HEADER_LEN (1 MB)
10         N           HEADER_JSON (plaintext UTF-8)
10+N       M           PAYLOAD (Arrow IPC/zstd, plaintext)
10+N+M     32          FILE_HMAC  ← HMAC-SHA256(_NO_KEY_HMAC_KEY, tudo acima)
```

**Atenção:** O HMAC usa a chave pública fixa `b"datalock-no-key-integrity-v1"`. Detecta **corrupção acidental** mas **não protege contra adulteração intencional**. Não use v4 para dados com PII não mascarado.

---

## 6. Serialização do Payload

O payload decifrado é um buffer binário com um magic marker nos primeiros 5 bytes que identifica o formato de serialização:

| Magic marker  | Formato                     | Versão do datalock |
|---------------|-----------------------------|-------------------|
| `b"IPC1\x00"` | Apache Arrow IPC (atual)    | ≥ v1.4            |
| `b"PQ1\x00"`  | Apache Parquet              | v1.3              |
| outros        | Arrow IPC sem marker        | v1.0–v1.2 (legado)|

**Arrow IPC** (formato atual): serialização binária colunar que mapeia diretamente o layout in-memory do Arrow, 3–5× mais rápido que Parquet para serialização em memória. Compressão interna via `IpcWriteOptions(compression=...)`.

**Compressões suportadas:**

| Valor no header `"compression"` | Algoritmo               | Versão |
|---------------------------------|-------------------------|--------|
| `"ipc_zstd"`                    | Arrow IPC + zstd        | atual  |
| `"ipc_lz4"`                     | Arrow IPC + lz4         | atual  |
| `"ipc_none"`                    | Arrow IPC sem compressão| atual  |
| `"parquet_zstd"`                | Parquet + zstd          | v1.3   |
| `"parquet_lz4"`                 | Parquet + lz4           | v1.3   |
| `"zlib"`                        | zlib externo            | v1.0–1.2 |
| `"none"`                        | sem compressão          | v4     |

---

## 7. Header JSON

O header é um objeto JSON UTF-8, cifrado (v2/v3) ou em claro (v1/v4). Campos comuns a todas as versões:

| Campo                   | Tipo     | Descrição |
|-------------------------|----------|-----------|
| `format`                | string   | sempre `"lgs"` |
| `version`               | string   | versão do formato (`"2.1"`, `"3.0"`, `"3.1"`, `"4.0"`) |
| `content_type`          | string   | tipo do payload (ver abaixo) |
| `label`                 | string   | rótulo livre para auditoria |
| `created_at`            | ISO 8601 | timestamp UTC de criação |
| `created_by`            | string   | `"datalock/{versão}"` |
| `encryption`            | string   | algoritmo cipher ou `"none"` |
| `kdf`                   | string   | `"HKDF-SHA256-v2"` ou `"none"` |
| `integrity`             | string   | `"HMAC-SHA256+MAK"` ou `"HMAC-SHA256 (public key)"` |
| `compression`           | string   | identificador do algoritmo (ver §6) |
| `plaintext_size_bytes`  | int      | tamanho do payload antes de cifrar |
| `metadata`              | object   | metadados arbitrários do usuário |
| `expires_at`            | ISO 8601 | data de expiração opcional (LGPD Art. 16) |

**Valores de `content_type`:**

| Valor                   | Significado |
|-------------------------|-------------|
| `"raw_dataframe"`       | Dados brutos, não mascarados |
| `"masked_dataframe"`    | Dados com mascaramento HMAC/REDACT/TRUNCATE aplicado |
| `"anonymous_dataframe"` | Dados anonimizados (v4, sem key) |
| `"multi_dataframe"`     | Múltiplos DataFrames (v3) |
| `"bytes"`               | Payload binário arbitrário |
| `"custom:<string>"`     | Tipo definido pelo usuário |

**Campos exclusivos de single-frame (v2, content_type ≠ multi):**

| Campo            | Tipo   | Descrição |
|------------------|--------|-----------|
| `shape`          | [int, int] | [linhas, colunas] do DataFrame |
| `schema`         | object | `{coluna: dtype_str}` |
| `columns`        | array  | lista de nomes de colunas |
| `column_stats`   | object | estatísticas por coluna (ver abaixo) |
| `masking_applied`| bool   | indica se mascaramento foi pré-aplicado |

**`column_stats` por coluna:**

Para `raw_dataframe`: apenas `dtype` (sem `n_nulls`, `n_unique`, `min`, `max` — suprimidos para evitar inferência sobre PII).

Para `masked_dataframe` e `anonymous_dataframe`: `dtype`, `n_nulls`, `n_unique`, e para tipos numéricos `min`/`max`.

**Campos exclusivos de multi-frame (v3):**

| Campo           | Tipo   | Descrição |
|-----------------|--------|-----------|
| `n_frames`      | int    | número de frames |
| `frame_names`   | array  | lista de nomes de frames |
| `frame_index`   | array  | array de objetos com `{name, filename, size_bytes, rows, cols, schema}` |
| `acl_enabled`   | bool   | `true` se ACL por frame ativo (v3.1) |
| `access_levels` | object | `{nome_frame: nivel}` quando `acl_enabled=true` |

---

## 8. Multi-frame: estrutura do ZIP interno

Quando `content_type = "multi_dataframe"`, o payload decifrado é um arquivo ZIP em memória (`zipfile.ZIP_STORED`, compressão delegada ao serializer Arrow). Estrutura do ZIP:

```
index.json          ← metadados de todos os frames (obrigatório)
clientes.parquet    ← Arrow IPC do frame "clientes"
pedidos.parquet     ← Arrow IPC do frame "pedidos"
...
```

**`index.json`:** array JSON onde cada elemento descreve um frame:

```json
[
  {
    "name":       "clientes",
    "filename":   "clientes.parquet",
    "size_bytes": 45231,
    "rows":       10000,
    "cols":       12,
    "schema":     {"cpf": "object", "renda": "float64", ...}
  },
  ...
]
```

A ordem dos elementos em `index.json` é preservada na leitura. Um leitor pode extrair um único frame sem desserializar os demais, lendo apenas a entrada correspondente.

---

## 9. ACL por frame (datalock ≥ 1.1.2)

Versão `3.1` do formato. Ativa quando `acl_enabled: true` no header.

Cada frame recebe uma DEK específica derivada do seu nível de acesso:

```
master_key
    └── HKDF-SHA256(salt_kdf, info="datalock-frame-dek-v1:{nivel}") → DEK_nivel
```

Estrutura do ZIP interno com ACL:

```
index.json                  ← lista com access_level por frame
clientes_nonce.bin          ← nonce (12 bytes) para o frame "clientes"
clientes_tag.bin            ← auth_tag GCM (16 bytes)
clientes.parquet.enc        ← Arrow IPC cifrado com DEK_nivel
salarios_nonce.bin
salarios_tag.bin
salarios.parquet.enc
...
```

O ZIP inteiro é **adicionalmente** cifrado pela DEK principal do arquivo (camada externa v2/v3). Um adversário que extraísse o ZIP sem quebrar a camada externa ainda precisaria de DEK_nivel para ler cada frame.

**Hierarquia recomendada de níveis:**

| Nível          | Uso típico |
|----------------|-----------|
| `"public"`     | Dados sem restrição de acesso |
| `"internal"`   | Colaboradores gerais |
| `"confidential"` | Dados sensíveis (RH, jurídico) |
| `"restricted"` | Dados altamente sensíveis (diretoria, M&A) |
| `"secret"`     | Dados pessoais críticos (LGPD art. 11) |

A hierarquia é convencional — o sistema trata os níveis como strings opacas. A distribuição de quais DEKs cada usuário recebe é responsabilidade do KMS da organização.

**Propriedade de segurança:** comprometer `DEK_user` não expõe `DEK_admin`, nem a `master_key`, nem DEKs de outros níveis. O erro de autorização (frame existe mas usuário não tem acesso) é silencioso — o frame é simplesmente omitido do resultado, evitando enumeração de frames confidenciais.

---

## 10. Canary Data

Linhas canary são injetadas no payload antes da cifragem (Nível 1) ou no DataFrame entregue ao usuário após a decifração (Nível 2, insider threat).

**Identificação:**
- Coluna oculta `__canary_sig__` presente em todas as linhas.
- Linhas reais: `__canary_sig__ = null`.
- Linhas canary: `__canary_sig__ = HMAC-SHA256(CANARY_SALT, "{pipeline_id}:{i}")[:16]`.

Remoção O(n) via filtro de nulos, sem inspecionar nenhum valor de dado.

**Geração de valores:** estatisticamente plausíveis para não distorcer distribuições. Valores canary em campos PII usam sentinels reconhecíveis (`canary.{fingerprint16}@datalock.internal`), rastreáveis em dumps de breach.

**Metadados no header (Nível 1):**

```json
{
  "canary": {
    "pipeline_id":   "crm_jan2025",
    "n_canary_rows": 3,
    "n_strata":      5,
    "fingerprints":  ["1ba472d8e3f9c0a1", "..."],
    "injected_at":   "2025-01-15T14:32:00Z",
    "original_rows": 10000
  }
}
```

**Distribuição estratificada:** as linhas são distribuídas em `n_strata` regiões do DataFrame usando posições derivadas dos fingerprints (não `random_state=42`). Qualquer corte parcial (head/tail 50%) conterá pelo menos uma linha canary.

**Segurança:** o `CANARY_SALT` deve ser configurado via `DATALOCK_CANARY_SALT` (env var) ou `dd.configure(canary_salt=...)`. O valor padrão hardcoded é público; com ele, um adversário que conheça o `pipeline_id` pode pré-calcular os fingerprints e remover as linhas antes de um vazamento.

---

## 11. Expiração de Arquivo

O campo `expires_at` (ISO 8601 UTC) no header define a data de expiração do arquivo. Ao ler um arquivo expirado, a biblioteca lança `ExpiredFileError` antes de decifrar o payload.

```python
from datalock.secure_file import ExpiredFileError
try:
    df = SecureFile.load("arquivo.dlk", key=KEY)
except ExpiredFileError as e:
    print(e)  # "Arquivo expirado em 2025-12-31. LGPD Art. 16..."
```

Alinhado com o princípio de limitação de retenção da LGPD (Art. 16) e do GDPR (Art. 5(1)(e)).

---

## 12. Retrocompatibilidade

| Versão lida | Versão da biblioteca | Suporte |
|-------------|---------------------|---------|
| v1 (0x01)   | qualquer            | ✓ leitura total |
| v2 (0x02)   | ≥ 1.3.2             | ✓ leitura e escrita |
| v3 (0x03)   | ≥ 1.5.0             | ✓ leitura e escrita |
| v4 (0x04)   | ≥ 1.1.2             | ✓ leitura e escrita |

Arquivos `.lgs` (formato anterior ao rename) são lidos identicamente via detecção automática.

Retrocompatibilidade do payload:

| Magic marker interno | Suporte |
|----------------------|---------|
| `b"IPC1\x00"` Arrow IPC | ✓ padrão atual |
| `b"PQ1\x00"` Parquet    | ✓ leitura retrocompat |
| sem marker (Arrow IPC legado) | ✓ leitura retrocompat |

---

## 13. Considerações de Segurança

**master_key e HKDF**
O HKDF-SHA256 não é um KDF de senhas (sem stretching). A `master_key` deve ser material de alta entropia: saída de KMS, `secrets.token_hex(32)`, ou vault corporativo. Para chaves derivadas de senhas humanas, aplique Argon2id antes de passar ao `pack()`.

**Reutilização de nonce**
Reutilizar o mesmo nonce com a mesma DEK em AES-GCM é catastrófico: expõe a XOR das plaintexts e permite forjar auth_tags. O formato gera `os.urandom(12)` por operação. Nunca forneça nonces manualmente.

**Limite de mensagens por chave (AES-GCM)**
Com nonces aleatórios de 96 bits, a probabilidade de colisão após `2^32` mensagens sob a mesma DEK chega a 50% (birthday bound). Dado que o salt_kdf é único por arquivo e produz uma DEK única, este limite é irrelevante na prática: cada arquivo usa uma DEK diferente.

**Verificação de integridade antes da decifração**
O padrão Verify-then-Decrypt é seguido rigorosamente. O HMAC de arquivo é verificado antes de qualquer decifração para prevenir ataques chosen-ciphertext. A auth_tag AEAD é verificada pelo próprio cipher antes de retornar qualquer byte.

**Limite de tamanho do header**
O campo `HEADER_CT_LEN` (uint32) é validado contra `MAX_HEADER_LEN = 1 MB` antes de qualquer alocação. Sem este limite, um arquivo malformado com `HEADER_CT_LEN = 0xFFFFFFFF` poderia causar OOM.

**Arquivo v4 e adulteração**
O HMAC do arquivo v4 usa chave pública fixa. Detecta corrupção acidental, mas não adulteração intencional (qualquer um pode recomputar o HMAC). Use v4 apenas para dados já anonimizados, nunca para dados pessoais brutos.

**Arquivo temporário e escrita paralela**
A escrita usa nome de arquivo `.tmp` com UUID4 único por chamada. Isto garante que dois processos escrevendo no mesmo arquivo destino simultaneamente não corrompam um ao outro antes do `rename()` atômico.

---

## 14. Referências Normativas

| Referência | Escopo |
|-----------|--------|
| NIST SP 800-38D | Recomendação para AES-GCM |
| NIST SP 800-175B | Diretriz para uso de padrões criptográficos |
| NIST SP 800-108r1 | KDF via PRF — separação de domínio |
| RFC 5869 | HMAC-based Key Derivation Function (HKDF) |
| RFC 8439 | ChaCha20 e Poly1305 para protocolos IETF |
| RFC 2104 | HMAC: Keyed-Hashing para autenticação |
| LGPD Art. 16 | Eliminação de dados pessoais após término do tratamento |
| GDPR Art. 5(1)(e) | Limitação de conservação |
