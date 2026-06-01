# Manual do Formato `.dlk` — Datalock

## O problema que motivou o formato

Quando um cientista de dados salva um DataFrame com dados pessoais em disco, o resultado é quase sempre um CSV ou Parquet. Esses formatos foram projetados para performance e interoperabilidade — segurança não fazia parte do projeto. Um CSV com CPFs e rendas de clientes é, do ponto de vista do sistema operacional, idêntico a qualquer arquivo de texto: pode ser aberto com um editor, copiado sem autenticação, transmitido por e-mail sem controle e modificado sem que ninguém perceba.

Ferramentas de infraestrutura como S3 SSE ou disco criptografado protegem os dados dentro de um ambiente específico. Mas quando o arquivo sai desse ambiente — para o notebook de um analista, um pendrive, um colaborador externo — a proteção desaparece junto.

O `.dlk` nasceu de uma pergunta: **é possível criar um formato de arquivo que carregue suas próprias garantias de segurança, independente de onde esteja armazenado?**

---

## O que o `.dlk` oferece que os formatos existentes não oferecem

| Propriedade | CSV | Parquet | GPG | Parquet+S3 SSE | `.dlk` |
|---|:---:|:---:|:---:|:---:|:---:|
| Confidencialidade | ✗ | ✗ | ✓ | ✓ (infra) | ✓ |
| Integridade autenticada | ✗ | ✗ | ✓ | ✗ | ✓ |
| Metadados auditáveis sem decifrar | ✗ | ✗ | ✗ | ✗ | ✓ |
| Expiração nativa | ✗ | ✗ | ✗ | ✗ | ✓ |
| Portabilidade (independente de infra) | ✓ | ✓ | ✓ | ✗ | ✓ |
| Canary Data forense | ✗ | ✗ | ✗ | ✗ | ✓ |
| Controle de acesso por frame | ✗ | ✗ | ✗ | ✗ | ✓ |

---

## A estrutura binária

Um arquivo `.dlk` é uma sequência de blocos com posições e tamanhos fixos ou prefixados por comprimento. A versão v2 (single-frame) tem esta estrutura:

```
[5  bytes]  MAGIC           = b"DLOCK"
[1  byte ]  VERSION         = 0x02
[1  byte ]  CIPHER          = 0x01 (AES-256-GCM) | 0x02 (ChaCha20-Poly1305)
[32 bytes]  SALT_KDF        — salt aleatório único por arquivo (os.urandom(32))
[12 bytes]  NONCE_HEADER    — nonce para cifragem do cabeçalho
[4  bytes]  HEADER_CT_LEN   — tamanho do bloco de cabeçalho cifrado
[N+16 bytes] HEADER_CT+TAG  — cabeçalho JSON cifrado com HEK + tag GCM
[12 bytes]  NONCE_PAYLOAD   — nonce para cifragem do payload
[M+16 bytes] PAYLOAD_CT+TAG — Arrow IPC/zstd cifrado com DEK + tag GCM
[32 bytes]  FILE_HMAC       — HMAC-SHA256 (via MAK) sobre todos os bytes anteriores
```

**Versões do formato:**

| Versão | Byte | Descrição |
|---|---|---|
| v1 | `0x01` | Header em claro — legado, leitura retrocompat |
| v2 | `0x02` | Header cifrado com HEK — single-frame atual |
| v3 | `0x03` | Multi-frame — payload é ZIP com um frame por arquivo |
| v4 | `0x04` | Sem criptografia — dados em claro, HMAC com chave fixa |

**Serialização interna do payload:** Arrow IPC com compressão zstd ou lz4, identificado pelo magic marker `b"IPC1\x00"` nos primeiros 5 bytes do payload decifrado. Retrocompatibilidade com arquivos mais antigos (`b"PQ1\x00"` para Parquet v1.3, sem marker para Arrow IPC v1.0-1.2) é mantida por detecção automática do marker.

Arrow IPC foi escolhido sobre Parquet porque o payload é sempre decifrado integralmente antes de qualquer operação — o overhead analítico do Parquet (row groups, column chunks, footer de predicados) é irrelevante nesse contexto. Arrow IPC é 3-5× mais rápido na serialização em memória com tamanho comparável.

---

## A hierarquia de chaves: DEK, HEK e MAK

A master key fornecida pelo usuário nunca é usada diretamente como chave de cifragem. Todas as chaves operacionais são derivadas via **HKDF-SHA256** (RFC 5869) com o SALT_KDF único por arquivo:

```
master_key + SALT_KDF  ──HKDF──►  DEK   info="datalock-dek-v1"       cifra o payload
                                ►  HEK   info="datalock-hek-v1"       cifra o cabeçalho
                                ►  MAK   info="datalock-mak-v1"       autentica o arquivo
                                ►  DEK_admin  info="datalock-frame-dek-v1:admin"
                                ►  DEK_user   info="datalock-frame-dek-v1:user"
                                ►  ...   (uma por nível de acesso, em arquivos ACL)
```

O campo `info` distinto por chave é o mecanismo de **separação de domínio criptográfico**: mesmo com master key e salt idênticos, cada derivação produz uma chave completamente diferente. Conhecer a DEK não revela nada sobre a HEK ou a MAK. Comprometer um arquivo não enfraquece nenhum outro, porque cada arquivo tem SALT_KDF gerado com `os.urandom(32)` independentemente.

**Por que isso importa:** a versão original do código usava a master key diretamente como chave HMAC para o FILE_HMAC — o mesmo material de chave alimentando tanto o HKDF quanto o HMAC. Isso viola o princípio de separação de domínio do NIST SP 800-108r1 e da RFC 5869. A correção introduziu a MAK como terceira chave derivada, especificamente para autenticação do arquivo.

---

## A cifragem: AES-256-GCM e autenticação integrada

O modo GCM é classificado como AEAD (Authenticated Encryption with Associated Data — NIST SP 800-38D). Ele cifra e autentica em uma única operação, produzindo uma tag de 128 bits que cobre o ciphertext.

A diferença fundamental em relação a modos como AES-CBC: no GCM, a tag é verificada **antes** de qualquer byte do plaintext ser retornado. Qualquer alteração no ciphertext — um bit flipado por erro de transmissão ou por adulteração intencional — invalida a tag e a decifração falha imediatamente, sem expor nada.

ChaCha20-Poly1305 (RFC 8439) é selecionado automaticamente em hardware sem suporte a instruções AES-NI, via benchmark de 1MB na primeira utilização.

---

## O FILE_HMAC e a proteção do layout

As tags GCM protegem o conteúdo de cada bloco cifrado. O FILE_HMAC nos últimos 32 bytes protege a estrutura do arquivo como um todo — incluindo os campos em claro: magic number, byte de versão, byte de cipher, SALT_KDF, nonces e tamanho do cabeçalho.

Calculado com a MAK sobre todos os bytes anteriores do arquivo, o HMAC é a **primeira** verificação na leitura, antes de qualquer tentativa de decifração. Arquivo com HMAC inválido não chega nem ao passo de decifrar o cabeçalho.

Os dois níveis são complementares: GCM detecta adulteração dentro de cada bloco; HMAC detecta reordenação, truncamento ou substituição de blocos e adulteração dos campos em claro.

---

## O cabeçalho e a auditabilidade sem exposição

O cabeçalho JSON é cifrado com a HEK — bloco independente do payload com seu próprio nonce e tag GCM. Contém:

```json
{
  "format":               "lgs",
  "version":              "2.1",
  "content_type":         "masked_dataframe",
  "label":                "exportacao-crm-jan26",
  "created_at":           "2026-01-15T10:30:00+00:00",
  "created_by":           "datalock/1.0.1",
  "shape":                [100000, 11],
  "schema":               {"cpf": "Utf8", "renda_mensal": "Float64"},
  "column_stats":         {"renda_mensal": {"n_nulls": 0, "n_unique": 8743}},
  "masking_applied":      true,
  "compression":          "ipc_zstd",
  "kdf":                  "HKDF-SHA256-v2",
  "encryption":           "AES256GCM",
  "integrity":            "HMAC-SHA256+MAK",
  "expires_at":           "2027-01-15T00:00:00+00:00",
  "pipeline_id":          "crm-jan-2026"
}
```

**Nota:** o campo `compression` agora usa o prefixo `ipc_` (ex: `"ipc_zstd"`, `"ipc_lz4"`) para distinguir arquivos Arrow IPC (v1.4+) de arquivos Parquet legados (`"parquet_zstd"`) e zlib externos (`"zlib"`).

**Nota:** `column_stats` inclui `min`/`max` apenas para `masked_dataframe` — para `raw_dataframe`, esses campos são suprimidos para evitar exposição de distribuições de dados sensíveis via cabeçalho.

A função `verify()` decifra apenas o cabeçalho, sem tocar no payload. Isso permite que um auditor verifique schema, shape, timestamps e expiração sem que nenhum dado pessoal seja carregado em memória — propriedade sem equivalente em qualquer formato tabular amplamente utilizado.

---

## A expiração nativa

O campo `expires_at` é verificado imediatamente após a decifração do cabeçalho. Arquivo expirado levanta `ExpiredFileError` antes de qualquer tentativa de processar o payload:

```
ExpiredFileError: Arquivo expirado em 2025-12-31.
Conforme a política de retenção LGPD (Art. 16),
dados após o prazo devem ser eliminados.
```

Campo `expires_at` mal-formatado emite `UserWarning` em vez de falhar silenciosamente — comportamento correto para uma verificação de segurança crítica.

---

## O suporte multi-frame (v3)

No v3, o payload cifrado com a DEK é um arquivo ZIP em memória. Dentro do ZIP há um arquivo Arrow IPC por DataFrame mais um `index.json` com nomes, tamanhos, shapes e schemas. A estrutura binária externa é idêntica ao v2. O ZIP existe apenas em memória — nunca em disco.

---

## Controle de acesso por frame (ACL)

Extensão do v3 onde cada frame é cifrado com uma DEK derivada do seu nível de acesso:

```
HKDF(master_key, salt, info="datalock-frame-dek-v1:restricted") → DEK_restricted
HKDF(master_key, salt, info="datalock-frame-dek-v1:internal")   → DEK_internal
```

Um usuário com acesso ao nível `internal` não pode calcular `DEK_restricted` — não há operação matemática que permita isso sem a master key. Frames não autorizados são omitidos silenciosamente da leitura (o sistema não revela sua existência).

**Caso de uso:** pipeline de RAG corporativo onde o modelo de linguagem nunca pode ver dados de RH ou financeiros ao atender usuários comuns — não por controle lógico contornável, mas porque as chaves para decifrar esses frames não estão disponíveis no contexto de execução.

---

## O Audit Report com assinatura HMAC

O `AuditReport.save(path, audit_key=KEY)` calcula `HMAC-SHA256` sobre o JSON canônico do relatório e armazena no campo `_integrity`. Um relatório de compliance adulterado é detectado por `AuditReport.verify_signature(path, audit_key)`. Sem `audit_key`, o campo recebe `"unsigned"` e um `UserWarning` é emitido.

---

## Canary Data — rastreamento forense em dois níveis

### Nível 1 — arquivo cifrado (adversário externo)

Linhas-armadilha com fingerprints HMAC únicos são injetadas no DataFrame antes da cifragem, distribuídas em 5 regiões estratificadas. Qualquer corte parcial do dataset (head ou tail de 20% a 80%) contém pelo menos uma linha canary. Na leitura, essas linhas são removidas silenciosamente.

### Nível 2 — DataFrame decifrado (insider threat)

Novas linhas canary são injetadas no DataFrame entregue ao usuário após a leitura. Se o usuário exportar como CSV ou banco de dados, os fingerprints acompanham a exportação. O arquivo `.dlk` em disco não é modificado. O `pipeline_id` do Nível 2 é diferenciado: `"{original}:read:{session_id[:8]}"`.

### Geração estatisticamente válida

| Tipo de coluna | Estratégia |
|---|---|
| PII (email, cpf) | Sentinel recognizável: `canary.{fp}@datalock.internal` |
| Numérica | Percentil derivado do fingerprint + perturbação < 0,5% do std |
| Categórica | Moda da coluna (não cria categorias novas) |
| Booleana | `False` |
| Data | Mediana da coluna |

PSI observado nos experimentos: < 0,02 em todos os cenários — impacto estatisticamente negligenciável (limiar convencional: PSI < 0,10).

---

## Dataset Watermarking para corpora de IA

Quando o `.dlk` armazena corpora de treino (`content_type="training_corpus"`), três técnicas de watermarking rastreável são disponíveis:

**Factual canary (texto):** fatos sintéticos plausíveis mas verificadamente falsos são injetados como sentenças em documentos selecionados. Um modelo treinado nesse corpus reproduzirá esses fatos com confiança anormalmente alta — detectável via `verify_text_watermark()`.

**Lexical n-gram:** sequências de palavras linguisticamente naturais mas estatisticamente raras. Detectáveis via log-likelihood anormal no modelo suspeito.

**Unicode marker:** caractere `⟦DL⟧` inserido em posição derivada do fingerprint — invisível na renderização HTML/PDF mas presente no tokenizer.

**Watermark vetorial:** para embeddings pré-computados, perturbação < 0,1% do L2-norm derivada deterministicamente do fingerprint. Similaridade cosseno preservada > 0,9999 — busca semântica não é afetada. Detectável por probe linear treinado na assinatura (Sablayrolles et al., ICML 2020 — "Radioactive Data").

O proprietário legítimo remove os watermarks com `strip_text_watermarks()` antes do uso próprio. Apenas o corpus exportado (potencialmente roubado) carrega as assinaturas.

---

## Referência completa de funções

### Escrita

| Função | Descrição |
|---|---|
| `SecureFile.pack(src, dst, key)` | Arquivo (CSV/Parquet) → `.dlk` cifrado |
| `SecureFile.pack_dataframe(df, dst, key)` | DataFrame em memória → `.dlk` cifrado |
| `SecureFile.pack_bytes(payload, dst, key)` | Bytes arbitrários → `.dlk` cifrado |
| `SecureFile.pack_open(df, dst)` | DataFrame → `.dlk` sem criptografia (v4) |
| `SecureFile.pack_frames(frames, dst, key)` | Multi-frame sem ACL (todos com mesma DEK) |
| `SecureFile.pack_frames_acl(frames, dst, key, levels)` | Multi-frame com DEK por nível de acesso |

### Leitura

| Função | Descrição |
|---|---|
| `SecureFile.load(path, key)` | Decifra, remove canary L1, injeta canary L2 |
| `SecureFile.load_raw(path, key)` | Decifra sem mascaramento adicional, sem canary L2 |
| `SecureFile.load_bytes(path, key)` | Lê payload binário |
| `SecureFile.load_open(path)` | Lê arquivo sem criptografia |
| `SecureFile.load_frames(path, key)` | Todos os frames (sem ACL) |
| `SecureFile.load_frame(path, key, frame=)` | Frame único (sem ACL) |
| `SecureFile.load_frames_acl(path, key, allowed_levels=)` | Frames autorizados pelo nível |
| `SecureFile.load_frame_acl(path, key, frame=, allowed_levels=)` | Frame único com verificação ACL |

### Verificação e auditoria

| Função | Descrição |
|---|---|
| `SecureFile.verify(path, key)` | Verifica integridade + retorna metadados sem payload |
| `AuditReport.save(path, audit_key=)` | Salva relatório com assinatura HMAC |
| `AuditReport.verify_signature(path, audit_key)` | Verifica integridade do relatório |

### Canary Data

| Função | Descrição |
|---|---|
| `canary.inject_canary(df, pipeline_id)` | Injeção L1 (para o arquivo) |
| `canary.inject_canary_on_read(df, pipeline_id)` | Injeção L2 (para o DataFrame entregue) |
| `canary.strip_canary(df)` | Remove linhas canary e a coluna `__canary_sig__` |
| `canary.canary_check(token)` | Busca fingerprint no manifesto local |
| `canary.inject_text_watermarks(texts, corpus_id)` | Watermark em corpus de texto |
| `canary.strip_text_watermarks(texts, meta)` | Remove watermarks (uso legítimo) |
| `canary.verify_text_watermark(output, meta)` | Detecta uso em modelo suspeito |
| `canary.inject_embedding_watermark(vecs, corpus_id)` | Watermark vetorial em embeddings |

---

## O que o formato não faz

**Zeroização de memória:** Python não garante quando objetos são coletados pelo GC. O plaintext existe na heap durante a desserialização.

**Eliminação física:** `expires_at` impede a leitura pela biblioteca, mas não apaga o arquivo do disco.

**Rotação de chave sem re-cifragem:** `rekey()` decifra e re-cifra o arquivo completo. Não existe mecanismo de envelope de chave.

**Anonimização formal:** o Datalock implementa pseudonimização — reversível com o salt. Conforme o Guia ANPD 2024, o dado pseudonimizado permanece sob o escopo da LGPD.

**Prova jurídica autônoma:** o watermarking produz evidência estatística forense. Para uso judicial, combine com hash do corpus registrado em cartório digital.
