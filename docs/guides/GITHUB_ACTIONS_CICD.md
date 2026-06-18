# 2.12 — GitHub Actions CI/CD Rehberi

> Bu maddeyi **sen** yapacaksın. Aşağısı goktugGPT'nin monorepo + multi-module
> Maven + repo-root Docker build context yapısına göre uyarlanmış tüyolar.

---

## Hedef

- Her servis için: **build + test + Docker image → GHCR (GitHub Container Registry)**
- **Path filter**: sadece değişen servis build edilir (8 servisi her push'ta
  build etmek dakikalarca sürer)
- Tag stratejisi: `sha-<short>` + `latest`
- (Opsiyonel) ArgoCD ile K8s deploy

---

## 0. Bu projeye özel 2 kritik nokta

1. **Docker build context = repo ROOT, dockerfile = `<service>/Dockerfile`.**
   Her Dockerfile parent `pom.xml` + tüm modül pom'larını kopyalıyor (Maven
   reactor için). CI'da `docker build`'i repo kökünden çalıştırmalısın:
   ```bash
   docker build -f asset-service/Dockerfile -t <img> .
   #                                              ^ context = .
   ```
2. **Yeni eklenen `config-server` modülü** tüm servis Dockerfile'larında
   `COPY config-server/pom.xml` satırına sahip. Matrix'e `config-server`'ı da ekle.

---

## 1. GHCR auth (otomatik token)

GHCR için ekstra secret gerekmez — `GITHUB_TOKEN` yeterli. Workflow'a izin ver:

```yaml
permissions:
  contents: read
  packages: write     # GHCR push için
```

## 2. Path filter — sadece değişen servisi build et

İki yaklaşım var. **Önerilen:** `dorny/paths-filter` ile değişen servisleri
tespit edip matrix'i dinamik kur.

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  packages: write

jobs:
  # ---- 1. Hangi servisler değişti? ----
  changes:
    runs-on: ubuntu-latest
    outputs:
      services: ${{ steps.filter.outputs.changes }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            config-server: ['config-server/**', 'pom.xml']
            api-gateway: ['api-gateway/**', 'pom.xml']
            identity-service: ['identity-service/**', 'pom.xml']
            conversation-service: ['conversation-service/**', 'pom.xml']
            inference-orchestrator: ['inference-orchestrator/**', 'pom.xml']
            asset-service: ['asset-service/**', 'pom.xml']
            billing-service: ['billing-service/**', 'pom.xml']
            telemetry-consumer: ['telemetry-consumer/**', 'pom.xml']
            notification-service: ['notification-service/**', 'pom.xml']

  # ---- 2. Değişen her servis için build + push ----
  build:
    needs: changes
    if: ${{ needs.changes.outputs.services != '[]' }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        service: ${{ fromJSON(needs.changes.outputs.services) }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up JDK 21
        uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '21'
          cache: maven          # ~/.m2 cache — build'i ciddi hızlandırır

      # Test'i Docker DIŞINDA çalıştır (daha hızlı, cache'li)
      - name: Build + Test (Maven)
        run: mvn -B -pl ${{ matrix.service }} -am verify

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build + Push image
        uses: docker/build-push-action@v6
        with:
          context: .                                   # ⚠️ repo root
          file: ${{ matrix.service }}/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/${{ matrix.service }}:latest
            ghcr.io/${{ github.repository_owner }}/${{ matrix.service }}:sha-${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

> **`-pl <service> -am`**: sadece o modülü + bağımlı olduğu modülleri (parent)
> build eder. `verify` testleri de çalıştırır (`package` çalıştırmaz). 2.13'teki
> Testcontainers testleri burada koşar — runner'da Docker var, Testcontainers çalışır.

## 3. Python servisleri (llm-server, guardrail, router)

Bunlar Maven değil. Ayrı bir job veya path-filter dalı ekle:

```yaml
            llm-server: ['llm-server/**']
            guardrail-service: ['guardrail-service/**']
            router-service: ['router-service/**']
```

Build adımı `docker/build-push-action` ile aynı, sadece test adımı `pytest`:

```yaml
      - run: pip install -r requirements.txt && pytest
```

> **llm-server uyarısı:** PyTorch CPU wheel'i ~200MB indirir. CI cache'i
> (`actions/cache` → pip + `~/.cache/torch`) ekle, yoksa her build yavaş.

## 4. Tag stratejisi

| Tag | Ne zaman | Kullanım |
|---|---|---|
| `sha-<full-sha>` | Her build | Immutable, rollback için |
| `latest` | main branch | Dev cluster auto-pull |
| `v1.2.3` | Git tag push | Prod release (ayrı workflow) |

Semantic version release için ayrı workflow:
```yaml
on:
  push:
    tags: ['v*']
```

## 5. ArgoCD ile deploy (opsiyonel — 2.11 Helm'e bağlı)

CI sadece image push eder. Deploy'u **GitOps** ile ayır:

1. CI image'ı GHCR'a push eder + Helm `values.yaml`'daki `image.tag`'i yeni
   sha ile günceller (ayrı bir "config repo" commit'i — senin `config-repo`
   pattern'ine benzer).
2. ArgoCD o repo'yu izler, değişiklik görünce cluster'a sync eder.

```bash
argocd app create goktug \
  --repo https://github.com/<you>/goktugGPT \
  --path infra/helm \
  --dest-namespace goktug \
  --sync-policy automated
```

> **Sıra önemli:** 2.11 (Helm) → 2.12 (CI/CD image push) → ArgoCD deploy.
> Helm chart'ları olmadan ArgoCD'nin sync edeceği bir şey yok.

---

## Kontrol listesi

- [ ] `.github/workflows/ci.yml` push'ta tetikleniyor
- [ ] Path filter çalışıyor — sadece değişen servis build oluyor (Actions log'da gör)
- [ ] Maven `~/.m2` cache hit veriyor (2. build belirgin hızlı)
- [ ] GHCR'da image'lar görünüyor (`ghcr.io/<you>/<service>`)
- [ ] `context: .` (repo root) — build context hatası yok
- [ ] config-server matrix'te var
- [ ] Testcontainers testleri runner'da geçiyor (Docker-in-runner)
- [ ] PR'da test fail → merge bloklanıyor (branch protection rule)

## Sık karşılaşılan hatalar

| Belirti | Sebep | Çözüm |
|---|---|---|
| `parent pom not found` | build context yanlış | `context: .` repo kökü olmalı |
| GHCR `denied: permission` | `packages: write` yok | `permissions:` bloğunu ekle |
| Her push tüm servisleri build ediyor | `pom.xml` filtresi çok geniş | parent pom değişikliğini ayrı düşün |
| Testcontainers `Could not find Docker` | runner'da Docker yok | `ubuntu-latest` kullan (Docker dahil) |
| Maven her seferinde her şeyi indiriyor | cache yok | `setup-java` içinde `cache: maven` |
