# goktug-platform — Helm Chart

goktugGPT uygulama servislerini (8 Java + 3 Python) Kubernetes'e deploy eden
**tek umbrella chart**.

## Neden 11 ayrı chart değil de tek chart?

11 servis büyük ölçüde aynı şekle sahip (Deployment + Service + probe + HPA).
11 ayrı chart = 11 kez kopyalanmış aynı template = bakım kâbusu ve junior sinyali.
Bunun yerine generic template'ler `values.yaml`'daki `services` map'i üzerinde
döner (`range`). Yeni servis eklemek = map'e ~6 satır. Tek `helm upgrade` ile
hepsi güncellenir.

> Gerçekten servis-başına bağımsız release isteyen ekipler için alternatif
> "library chart + subchart" desenidir; bu projenin ölçeğinde umbrella daha temiz.

## Kapsam

Bu chart **uygulama servislerini** deploy eder. Altyapı (PostgreSQL, Kafka,
Keycloak, Redis, MinIO, Elasticsearch) cluster'da **dışarıdan** sağlanır
(operator / managed service / ayrı chart) ve `sharedEnv` + Secret üzerinden
bağlanır. Sebep: stateful altyapıyı uygulama chart'ına gömmek prod'da
anti-pattern'dir (DB'yi `helm upgrade` ile yönetmek istemezsin).

## Dosya yapısı

```
goktug-platform/
  Chart.yaml
  values.yaml          # taban: services map + defaults + sharedEnv
  values-dev.yaml      # 1 replica, düşük kaynak, HPA kapalı
  values-prod.yaml     # 2+ replica, HPA açık, TLS, immutable sha tag
  templates/
    _helpers.tpl       # label/image/healthPath helper'ları
    configmap.yaml     # sharedEnv → ConfigMap (envFrom)
    deployment.yaml    # services map'inde dönen generic Deployment
    service.yaml       # generic ClusterIP Service
    hpa.yaml           # hpa.enabled olan servisler için HPA
    ingress.yaml       # sadece public:true (api-gateway), SSE-uyumlu
```

## Kullanım

```bash
# Lint + render (cluster gerekmez, indirme yok)
helm lint ./infra/helm/goktug-platform
helm template goktug ./infra/helm/goktug-platform -f infra/helm/goktug-platform/values-dev.yaml

# Secret (DB şifreleri vb. — Faz 3'te Vault'a taşınır)
kubectl create secret generic goktug-secrets \
  --from-literal=DB_PASSWORD=... \
  --from-literal=MINIO_SECRET_KEY=... \
  --from-literal=KEYCLOAK_ADMIN_SECRET=...

# Dev
helm upgrade --install goktug ./infra/helm/goktug-platform -f infra/helm/goktug-platform/values-dev.yaml

# Prod (CI/CD immutable sha geçer)
helm upgrade --install goktug ./infra/helm/goktug-platform \
  -f infra/helm/goktug-platform/values-prod.yaml \
  --set global.imageTag=sha-$GIT_SHA
```

## Tasarım notları

- **Probe'lar:** Java servisleri `management.endpoint.health.probes.enabled=true`
  ile gelen `/actuator/health/liveness|readiness` yollarını kullanır (bu ayar
  shared config'te zaten açık). Python servisleri `healthPath` ile özelleştirilir.
- **SSE:** ingress'te `proxy-buffering: off` + `proxy-read-timeout: 600` —
  inference token stream'inin kesilmemesi için.
- **Prometheus:** Pod annotation'ları (`prometheus.io/scrape`) ile otomatik
  scrape; mevcut Prometheus kurulumuyla uyumlu.
- **Image:** `ghcr.io/goktug/<service>:<tag>` — CI/CD (GitHub Actions) buraya push eder.
- **Güvenlik:** gizli değerler Secret'tan `envFrom` ile; chart'ta plaintext yok.
```
