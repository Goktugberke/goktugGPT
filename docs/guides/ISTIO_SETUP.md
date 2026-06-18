# 2.8 — Service Mesh (Istio) Kurulum Rehberi

> Bu maddeyi **sen** yapacaksın. Aşağısı adım adım tüyolar + bu projeye özel
> dikkat noktaları. goktugGPT'nin mimarisine (SSE streaming, gateway'in JWT→header
> filtresi, 10+ servis) göre uyarlandı.

---

## 0. Ön koşul: Kubernetes gerekiyor (Docker Compose yetmez)

Istio bir **sidecar mesh**'tir; her pod'a Envoy proxy enjekte eder. Bu yüzden
önce bir K8s cluster lazım. Lokal için seçenekler:

| Seçenek | Komut | Not |
|---|---|---|
| **Docker Desktop K8s** | Settings → Kubernetes → Enable | En kolay, zaten Docker var |
| **kind** | `kind create cluster --name goktug` | Hafif, CI'a uygun |
| **minikube** | `minikube start --cpus 4 --memory 8192` | Istio için min 4 CPU / 8GB |

> goktugGPT şu an Docker Compose ile çalışıyor. Istio'ya geçmek = K8s manifest'lerine
> (veya 2.11'deki Helm chart'larına) geçmek demek. **Bu yüzden 2.11 Helm Charts'ı
> Istio'dan ÖNCE bitirmek mantıklı** — Istio'yu chart'ların üstüne kurarsın.

---

## 1. istioctl kurulumu

```bash
# Windows (PowerShell) — en güncel sürüm
winget install Istio.istioctl
# veya manuel:
#   https://github.com/istio/istio/releases → istioctl.exe PATH'e koy

istioctl version
```

## 2. Istio'yu cluster'a kur

`demo` profili lokal/dev için ideal (Kiali, Jaeger, Prometheus, Grafana dahil
gelir — 2.6 tracing ile çakışmaması için aşağıdaki nota bak):

```bash
istioctl install --set profile=demo -y

# Doğrula
kubectl get pods -n istio-system
```

> **2.6 ile çakışma:** Istio demo profili kendi Jaeger'ını kurar. Sen zaten
> docker-compose'da Jaeger çalıştırıyorsun. K8s'e geçtiğinde ya Istio'nun
> Jaeger'ını kullan, ya da `--set values.global.tracer.zipkin.address=<kendi-jaeger>`
> ile kendi collector'ına yönlendir. İkisini birden çalıştırma.

## 3. Sidecar injection'ı aç

```bash
kubectl create namespace goktug
kubectl label namespace goktug istio-injection=enabled

# Bu label'dan sonra deploy edilen her pod'a otomatik Envoy sidecar eklenir.
# Mevcut pod'lar için: kubectl rollout restart deployment -n goktug
```

## 4. mTLS — otomatik (STRICT mode)

Servisler arası tüm trafik otomatik şifrelenir. `goktug` namespace'ine STRICT
mTLS uygula:

```yaml
# infra/istio/peer-authentication.yaml
apiVersion: security.istio.io/v1
kind: PeerAuthentication
metadata:
  name: default
  namespace: goktug
spec:
  mtls:
    mode: STRICT
```

```bash
kubectl apply -f infra/istio/peer-authentication.yaml
```

> **Kazanç:** Faz 3'te planlanan "servisler arası mTLS" maddesi bununla çözülür.
> `JwtToHeadersFilter` (api-gateway) hâlâ JWT'yi doğrular; Istio bunun ÜSTÜNE
> transport-level mTLS ekler. İkisi farklı katman, çakışmaz.

## 5. Ingress Gateway — api-gateway'i dışarı aç

```yaml
# infra/istio/gateway.yaml
apiVersion: networking.istio.io/v1
kind: Gateway
metadata:
  name: goktug-gateway
  namespace: goktug
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts: ["*"]
---
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: api-gateway-vs
  namespace: goktug
spec:
  hosts: ["*"]
  gateways: ["goktug-gateway"]
  http:
    - match:
        - uri:
            prefix: /api/
      route:
        - destination:
            host: api-gateway        # K8s Service adı
            port:
              number: 8080
      # ⚠️ SSE STREAMING İÇİN KRİTİK — aşağıdaki nota bak
      timeout: 0s                    # /api/v1/inference için timeout'u KAPAT
```

### ⚠️ SSE / Token Streaming Tuzağı (bu projeye özel)

`inference-orchestrator` token'ları Server-Sent Events ile stream ediyor.
Istio/Envoy default'ları streaming'i bozar:

1. **`timeout: 0s`** — default 15s timeout uzun stream'i keser. Inference
   route'unda sıfırla.
2. **Envoy buffering** — Envoy response'u buffer'larsa token'lar tek tek
   gelmez, sonda toplu düşer. SSE route'una şu annotation'ı ekle:
   ```yaml
   # DestinationRule veya route üzerinde
   # Envoy zaten SSE (text/event-stream) için flush eder, ama emin olmak için
   # idle timeout'u da uzat:
   ```
3. En temizi: inference path'ini ayrı bir `VirtualService` match'i yap, sadece
   ona `timeout: 0s` ver, diğer route'lar 15s'de kalsın.

```yaml
  http:
    - match:
        - uri:
            prefix: /api/v1/inference   # SSE — timeout yok
      timeout: 0s
      route:
        - destination: { host: api-gateway, port: { number: 8080 } }
    - match:
        - uri:
            prefix: /api/                # geri kalan — normal timeout
      route:
        - destination: { host: api-gateway, port: { number: 8080 } }
```

## 6. Canary Deployment — Traffic Split

En anlamlı kullanım: **yeni model versiyonunu** (örn `goktug-pro-v2`) trafiğin
%10'una ver. `inference-orchestrator` veya `llm-server` için:

```yaml
# infra/istio/canary.yaml
apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata:
  name: llm-server-dr
  namespace: goktug
spec:
  host: llm-server
  subsets:
    - name: v1
      labels: { version: v1 }
    - name: v2
      labels: { version: v2 }
---
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: llm-server-vs
  namespace: goktug
spec:
  hosts: ["llm-server"]
  http:
    - route:
        - destination: { host: llm-server, subset: v1 }
          weight: 90
        - destination: { host: llm-server, subset: v2 }
          weight: 10
```

Deployment'larına `version: v1` / `version: v2` label'ı koyman gerekir. Trafiği
kademeli kaydır: 90/10 → 50/50 → 0/100.

## 7. Gözlemlenebilirlik

```bash
istioctl dashboard kiali      # servis grafiği, trafik akışı, mTLS durumu
istioctl dashboard jaeger     # distributed trace (2.6 ile aynı veri)
```

Kiali'de Saga akışını (gateway → orchestrator → guardrail → router → llm-server
→ billing) görsel graf olarak görürsün — sunum/CV için harika ekran görüntüsü.

---

## Kontrol listesi

- [ ] K8s cluster ayakta (`kubectl get nodes`)
- [ ] `istioctl install` başarılı
- [ ] `goktug` namespace'i `istio-injection=enabled` label'lı
- [ ] STRICT mTLS uygulandı, Kiali'de kilit ikonları yeşil
- [ ] api-gateway ingress'ten erişilebilir
- [ ] **inference route'unda `timeout: 0s` — SSE bozulmuyor** (smoke test [4] geçiyor)
- [ ] Canary 90/10 split çalışıyor
- [ ] Kiali graph + Jaeger trace görünüyor

## Sık karşılaşılan hatalar

| Belirti | Sebep | Çözüm |
|---|---|---|
| SSE token'ları sonda toplu geliyor | Envoy buffering / timeout | inference route `timeout: 0s` |
| Servisler birbirini görmüyor | mTLS STRICT ama bir serviste sidecar yok | `kubectl get pod -o wide`, sidecar var mı bak |
| `503 UC` hataları | Pod hazır değilken trafik | readinessProbe ekle (Helm chart'ında) |
| Keycloak token reddi | Saat kayması / iss mismatch | mevcut `jwk-set-uri` ayarı K8s'te de geçerli olmalı |
