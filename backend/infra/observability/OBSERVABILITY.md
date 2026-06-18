# Observability Stack

> Distributed tracing, metrics ve logs için tek bir standart: **OpenTelemetry**.

## Mimari

```
┌─────────────────┐      OTLP/gRPC     ┌──────────┐
│ Service (Java)  │ ─────────────────▶ │  Jaeger  │ traces
│  + OTel agent   │                    └──────────┘
└────────┬────────┘
         │ /actuator/prometheus
         ▼
┌────────────────┐                     ┌──────────────┐
│  Prometheus    │ ──────────────────▶ │   Grafana    │ dashboards
└────────────────┘                     └──────────────┘

┌─────────────────┐      OTLP          ┌──────────┐    ┌──────────┐
│ Service (Py)    │ ─────────────────▶ │  Jaeger  │    │ Kibana   │ logs
│ otel-instrument │                    └──────────┘    └────┬─────┘
└─────────────────┘                                         │
                                                            │
        Filebeat → Logstash → Elasticsearch ────────────────┘
```

## Java Servisleri — OTel Agent (Auto Instrumentation)

Önerilen yaklaşım: **OpenTelemetry Java Agent** kullan. Kod değişikliği gerektirmez,
servisin JAR'ını çalıştırırken `-javaagent` flag'i ile yükle.

### Dockerfile patch (her Java servisi için aynı):

```dockerfile
# OTel agent download (build stage'inde tek seferlik)
ADD https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/latest/download/opentelemetry-javaagent.jar /opt/otel/otel.jar

# CMD'yi şuna güncelle:
ENTRYPOINT ["java","-javaagent:/opt/otel/otel.jar","-jar","/app/app.jar"]
```

### Environment variables (docker-compose'da her servis):

```yaml
environment:
  OTEL_SERVICE_NAME: identity-service       # her servis için farklı
  OTEL_RESOURCE_ATTRIBUTES: "service.namespace=goktug,service.version=0.1.0,deployment.environment=dev"
  OTEL_EXPORTER_OTLP_ENDPOINT: http://jaeger:4317
  OTEL_EXPORTER_OTLP_PROTOCOL: grpc
  OTEL_TRACES_EXPORTER: otlp
  OTEL_METRICS_EXPORTER: none               # metrics Prometheus'tan, OTel'den değil
  OTEL_LOGS_EXPORTER: none                  # logs ELK üzerinden
  OTEL_INSTRUMENTATION_KAFKA_ENABLED: "true"
  OTEL_INSTRUMENTATION_SPRING_KAFKA_ENABLED: "true"
  OTEL_INSTRUMENTATION_R2DBC_ENABLED: "true"
  OTEL_INSTRUMENTATION_JDBC_ENABLED: "true"
  OTEL_INSTRUMENTATION_REACTOR_NETTY_ENABLED: "true"
  # Sampling — production'da 0.05 (5%), dev'de 1.0 (her trace)
  OTEL_TRACES_SAMPLER: parentbased_traceidratio
  OTEL_TRACES_SAMPLER_ARG: "1.0"
```

Agent şu kütüphaneleri otomatik instrument eder:
- Servlet (Tomcat) + Spring WebFlux (Netty)
- JDBC + R2DBC + Hibernate
- Spring Kafka (producer + consumer)
- WebClient + RestTemplate
- Resilience4j

→ Tek satır kod yazmadan distributed trace çıkar.

## Python Servisleri — OTel SDK (Manual Instrumentation)

```bash
pip install opentelemetry-distro opentelemetry-exporter-otlp \
            opentelemetry-instrumentation-fastapi
```

Run command:
```bash
opentelemetry-instrument \
  --traces_exporter otlp \
  --service_name inference-worker-goktug \
  --exporter_otlp_endpoint http://jaeger:4317 \
  uvicorn app.main:app --host 0.0.0.0 --port 9001
```

Veya `app/main.py` başında:

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(app)
```

## Trace Context Propagation

Gateway → downstream akışı `traceparent` (W3C standard) header'ı ile devam eder.
Spring Cloud Gateway + Spring Webflux WebClient + RestTemplate auto-propagate.
Spring Kafka producer/consumer da otomatik (bağlam header'lara konur, consumer
tarafında okunur).

→ Sonuç: bir kullanıcı isteği `api-gateway` → `inference-orchestrator` → `guardrail-service`
→ `inference-worker-goktug` → `inference.completed.v1` (Kafka) → `conversation-service`
zincirinin **tek bir trace** olarak Jaeger'da görünür.

## Metrics — Prometheus + Grafana

Java servisleri Micrometer + `micrometer-registry-prometheus` ile `/actuator/prometheus`
endpoint expose eder. `prometheus.yml` zaten her servisi scrape ediyor (bkz: prometheus.yml).

Python servisleri `prometheus_client` paketi ile `/metrics` expose eder.

### Önerilen Grafana dashboard'ları (Faz 2'de eklenecek):

1. **Service health overview:** her servisin uptime, request rate, error rate, p95 latency
2. **Saga performance:** her saga step'inin p50/p95 süresi, failure rate
3. **Inference worker:** GPU utilization, tokens/sec, queue depth
4. **Kafka consumer lag:** her consumer group için lag
5. **Database:** Postgres connection pool, slow queries

## Logs — ELK Stack

Spring servisleri `spring.logging.pattern.level: "%5p [traceId=%X{traceId:-} spanId=%X{spanId:-}]"`
zaten konfigli — log line'larında trace ID görünüyor.

Filebeat (Faz 2 — şu an docker-compose'da yok) container log'ları toplar →
Logstash parse eder → Elasticsearch'e yazar → Kibana'da `traceId` ile arama yapılabilir.

→ Sonuç: Jaeger'da problem gör, traceId'yi Kibana'ya yapıştır, ilgili log line'larını gör.
