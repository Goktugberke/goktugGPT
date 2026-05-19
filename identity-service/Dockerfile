# Build context: repository root (see docker-compose.yml)
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /build
RUN apk add --no-cache maven

COPY pom.xml ./pom.xml
COPY api-gateway/pom.xml ./api-gateway/pom.xml
COPY identity-service/pom.xml ./identity-service/pom.xml
COPY conversation-service/pom.xml ./conversation-service/pom.xml
COPY inference-orchestrator/pom.xml ./inference-orchestrator/pom.xml
COPY asset-service/pom.xml ./asset-service/pom.xml
COPY billing-service/pom.xml ./billing-service/pom.xml
COPY telemetry-consumer/pom.xml ./telemetry-consumer/pom.xml
COPY notification-service/pom.xml ./notification-service/pom.xml

COPY identity-service/src ./identity-service/src

RUN mvn -B -pl identity-service -am -DskipTests package

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
# OpenTelemetry javaagent KASITLI olarak kaldırıldı — agent OkHttp span exporter
# thread pool'u Reactor Netty event loop'una blokladığı için register/login
# Mono.block() sonsuza dek hung kalıyordu. Tracing'e gerçekten ihtiyaç olduğunda
# tekrar eklenir, ama tek bir JVM-wide flag ile değil, dedicated bir setup ile.
COPY --from=builder /build/identity-service/target/*.jar app.jar
EXPOSE 8081
ENTRYPOINT ["java","-jar","/app/app.jar"]
