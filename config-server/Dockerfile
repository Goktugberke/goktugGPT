# Build context: repository root (see docker-compose.yml)
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /build
RUN apk add --no-cache maven

COPY pom.xml ./pom.xml
COPY config-server/pom.xml ./config-server/pom.xml
COPY api-gateway/pom.xml ./api-gateway/pom.xml
COPY identity-service/pom.xml ./identity-service/pom.xml
COPY conversation-service/pom.xml ./conversation-service/pom.xml
COPY inference-orchestrator/pom.xml ./inference-orchestrator/pom.xml
COPY asset-service/pom.xml ./asset-service/pom.xml
COPY billing-service/pom.xml ./billing-service/pom.xml
COPY telemetry-consumer/pom.xml ./telemetry-consumer/pom.xml
COPY notification-service/pom.xml ./notification-service/pom.xml

COPY config-server/src ./config-server/src

RUN mvn -B -pl config-server -am -DskipTests package

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
# Git client gerek değil — file:// veya HTTPS üzerinden çalışıyoruz.
COPY --from=builder /build/config-server/target/*.jar app.jar
EXPOSE 8888
ENTRYPOINT ["java","-jar","/app/app.jar"]
