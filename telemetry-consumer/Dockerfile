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

COPY telemetry-consumer/src ./telemetry-consumer/src

RUN mvn -B -pl telemetry-consumer -am -DskipTests package

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
ADD https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/download/v2.7.0/opentelemetry-javaagent.jar /opt/otel/otel.jar
COPY --from=builder /build/telemetry-consumer/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java","-javaagent:/opt/otel/otel.jar","-jar","/app/app.jar"]
