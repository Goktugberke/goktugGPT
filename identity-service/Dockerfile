FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /build
COPY pom.xml .
COPY src ./src
RUN apk add --no-cache maven && mvn -B -DskipTests package

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app

# OpenTelemetry Java agent — auto instrumentation, kod değişikliği gerektirmez.
ADD https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/download/v2.7.0/opentelemetry-javaagent.jar /opt/otel/otel.jar

COPY --from=builder /build/target/*.jar app.jar
EXPOSE 8081
ENTRYPOINT ["java","-javaagent:/opt/otel/otel.jar","-jar","/app/app.jar"]
