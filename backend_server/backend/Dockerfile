# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build the application
# ─────────────────────────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jdk-jammy AS builder

WORKDIR /build

# Maven wrapper needs curl/wget to bootstrap Maven
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency descriptor first for layer caching
COPY pom.xml .
COPY .mvn/ .mvn/
COPY mvnw .
COPY /src/main/resources/.env ./.env

# Pre-fetch dependencies (fails gracefully if mvnw is absent — also supports
# plain `mvn` invocation in CI)
RUN chmod +x mvn && ./mvn dependency:go-offline -q 2>/dev/null || true

# Copy source and build the fat JAR
COPY src/ src/
RUN ./mvnw package -DskipTests -q

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Minimal runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jre-jammy

# Non-root user for principle of least privilege
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 -m appuser

WORKDIR /app

# Install Playwright browsers + system deps at build time
COPY pom.xml /tmp/pw/
COPY .mvn/ /tmp/pw/.mvn/
COPY mvnw /tmp/pw/
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && chmod +x /tmp/pw/mvnw \
    && (cd /tmp/pw && ./mvnw -q -DskipTests exec:java -e \
        -Dexec.mainClass=com.microsoft.playwright.CLI \
        -Dexec.args="install --with-deps chromium") \
    && rm -rf /var/lib/apt/lists/* /tmp/pw /root/.m2

# Copy only the fat JAR from the build stage
COPY --from=builder /build/target/*.jar app.jar

# Playwright runtime settings
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

# Drop to non-root
USER appuser:appgroup

EXPOSE 8082

# Tune JVM for container environments
ENTRYPOINT ["java", \
    "-XX:+UseContainerSupport", \
    "-XX:MaxRAMPercentage=75.0", \
    "-Duser.home=/tmp", \
    "-Djava.security.egd=file:/dev/./urandom", \
    "-jar", "app.jar"]
