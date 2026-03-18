# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build the application
# ─────────────────────────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jdk-alpine AS builder

WORKDIR /build

# Copy dependency descriptor first for layer caching
COPY pom.xml .
COPY .mvn/ .mvn/
COPY mvn .

# Pre-fetch dependencies (fails gracefully if mvnw is absent — also supports
# plain `mvn` invocation in CI)
RUN chmod +x mvn && ./mvn dependency:go-offline -q 2>/dev/null || true

# Copy source and build the fat JAR
COPY src/ src/
RUN ./mvn package -DskipTests -q

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Minimal runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jre-alpine

# Non-root user for principle of least privilege
RUN addgroup -S appgroup && adduser -S appuser -G appgroup -u 1000

WORKDIR /app

# Copy only the fat JAR from the build stage
COPY --from=builder /build/target/*.jar app.jar

# Install Playwright Chromium dependencies (needed at runtime)
# Playwright downloads browsers to ~/.cache/ms-playwright at first run;
# we pre-install them here to avoid delays at startup.
RUN apk add --no-cache \
        chromium \
        nss \
        freetype \
        harfbuzz \
        ca-certificates \
        ttf-freefont \
    && rm -rf /var/cache/apk/*

# Tell Playwright to use the system Chromium instead of downloading its own
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/bin
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Drop to non-root
USER appuser:appgroup

EXPOSE 8080

# Tune JVM for container environments
ENTRYPOINT ["java", \
    "-XX:+UseContainerSupport", \
    "-XX:MaxRAMPercentage=75.0", \
    "-Djava.security.egd=file:/dev/./urandom", \
    "-jar", "app.jar"]
