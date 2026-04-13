package com.thesis.qrquishing;

import com.thesis.qrquishing.integrations.ai.AIAnalyzer;
import com.thesis.qrquishing.integrations.ai.GeminiAnalyzer;
import com.thesis.qrquishing.integrations.ai.OpenaiAnalyzer;
import com.thesis.qrquishing.integrations.blacklist.TotalVirus;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * REST controller exposing the {@code POST /validate} endpoint.
 *
 * <p>Flow:
 * <ol>
 *   <li>Input validation (bean-validation + URL sanity checks)</li>
 *   <li>Per-IP rate limiting (10 requests / 60 s, in-memory)</li>
 *   <li>Feature extraction via {@link UrlFeatureExtractor} (Playwright)</li>
 *   <li>LLM analysis via {@link OpenaiAnalyzer} (OpenAI)</li>
 * </ol>
 */
@RestController
@RequestMapping("/validate")
public class UrlValidatorController {

    private static final Logger log = LoggerFactory.getLogger(UrlValidatorController.class);

    private static final int RATE_LIMIT = 10;
    private static final long RATE_WINDOW_MS = 60_000L;

    private final UrlFeatureExtractor featureExtractor;
    private final AIAnalyzer aiAnalyzer;
    private final TotalVirus totalVirus;

    /** Per-IP sliding-window counters: IP → [count, windowStartEpochMs] */
    private final ConcurrentHashMap<String, long[]> rateLimitMap = new ConcurrentHashMap<>();

    public UrlValidatorController(UrlFeatureExtractor featureExtractor, GeminiAnalyzer aiAnalyzer, TotalVirus totalVirus) {
        this.featureExtractor = featureExtractor;
        this.aiAnalyzer = aiAnalyzer;
        this.totalVirus = totalVirus;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // POST /validate
    // ─────────────────────────────────────────────────────────────────────────

    @PostMapping
    public ResponseEntity<?> validate(
            @Valid @RequestBody ValidationRequest request,
            jakarta.servlet.http.HttpServletRequest httpRequest) {

        String clientIp = resolveClientIp(httpRequest);

        if (isRateLimited(clientIp)) {
            log.warn("Rate limit exceeded for IP: {}", clientIp);
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .body(Map.of("error", "Rate limit exceeded. Try again later."));
        }

        String url = request.url().strip();
        log.info("Validating URL from {}: {}", clientIp, url);

//        if (totalVirus.checkUrl(url)) {
//            log.warn("VirusTotal check failed for {}: URL is malicious", url);
//            return ResponseEntity.ok(new ValidationResponse("malicious", 1.0, Map.of()));
//        }

        try {
            UrlFeatures features = featureExtractor.extract(url);
            ValidationResponse response = aiAnalyzer.analyze(url, features);
            log.info("Result for {}: verdict={}, confidence={}", url, response.verdict(), response.confidence());
            return ResponseEntity.ok(response);

        } catch (UrlFeatureExtractor.SsrfBlockedException e) {
            log.warn("SSRF attempt blocked: {}", url);
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("error", "URL target is not permitted."));

        } catch (UrlFeatureExtractor.ExtractionException e) {
            log.error("Feature extraction failed for {}: {}", url, e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(Map.of("error", "Could not retrieve URL features."));

        } catch (OpenaiAnalyzer.LlmException e) {
            log.error("LLM analysis failed: {}", e.getMessage());
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Analysis service temporarily unavailable."));

        } catch (Exception e) {
            log.error("Unexpected error during validation", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Internal server error."));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Extract the real client IP, honouring the {@code X-Forwarded-For} header
     * that may be set by a reverse proxy.
     */
    private String resolveClientIp(jakarta.servlet.http.HttpServletRequest req) {
        String xff = req.getHeader("X-Forwarded-For");
        if (xff != null && !xff.isBlank()) {
            // Take the leftmost (original client) address
            return xff.split(",")[0].strip();
        }
        return req.getRemoteAddr();
    }

    /**
     * Simple fixed-window in-memory rate limiter.
     *
     * @param ip Client IP address
     * @return {@code true} if the request should be rejected
     */
    private boolean isRateLimited(String ip) {
        long now = Instant.now().toEpochMilli();
        long[] state = rateLimitMap.compute(ip, (k, v) -> {
            if (v == null || now - v[1] >= RATE_WINDOW_MS) {
                return new long[]{1L, now};
            }
            v[0]++;
            return v;
        });
        return state[0] > RATE_LIMIT;
    }

    /**
     * Periodically evict stale rate-limit entries (windows older than
     * {@code RATE_WINDOW_MS}) to prevent unbounded memory growth.
     * Runs every minute.
     */
    @Scheduled(fixedDelay = 60_000L)
    void evictStaleRateLimitEntries() {
        long cutoff = Instant.now().toEpochMilli() - RATE_WINDOW_MS;
        rateLimitMap.entrySet().removeIf(e -> e.getValue()[1] < cutoff);
    }
}
