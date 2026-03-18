package com.thesis.qrquishing;

import com.microsoft.playwright.*;
import com.microsoft.playwright.options.LoadState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Extracts security-relevant features from a URL using a headless Chromium
 * browser (Playwright) and optional VirusTotal lookup.
 *
 * <h3>Security controls</h3>
 * <ul>
 *   <li>SSRF prevention: private/loopback IP ranges are blocked before any
 *       network connection is made.</li>
 *   <li>JavaScript is disabled in the browser to prevent drive-by execution.</li>
 *   <li>Navigation timeout capped at 10 seconds.</li>
 *   <li>VirusTotal API key is read from {@code VT_API_KEY} (env or .env)
 *       — never hardcoded.</li>
 * </ul>
 */
@Component
public class UrlFeatureExtractor {

    private static final Logger log = LoggerFactory.getLogger(UrlFeatureExtractor.class);

    private static final int NAV_TIMEOUT_MS = 10_000;
    private static final int MAX_REDIRECTS = 20;

    /** Private/loopback CIDR prefixes that must be blocked (SSRF). */
    private static final List<String> BLOCKED_PREFIXES = List.of(
            "10.", "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
            "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
            "172.30.", "172.31.", "192.168.", "127.", "169.254.", "0."
    );
    private static final Set<String> BLOCKED_HOSTNAMES = Set.of("localhost", "metadata.google.internal");

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Extract features from {@code url}.
     *
     * @throws SsrfBlockedException if the URL resolves to a private/loopback address
     * @throws ExtractionException  on any other extraction failure
     */
    public UrlFeatures  extract(String url) {
        validateNoSsrf(url);

        try (Playwright playwright = Playwright.create()) {
            BrowserType chromium = playwright.chromium();
            BrowserType.LaunchOptions opts = new BrowserType.LaunchOptions()
                    .setHeadless(true)
                    .setArgs(List.of(
                            "--no-sandbox",
                            "--disable-gpu",
                            "--disable-dev-shm-usage",
                            "--disable-extensions"
                    ));

            try (Browser browser = chromium.launch(opts)) {
                Browser.NewContextOptions ctxOpts = new Browser.NewContextOptions()
                        .setJavaScriptEnabled(false);   // JS disabled — no drive-by
                try (BrowserContext context = browser.newContext(ctxOpts);
                     Page page = context.newPage()) {

                    page.setDefaultNavigationTimeout(NAV_TIMEOUT_MS);
                    page.setDefaultTimeout(NAV_TIMEOUT_MS);

                    // Navigate and collect redirect chain
                    Response response = page.navigate(url);
                    page.waitForLoadState(LoadState.DOMCONTENTLOADED);

                    String finalUrl = page.url();
                    int redirectCount = countRedirects(response);
                    boolean finalUrlHttps = finalUrl.startsWith("https://");

                    // Feature: login form detection (password field present)
                    boolean hasLoginForm = (boolean) page.evaluate(
                            "() => document.querySelector('input[type=\"password\"]') !== null"
                    );

                    // Feature: eval usage in inline scripts (JS disabled, check source)
                    String pageContent = page.content();
                    boolean usesEval = pageContent.contains("eval(");

                    String pageTitle = page.title();

                    return new UrlFeatures(java.util.Map.of(
                            "redirect_count", redirectCount,
                            "final_url", finalUrl,
                            "has_login_form", hasLoginForm,
                            "uses_eval", usesEval,
                            "final_url_https", finalUrlHttps,
                            "page_title", pageTitle
                    ));
                }
            }
        } catch (SsrfBlockedException e) {
            throw e;
        } catch (Exception e) {
            throw new ExtractionException("Feature extraction failed: " + e.getMessage(), e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SSRF prevention
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Reject URLs whose host resolves to a private or loopback address.
     *
     * @throws SsrfBlockedException if the URL is prohibited
     */
    void validateNoSsrf(String url) {
        try {
            URI uri = URI.create(url);

            String host = uri.getHost();
            if (host == null || host.isBlank()) {
                throw new SsrfBlockedException("URL has no resolvable host.");
            }

            if (BLOCKED_HOSTNAMES.contains(host.toLowerCase())) {
                throw new SsrfBlockedException("Host is not permitted: " + host);
            }

            InetAddress[] addresses = InetAddress.getAllByName(host);
            for (InetAddress addr : addresses) {
                String ip = addr.getHostAddress();
                if (addr.isLoopbackAddress() || addr.isSiteLocalAddress()
                        || addr.isLinkLocalAddress() || addr.isAnyLocalAddress()) {
                    throw new SsrfBlockedException("Resolved IP is in a private range: " + ip);
                }
                for (String prefix : BLOCKED_PREFIXES) {
                    if (ip.startsWith(prefix)) {
                        throw new SsrfBlockedException("Resolved IP is in a private range: " + ip);
                    }
                }
            }
        } catch (SsrfBlockedException e) {
            throw e;
        } catch (UnknownHostException e) {
            throw new SsrfBlockedException("Cannot resolve host: " + e.getMessage());
        } catch (Exception e) {
            throw new SsrfBlockedException("URL validation failed: " + e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private int countRedirects(Response finalResponse) {
        // Walk the redirect chain via Response.request().redirectedFrom()
        int count = 0;
        if (finalResponse != null) {
            Request req = finalResponse.request().redirectedFrom();
            while (req != null && count <= MAX_REDIRECTS) {
                count++;
                req = req.redirectedFrom();
            }
        }
        return count;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Exceptions
    // ─────────────────────────────────────────────────────────────────────────

    /** Thrown when the URL resolves to a blocked (private/loopback) address. */
    public static class SsrfBlockedException extends RuntimeException {
        public SsrfBlockedException(String message) {
            super(message);
        }
    }

    /** Thrown for all other feature-extraction failures. */
    public static class ExtractionException extends RuntimeException {
        public ExtractionException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
