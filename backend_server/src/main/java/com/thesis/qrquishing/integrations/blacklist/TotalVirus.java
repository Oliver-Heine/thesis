package com.thesis.qrquishing.integrations.blacklist;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

@Component
public class TotalVirus {

    @Value("${vt.api.key:}")
    private String vtApiKey;

    Logger log = LoggerFactory.getLogger(TotalVirus.class);

    /**
     * Query the VirusTotal URL report API.
     *
     * @return {@code true} if any engine flags the URL as malicious
     */
    public boolean checkUrl(String url) {
        if (vtApiKey == null || vtApiKey.isBlank()) {
            log.debug("VT_API_KEY not set — skipping VirusTotal check");
            return false;
        }

        try {
            // VT v3: encode URL as base64url (no padding) for the endpoint
            String encoded = java.util.Base64.getUrlEncoder()
                    .withoutPadding()
                    .encodeToString(url.getBytes(java.nio.charset.StandardCharsets.UTF_8));

            HttpClient client = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(5))
                    .build();

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://www.virustotal.com/api/v3/urls/" + encoded))
                    .header("x-apikey", vtApiKey)
                    .timeout(Duration.ofSeconds(10))
                    .GET()
                    .build();

            HttpResponse<String> resp = client.send(request, HttpResponse.BodyHandlers.ofString());

            if (resp.statusCode() == 200) {
                String body = resp.body();
                // Quick heuristic: "malicious" count > 0 in stats
                return body.contains("\"malicious\":") && !body.contains("\"malicious\":0");
            }
        } catch (Exception e) {
            log.warn("VirusTotal check failed: {}", e.getMessage());
        }
        return false;
    }
}
