package com.thesis.qrquishing.integrations.ai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.genai.Client;
import com.google.genai.types.GenerateContentResponse;
import com.thesis.qrquishing.UrlFeatures;
import com.thesis.qrquishing.ValidationResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.Objects;

/**
 * Calls the OpenAI Chat Completions API to classify a URL given its
 * extracted features.
 *
 * <h3>Security design</h3>
 * <ul>
 *   <li>The OpenAI API key is read from {@code OPENAI_API_KEY} (env or .env)
 *       — never hardcoded or logged.</li>
 *   <li>URL features are serialised to a JSON object and embedded inside the
 *       user message as a structured value, <em>not</em> interpolated as raw
 *       strings.  This prevents prompt-injection via adversarial URL content.</li>
 *   <li>The model is instructed to respond <em>only</em> with a JSON object;
 *       the response is parsed strictly — any deviation is treated as an error.</li>
 * </ul>
 */
@Component
public class GeminiAnalyzer implements AIAnalyzer{

    private static final Logger log = LoggerFactory.getLogger(GeminiAnalyzer.class);

    private static final String SYSTEM_PROMPT =
            "You are a cybersecurity analyst specialising in phishing and quishing (QR phishing) detection. "
            + "You will be given a JSON object containing features extracted from a URL. "
            + "Analyse these features and determine whether the URL is malicious, benign, or uncertain. "
            + "Respond ONLY with a valid JSON object in exactly this format (no markdown, no extra text): "
            + "{\"verdict\": \"malicious\"|\"benign\"|\"uncertain\", "
            + "\"confidence\": <float 0.0-1.0>, "
            + "\"reasoning\": \"<one sentence explanation>\"}";

    private final ObjectMapper objectMapper = new ObjectMapper();

    // Optional: when missing, we degrade to "uncertain" rather than failing startup.

    @Value("${GEMINI_API_KEY:}")
    private String geminiApiKey;

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Analyse a URL and its features using the LLM.
     *
     * @param url      The original URL string (used for logging only)
     * @param features Extracted features
     * @return {@link ValidationResponse} with verdict, confidence, and features
     * @throws LlmException if the API call fails or returns an unparseable response
     */
    public ValidationResponse analyze(String url, UrlFeatures  features) {
        if (geminiApiKey == null || geminiApiKey.isBlank()) {
            log.warn("GEMINI_API_KEY not set — returning uncertain verdict");
            return new ValidationResponse("uncertain", 0.0, features.features());
        }

        String userContent = buildUserMessage(features);
        log.info("Calling Gemini with user message: {}", userContent);

        try {

            Client client = Client.builder()
                    .apiKey(geminiApiKey)
                    .build();

            GenerateContentResponse response =
                    client.models.generateContent(
                            "gemini-3-flash-preview",
                            userContent,
                            null);

            if (response.candidates().isEmpty() || Objects.requireNonNull(response.text()).isEmpty()) {
                throw new LlmException("Gemini response empty");
            }

            return parseGeminiResponse(response, features);

        } catch (LlmException e) {
            throw e;
        } catch (Exception e) {
            throw new LlmException("LLM request failed: " + e.getMessage(), e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Build the user message as a JSON string containing the feature object.
     *
     * <p>Features are serialised via Jackson, so special characters in URL
     * fields are properly escaped and cannot break the JSON structure or
     * inject additional prompt instructions.
     */
    private String buildUserMessage(UrlFeatures  features) {
        try {
            return "Analyse the following URL features and classify the URL:\n"
                    + objectMapper.writeValueAsString(features)
                    + "\n\n" + SYSTEM_PROMPT;
        } catch (Exception e) {
            throw new LlmException("Failed to serialise features", e);
        }
    }

    private ValidationResponse parseGeminiResponse(GenerateContentResponse responseBody, UrlFeatures features) {
        try {
            JsonNode result = objectMapper.readTree(responseBody.text());
            String verdict = result.path("verdict").asText("uncertain").toLowerCase();
            double confidence = result.path("confidence").asDouble(0.0);

            // Clamp confidence to [0, 1]
            confidence = Math.max(0.0, Math.min(1.0, confidence));

            if (!verdict.equals("malicious") && !verdict.equals("benign") && !verdict.equals("uncertain")) {
                log.warn("Unexpected verdict from LLM: {}; defaulting to 'uncertain'", verdict);
                verdict = "uncertain";
            }

            return new ValidationResponse(verdict, confidence, features.features());

        } catch (Exception e) {
            throw new OpenaiAnalyzer.LlmException("Failed to parse LLM response: " + e.getMessage(), e);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Exception type
    // ─────────────────────────────────────────────────────────────────────────

    /** Thrown when the LLM API call fails or returns an invalid response. */
    public static class LlmException extends RuntimeException {
        public LlmException(String message) {
            super(message);
        }

        public LlmException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
