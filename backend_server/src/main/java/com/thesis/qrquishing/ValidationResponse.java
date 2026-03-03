package com.thesis.qrquishing;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.Map;

/**
 * Response DTO for {@code POST /validate}.
 *
 * @param verdict    Classification result: {@code "malicious"}, {@code "benign"}, or {@code "uncertain"}
 * @param confidence Probability score in the range [0.0, 1.0]
 * @param features   The extracted URL features that informed the verdict
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ValidationResponse(
        String verdict,
        double confidence,
        Map<String, Object> features
) {}
