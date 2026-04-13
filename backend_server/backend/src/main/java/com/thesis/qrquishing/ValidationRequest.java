package com.thesis.qrquishing;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/**
 * Request DTO for {@code POST /validate}.
 *
 * @param url The URL extracted from a QR code. Must be a non-blank
 *            http or https URL, maximum 2048 characters.
 */
public record ValidationRequest(

        @NotBlank(message = "url must not be blank")
        @Size(max = 2048, message = "url must not exceed 2048 characters")
        @Pattern(
                regexp = "^https?://.*",
                message = "url must start with http:// or https://"
        )
        String url

) {}
