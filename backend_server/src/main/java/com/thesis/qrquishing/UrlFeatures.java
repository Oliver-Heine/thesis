package com.thesis.qrquishing;

/**
 * Immutable value object holding security-relevant features extracted from a URL.
 *
 * @param redirectCount  Number of HTTP redirects observed before reaching the final URL
 * @param finalUrl       The URL after all redirects have been followed
 * @param hasLoginForm   {@code true} if the page contains a password input field
 * @param usesEval       {@code true} if the page source contains {@code eval(} calls
 * @param finalUrlHttps  {@code true} if the final URL uses HTTPS
 * @param vtFlag         {@code true} if VirusTotal flagged the URL as malicious
 * @param pageTitle      The HTML {@code <title>} of the landing page
 */
public record UrlFeatures(
        int redirectCount,
        String finalUrl,
        boolean hasLoginForm,
        boolean usesEval,
        boolean finalUrlHttps,
        boolean vtFlag,
        String pageTitle
) {}
