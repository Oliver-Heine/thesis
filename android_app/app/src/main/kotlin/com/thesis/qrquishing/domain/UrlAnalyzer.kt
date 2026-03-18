package com.thesis.qrquishing.domain

import java.net.URI

class UrlAnalyzer {

    /**
     * Normalizes URL to match the requested structure:
     * "< tag > value < tag > value ..."
     */
    fun normalize(url: String): String {
        val cleanUrl = url.lowercase().trim()
            .replace(Regex("^https?://"), "")
            .replace(Regex("^www\\."), "")

        return try {
            val uri = URI("http://$cleanUrl")
            val host = uri.host ?: ""
            val hostParts = host.split(".").filter { it.isNotEmpty() }

            val subdomainParts = if (hostParts.size > 2) hostParts.dropLast(2) else emptyList()
            val domainPart = hostParts.getOrNull(hostParts.size - 2) ?: hostParts.getOrNull(0) ?: ""
            val suffixParts = hostParts.takeLast(1)

            val tokens = mutableListOf<String>()

            fun addSection(label: String, parts: List<String>) {
                if (parts.isNotEmpty()) {
                    tokens.addAll(listOf("<", label, ">"))
                    tokens.addAll(parts)
                }
            }

            addSection("subdomain", subdomainParts)
            addSection("domain", listOf(domainPart).filter { it.isNotEmpty() })
            addSection("suffix", suffixParts)

            uri.path
                ?.takeIf { it != "/" }
                ?.split(Regex("[/\\-_.?=&]"))
                ?.filter { it.isNotEmpty() }
                ?.let { addSection("path", it) }

            uri.query
                ?.takeIf { it.isNotBlank() }
                ?.split(Regex("[=&]"))
                ?.filter { it.isNotEmpty() }
                ?.let { addSection("query", it) }

            tokens.joinToString(" ")
        } catch (e: Exception) {
            cleanUrl
        }
    }
}