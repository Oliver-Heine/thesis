package com.thesis.qrquishing.model.tokenizers

import java.util.Locale

class SentencePieceTokenizer(
    private val vocab: Map<String, Int>,
    private val doLowerCase: Boolean = true
) : Tokenizer {
    private val unknownId = vocab["<unk>"] ?: 0
    private val spaceMarker = '▁'

    /**
     * Tokenize input text into token IDs.
     */
    override fun tokenize(text: String): List<Int> {
        val normalized = normalize(text)
        val pieces = mutableListOf<Int>()

        var i = 0
        while (i < normalized.length) {
            var end = normalized.length
            var matchedId: Int? = null

            while (end > i) {
                val candidate = normalized.substring(i, end)
                matchedId = vocab[candidate]
                if (matchedId != null) {
                    break
                }
                end--
            }

            if (matchedId != null) {
                pieces.add(matchedId)
                i = end
            } else {
                // No piece matched: emit <unk> and advance one character
                pieces.add(unknownId)
                i++
            }
        }

        return pieces
    }

    /**
     * Convert text into SentencePiece-like normalized stream:
     * "google.com" -> "▁google.com"
     * "<domain> google <suffix> com" -> "▁<domain>▁google▁<suffix>▁com"
     */
    private fun normalize(text: String): String {
        var s = text.trim()
        if (doLowerCase) {
            s = s.lowercase(Locale.ROOT)
        }

        // Collapse whitespace
        s = s.replace(Regex("\\s+"), " ")

        if (s.isEmpty()) return ""

        val words = s.split(" ")
        return buildString {
            for (word in words) {
                append(spaceMarker)
                append(word)
            }
        }
    }
}