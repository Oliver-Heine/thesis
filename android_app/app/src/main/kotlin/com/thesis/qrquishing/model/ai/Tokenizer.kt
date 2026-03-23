package com.thesis.qrquishing.model.ai

/**
 * Standard WordPiece tokenizer.
 */
class Tokenizer(private val vocab: Map<String, Int>) {
    private val unknownId = vocab["[UNK]"] ?: 100
    private val prefix = "##"

    fun tokenize(text: String): List<Int> {
        val tokens = mutableListOf<Int>()

        // Split by whitespace
        for (word in text.split(Regex("\\s+")).filter { it.isNotEmpty() }) {
            var start = 0
            while (start < word.length) {
                var end = word.length
                var matchedId: Int? = null

                // Try the longest substring first
                while (start < end) {
                    val subword = if (start == 0) word.substring(start, end) else prefix + word.substring(start, end)
                    matchedId = vocab[subword]
                    if (matchedId != null) break
                    end--
                }

                if (matchedId == null) {
                    // No match: unknown token
                    tokens.add(unknownId)
                    break
                } else {
                    tokens.add(matchedId)
                    start = end
                }
            }
        }

        return tokens
    }
}
