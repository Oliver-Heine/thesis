package com.thesis.qrquishing.model.tokenizers

interface Tokenizer {
    fun tokenize(text: String): List<Int>
}