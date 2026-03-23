package com.thesis.qrquishing.benchmarkrunner

import android.content.Context
data class UrlSample(val url: String, val label: Int)

class DatasetLoader(
    private val fileName: String = "benchmark_dataset.csv"
) {

    fun getTotalSamples(context: Context, skipHeader: Boolean = true): Int {
        context.assets.open(fileName).bufferedReader().use { reader ->
            var count = 0
            reader.forEachLine { line ->
                if (skipHeader && count == 0) {
                    count++ // skip header
                    return@forEachLine
                }
                count++
            }
            return if (skipHeader) count - 1 else count
        }
    }

    fun streamDatasetInChunks(
        context: Context,
        chunkSize: Int = 10_000,
        skipHeader: Boolean = true,
        maxSamples: Int? = 10000,
        onChunkLoaded: (chunk: List<UrlSample>, chunkIndex: Int) -> Unit
    ) {
        context.assets.open(fileName).bufferedReader().use { reader ->

            var chunk = ArrayList<UrlSample>(chunkSize)
            var chunkIndex = 0
            var isFirstLine = true
            var totalRead = 0

            reader.forEachLine { line ->

                // Skip header if needed
                if (skipHeader && isFirstLine) {
                    isFirstLine = false
                    return@forEachLine
                }

                if (maxSamples != null && totalRead >= maxSamples) return@forEachLine

                val sample = parseLine(line)
                if (sample != null) {
                    chunk.add(sample)
                    totalRead++
                }

                // Emit chunk when full
                if (chunk.size >= chunkSize) {
                    onChunkLoaded(chunk, chunkIndex)
                    chunk = ArrayList(chunkSize)
                    chunkIndex++
                }
            }

            // Emit remaining data
            if (chunk.isNotEmpty()) {
                onChunkLoaded(chunk, chunkIndex)
            }
        }
    }

    private fun parseLine(line: String): UrlSample? {
        val parts = line.split(",", limit = 2)

        if (parts.size < 2) return null

        val url = parts[0].trim()
        val label = parts[1].trim().toIntOrNull() ?: return null

        return UrlSample(url, label)
    }
}