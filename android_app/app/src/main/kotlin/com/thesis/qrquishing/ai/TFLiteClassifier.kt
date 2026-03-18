package com.thesis.qrquishing.ai

import android.util.Log
import com.thesis.qrquishing.domain.UrlAnalyzer
import com.thesis.qrquishing.domain.Verdict
import org.tensorflow.lite.Interpreter
import kotlin.math.exp

class TFLiteClassifier(
    private val tflite: Interpreter,
    private val tokenizer: Tokenizer,
    vocab: Map<String, Int>) {

    companion object {
        private const val TAG = "QRQuishing"
        private const val CONFIDENCE_THRESHOLD = 0.80f
        private const val MODEL_SEQUENCE_LENGTH = 128
    }

    private val clsId = vocab["[CLS]"]?.toLong() ?: 101L
    private val sepId = vocab["[SEP]"]?.toLong() ?: 102L
    private val padId = vocab["[PAD]"]?.toLong() ?: 0L

    private val urlAnalyzer = UrlAnalyzer()

    fun classify(url: String): Pair<Verdict, Float> = try {
        val input = prepareInput(url)
        val logits = runInference(input)
        val (pBenign, pMalicious) = softmax(logits)
        decideVerdict(pBenign, pMalicious)

    } catch (e: Exception) {
        Log.e(TAG, "Inference error", e)
        Verdict.UNCERTAIN to 0f
    }

    private fun prepareInput(url: String): ModelInput {
        val normalized = urlAnalyzer.normalize(url)
        val tokens = tokenizer.tokenize(normalized).take(MODEL_SEQUENCE_LENGTH - 2)

        val inputIds = LongArray(MODEL_SEQUENCE_LENGTH) { padId }
        val attentionMask = LongArray(MODEL_SEQUENCE_LENGTH)

        inputIds[0] = clsId
        attentionMask[0] = 1L

        tokens.forEachIndexed { i, tokenId ->
            inputIds[i + 1] = tokenId.toLong()
            attentionMask[i + 1] = 1L
        }

        val sepIndex = tokens.size + 1
        if (sepIndex < MODEL_SEQUENCE_LENGTH) {
            inputIds[sepIndex] = sepId
            attentionMask[sepIndex] = 1L
        }

        return ModelInput(inputIds, attentionMask)
    }

    private fun runInference(input: ModelInput): FloatArray {
        val inputIdsBatch = arrayOf(input.inputIds)
        val attentionMaskBatch = arrayOf(input.attentionMask)

        val inputs = Array(tflite.inputTensorCount) { i ->
            val name = tflite.getInputTensor(i).name().lowercase()
            when {
                "input_ids" in name -> inputIdsBatch
                "attention_mask" in name -> attentionMaskBatch
                else -> throw IllegalStateException("Unexpected input tensor: $name")
            }
        }

        val outputBuffer = Array(1) { FloatArray(2) }  // batch=1, 2 classes
        tflite.runForMultipleInputsOutputs(inputs, mapOf(0 to outputBuffer))

        return outputBuffer[0]
    }

    private fun softmax(logits: FloatArray): Pair<Float, Float> {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp((it - maxLogit).toDouble()).toFloat() }
        val sum = exps.sum()
        return exps[0] / sum to exps[1] / sum
    }

    private fun decideVerdict(pBenign: Float, pMalicious: Float): Pair<Verdict, Float> = when {
        pMalicious >= CONFIDENCE_THRESHOLD -> Verdict.MALICIOUS to pMalicious
        pBenign >= CONFIDENCE_THRESHOLD -> Verdict.BENIGN to pBenign
        else -> Verdict.UNCERTAIN to pMalicious
    }
}