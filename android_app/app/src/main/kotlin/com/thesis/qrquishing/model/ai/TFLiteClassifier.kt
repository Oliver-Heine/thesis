package com.thesis.qrquishing.model.ai

import android.util.Log
import com.thesis.qrquishing.model.UrlAnalyzer
import com.thesis.qrquishing.model.dto.Verdict
import com.thesis.qrquishing.model.tokenizers.Tokenizer
import org.tensorflow.lite.Interpreter
import kotlin.math.exp

class TFLiteClassifier(
    private val tflite: Interpreter,
    private val tokenizer: Tokenizer,
    vocab: Map<String, Int>) {

    companion object {
        private const val TAG = "QRQuishing"
        private const val CONFIDENCE_THRESHOLD = 0.90f
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
        val tokenTypeIds = LongArray(MODEL_SEQUENCE_LENGTH)

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

        return ModelInput(inputIds, attentionMask, tokenTypeIds)
    }

    private fun runInference(input: ModelInput): FloatArray {
        // TFLite runForMultipleInputsOutputs expects an Object[] for inputs
        val inputs = arrayOfNulls<Any>(tflite.inputTensorCount)

        for (i in 0 until tflite.inputTensorCount) {
            val tensor = tflite.getInputTensor(i)
            val name = tensor.name().lowercase()
            val dataType = tensor.dataType()

            // Select the appropriate source data based on tensor name
            val sourceArray = when {
                "input_ids" in name -> input.inputIds
                "attention_mask" in name -> input.attentionMask
                "token_type_ids" in name -> input.tokenTypeIds
                else -> throw IllegalArgumentException("Unexpected input tensor: $name")
            }

            // Map sourceArray (LongArray) to the data type required by the tensor
            // and wrap it in another array to match the [1, 128] shape (batch size 1)
            if (dataType == org.tensorflow.lite.DataType.INT64) {
                // long[][] is compatible with INT64 tensor of shape [1, 128]
                inputs[i] = arrayOf(sourceArray)
            } else if (dataType == org.tensorflow.lite.DataType.INT32) {
                // int[][] is compatible with INT32 tensor of shape [1, 128]
                val intArray = IntArray(sourceArray.size) { sourceArray[it].toInt() }
                inputs[i] = arrayOf(intArray)
            } else {
                throw IllegalArgumentException("Unsupported tensor type: $dataType for $name")
            }
        }

        // Output buffer for logits (2 classes: benign, malicious)
        val outputBuffer = Array(1) { FloatArray(2) }
        tflite.runForMultipleInputsOutputs(inputs, mapOf(0 to outputBuffer))

        return outputBuffer[0]
    }

    fun close() {
        try {
            tflite.close()
            Log.d(TAG, "TFLiteInterpreter closed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing TFLiteInterpreter", e)
        }
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