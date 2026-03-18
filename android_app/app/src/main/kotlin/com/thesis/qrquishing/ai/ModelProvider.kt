package com.thesis.qrquishing.ai

import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader

object ModelProvider {
    private const val TAG = "QRQuishing"
    private const val MODEL_SEQUENCE_LENGTH = 128

    fun create(activity: AppCompatActivity): TFLiteClassifier {
        val tflite = loadModel(activity)
        val vocab = loadVocab(activity)
        val tokenizer = Tokenizer(vocab)
        return TFLiteClassifier(tflite, tokenizer, vocab)
    }

    private fun loadModel(activity: AppCompatActivity): Interpreter {
        return try {
            val modelBuffer = FileUtil.loadMappedFile(activity, "distilbert_model.tflite")
            val options = Interpreter.Options().apply { numThreads = 2 }
            val tflite = Interpreter(modelBuffer, options)

            // Resize inputs to match the expected sequence length
            val inputShape = intArrayOf(1, MODEL_SEQUENCE_LENGTH)
            for (i in 0 until tflite.inputTensorCount) {
                tflite.resizeInput(i, inputShape)
            }
            tflite.allocateTensors()

            // Log input/output info
            for (i in 0 until tflite.inputTensorCount) {
                val t = tflite.getInputTensor(i)
                Log.i(TAG, "Input[$i]: name=${t.name()}, shape=${t.shape().contentToString()}, dtype=${t.dataType()}")
            }
            for (i in 0 until tflite.outputTensorCount) {
                val t = tflite.getOutputTensor(i)
                Log.i(TAG, "Output[$i]: name=${t.name()}, shape=${t.shape().contentToString()}, dtype=${t.dataType()}")
            }

            Log.i(TAG, "TFLite model loaded and resized to $MODEL_SEQUENCE_LENGTH")
            tflite
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model", e)
            throw e
        }
    }

    private fun loadVocab(activity: AppCompatActivity): Map<String, Int>{
        return try {
            val mutableVocab = mutableMapOf<String, Int>()
            activity.assets.open("vocab.txt").use { stream ->
                BufferedReader(InputStreamReader(stream))
                    .lineSequence()
                    .forEachIndexed { idx, line ->
                        val token = line.trim()
                        if (token.isNotEmpty()) mutableVocab[token] = idx
                    }
            }

            val required = listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<", ">")
            val missing = required.filter { !mutableVocab.containsKey(it) }
            if (missing.isNotEmpty()) {
                throw IllegalStateException("vocab.txt is missing required tokens: $missing")
            }

            Log.i(TAG, "Vocab loaded: ${mutableVocab.size} tokens")
            mutableVocab
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocab", e)
            throw e
        }
    }
}