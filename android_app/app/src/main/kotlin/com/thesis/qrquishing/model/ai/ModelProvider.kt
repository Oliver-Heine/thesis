package com.thesis.qrquishing.model.ai

import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
private const val VOCAB_FILE_NAME = "vocab.txt"

object ModelProvider {
    private const val TAG = "QRQuishing"
    private const val MODEL_SEQUENCE_LENGTH = 128

    fun create(activity: AppCompatActivity, modelName: String): TFLiteClassifier {
        val tflite = loadModel(activity, modelName)
        val vocab = loadVocab(activity, modelName)
        val tokenizer = Tokenizer(vocab)
        return TFLiteClassifier(tflite, tokenizer, vocab)
    }

    private fun loadModel(activity: AppCompatActivity, modelName: String): Interpreter {
        return try {
            val modelPath = modelName.replace(".tflite", "")
            val modelBuffer = FileUtil.loadMappedFile(activity, "tflite/" + modelPath + "/" + modelName)
            val options = Interpreter.Options().apply { numThreads = 2 }
            val tflite = Interpreter(modelBuffer, options)

            // Resize inputs to match the expected sequence length
            val inputShape = intArrayOf(1, MODEL_SEQUENCE_LENGTH)
            for (i in 0 until tflite.inputTensorCount) {
                tflite.resizeInput(i, inputShape)
            }

            tflite.allocateTensors()
            tflite
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model", e)
            throw e
        }
    }

    private fun loadVocab(activity: AppCompatActivity, modelName: String): Map<String, Int>{
        return try {
            val modelPath = modelName.replace(".tflite", "")
            val mutableVocab = mutableMapOf<String, Int>()
            activity.assets.open("tflite/" + modelPath + "/"+ VOCAB_FILE_NAME).use { stream ->
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

            Log.d(TAG, "Vocab loaded: ${mutableVocab.size} tokens")
            mutableVocab
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocab", e)
            throw e
        }
    }
}