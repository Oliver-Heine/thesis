package com.thesis.qrquishing.model.ai

import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.thesis.qrquishing.model.tokenizers.SentencePieceTokenizer
import com.thesis.qrquishing.model.tokenizers.Tokenizer
import com.thesis.qrquishing.model.tokenizers.WordPieceTokenizer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.InputStreamReader
private const val VOCAB_FILE_NAME = "vocab.txt"

private const val DISTILBERT_BASE_UNCASED = "distilbert-base-uncased"
private const val TINYBERT_GENERAL_4L_312D = "huawei-noah_TinyBERT_General_4L_312D"
private const val TINYBERT_GENERAL_6L_768D = "huawei-noah_TinyBERT_General_6L_768D"
private const val ALBERT_BASE_V2 = "albert-base-v2"
private const val MOBILEBERT_UNCASED = "google_mobilebert-uncased"


object ModelProvider {
    private const val TAG = "QRQuishing"
    private const val MODEL_SEQUENCE_LENGTH = 128

    fun create(activity: AppCompatActivity, modelName: String): TFLiteClassifier {
        val tflite = loadModel(activity, modelName)
        val vocab = loadVocab(activity, modelName)

        val modelKey = modelName.replace(".tflite", "")
        val tokenizer : Tokenizer = when (modelKey) {
            DISTILBERT_BASE_UNCASED -> WordPieceTokenizer(vocab)
            TINYBERT_GENERAL_4L_312D -> WordPieceTokenizer(vocab)
            TINYBERT_GENERAL_6L_768D -> WordPieceTokenizer(vocab)
            ALBERT_BASE_V2 -> SentencePieceTokenizer(vocab)
            MOBILEBERT_UNCASED -> WordPieceTokenizer(vocab)
            else -> throw IllegalArgumentException("Unsupported model: $modelName")
        }

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

            Log.d(TAG, "Vocab loaded: ${mutableVocab.size} tokens")
            mutableVocab
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load vocab", e)
            throw e
        }
    }
}