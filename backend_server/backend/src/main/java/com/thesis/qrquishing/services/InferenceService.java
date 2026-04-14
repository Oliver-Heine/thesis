package com.thesis.qrquishing.services;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.thesis.qrquishing.UrlFeatures;
import com.thesis.qrquishing.ValidationResponse;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class InferenceService {

    private static final Logger log = LoggerFactory.getLogger(InferenceService.class);

    private final ResourceLoader resourceLoader;
    private final String modelPath;

    private OrtEnvironment environment;
    private OrtSession session;
    private Map<String, NodeInfo> inputInfo = Collections.emptyMap();
    private List<String> outputNames = List.of();

    public InferenceService(
            @Value("${app.inference.model-path}") String modelPath,
            ResourceLoader resourceLoader
    ) {
        this.modelPath = modelPath;
        this.resourceLoader = resourceLoader;
    }

    @PostConstruct
    void init() {
        loadModel();
    }

    @PreDestroy
    void shutdown() {
        closeSession();
    }

    /**
     * Reload the ONNX model from the configured path.
     */
    public synchronized void reloadModel() {
        closeSession();
        loadModel();
    }

    /**
     * Run inference for models with a single input. Throws if the model
     * declares multiple inputs.
     */
    public Object inferSingle(Object input) {
        ensureReady();
        if (inputInfo.size() != 1) {
            throw new IllegalStateException(
                    "Model has " + inputInfo.size() + " inputs; use infer(Map) instead."
            );
        }
        String inputName = inputInfo.keySet().iterator().next();
        Map<String, Object> outputs = infer(Map.of(inputName, input));
        if (outputs.size() == 1) {
            return outputs.values().iterator().next();
        }
        return outputs;
    }

    /**
     * Run inference using extracted URL features and map outputs to a
     * {@link ValidationResponse}.
     */
    public ValidationResponse infer(UrlFeatures features) {
        ensureReady();
        if (features == null || features.features() == null || features.features().isEmpty()) {
            throw new IllegalArgumentException("Features must not be empty.");
        }

        Map<String, Object> modelInputs = buildInputsFromFeatures(features.features());
        Map<String, Object> outputs = infer(modelInputs);
        return toValidationResponse(features, outputs);
    }

    /**
     * Run inference using a map of input name to Java arrays
     * (e.g., float[], long[][]). The input names must match the model
     * signature.
     */
    public Map<String, Object> infer(Map<String, ?> inputs) {
        ensureReady();
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("Inputs must not be empty.");
        }

        for (String name : inputs.keySet()) {
            if (!inputInfo.containsKey(name)) {
                throw new IllegalArgumentException(
                        "Unknown input '" + name + "'. Expected one of " + inputInfo.keySet()
                );
            }
        }

        Map<String, OnnxTensor> onnxInputs = new HashMap<>();
        List<OnnxTensor> toClose = new ArrayList<>();

        try {
            for (Map.Entry<String, ?> entry : inputs.entrySet()) {
                Object value = entry.getValue();
                if (value == null) {
                    throw new IllegalArgumentException("Input '" + entry.getKey() + "' is null.");
                }
                OnnxTensor tensor = OnnxTensor.createTensor(environment, value);
                onnxInputs.put(entry.getKey(), tensor);
                toClose.add(tensor);
            }

            try (OrtSession.Result result = session.run(onnxInputs)) {
                Map<String, Object> outputs = new LinkedHashMap<>();
                for (Map.Entry<String, OnnxValue> entry : result) {
                    outputs.put(entry.getKey(), entry.getValue().getValue());
                }
                return outputs;
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ONNX inference failed: " + e.getMessage(), e);
        } finally {
            for (OnnxTensor tensor : toClose) {
                tensor.close();
            }
        }
    }

    /** @return the model input names for building request tensors. */
    public List<String> getInputNames() {
        return List.copyOf(inputInfo.keySet());
    }

    /** @return the model output names in execution order. */
    public List<String> getOutputNames() {
        return List.copyOf(outputNames);
    }

    private void ensureReady() {
        if (session == null) {
            throw new IllegalStateException("ONNX session not initialized.");
        }
    }

    private Map<String, Object> buildInputsFromFeatures(Map<String, Object> features) {
        Map<String, Object> inputs = new LinkedHashMap<>();
        for (Map.Entry<String, NodeInfo> entry : inputInfo.entrySet()) {
            String inputName = entry.getKey();
            if (!features.containsKey(inputName)) {
                throw new IllegalArgumentException(
                        "Missing feature for model input '" + inputName + "'. Available: " + features.keySet()
                );
            }
            Object value = features.get(inputName);
            inputs.put(inputName, coerceInputValue(value, entry.getValue()));
        }
        return inputs;
    }

    private Object coerceInputValue(Object value, NodeInfo info) {
        if (value == null) {
            throw new IllegalArgumentException("Model input value must not be null.");
        }
        if (value.getClass().isArray()) {
            return value;
        }

        OnnxJavaType type = resolveOnnxType(info);

        if (value instanceof List<?> list) {
            return coerceList(list, type);
        }

        return switch (type) {
            case STRING -> new String[]{String.valueOf(value)};
            case BOOL -> new boolean[]{toBoolean(value)};
            case INT8, UINT8 -> new byte[]{(byte) toLong(value)};
            case INT16 -> new short[]{(short) toLong(value)};
            case INT32 -> new int[]{(int) toLong(value)};
            case INT64 -> new long[]{toLong(value)};
            case DOUBLE -> new double[]{toDouble(value)};
            case FLOAT, FLOAT16, BFLOAT16, UNKNOWN -> new float[]{(float) toDouble(value)};
        };
    }

    private OnnxJavaType resolveOnnxType(NodeInfo info) {
        if (info == null || info.getInfo() == null) {
            return OnnxJavaType.UNKNOWN;
        }
        if (info.getInfo() instanceof TensorInfo tensorInfo) {
            return tensorInfo.type;
        }
        return OnnxJavaType.UNKNOWN;
    }

    private Object coerceList(List<?> list, OnnxJavaType type) {
        return switch (type) {
            case STRING -> list.stream().map(String::valueOf).toArray(String[]::new);
            case BOOL -> {
                boolean[] arr = new boolean[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = toBoolean(list.get(i));
                }
                yield arr;
            }
            case INT8, UINT8 -> {
                byte[] arr = new byte[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = (byte) toLong(list.get(i));
                }
                yield arr;
            }
            case INT16 -> {
                short[] arr = new short[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = (short) toLong(list.get(i));
                }
                yield arr;
            }
            case INT32 -> {
                int[] arr = new int[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = (int) toLong(list.get(i));
                }
                yield arr;
            }
            case INT64 -> {
                long[] arr = new long[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = toLong(list.get(i));
                }
                yield arr;
            }
            case DOUBLE -> {
                double[] arr = new double[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = toDouble(list.get(i));
                }
                yield arr;
            }
            case FLOAT, FLOAT16, BFLOAT16, UNKNOWN -> {
                float[] arr = new float[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    arr[i] = (float) toDouble(list.get(i));
                }
                yield arr;
            }
        };
    }

    private boolean toBoolean(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof Number num) {
            return num.intValue() != 0;
        }
        if (value instanceof String str) {
            return Boolean.parseBoolean(str);
        }
        throw new IllegalArgumentException("Cannot convert value to boolean: " + value);
    }

    private long toLong(Object value) {
        if (value instanceof Number num) {
            return num.longValue();
        }
        if (value instanceof Boolean bool) {
            return bool ? 1L : 0L;
        }
        if (value instanceof String str) {
            try {
                return Long.parseLong(str);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Cannot convert value to long: " + value, e);
            }
        }
        throw new IllegalArgumentException("Cannot convert value to long: " + value);
    }

    private double toDouble(Object value) {
        if (value instanceof Number num) {
            return num.doubleValue();
        }
        if (value instanceof Boolean bool) {
            return bool ? 1.0 : 0.0;
        }
        if (value instanceof String str) {
            try {
                return Double.parseDouble(str);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Cannot convert value to double: " + value, e);
            }
        }
        throw new IllegalArgumentException("Cannot convert value to double: " + value);
    }

    private ValidationResponse toValidationResponse(UrlFeatures features, Map<String, Object> outputs) {
        double[] scores = extractScores(outputs);
        if (scores == null || scores.length == 0) {
            log.warn("Model produced no usable outputs; returning 'uncertain'.");
            return new ValidationResponse("uncertain", 0.0, features.features());
        }

        double[] probabilities = normalizeScores(scores);
        if (probabilities.length == 0) {
            log.warn("Model outputs contained non-finite values; returning 'uncertain'.");
            return new ValidationResponse("uncertain", 0.0, features.features());
        }

        String verdict;
        double confidence;
        if (probabilities.length == 1) {
            double p = clamp01(probabilities[0]);
            verdict = p >= 0.5 ? "malicious" : "benign";
            confidence = p >= 0.5 ? p : 1.0 - p;
        } else {
            int maxIdx = argMax(probabilities);
            confidence = clamp01(probabilities[maxIdx]);
            if (probabilities.length >= 3) {
                verdict = switch (maxIdx) {
                    case 0 -> "benign";
                    case 1 -> "malicious";
                    default -> "uncertain";
                };
            } else {
                verdict = (maxIdx == 0) ? "benign" : "malicious";
            }
        }

        return new ValidationResponse(verdict, confidence, features.features());
    }

    private double[] extractScores(Map<String, Object> outputs) {
        if (outputs == null || outputs.isEmpty()) {
            return null;
        }

        Object primary;
        if (outputs.size() == 1) {
            primary = outputs.values().iterator().next();
        } else if (outputNames != null && !outputNames.isEmpty() && outputs.containsKey(outputNames.get(0))) {
            primary = outputs.get(outputNames.get(0));
        } else if (outputs.containsKey("probabilities")) {
            primary = outputs.get("probabilities");
        } else if (outputs.containsKey("logits")) {
            primary = outputs.get("logits");
        } else {
            primary = outputs.values().iterator().next();
        }

        return toDoubleArray(primary);
    }

    private double[] toDoubleArray(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof double[] arr) {
            return arr;
        }
        if (value instanceof float[] arr) {
            double[] out = new double[arr.length];
            for (int i = 0; i < arr.length; i++) {
                out[i] = arr[i];
            }
            return out;
        }
        if (value instanceof long[] arr) {
            double[] out = new double[arr.length];
            for (int i = 0; i < arr.length; i++) {
                out[i] = arr[i];
            }
            return out;
        }
        if (value instanceof int[] arr) {
            double[] out = new double[arr.length];
            for (int i = 0; i < arr.length; i++) {
                out[i] = arr[i];
            }
            return out;
        }
        if (value instanceof double[][] arr) {
            return arr.length > 0 ? Arrays.copyOf(arr[0], arr[0].length) : new double[0];
        }
        if (value instanceof float[][] arr) {
            return arr.length > 0 ? toDoubleArray(arr[0]) : new double[0];
        }
        if (value instanceof long[][] arr) {
            return arr.length > 0 ? toDoubleArray(arr[0]) : new double[0];
        }
        if (value instanceof int[][] arr) {
            return arr.length > 0 ? toDoubleArray(arr[0]) : new double[0];
        }
        if (value instanceof Number num) {
            return new double[]{num.doubleValue()};
        }
        if (value instanceof Boolean bool) {
            return new double[]{bool ? 1.0 : 0.0};
        }
        if (value instanceof List<?> list) {
            double[] out = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                out[i] = toDouble(list.get(i));
            }
            return out;
        }
        if (value instanceof Object[] arr) {
            double[] out = new double[arr.length];
            for (int i = 0; i < arr.length; i++) {
                out[i] = toDouble(arr[i]);
            }
            return out;
        }

        log.warn("Unsupported output type from model: {}", value.getClass().getName());
        return null;
    }

    private double[] normalizeScores(double[] scores) {
        boolean needsNormalization = false;
        for (double s : scores) {
            if (!Double.isFinite(s)) {
                return new double[0];
            }
            if (s < 0.0 || s > 1.0) {
                needsNormalization = true;
            }
        }
        if (!needsNormalization) {
            return scores;
        }
        if (scores.length == 1) {
            return new double[]{sigmoid(scores[0])};
        }
        return softmax(scores);
    }

    private double[] softmax(double[] logits) {
        double max = Arrays.stream(logits).max().orElse(0.0);
        double[] exp = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            double v = Math.exp(logits[i] - max);
            exp[i] = v;
            sum += v;
        }
        if (sum == 0.0) {
            return new double[0];
        }
        for (int i = 0; i < exp.length; i++) {
            exp[i] = exp[i] / sum;
        }
        return exp;
    }

    private double sigmoid(double x) {
        if (x >= 0.0) {
            double z = Math.exp(-x);
            return 1.0 / (1.0 + z);
        }
        double z = Math.exp(x);
        return z / (1.0 + z);
    }

    private int argMax(double[] values) {
        int maxIdx = 0;
        double max = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private void loadModel() {
        try {
            Resource resource = resourceLoader.getResource(modelPath);
            if (!resource.exists()) {
                throw new IllegalStateException("Model not found at: " + modelPath);
            }

            environment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            session = createSession(resource, options);

            inputInfo = session.getInputInfo();
            outputNames = new ArrayList<>(session.getOutputNames());

            log.info("Loaded ONNX model from {} (inputs={}, outputs={})",
                    modelPath, inputInfo.keySet(), outputNames);

        } catch (OrtException e) {
            String message = e.getMessage();
            if (message != null && message.contains(".onnx.data")) {
                throw new IllegalStateException(
                        "Failed to load ONNX model with external data. Ensure the .onnx.data file is next "
                                + "to the .onnx and use a file: path (not classpath:) so ONNX Runtime can resolve it.",
                        e);
            }
            throw new IllegalStateException("Failed to load ONNX model: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to load ONNX model: " + e.getMessage(), e);
        }
    }

    private OrtSession createSession(Resource modelResource, OrtSession.SessionOptions options) throws IOException, OrtException {
        if (modelResource.isFile()) {
            return environment.createSession(modelResource.getFile().getAbsolutePath(), options);
        }

        String filename = modelResource.getFilename();
        if (filename == null || filename.isBlank()) {
            throw new IllegalStateException("Model resource filename is missing for: " + modelPath);
        }

        Path tempDir = Files.createTempDirectory("onnx-model-");
        Path onnxPath = tempDir.resolve(filename);
        try (InputStream in = modelResource.getInputStream()) {
            Files.copy(in, onnxPath, StandardCopyOption.REPLACE_EXISTING);
        }

        if (filename.endsWith(".onnx")) {
            String dataFilename = filename + ".data";
            String dataResourcePath = modelPath.replace(filename, dataFilename);
            Resource dataResource = resourceLoader.getResource(dataResourcePath);
            if (dataResource.exists()) {
                try (InputStream in = dataResource.getInputStream()) {
                    Files.copy(in, tempDir.resolve(dataFilename), StandardCopyOption.REPLACE_EXISTING);
                }
            } else {
                log.warn("ONNX external data file not found at {}. Expected {}", dataResourcePath, dataFilename);
            }
        }

        return environment.createSession(onnxPath.toString(), options);
    }

    private void closeSession() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                log.warn("Failed to close ONNX session", e);
            }
            session = null;
        }
        if (environment != null) {
            environment.close();
            environment = null;
        }
        inputInfo = Collections.emptyMap();
        outputNames = List.of();
    }
}
