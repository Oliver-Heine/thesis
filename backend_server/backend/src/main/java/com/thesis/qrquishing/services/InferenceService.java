package com.thesis.qrquishing.services;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
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
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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
