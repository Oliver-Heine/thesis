package com.thesis.qrquishing.integrations.ai;

import com.thesis.qrquishing.UrlFeatures;
import com.thesis.qrquishing.ValidationResponse;

public interface AIAnalyzer {
    public ValidationResponse analyze(String url, UrlFeatures features);
}
