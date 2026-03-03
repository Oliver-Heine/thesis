package com.thesis.qrquishing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Entry point for the QR Quishing Detector backend service.
 */
@SpringBootApplication
public class QrQuishingApplication {

    public static void main(String[] args) {
        SpringApplication.run(QrQuishingApplication.class, args);
    }
}
