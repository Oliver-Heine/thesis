package com.thesis.qrquishing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Entry point for the QR Quishing Detector backend service.
 */
@SpringBootApplication
@EnableScheduling
public class QrQuishingApplication {

    public static void main(String[] args) {
        SpringApplication.run(QrQuishingApplication.class, args);
    }
}
