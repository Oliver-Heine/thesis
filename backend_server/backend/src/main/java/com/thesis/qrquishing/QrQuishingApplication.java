package com.thesis.qrquishing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.PropertySources;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Entry point for the QR Quishing Detector backend service.
 */
@SpringBootApplication
@EnableScheduling
@PropertySources({
        @PropertySource(value = "file:.env", ignoreResourceNotFound = true),
        @PropertySource(value = "file:../.env", ignoreResourceNotFound = true),
        @PropertySource(value = "file:./backend_server/.env", ignoreResourceNotFound = true)
})
public class QrQuishingApplication {

    public static void main(String[] args) {
        SpringApplication.run(QrQuishingApplication.class, args);
    }
}
