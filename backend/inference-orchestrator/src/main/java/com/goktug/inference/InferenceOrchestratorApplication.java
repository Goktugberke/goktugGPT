package com.goktug.inference;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableKafka
@EnableAsync
@EnableScheduling   // SagaRecoveryWorker @Scheduled için
public class InferenceOrchestratorApplication {
    public static void main(String[] args) {
        SpringApplication.run(InferenceOrchestratorApplication.class, args);
    }
}
