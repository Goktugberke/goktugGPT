package com.goktug.conversation;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableKafka
@EnableScheduling   // OutboxPoller @Scheduled için
public class ConversationServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(ConversationServiceApplication.class, args);
    }
}
