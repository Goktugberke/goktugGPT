package com.goktug.notification.event;

import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class NotificationConsumer {

    

    @KafkaListener(topics = "user.events", groupId = "notification-group")
    public void consumeUserEvent(String message) {
        log.info("Processing user event for notification: {}", message);
        // Logic to send email or push notification based on event type
    }
}
