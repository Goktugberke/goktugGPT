package com.goktug.conversation;

import com.goktug.conversation.api.dto.ChatDto;
import com.goktug.conversation.api.dto.CreateChatRequest;
import com.goktug.conversation.domain.ChatRepository;
import com.goktug.conversation.service.ChatService;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.testcontainers.containers.KafkaContainer;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.elasticsearch.ElasticsearchContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;
import java.util.List;
import java.util.Properties;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * End-to-end integration test of the Transactional Outbox flow, backed by real
 * Postgres + Kafka + Elasticsearch via Testcontainers (image tags match
 * docker-compose so no extra image pulls).
 *
 * Verifies: ChatService.create() persists the chat (Postgres + Flyway schema) AND
 * the OutboxPoller publishes the resulting chat.created.v1 event to Kafka.
 *
 * The same pattern (one @SpringBootTest + @ServiceConnection containers) extends to
 * the other services' happy/failure paths.
 */
@SpringBootTest(properties = {
    "spring.cloud.config.enabled=false",   // no config-server in tests
    "management.tracing.enabled=false",    // no Jaeger export noise
    "outbox.poll-interval-ms=500"          // speed up the poller
})
@Testcontainers
class OutboxIntegrationTest {

    @Container
    @ServiceConnection
    static PostgreSQLContainer<?> postgres =
        new PostgreSQLContainer<>(DockerImageName.parse("postgres:16-alpine"));

    @Container
    @ServiceConnection
    static KafkaContainer kafka =
        new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:7.6.1"));

    @Container
    @ServiceConnection
    static ElasticsearchContainer elasticsearch =
        new ElasticsearchContainer(DockerImageName.parse("docker.elastic.co/elasticsearch/elasticsearch:8.13.4"))
            .withEnv("xpack.security.enabled", "false")
            .withEnv("ES_JAVA_OPTS", "-Xms512m -Xmx512m")   // smaller heap = faster, lighter startup
            .withStartupTimeout(Duration.ofMinutes(3));      // ES 8 can be slow on a loaded box

    @Autowired ChatService chatService;
    @Autowired ChatRepository chatRepository;

    @Test
    void createChat_persistsToPostgres_andPublishesOutboxEventToKafka() {
        UUID userId = UUID.randomUUID();

        // when: create a chat — writes the chat row + outbox row in one transaction
        ChatDto chat = chatService.create(userId, new CreateChatRequest("Testcontainers chat"), "trace-it");

        // then: persisted in Postgres (schema applied by Flyway against the container)
        assertThat(chatRepository.findByIdAndUserIdAndDeletedAtIsNull(chat.id(), userId)).isPresent();

        // and: the OutboxPoller publishes chat.created.v1 to Kafka within a few seconds
        try (KafkaConsumer<String, String> consumer = newConsumer()) {
            consumer.subscribe(List.of("chat.events"));
            String event = pollFor(consumer, chat.id().toString(), Duration.ofSeconds(25));
            assertThat(event).contains("chat.created.v1");
            assertThat(event).contains(chat.id().toString());
        }
    }

    private KafkaConsumer<String, String> newConsumer() {
        Properties p = new Properties();
        p.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, kafka.getBootstrapServers());
        p.put(ConsumerConfig.GROUP_ID_CONFIG, "it-" + UUID.randomUUID());
        p.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        p.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        p.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        return new KafkaConsumer<>(p);
    }

    /** Poll until a record value contains the marker, or fail after the timeout. */
    private String pollFor(KafkaConsumer<String, String> consumer, String marker, Duration timeout) {
        long deadline = System.currentTimeMillis() + timeout.toMillis();
        while (System.currentTimeMillis() < deadline) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));
            for (ConsumerRecord<String, String> r : records) {
                if (r.value() != null && r.value().contains(marker)) {
                    return r.value();
                }
            }
        }
        throw new AssertionError("chat.events received no event containing '" + marker + "' within " + timeout);
    }
}
