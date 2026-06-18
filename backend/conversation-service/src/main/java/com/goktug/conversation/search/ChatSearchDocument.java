package com.goktug.conversation.search;

import lombok.*;
import org.springframework.data.annotation.Id;
import org.springframework.data.elasticsearch.annotations.Document;
import org.springframework.data.elasticsearch.annotations.Field;
import org.springframework.data.elasticsearch.annotations.FieldType;

import java.time.OffsetDateTime;

/**
 * Elasticsearch read model for chat search.
 *
 * NOT a JPA entity — bu Elasticsearch document. Source of truth değil,
 * `chats` tablosundan event-driven olarak türetilen DENORMALIZED projection.
 * Bu sınıf üzerinden yazılacak değer DB'ye değil ES'e gider.
 */
@Document(indexName = "chat-search")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatSearchDocument {

    @Id
    private String id;   // chatId

    @Field(type = FieldType.Keyword)
    private String userId;

    @Field(type = FieldType.Text, analyzer = "standard")
    private String title;

    @Field(type = FieldType.Date)
    private OffsetDateTime createdAt;

    @Field(type = FieldType.Date)
    private OffsetDateTime updatedAt;

    @Field(type = FieldType.Boolean)
    private boolean deleted;
}
