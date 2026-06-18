package com.goktug.identity.domain;

import jakarta.persistence.*;

import lombok.*;

import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "custom_instructions")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CustomInstructionsEntity {

    @Id
    @Column(name = "user_id")
    private UUID userId;

    @Column(name = "about_user", columnDefinition = "TEXT")
    private String aboutUser;

    @Column(name = "response_style", columnDefinition = "TEXT")
    private String responseStyle;

    @Column(nullable = false)
    private Boolean enabled;

    @Column(name = "updated_at", nullable = false)
    private OffsetDateTime updatedAt;

    @PrePersist
    @PreUpdate
    void onSave() {
        updatedAt = OffsetDateTime.now();
        if (enabled == null) enabled = true;
    }
}
