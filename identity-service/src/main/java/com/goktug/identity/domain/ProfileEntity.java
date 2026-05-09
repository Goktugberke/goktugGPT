package com.goktug.identity.domain;

import jakarta.persistence.*;

import lombok.*;

import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "profiles")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ProfileEntity {

    @Id
    @Column(name = "user_id")
    private UUID userId;

    @Column(nullable = false, length = 320, unique = true)
    private String email;

    @Column(name = "display_name", length = 120)
    private String displayName;

    @Column(name = "avatar_url", length = 1024)
    private String avatarUrl;

    @Column(nullable = false, length = 8)
    private String language;

    @Column(nullable = false, length = 16)
    private String theme;

    @Column(name = "created_at", nullable = false)
    private OffsetDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private OffsetDateTime updatedAt;

    @PrePersist
    void onCreate() {
        OffsetDateTime now = OffsetDateTime.now();
        if (createdAt == null) createdAt = now;
        if (updatedAt == null) updatedAt = now;
        if (language == null) language = "tr";
        if (theme == null) theme = "system";
    }

    @PreUpdate
    void onUpdate() {
        updatedAt = OffsetDateTime.now();
    }
}
