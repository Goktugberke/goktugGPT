package com.goktug.asset.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;
import java.util.UUID;

public interface AssetRepository extends JpaRepository<AssetEntity, UUID> {
    Optional<AssetEntity> findByIdAndUserIdAndDeletedAtIsNull(UUID id, UUID userId);
}
