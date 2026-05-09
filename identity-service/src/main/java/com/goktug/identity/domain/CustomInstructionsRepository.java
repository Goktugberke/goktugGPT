package com.goktug.identity.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;

public interface CustomInstructionsRepository extends JpaRepository<CustomInstructionsEntity, UUID> {
}
