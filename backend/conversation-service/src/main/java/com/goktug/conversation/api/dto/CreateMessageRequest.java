package com.goktug.conversation.api.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record CreateMessageRequest(
    @NotBlank @Size(max = 32000) String content,
    String modelHint,
    JsonNode attachments
) {}
