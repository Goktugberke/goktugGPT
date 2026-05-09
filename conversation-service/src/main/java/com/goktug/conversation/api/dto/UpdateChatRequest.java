package com.goktug.conversation.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record UpdateChatRequest(
    @NotBlank @Size(max = 256) String title
) {}
