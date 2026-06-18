package com.goktug.conversation.api.dto;

import jakarta.validation.constraints.Size;

public record CreateChatRequest(
    @Size(max = 256) String title
) {}
