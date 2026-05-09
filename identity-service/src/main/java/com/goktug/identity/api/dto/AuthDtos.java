package com.goktug.identity.api.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class AuthDtos {

    public record RegisterRequest(
        @Email @NotBlank String email,
        @NotBlank @Size(min = 8, max = 128) String password,
        @Size(max = 120) String displayName
    ) {}

    public record LoginRequest(
        @Email @NotBlank String email,
        @NotBlank String password
    ) {}

    public record RefreshRequest(@NotBlank String refreshToken) {}

    public record TokenResponse(
        String accessToken,
        String refreshToken,
        Integer expiresIn,
        String userId,
        String email
    ) {}

    public record ProfileDto(
        String userId,
        String email,
        String displayName,
        String avatarUrl,
        String language,
        String theme
    ) {}

    public record UpdateProfileRequest(
        @Size(max = 120) String displayName,
        @Size(max = 1024) String avatarUrl,
        @Size(max = 8) String language,
        @Size(max = 16) String theme
    ) {}

    public record CustomInstructionsDto(
        String aboutUser,
        String responseStyle,
        Boolean enabled
    ) {}
}
