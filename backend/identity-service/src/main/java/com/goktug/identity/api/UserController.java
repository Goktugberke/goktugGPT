package com.goktug.identity.api;

import com.goktug.identity.api.dto.AuthDtos.*;
import com.goktug.identity.error.IdentityException;
import com.goktug.identity.service.IdentityService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1/users/me")
@RequiredArgsConstructor
public class UserController {

    private final IdentityService identityService;

    @GetMapping
    public ProfileDto me(HttpServletRequest request) {
        return identityService.getProfile(userIdOrThrow(request));
    }

    @PutMapping("/profile")
    public ProfileDto updateProfile(@RequestBody @Valid UpdateProfileRequest req,
                                    HttpServletRequest request) {
        return identityService.updateProfile(userIdOrThrow(request), req);
    }

    @GetMapping("/custom-instructions")
    public CustomInstructionsDto getInstructions(HttpServletRequest request) {
        return identityService.getCustomInstructions(userIdOrThrow(request));
    }

    @PutMapping("/custom-instructions")
    public CustomInstructionsDto updateInstructions(@RequestBody @Valid CustomInstructionsDto dto,
                                                    HttpServletRequest request) {
        return identityService.updateCustomInstructions(userIdOrThrow(request), dto);
    }

    private UUID userIdOrThrow(HttpServletRequest request) {
        String header = request.getHeader("X-User-Id");
        if (header == null || header.isBlank()) {
            throw new IdentityException(HttpStatus.UNAUTHORIZED, "Missing X-User-Id");
        }
        try { return UUID.fromString(header); }
        catch (IllegalArgumentException ex) {
            throw new IdentityException(HttpStatus.UNAUTHORIZED, "Invalid user id");
        }
    }
}
