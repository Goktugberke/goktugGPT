package com.goktug.identity.api;

import com.goktug.identity.api.dto.AuthDtos.*;
import com.goktug.identity.service.IdentityService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final IdentityService identityService;

    @PostMapping("/register")
    public ResponseEntity<TokenResponse> register(@RequestBody @Valid RegisterRequest req) {
        return ResponseEntity.status(HttpStatus.CREATED).body(identityService.register(req));
    }

    @PostMapping("/login")
    public TokenResponse login(@RequestBody @Valid LoginRequest req) {
        return identityService.login(req);
    }

    @PostMapping("/refresh")
    public TokenResponse refresh(@RequestBody @Valid RefreshRequest req) {
        return identityService.refresh(req.refreshToken());
    }

    @PostMapping("/logout")
    public ResponseEntity<Void> logout() {
        // Stateless JWT — frontend token'ı kendisi siler.
        // Token revocation listesi (Redis) eklemek istersen Faz 2'de.
        return ResponseEntity.noContent().build();
    }
}
