package com.goktug.conversation.config;

import com.goktug.conversation.error.UnauthorizedException;
import jakarta.servlet.http.HttpServletRequest;
import lombok.experimental.UtilityClass;

import java.util.UUID;

/**
 * API Gateway downstream'lere JWT'yi parse edip kullanıcı kimliğini header
 * olarak iletir (X-User-Id). Bu sınıf bu header'ı UUID olarak çeker.
 *
 * Faz 2'de gateway eklendikten sonra direkt JWT validation kaldırılır
 * (zaten gateway'de yapıldı). O zamana kadar dev profil için fallback:
 * `X-User-Id` header'ı yoksa 401.
 */
@UtilityClass
public class AuthContext {

    public static final String USER_ID_HEADER = "X-User-Id";

    public static UUID requireUserId(HttpServletRequest request) {
        String raw = request.getHeader(USER_ID_HEADER);
        if (raw == null || raw.isBlank()) {
            throw new UnauthorizedException("Missing " + USER_ID_HEADER + " header");
        }
        try {
            return UUID.fromString(raw);
        } catch (IllegalArgumentException ex) {
            throw new UnauthorizedException("Invalid user id: " + raw);
        }
    }
}
