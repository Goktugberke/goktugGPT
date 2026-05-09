package com.goktug.identity.error;

import org.springframework.http.HttpStatus;

public class IdentityException extends RuntimeException {
    private final HttpStatus status;

    public IdentityException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public HttpStatus getStatus() { return status; }
}
