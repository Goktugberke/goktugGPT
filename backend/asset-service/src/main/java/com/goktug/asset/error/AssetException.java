package com.goktug.asset.error;

import org.springframework.http.HttpStatus;

public class AssetException extends RuntimeException {
    private final HttpStatus status;
    public AssetException(HttpStatus status, String message) { super(message); this.status = status; }
    public HttpStatus getStatus() { return status; }
}
