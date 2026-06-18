package com.goktug.conversation.api;

import com.goktug.conversation.api.dto.PageResponse;
import com.goktug.conversation.config.AuthContext;
import com.goktug.conversation.search.ChatSearchDocument;
import com.goktug.conversation.search.ChatSearchRepository;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.time.OffsetDateTime;
import java.util.UUID;

/**
 * Read endpoint — Elasticsearch'e gider (CQRS read side).
 *
 * Write endpoint'leri (POST /api/v1/chats vs.) ChatController'da, PostgreSQL'e gider.
 * Bu ayrım sayesinde search Elasticsearch ölçeklenmesinden bağımsız hızlı çalışır.
 */
@RestController
@RequestMapping("/api/v1/chats/search")
@RequiredArgsConstructor
public class ChatSearchController {

    private final ChatSearchRepository searchRepository;

    @GetMapping
    public PageResponse<ChatSearchResult> search(
            @RequestParam String q,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            HttpServletRequest request) {

        UUID userId = AuthContext.requireUserId(request);

        Page<ChatSearchDocument> results = searchRepository
            .findByUserIdAndTitleContainingIgnoreCaseAndDeletedFalse(
                userId.toString(), q, PageRequest.of(page, Math.min(size, 100)));

        return PageResponse.from(results, ChatSearchResult::from);
    }

    public record ChatSearchResult(
        UUID id, String title, OffsetDateTime updatedAt
    ) {
        static ChatSearchResult from(ChatSearchDocument d) {
            return new ChatSearchResult(
                UUID.fromString(d.getId()), d.getTitle(), d.getUpdatedAt());
        }
    }
}
