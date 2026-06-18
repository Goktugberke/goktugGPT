package com.goktug.conversation.search;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.elasticsearch.repository.ElasticsearchRepository;

public interface ChatSearchRepository extends ElasticsearchRepository<ChatSearchDocument, String> {

    Page<ChatSearchDocument> findByUserIdAndTitleContainingIgnoreCaseAndDeletedFalse(
        String userId, String titleFragment, Pageable pageable);
}
