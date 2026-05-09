package com.goktug.conversation.outbox;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;

import jakarta.persistence.LockModeType;
import java.util.List;
import java.util.UUID;

public interface OutboxRepository extends JpaRepository<OutboxEntity, UUID> {

    /**
     * İşlenmemiş outbox satırlarını yaşa göre çek + PESSIMISTIC_WRITE lock al.
     *
     * Birden fazla poller instance varsa (multi-replica deployment),
     * `FOR UPDATE SKIP LOCKED` semantiği bu lock ile sağlanır → her satırı
     * sadece bir worker işler. Bu Postgres'in kritik özelliklerinden biri.
     */
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("""
        SELECT o FROM OutboxEntity o
        WHERE o.processedAt IS NULL
        ORDER BY o.createdAt ASC
        """)
    List<OutboxEntity> findUnprocessed(Pageable pageable);
}
