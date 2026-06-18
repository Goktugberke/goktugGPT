package com.goktug.inference.domain;

import com.goktug.inference.saga.SagaState;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

public interface InferenceJobRepository extends JpaRepository<InferenceJob, UUID> {

    /**
     * Recovery için: terminal olmayan ama uzun süredir update edilmemiş jobları çek.
     * (orchestrator crash, network partition, vs.)
     */
    @Query("""
        SELECT j FROM InferenceJob j
        WHERE j.state IN :states
        AND j.updatedAt < :before
        ORDER BY j.updatedAt ASC
        """)
    List<InferenceJob> findStaleJobs(
        @Param("states") List<SagaState> states,
        @Param("before") OffsetDateTime before
    );
}
