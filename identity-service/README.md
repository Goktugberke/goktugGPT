# identity-service

> **Kullanıcı kayıt, profil, custom instructions** — Keycloak ile entegre.

## Sorumluluklar

- Kullanıcı kayıt (Keycloak Admin API üzerinden user oluşturur)
- Login/Refresh token (Keycloak token endpoint'ine proxy yapar — frontend için tek endpoint sağlar)
- Profile CRUD (display name, avatar, language, theme)
- Custom Instructions ("about me" / "how should I respond")
- `user.registered.v1` event publish

## Mimari Karar

**Neden Keycloak + ayrı bir DB?** Keycloak credentials, OAuth flows, JWT issuance, MFA gibi şeyler için olgun. Ama uygulama-spesifik profil verilerini (custom instructions, theme tercihi vs.) Keycloak'a koymak yanlış — schema flexibility ve ownership açısından kötü. Bu yüzden:

- **Keycloak DB:** username, password hash, roles, MFA, sessions
- **identity-service DB:** profile, custom_instructions, outbox events

`userId` her iki yerde de Keycloak'ın `sub` (UUID) claim'iyle aynı.

## Endpoints

| Method | Path | Public | Açıklama |
|--------|------|:------:|----------|
| POST | /api/v1/auth/register | ✓ | Yeni kullanıcı (Keycloak'a delege) |
| POST | /api/v1/auth/login | ✓ | Token endpoint proxy |
| POST | /api/v1/auth/refresh | ✓ | Refresh token |
| POST | /api/v1/auth/logout | | |
| GET | /api/v1/users/me | | Profil getir |
| PUT | /api/v1/users/me/profile | | Profil güncelle |
| GET | /api/v1/users/me/custom-instructions | | |
| PUT | /api/v1/users/me/custom-instructions | | |

## Eventler

**Publish:**
- `user.registered.v1` — billing (free plan oluştur), notification (welcome email)
- `user.profile-updated.v1`
- `user.custom-instructions-updated.v1` — inference-orchestrator dinleyip cache invalidate edebilir
- `user.deleted.v1` — conversation, asset cascade cleanup

**Subscribe:** (yok — root identity service)

## Patterns

- **Transactional Outbox** — register flow: Keycloak'a user oluştur → DB'ye profile insert → outbox event row, hepsi tek logical operation. Kafka publish ayrı thread.
- **Idempotency** (`Idempotency-Key` header) — register endpoint için zorunlu (frontend retry safe).
- **Saga participation** (Faz 2): `user.deleted.v1` consume eden conversation/asset servisleri kendi cleanup'ını yapar; identity orchestrator değil — choreography.

## Port

`8081` (internal, sadece api-gateway'den erişilir)

## TODO (bir sonraki session)

1. **JPA entity'leri:** `Profile`, `CustomInstructions`, `OutboxEvent` (Lombok + Hibernate JsonType)
2. **Repository'ler:** Spring Data JPA
3. **AuthController:** register / login / refresh / logout (Keycloak Admin Client + WebClient)
4. **UserController:** profile + custom instructions
5. **OutboxPoller:** `@Scheduled(fixedDelay = 1000)` → unprocessed outbox rows → Kafka publish → mark processed
6. **KeycloakClient:** Admin token alma + user create/disable/delete
7. **GlobalExceptionHandler:** unified error response
8. **OpenAPI annotation'ları** — `/swagger-ui/index.html`
9. **Testler:** Testcontainers (Postgres + Keycloak) ile integration test

## Çalıştırma

```bash
# infra önce ayağa kalksın
docker compose --profile infra up -d

# sonra servisi build + run
mvn -pl services/identity-service -am spring-boot:run
```
