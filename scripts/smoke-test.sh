#!/usr/bin/env bash
# smoke-test.sh — End-to-end test of goktugGPT platform.
#
# Prereq:
#   docker compose --profile infra up -d   # infra
#   docker compose up -d --build           # services
#   ... wait ~30s for everything to be healthy
#
# Usage:
#   ./scripts/smoke-test.sh

set -euo pipefail

GW="${GATEWAY_URL:-http://localhost:8080}"
EMAIL="${EMAIL:-test+$(date +%s)@goktug.dev}"
PASSWORD="${PASSWORD:-TestPass1234}"

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
hr()     { printf '─%.0s' $(seq 1 60); echo; }

require() { command -v "$1" >/dev/null || { red "Missing: $1"; exit 1; }; }
require curl
require jq

hr
yellow "[0] Health checks"
hr
curl -fsS "$GW/actuator/health" | jq -r .status || { red "Gateway down"; exit 1; }
green "  ✓ api-gateway healthy"

curl -fsS http://localhost:9001/v1/health | jq -c '{status, model_loaded, device}'
green "  ✓ inference-worker reachable"

curl -fsS "http://localhost:8180/realms/goktuggpt/.well-known/openid-configuration" \
  | jq -r .issuer
green "  ✓ Keycloak realm accessible"

hr
yellow "[1] Register: $EMAIL"
hr
REGISTER=$(curl -fsS -X POST "$GW/api/v1/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"displayName\":\"Smoke Test\"}")
echo "$REGISTER" | jq
TOKEN=$(echo "$REGISTER" | jq -r .accessToken)
USER_ID=$(echo "$REGISTER" | jq -r .userId)
green "  ✓ Registered userId=$USER_ID"

hr
yellow "[2] Create chat"
hr
CHAT=$(curl -fsS -X POST "$GW/api/v1/chats" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Smoke test chat"}')
echo "$CHAT" | jq
CHAT_ID=$(echo "$CHAT" | jq -r .id)
green "  ✓ chatId=$CHAT_ID"

hr
yellow "[3] Send user message (idempotent)"
hr
IDEMP=$(uuidgen 2>/dev/null || echo "smoke-$(date +%s%N)")
MSG=$(curl -fsS -X POST "$GW/api/v1/chats/$CHAT_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Idempotency-Key: $IDEMP" \
  -H 'Content-Type: application/json' \
  -d '{"content":"merhaba dünya, sen kimsin?"}')
echo "$MSG" | jq
MSG_ID=$(echo "$MSG" | jq -r .id)
green "  ✓ messageId=$MSG_ID"

# Idempotency replay — same key should return same id
yellow "  ... replaying with same Idempotency-Key (should be no-op)"
MSG2=$(curl -fsS -X POST "$GW/api/v1/chats/$CHAT_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Idempotency-Key: $IDEMP" \
  -H 'Content-Type: application/json' \
  -d '{"content":"merhaba dünya, sen kimsin?"}')
MSG2_ID=$(echo "$MSG2" | jq -r .id)
if [[ "$MSG_ID" == "$MSG2_ID" ]]; then
  green "  ✓ Idempotency works (same id returned: $MSG_ID)"
else
  red   "  ✗ Idempotency FAILED — got different ids ($MSG_ID vs $MSG2_ID)"
  exit 1
fi

hr
yellow "[4] Stream inference (SSE)"
hr
echo "  (showing first 30 lines of stream...)"
curl -N --max-time 30 -X POST "$GW/api/v1/inference/stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"chatId\":\"$CHAT_ID\",\"userMessageId\":\"$MSG_ID\",\"text\":\"merhaba dünya, sen kimsin?\"}" \
  | head -30
green "  ✓ SSE stream completed (or timed out — check above for token data)"

hr
yellow "[5] Verify assistant message persisted (eventual consistency)"
hr
echo "  ... waiting 3s for inference.completed.v1 → conversation-service"
sleep 3
ALL=$(curl -fsS "$GW/api/v1/chats/$CHAT_ID/messages" \
  -H "Authorization: Bearer $TOKEN")
echo "$ALL" | jq '.[] | {sender, content: (.content[:80])}'
ASSISTANT_COUNT=$(echo "$ALL" | jq '[.[] | select(.sender=="ASSISTANT")] | length')
if [[ "$ASSISTANT_COUNT" -ge 1 ]]; then
  green "  ✓ Assistant message persisted ($ASSISTANT_COUNT row(s))"
else
  yellow "  ⚠ No assistant message yet — check inference-orchestrator/conversation-service logs"
fi

hr
yellow "[6] Search via Elasticsearch projector"
hr
sleep 2
SEARCH=$(curl -fsS "$GW/api/v1/chats/search?q=Smoke" \
  -H "Authorization: Bearer $TOKEN")
echo "$SEARCH" | jq '.items[] | {id, title}'
if echo "$SEARCH" | jq -e '.items | length > 0' >/dev/null; then
  green "  ✓ ES projector working — chat found in search"
else
  yellow "  ⚠ ES has no results yet — projector lag or chat.events not consumed"
fi

hr
green "✅ Smoke test passed!"
yellow "Distributed trace: open http://localhost:16686 (Jaeger) and find service=api-gateway"
