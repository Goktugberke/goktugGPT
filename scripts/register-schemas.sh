#!/usr/bin/env bash
# Registers all event JSON Schemas (schemas/events/*.json) to Apicurio Registry
# with a BACKWARD compatibility rule per artifact + a global default rule.
#
# Usage:  ./scripts/register-schemas.sh
# Env:    REGISTRY_URL (default http://localhost:8086)
#         GROUP        (default goktug-events)
set -euo pipefail

REGISTRY_URL="${REGISTRY_URL:-http://localhost:8086}"
GROUP="${GROUP:-goktug-events}"
API_BASE="$REGISTRY_URL/apis/registry/v2"
SCHEMA_DIR="$(cd "$(dirname "$0")/../schemas/events" && pwd)"

echo "Registry : $REGISTRY_URL"
echo "Group    : $GROUP"
echo "Schemas  : $SCHEMA_DIR"
echo

# 1) Wait for the registry
printf "Waiting for Apicurio..."
for i in $(seq 1 30); do
  if curl -sf "$API_BASE/system/info" >/dev/null 2>&1; then echo " up."; break; fi
  sleep 2; printf "."
  [ "$i" -eq 30 ] && { echo " timeout"; exit 1; }
done

# 2) Global default COMPATIBILITY = BACKWARD (ignore 409)
curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/admin/rules" \
  -H "Content-Type: application/json" \
  -d '{"type":"COMPATIBILITY","config":"BACKWARD"}' | grep -qE '20|409' \
  && echo "[OK] global COMPATIBILITY=BACKWARD"

# 3) Register / update each schema
count=0
for f in "$SCHEMA_DIR"/*.json; do
  artifact_id="$(basename "$f" .json)"
  curl -s -o /dev/null -X POST "$API_BASE/groups/$GROUP/artifacts?ifExists=UPDATE" \
    -H "X-Registry-ArtifactId: $artifact_id" \
    -H "X-Registry-ArtifactType: JSON" \
    -H "Content-Type: application/json" \
    --data-binary "@$f"
  echo "[OK] registered $artifact_id"

  curl -s -o /dev/null -X POST "$API_BASE/groups/$GROUP/artifacts/$artifact_id/rules" \
    -H "Content-Type: application/json" \
    -d '{"type":"COMPATIBILITY","config":"BACKWARD"}' || true
  count=$((count+1))
done

echo
echo "Done. $count schema(s) registered under group '$GROUP'."
echo "Browse: $REGISTRY_URL/ui"
