#!/usr/bin/env bash
# Initializes the runtime config-repo/ that Spring Cloud Config Server reads
# via its Git backend (file:///opt/config-repo). config-repo/ is .gitignore'd
# (it's a JGit repo), so it must be recreated from the version-controlled seed
# at config-server/seed-config/ on a fresh clone. Idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_REPO="$REPO_ROOT/config-repo"
SEED_DIR="$REPO_ROOT/config-server/seed-config"

if [ -d "$CONFIG_REPO/.git" ]; then
  echo "[init-config-repo] already initialized — syncing seed."
  cp -rf "$SEED_DIR"/* "$CONFIG_REPO"/
  cd "$CONFIG_REPO"
  git add -A
  if ! git diff --cached --quiet; then
    git -c user.email=config@goktug.local -c user.name="Config Bot" commit -m "chore: sync shared config from seed" >/dev/null
    echo "[init-config-repo] committed seed changes."
  else
    echo "[init-config-repo] no changes."
  fi
  exit 0
fi

echo "[init-config-repo] creating config-repo/ from seed..."
mkdir -p "$CONFIG_REPO"
cp -rf "$SEED_DIR"/* "$CONFIG_REPO"/
cd "$CONFIG_REPO"
git init -b main >/dev/null
git add -A
git -c user.email=config@goktug.local -c user.name="Config Bot" commit -m "init: shared platform config" >/dev/null
echo "[init-config-repo] done."
