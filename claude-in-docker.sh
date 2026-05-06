#!/usr/bin/env bash
# claude-in-docker.sh — start an interactive Claude Code session inside the
# demgis container with full agent permissions.
#
# The container is itself a sandbox (no host filesystem access outside the
# bind-mounted volumes), so --dangerously-skip-permissions is safe here. The
# host's Claude auth state is mounted in via $HOME/.claude so no re-login.
#
# Usage:
#   ./claude-in-docker.sh              — start a fresh agent session
#   ./claude-in-docker.sh -c           — continue most recent session
#   ./claude-in-docker.sh "prompt..."  — one-shot prompt

set -euo pipefail

cd "$(dirname "$0")"

if ! docker image inspect demgis:latest >/dev/null 2>&1; then
    echo "demgis:latest image not found. Building..."
    docker compose build
fi

exec docker compose run --rm \
    -e TERM \
    demgis \
    claude --dangerously-skip-permissions "$@"
