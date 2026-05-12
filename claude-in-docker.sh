#!/usr/bin/env bash
# claude-in-docker.sh — start an interactive Claude Code session inside the
# demgis container with full agent permissions.
#
# The container is a sandbox (no host filesystem access outside the bind-
# mounted volumes) and runs as non-root user "demgis", so claude can use
# --dangerously-skip-permissions safely. The host's Claude auth state is
# mounted in via $HOME/.claude AND $HOME/.claude.json so no re-login.
#
# Usage:
#   ./claude-in-docker.sh              — start a fresh agent session
#   ./claude-in-docker.sh -c           — continue most recent session
#   ./claude-in-docker.sh "prompt..."  — one-shot prompt

set -euo pipefail

cd "$(dirname "$0")"

# Sanity: the bind mounts in docker-compose.yml require these to exist on
# the host BEFORE the container starts. If a file mount target doesn't
# exist, docker silently creates a DIRECTORY there, which breaks Claude
# auth in mysterious ways.
if [[ ! -d "$HOME/.claude" ]]; then
    echo "ERROR: $HOME/.claude does not exist on host."
    echo "  Run 'claude' once on the host to authenticate, then retry."
    exit 1
fi
if [[ ! -f "$HOME/.claude.json" ]]; then
    echo "ERROR: $HOME/.claude.json does not exist on host."
    echo "  Run 'claude' once on the host to authenticate, then retry."
    exit 1
fi

if ! docker image inspect demgis:latest >/dev/null 2>&1; then
    echo "demgis:latest image not found. Building..."
    docker compose build
fi

exec docker compose run --rm \
    -e TERM \
    demgis \
    claude --dangerously-skip-permissions "$@"
