# Dockerfile for demgis — DEM-to-STL pipeline.
#
# Built atop conda-forge miniforge for fast, portable env builds (mamba solver,
# conda-forge default channel matches environment.yml). Adds GDAL CLI tools,
# s5cmd for bulk Copernicus tile downloads, and Node.js + Claude Code CLI so
# agents can run with elevated permissions (--dangerously-skip-permissions)
# inside the container's natural sandbox.
#
# GUI tools (stl_viewer_gui.py / stl_align_gui.py / stl_fit_tool.py) are not
# expected to run in the container — they need an X11 display. Run them on the
# host directly via the conda demgis env.
#
# Build: docker compose build
# Run:   docker compose run --rm demgis bash
# Agent: ./claude-in-docker.sh

FROM condaforge/miniforge3:latest

# ---- System packages ----
# build-essential for any pip wheels that compile from source.
# git so the container can do its own git ops (the host harness was the
# blocker; a sandboxed container with skip-permissions fixes that).
# tini as init so signals propagate cleanly to interactive shells.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        less \
        tini \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# ---- s5cmd for parallel S3 tile downloads ----
ARG S5CMD_VERSION=2.3.0
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
        x86_64)  S5CMD_ARCH=Linux-64bit ;; \
        aarch64) S5CMD_ARCH=Linux-arm64 ;; \
        *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
    esac && \
    curl -sSL "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_${S5CMD_ARCH}.tar.gz" \
        | tar -xz -C /usr/local/bin/ s5cmd && \
    chmod +x /usr/local/bin/s5cmd

# ---- Node.js + Claude Code CLI ----
# Claude Code installed inside the container so agents can run with full
# permissions; the user's auth state is bind-mounted from host (~/.claude)
# via docker-compose so re-auth isn't needed per container.
ARG NODE_MAJOR=22
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g @anthropic-ai/claude-code

# ---- demgis conda env ----
COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && mamba clean --all --yes

# Activate the env for every subsequent RUN and for interactive shells.
ENV CONDA_DEFAULT_ENV=demgis
ENV PATH=/opt/conda/envs/demgis/bin:$PATH
RUN echo "conda activate demgis" >> /root/.bashrc

# ---- Workspace + entrypoint ----
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
