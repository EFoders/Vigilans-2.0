# syntax=docker/dockerfile:1
#
# Vigilans and its simulated sensor hub, as separate images; the build context is the
# repository root, because the engine, the hub and the contract are one uv workspace.
#
#   test     ruff, mypy --strict and pytest -- what CI runs
#   runtime  the engine as an unprivileged user (the default). It runs no simulator: it
#            listens for observation.v1 over HTTP, from whatever hub is pointed at it.
#   hub      the simulated sensor hub (ADR-0010): plays scenario.v2 worlds and sends
#            observation.v1 to Vigilans. Swap it for real equipment and Vigilans cannot tell.
#
# The workspace is installed editable on purpose: scenarios and the synthetic library are
# found beside the source tree, exactly as in a checkout.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first, keyed on the lockfile, so a source change does not re-resolve.
COPY pyproject.toml uv.lock .python-version ./
COPY contract/pyproject.toml contract/README.md contract/
COPY vigilans/pyproject.toml vigilans/README.md vigilans/
COPY hub/pyproject.toml hub/README.md hub/

FROM base AS runtime-deps
RUN uv sync --locked --no-dev --package vigilans --no-install-workspace

FROM runtime-deps AS runtime-build
COPY contract/ contract/
COPY vigilans/ vigilans/
RUN uv sync --locked --no-dev --package vigilans

FROM base AS hub-build
RUN uv sync --locked --no-dev --package vigilans-hub --no-install-workspace
COPY contract/ contract/
COPY vigilans/ vigilans/
COPY hub/ hub/
RUN uv sync --locked --no-dev --package vigilans-hub

FROM base AS test
RUN uv sync --locked --all-packages --no-install-workspace
COPY . .
RUN uv sync --locked --all-packages
ENV PATH="/app/.venv/bin:$PATH"
CMD ["sh", "-c", "ruff check . && ruff format --check . && mypy && pytest -q"]

FROM ${PYTHON_IMAGE} AS hub
RUN useradd --create-home --uid 10002 hub
WORKDIR /app
COPY --from=hub-build --chown=hub:hub /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HUB_VIGILANS_URL=http://vigilans:8092 \
    HUB_SCENARIO=mixed \
    HUB_RATE=1
USER hub
ENTRYPOINT ["vigilans-hub"]
CMD ["sim"]

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 vigilans
WORKDIR /app
COPY --from=runtime-build --chown=vigilans:vigilans /app /app
# Defaults as environment, not as flags in CMD: a flag outranks the environment, so a flag
# baked into CMD would silently override the variable (spec §12).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    VIGILANS_LISTEN=0.0.0.0:8091 \
    VIGILANS_INGEST=0.0.0.0:8092
USER vigilans
EXPOSE 8091 8092
# Healthy once it can take observations: the picture feed only starts with the first one.
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8092/observations/v1/health', timeout=2)" || exit 1
ENTRYPOINT ["vigilans"]
CMD ["run"]
