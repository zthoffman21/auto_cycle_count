FROM node:22-slim AS dashboard-build

WORKDIR /dashboard

COPY dashboard/package.json ./package.json
COPY dashboard/package-lock.json ./package-lock.json
RUN npm ci

COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app \
    && useradd --system --create-home --home-dir /home/app --gid app app \
    && mkdir -p /home/app/.cache \
    && chown -R app:app /home/app

COPY pyproject.toml ./

FROM python-base AS development-dependencies

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    HOME=/root \
    PIP_CACHE_DIR=/root/.cache/pip \
    mkdir -p tote_vision \
    && touch tote_vision/__init__.py \
    && pip install . \
    && rm -rf tote_vision

FROM python-base AS vision-dependencies

RUN mkdir -p tote_vision \
    && touch tote_vision/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    HOME=/root \
    PIP_CACHE_DIR=/root/.cache/pip \
    pip install \
        torch==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && HOME=/root \
       PIP_CACHE_DIR=/root/.cache/pip \
       pip install ".[vision]" --extra-index-url https://download.pytorch.org/whl/cu126 \
    && rm -rf tote_vision

FROM python-base AS app-source
COPY tote_vision ./tote_vision
COPY --from=dashboard-build /dashboard/dist ./dashboard/dist
RUN mkdir -p /app/data/artifacts /app/data/training && chown -R app:app /app/data

USER app
EXPOSE 8000

CMD ["uvicorn", "tote_vision.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM app-source AS development
USER root
COPY --from=development-dependencies /usr/local /usr/local
RUN pip install . --no-deps --force-reinstall
USER app

FROM app-source AS vision
USER root
COPY --from=vision-dependencies /usr/local /usr/local
RUN pip install . --no-deps --force-reinstall
USER app

FROM development AS runtime
