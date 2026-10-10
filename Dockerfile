# syntax=docker/dockerfile:1
# Local development and CI (docker-compose.yml builds one stage per service).
# The app itself is static files and needs no build.

# Unit tests and tooling: docker compose run --rm --build dev node --test tests/*.test.js
FROM node:26.9.0-slim AS dev
WORKDIR /src
COPY --exclude=models . .

# Dev server: docker compose up --watch web → http://localhost:8090
FROM nginx:1.30.5-alpine AS web
RUN printf 'server {\n  listen 80;\n  root /usr/share/nginx/html;\n  add_header Cache-Control "no-store";\n}\n' > /etc/nginx/conf.d/default.conf
# Separate layer: code edits don't re-copy the ~3 GB of models
COPY models/ /usr/share/nginx/html/models/
COPY --exclude=models . /usr/share/nginx/html
