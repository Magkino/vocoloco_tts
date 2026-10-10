#!/bin/sh
# Rebuilds tailwind.css from the classes used in index.html, *.js and workers/*.js, with the
# versions in package-lock.json (writes the CSS to stdout). Run:
#   docker compose run --rm --build dev sh build-tailwind.sh > tailwind.css
set -e
npm ci --omit=optional --no-audit --no-fund >&2
npm run --silent build:css
