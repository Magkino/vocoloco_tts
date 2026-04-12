#!/bin/sh
# Rebuild tailwind.css from index.html + app.js class usage.
# Run: docker run --rm -v "$(pwd)":/src node:20-slim sh /src/build-tailwind.sh
set -e
mkdir -p /build
cp /src/tailwind.config.js /src/tailwind-input.css /build/
cp /src/*.html /src/*.js /build/ 2>/dev/null || true
cp -r /src/workers /build/workers 2>/dev/null || true
cd /build
npm init -y >/dev/null 2>&1
npm i -D tailwindcss@3 @tailwindcss/forms @tailwindcss/container-queries >/dev/null 2>&1
npx tailwindcss -i tailwind-input.css -o /src/tailwind.css --minify
echo "Done. Output: /src/tailwind.css"
