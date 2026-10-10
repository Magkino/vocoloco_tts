# Contributing to VocoLoco

Thanks for your interest! Bug reports, fixes and improvements are welcome. For anything larger than a small fix, please [open an issue](https://github.com/Magkino/vocoloco_tts/issues) first, so we can agree on the approach before you spend time on it.

Security problems: please don't open an issue, see [SECURITY.md](SECURITY.md).

## Development setup

Everything runs in Docker; you don't need Node or Python on your machine. See [Development](README.md#development) for the details and the local model mirror.

```bash
docker compose up --watch web                                     # app on http://localhost:8090
docker compose run --rm --build dev node --test tests/*.test.js   # unit tests
```

## Guidelines

- **No build step.** The app is plain ES modules served as static files. Don't add a bundler, a framework or npm runtime dependencies.
- **Privacy first.** The app must not send text, recordings or audio anywhere, and must not load code or data from new hosts. New third-party code needs a compatible licence and an entry in the README's licence and attribution tables.
- **Style.** Match the surrounding code: 2-space indentation, `const`/`let`, small functions, comments that explain *why*. `.editorconfig` has the basics.
- **Tests.** Logic that runs without a GPU (text handling, audio processing, schedules, the tokenizer) gets a unit test in `tests/` (`node:test`). The CI runs the tests and a syntax check on every push and pull request.
- **Cached modules.** GitHub Pages and browsers cache the scripts, and a stale copy next to a new one can break the app. Imports carry a `?v=N` query for this: when you change a module, bump `N` where it is imported. A new file the app loads must also be listed in `sw.js` (a test checks this).
- **Engine changes** (`workers/engine/`): every tile configuration the tuner can pick must give bit-identical results; tuning may only change the speed. In the pull request, say how you checked the output (for example the same tokens or audio for the same seed as before) and on which browsers and devices.

## Pull requests

- One topic per pull request, with a short description of what changed and why.
- Say which browsers and devices you tested on (WebGPU behaviour differs a lot between Chrome, Safari and phones).
- Make sure the unit tests pass.

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE), like the rest of the app.
