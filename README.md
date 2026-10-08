# VocoLoco

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/demo-live-brightgreen)](https://magkino.github.io/vocoloco_tts/)

**Text-to-speech that runs entirely in your browser.** No server, no API keys, no data leaves your device.

VocoLoco uses WebGPU and WebAssembly to run a 600M-parameter diffusion TTS model client-side. Type text, pick a voice, and get natural speech, all locally.

> **Try it now:** [magkino.github.io/vocoloco_tts](https://magkino.github.io/vocoloco_tts/)

---

## Features

- **600+ languages**: multilingual TTS powered by OmniVoice
- **Streamed generation**: long texts (up to 2000 characters) are split into sentences and start playing within seconds
- **Voice design**: control gender and pitch with simple toggles, lock a voice you like for reuse
- **Guided voice cloning**: a step-by-step wizard with curated reading scripts (the transcript fills itself in), live level meter, and instant voice testing
- **Trim editor**: upload or record longer audio (or use any part of a generation) and pick exactly the 3-15 s to clone, with snap-to-pause handles and a preview playhead
- **Saved voices**: cloned voices are analyzed once and cached locally, so generation with them starts fast
- **Generation library**: replay, reuse, delete, and download past generations as MP3 with AI-provenance metadata
- **GPU-accelerated**: WebGPU for model inference, with a custom compute shader for post-processing
- **CPU fallback**: works without WebGPU via WebAssembly (slower, but functional)
- **100% private**: all synthesis runs in your browser; no audio or text ever leaves your device

## Requirements

| Requirement | Details |
|---|---|
| **Browser** | Chrome 113+ or Edge 113+ (WebGPU required for full speed) |
| **GPU** | Dedicated GPU with WebGPU support recommended |
| **Storage** | ~3 GB for cached models (one-time download) |
| **Fallback** | Firefox and non-WebGPU browsers work via WASM (significantly slower) |

## How It Works

VocoLoco runs [OmniVoice](https://github.com/k2-fsa/OmniVoice), a diffusion-based TTS model, in the browser using [ONNX Runtime Web](https://onnxruntime.ai/). The full pipeline:

1. **Text tokenization**: Qwen2 BPE tokenizer via [transformers.js](https://huggingface.co/docs/transformers.js)
2. **Iterative masked diffusion**: 8-32 denoising steps with classifier-free guidance
3. **Post-processing**: log-softmax + CFG fusion + argmax, offloaded to a WebGPU compute shader when available (`workers/gpu-postprocess.js`)
4. **Audio decoding**: HiggsAudioV2 codec converts tokens to 24kHz PCM
5. **MP3 export**: client-side encoding with ID3v2 metadata marking audio as AI-generated

### Models

| Component | Size | Description |
|---|---|---|
| Main model | 2.3 GB (sharded) | Qwen3-0.6B backbone, iterative diffusion transformer |
| Audio decoder | 83 MB | HiggsAudioV2, token-to-waveform |
| Audio encoder | 624 MB | HiggsAudioV2, waveform-to-token (voice cloning) |
| Tokenizer | ~2 MB | Qwen2 BPE (loaded via transformers.js) |

Models are hosted on [Hugging Face](https://huggingface.co/Gigsu/vocoloco-onnx) and cached in the browser after first download.

## Project Structure

```
vocoloco_tts/
├── index.html              # Main page (Tailwind CSS, pre-built)
├── app.js                  # UI logic, streaming orchestration, voices, library, settings
├── player.js               # StreamingPlayer: chunked playback, seek, playhead, waveform
├── text-chunker.js         # Sentence-aligned chunking for streamed generation
├── ui-dialogs.js           # Toasts and confirm dialogs (replaces native dialogs)
├── trim-editor.js          # Pick the part of a longer clip to use as a voice reference
├── workers/
│   ├── tts-worker.js       # ONNX inference, diffusion loop, reference encoding, cancel
│   ├── unmask-schedule.js  # Diffusion unmasking schedule (port of OmniVoice)
│   └── gpu-postprocess.js  # WebGPU compute shader for post-processing
├── audio-postprocess.js    # Chunk joins, reference prep, trim selection helpers
├── duration-estimator.js   # Estimates output length from input text
├── sentence-buffer.js      # Abbreviation-aware sentence splitting
├── lib/
│   └── lamejs.min.js       # MP3 encoder (self-hosted)
├── tests/                  # Unit tests (node:test, run in Docker)
├── scripts/
│   └── fetch-models.sh     # Mirror the models into models/ for local dev
├── docker-compose.yml      # Local dev server + test container
├── tailwind.css            # Pre-built Tailwind CSS
├── tailwind.config.js      # Tailwind config (for rebuilds)
└── build-tailwind.sh       # Rebuild CSS via Docker
```

## Development

The project is vanilla JavaScript with no build step required. Tailwind CSS is pre-built and committed.

Everything runs in Docker, no local Node or Python needed.

**Serve locally** (edits sync into the container, open http://localhost:8090):

```bash
docker compose up --watch web
```

**Local model mirror** (recommended): download the models once into `models/` (git-ignored). On localhost the app then loads them from the dev server instead of re-downloading ~3 GB on every reload:

```bash
docker run --rm -v "$PWD":/src -w /src curlimages/curl sh scripts/fetch-models.sh
docker compose up --build --watch web
```

**Run the unit tests:**

```bash
docker compose run --rm --build dev node --test
```

**Rebuild Tailwind CSS** (after changing HTML classes):

```bash
docker run --rm -v "$(pwd)":/src node:20-slim sh /src/build-tailwind.sh
```

**Force CPU mode** (for testing):

```
http://localhost:8090?cpu
```

## EU AI Act Compliance

VocoLoco implements transparency measures in accordance with EU AI Act Article 50:

- **Machine-readable metadata**: all downloaded MP3 files contain ID3v2 tags identifying the audio as AI-generated synthetic speech
- **User disclosure**: the app displays legal obligations for users who publish or distribute generated audio
- **Provenance tracking**: metadata includes software identification, creation timestamps, and an AI-generation disclaimer

See the in-app disclaimer for full details.

## Third-Party Services

All synthesis runs locally, but the app fetches resources from external services on first load:

| Service | Purpose | When |
|---|---|---|
| [Hugging Face](https://huggingface.co/Gigsu/vocoloco-onnx) | Model weights (~3 GB) | First use / after cache clear |
| [jsDelivr](https://www.jsdelivr.com/) | ONNX Runtime + transformers.js | First use / after cache clear |
| [GitHub Pages](https://pages.github.com/) | Hosting the app itself | Every visit |

Once models are cached, no network requests are made during synthesis.

## License

- **VocoLoco app**: [Apache License 2.0](LICENSE)
- **ONNX models**: [CC BY-NC](https://huggingface.co/k2-fsa/OmniVoice#license) (non-commercial), converted to ONNX from the [OmniVoice](https://github.com/k2-fsa/OmniVoice) weights by Xiaomi/k2-fsa

> **Licence history:** VocoLoco and its ONNX exports were created on 2026-04-09, when the OmniVoice model card listed the weights as Apache 2.0. On 2026-07-03 the OmniVoice authors relicensed the pre-trained model as CC BY-NC, citing their training data ([model card change](https://huggingface.co/k2-fsa/OmniVoice/commit/c5fdb5ccb189668d56333f77ba2629f4cd7535f4)). The weight files themselves were not changed. VocoLoco follows the current upstream licence.

## Attribution

Built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi Corp (k2-fsa). Uses [ONNX Runtime Web](https://onnxruntime.ai/) by Microsoft. MP3 encoding by [lamejs](https://github.com/zhuker/lamejs).

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.
