# VocoLoco: free, private text-to-speech and voice cloning in your browser

[![CI](https://github.com/Magkino/vocoloco_tts/actions/workflows/ci.yml/badge.svg)](https://github.com/Magkino/vocoloco_tts/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/demo-live-brightgreen)](https://magkino.github.io/vocoloco_tts/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-yellow)](https://huggingface.co/Gigsu/vocoloco-onnx)

VocoLoco is an open-source, local-first text-to-speech web app. It runs [OmniVoice](https://github.com/k2-fsa/OmniVoice), a 600M-parameter diffusion TTS model, on its own WebGPU engine in your browser, on desktops and phones: 600+ languages, voice cloning with on-device Whisper transcription, and MP3 export. No server, no API keys, no uploads.

**Try it: [https://magkino.github.io/vocoloco_tts/](https://magkino.github.io/vocoloco_tts/)**

---

## Features

- **600+ languages**: multilingual TTS powered by OmniVoice
- **Long texts**: up to 2000 characters, split into sentence-aligned parts and joined with natural pauses
- **Pauses**: `[pause]` and `[pause 2s]` buttons, or type `[pause 1.5]` / `[pause 500ms]` anywhere in the text
- **Voice design**: control gender and pitch with simple toggles, lock a voice you like for reuse
- **Speed control**: 0.85× to 1.1× tempo, applied after generation so the voice and its pitch stay the same
- **Guided voice cloning**: record yourself (just talk) or upload audio, with a live level meter and instant voice testing
- **Trim editor**: upload or record long audio (or use any part of a generation) and pick exactly the 3-30 s to clone; a new selection snaps to pauses, the handles then move freely. Preview playhead included
- **Automatic transcription**: Whisper large-v3-turbo transcribes the selected part on your device (~100 languages, language detected automatically)
- **One download, chosen up front**: speech only (1.2 GB), or speech and voice cloning together (2.7 GB); voice cloning can be added later in one step
- **Works offline** once the models are downloaded
- **Saved voices**: cloned voices are analyzed once and cached locally, so generation with them starts fast
- **Generation library**: replay, reuse, delete, and download past generations as MP3 with AI-provenance metadata
- **Own WebGPU engine**: hand-written WGSL kernels for the model, the audio decoder, the voice encoder and Whisper, tuned to each device on first use, and its own tokenizer
- **Phone app**: a touch layout with tabs and sheets; add it to the home screen for a full-screen app (iPhone, iPad, Android)
- **Fallback**: browsers without the engine's WebGPU features run on ONNX Runtime (WebGPU or WebAssembly, slower)
- **100% private**: all synthesis runs in your browser; no analytics, no tracking, no account. The app asks before downloading models and never sends audio or text anywhere

## Requirements

| Requirement | Details |
|---|---|
| **Browser** | WebGPU with 16-bit floats (`shader-f16`): Chrome or Edge 121+ on desktop and Android, Safari 26 on Mac, any browser on iPhone and iPad with iOS 26 |
| **Storage** | ~1.2 GB for the voice model (one-time download), +1.5 GB for voice cloning (voice encoder and speech recognition) |
| **Fallback** | Other browsers (e.g. Firefox) run ONNX Runtime Web: ~2.4 GB download, slower |

## FAQ

**Is it free?** Yes, there is no account, no payment and no API key. The app is [Apache 2.0](LICENSE); the model weights are CC BY-NC (non-commercial), see [License](#license) before using it commercially.

**Does it work offline?** Yes, once the models are downloaded. The app keeps its own files on your device (a service worker) next to the models, so it opens, generates speech and clones voices without a connection. While you are online it always loads the newest version first, so updates arrive as usual.

**Is anything uploaded?** No. Text, recordings and generated audio stay in your browser; synthesis and transcription run on your device. The only network traffic is downloading the app and the model files, see [Third-Party Services](#third-party-services).

**Which browsers and phones work?** Any with WebGPU and 16-bit floats: Chrome or Edge 121+ on desktop and Android, Safari 26 on Mac, any browser on iPhone and iPad with iOS 26. Other desktop browsers fall back to ONNX Runtime Web, which is much slower; phones and tablets without these WebGPU features cannot run the app.

**How big is the download?** On the first visit you choose: about 1.2 GB for speech only, or about 2.7 GB with voice cloning (the voice encoder and speech recognition). Voice cloning can also be added later, as one download, when you create your first voice. The files are stored on your device, so it is a one-time download unless you clear the site data. The ONNX Runtime fallback needs about 2.4 GB.

## How It Works

VocoLoco runs [OmniVoice](https://github.com/k2-fsa/OmniVoice), a diffusion-based TTS model, in the browser. The pipeline:

1. **Text tokenization**: Qwen2 byte-level BPE (`workers/engine/qwen2-tokenizer.js`, id for id the same as Hugging Face tokenizers)
2. **Iterative masked diffusion**: 8-32 denoising steps with classifier-free guidance; the conditional and unconditional passes run as one batch on the engine (`workers/engine/omnivoice-engine.js`): f16 weights, f32 accumulation
3. **Post-processing**: log-softmax + CFG fusion + argmax on the GPU, inside the engine's step (ONNX Runtime fallback: `workers/gpu-postprocess.js`)
4. **Audio decoding**: the HiggsAudioV2 codec converts tokens to 24 kHz PCM on WebGPU (`workers/engine/audio-decoder.js`)
5. **MP3 export**: client-side encoding with ID3v2 metadata marking audio as AI-generated

Voice cloning first encodes the reference recording into audio tokens with the HiggsAudioV2 encoder (`workers/engine/audio-encoder.js`). Automatic transcription runs Whisper on the same engine (`workers/engine/whisper-engine.js`); its output matches PyTorch token for token on the test clips.

The engine has no dependencies and works outside VocoLoco too; its API is described in [`workers/engine/README.md`](workers/engine/README.md).

### Models

| Component | Size | Description |
|---|---|---|
| Main model | 1.1 GB (f16) | Qwen3-0.6B backbone, iterative diffusion transformer |
| Audio decoder | 85 MB | HiggsAudioV2, token-to-waveform |
| Audio encoder | 622 MB | HiggsAudioV2, waveform-to-token (voice cloning) |
| Speech recognition | 936 MB (int8) | Whisper large-v3-turbo (voice cloning) |
| Tokenizer | 11 MB | Qwen2 BPE vocabulary and merges (`tokenizer.json`) |

The ONNX Runtime fallback uses ONNX exports of the same models (main model 2.3 GB, decoder 83 MB, encoder 624 MB).

Models are hosted on [Hugging Face](https://huggingface.co/Gigsu/vocoloco-onnx) (the engine's files in `engine/`) and stored on your device after the first download. Large files are fetched as parallel 32 MB ranges, so the first download is fast and resumes after an interruption.

## Project Structure

```
vocoloco_tts/
├── index.html              # Main page (Tailwind CSS, pre-built)
├── app.js                  # UI logic, generation, voices, library, settings
├── mobile.css, mobile.js   # Phone layout: tabs, sheets, keyboard handling
├── manifest.json, icons/   # Home-screen app
├── sw.js                   # Service worker: offline use, newest version first while online
├── player.js               # StreamingPlayer: chunked playback, seek, playhead, waveform
├── text-chunker.js         # Sentence-aligned parts and [pause] tags
├── ui-dialogs.js           # Toasts and confirm dialogs (replaces native dialogs)
├── trim-editor.js          # Pick the part of a longer clip to use as a voice reference
├── workers/
│   ├── tts-worker.js       # Diffusion loop, reference encoding, cancel; engine or ONNX Runtime
│   ├── engine/             # The WebGPU engine: main model, audio decoder, voice encoder, Whisper, tokenizer
│   ├── asr-worker.js       # Whisper transcription for the voice wizard
│   ├── encoder-worker.js   # Voice encoder on its own, for phones (one model in memory at a time)
│   ├── download-worker.js  # Voice cloning's models as one download
│   ├── model-files.js      # Parallel ranged downloads, Cache API and on-disk (OPFS) storage
│   ├── voice-pitch.js      # A reference voice's pitch, for the gender hint
│   ├── unmask-schedule.js  # Diffusion unmasking schedule (port of OmniVoice)
│   └── gpu-postprocess.js  # WebGPU post-processing for the ONNX Runtime fallback
├── audio-postprocess.js    # Chunk joins, reference prep, trim selection helpers
├── time-stretch.js         # Pitch-preserving tempo change (WSOLA) for the speed slider
├── duration-estimator.js   # Estimates output length from input text (port of OmniVoice)
├── sentence-buffer.js      # Abbreviation-aware sentence splitting
├── lib/
│   ├── lamejs.min.js       # MP3 encoder (self-hosted)
│   └── LICENSE-lamejs      # its licence (LGPL-3.0)
├── tests/                  # Unit tests (node:test, run in Docker)
├── scripts/
│   └── fetch-models.sh     # Mirror the models into models/ for local dev
├── .github/                # CI, Dependabot, issue and pull request templates
├── Dockerfile              # Dev server and test container (also used by CI)
├── docker-compose.yml      # Runs them: web (port 8090), dev (tests, tools)
├── package.json            # Tool versions: Tailwind, and the ONNX Runtime the app loads from a CDN
├── tailwind.css            # Pre-built Tailwind CSS
├── tailwind.config.cjs     # Tailwind config (for rebuilds)
└── build-tailwind.sh       # Rebuild CSS via Docker
```

## Development

The project is vanilla JavaScript with no build step required. Tailwind CSS is pre-built and committed.

Everything runs in Docker, no local Node or Python needed.

**Serve locally** (edits sync into the container, open http://localhost:8090):

```bash
docker compose up --watch web
```

**Local model mirror** (recommended): download the models once into `models/` (git-ignored). On localhost the app then loads them from the dev server instead of re-downloading them on every reload (~5.5 GB, engine and ONNX files):

```bash
docker run --rm -v "$PWD":/src -w /src curlimages/curl sh scripts/fetch-models.sh
docker compose up --build --watch web
```

**Run the unit tests:**

```bash
docker compose run --rm --build dev node --test tests/*.test.js
```

The same tests and a syntax check run on GitHub Actions for every push and pull request (`.github/workflows/ci.yml`).

**Rebuild Tailwind CSS** (after changing HTML classes):

```bash
docker compose run --rm --build dev sh build-tailwind.sh > tailwind.css
```

**Force the ONNX Runtime CPU fallback** (for testing):

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
| [Hugging Face](https://huggingface.co/Gigsu/vocoloco-onnx) | Model weights (~1.2 GB, +1.5 GB for voice cloning) | First download / after clearing site data |
| [jsDelivr](https://www.jsdelivr.com/) | ONNX Runtime, only for the fallback without WebGPU | First use / after cache clear |
| [GitHub Pages](https://pages.github.com/) | Hosting the app itself | Visits while online (it then works offline) |

Once models are cached, no network requests are made during synthesis.

## License

- **VocoLoco app** (this repository): [Apache License 2.0](LICENSE)
- **Model weights** (downloaded by the app from [Hugging Face](https://huggingface.co/Gigsu/vocoloco-onnx), not part of this repository), converted to ONNX and to the engine's format:
  - **OmniVoice main model** by Xiaomi Corp. (k2-fsa): [CC BY-NC](https://huggingface.co/k2-fsa/OmniVoice#license) (non-commercial)
  - **Audio tokenizer** (voice encoder and audio decoder, shipped with OmniVoice): the [Higgs Audio v2 tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer) by Boson AI, under the [Boson Higgs Audio 2 Community License](https://huggingface.co/k2-fsa/OmniVoice/blob/main/audio_tokenizer/LICENSE), which is based on the Meta Llama 3 Community License. Use must follow the [Llama 3 Acceptable Use Policy](https://www.llama.com/llama3/use-policy); services with more than 100,000 annual active users need an expanded licence from Boson AI.
  - **Speech recognition**: [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) by OpenAI (MIT)
  - **Text tokenizer**: Qwen2 BPE (Apache 2.0, Qwen team), as used by OmniVoice

Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Rights Reserved.

> **Licence history (OmniVoice main model):** VocoLoco and its ONNX exports were created on 2026-04-09 ([export upload](https://huggingface.co/Gigsu/vocoloco-onnx/commit/0cf41e88c9b3c594babf64a6dd613196c98b4900)), when the OmniVoice model card listed the weights as Apache 2.0 ([card as of 2026-04-05](https://huggingface.co/k2-fsa/OmniVoice/blob/29cde0ee295ee673d33e9ab570e7bbbe761c33b3/README.md), `license: apache-2.0`). On 2026-07-03 the OmniVoice authors relicensed the pre-trained model as CC BY-NC, citing their training data ([model card change](https://huggingface.co/k2-fsa/OmniVoice/commit/c5fdb5ccb189668d56333f77ba2629f4cd7535f4)). The weight files themselves were not changed. VocoLoco follows the current upstream licence.

## Attribution

Built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi Corp (k2-fsa), the [Higgs Audio v2 tokenizer](https://github.com/boson-ai/higgs-audio) by Boson AI and [Whisper](https://github.com/openai/whisper) by OpenAI. The WebGPU engine (`workers/engine/`) is VocoLoco's own implementation of these architectures.

| Third-party code | Use | Licence |
|---|---|---|
| [lamejs](https://github.com/zhuker/lamejs) (`lib/lamejs.min.js`, unmodified) | MP3 export | LGPL-3.0 ([`lib/LICENSE-lamejs`](lib/LICENSE-lamejs)); a separate file you can replace |
| [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (loaded from jsDelivr) | Fallback without the engine | MIT, Microsoft |
| [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss) (build only) | `tailwind.css` | MIT |
| Ports of [OmniVoice](https://github.com/k2-fsa/OmniVoice) code: `duration-estimator.js`, `workers/unmask-schedule.js`, `addEndPunctuation` in `sentence-buffer.js` | Duration estimate, unmasking schedule, reference punctuation | Apache 2.0, Copyright 2026 Xiaomi Corp.; translated to JavaScript |

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md), and please open an issue first for larger changes. To report a security problem, see [SECURITY.md](SECURITY.md).
