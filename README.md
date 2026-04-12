# VocoLoco

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/demo-live-brightgreen)](https://magkino.github.io/vocoloco_tts/)

**Text-to-speech that runs entirely in your browser.** No server, no API keys, no data leaves your device.

VocoLoco uses WebGPU and WebAssembly to run a 600M-parameter diffusion TTS model client-side. Type text, pick a voice, and get natural speech, all locally.

> **Try it now:** [magkino.github.io/vocoloco_tts](https://magkino.github.io/vocoloco_tts/)

---

## Features

- **600+ languages**: multilingual TTS powered by OmniVoice
- **Voice design**: control gender and pitch with simple toggles
- **Voice cloning**: upload or record a 5-10s audio sample to clone any voice
- **Saved voices**: store cloned voices locally in the browser for reuse
- **Generation library**: replay and download past generations as MP3 with AI-provenance metadata
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
├── app.js                  # UI logic, audio playback, history, MP3 export
├── workers/
│   ├── tts-worker.js       # ONNX inference, diffusion loop
│   └── gpu-postprocess.js  # WebGPU compute shader for post-processing
├── duration-estimator.js   # Estimates output length from input text
├── sentence-buffer.js      # Text splitting and buffering
├── lib/
│   └── lamejs.min.js       # MP3 encoder (self-hosted)
├── tailwind.css            # Pre-built Tailwind CSS
├── tailwind.config.js      # Tailwind config (for rebuilds)
└── build-tailwind.sh       # Rebuild CSS via Docker
```

## Development

The project is vanilla JavaScript with no build step required. Tailwind CSS is pre-built and committed.

**Serve locally:**

```bash
python3 -m http.server 8080
```

**Rebuild Tailwind CSS** (after changing HTML classes):

```bash
docker run --rm -v "$(pwd)":/src node:20-slim sh /src/build-tailwind.sh
```

**Force CPU mode** (for testing):

```
http://localhost:8080?cpu
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
- **ONNX models**: Apache 2.0, derived from [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi/k2-fsa

## Attribution

Built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi Corp (k2-fsa). Uses [ONNX Runtime Web](https://onnxruntime.ai/) by Microsoft. MP3 encoding by [lamejs](https://github.com/zhuker/lamejs).

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.
