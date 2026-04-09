# VocoLoco

**Local text-to-speech that runs entirely in your browser.**

No server, no API keys, no data leaves your device. VocoLoco uses WebAssembly and WebGPU to run a 600M parameter diffusion TTS model client-side.

## Features

- **Text-to-speech** in 600+ languages
- **Voice design** — control gender and pitch
- **Voice cloning** — record or upload a reference audio to clone any voice
- **Save voices** — store cloned voices locally for reuse
- **Generation library** — replay and download past generations as WAV
- **100% private** — everything runs in your browser, nothing is sent to any server

## How It Works

VocoLoco runs [OmniVoice](https://github.com/k2-fsa/OmniVoice) (a diffusion-based TTS model) in the browser using [ONNX Runtime Web](https://onnxruntime.ai/). The iterative masked-diffusion generation loop is implemented in JavaScript. Models are hosted on [HuggingFace](https://huggingface.co/Gigsu/vocoloco-onnx) and cached locally after first download.

**Models:**
- Main model: Qwen3-0.6B backbone (2.3 GB, sharded)
- Audio decoder: HiggsAudioV2 (83 MB)
- Audio encoder: HiggsAudioV2 for voice cloning (624 MB)
- Text tokenizer: Qwen2 BPE via [transformers.js](https://huggingface.co/docs/transformers.js)

## Requirements

- Desktop browser with WebGPU support (Chrome 113+, Edge 113+)
- GPU with sufficient VRAM (WebGPU accelerated)
- ~3 GB storage for cached models (one-time download)

## License

- **VocoLoco app**: MIT
- **ONNX models**: Apache 2.0 (derived from [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi/k2-fsa)

## Attribution

Built on [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi Corp (k2-fsa). Uses [ONNX Runtime Web](https://onnxruntime.ai/) by Microsoft.
