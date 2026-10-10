# VocoLoco engine

The WebGPU inference engine behind [VocoLoco](https://magkino.github.io/vocoloco_tts/): hand-written WGSL kernels for the [OmniVoice](https://github.com/k2-fsa/OmniVoice) text-to-speech model, its audio codec (the [Higgs Audio v2 tokenizer](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer): audio decoder and voice encoder) and [Whisper](https://github.com/openai/whisper) large-v3-turbo speech recognition, plus a Qwen2 byte-level BPE tokenizer in plain JavaScript.

Plain ES modules: no dependencies, no build step, no WebAssembly. Model weights are not included (see [Weights](#weights)).

## Modules

| Module | Entry point (`package.json`) | Exports | Does |
|---|---|---|---|
| `omnivoice-engine.js` | `./omnivoice` | `OmniVoiceEngine`, `httpSource` | One diffusion step of OmniVoice: forward pass and classifier-free guidance on the GPU |
| `audio-decoder.js` | `./audio-decoder` | `AudioDecoder` | Audio tokens → 24 kHz audio |
| `audio-encoder.js` | `./audio-encoder` | `AudioEncoder` | 24 kHz audio → audio tokens (voice cloning) |
| `whisper-engine.js` | `./whisper` | `WhisperEngine` | 16 kHz audio → text and language |
| `qwen2-tokenizer.js` | `./qwen2-tokenizer` | `Qwen2Tokenizer` | Text → token ids, the same ids as Hugging Face tokenizers; runs anywhere, no GPU |

The other exports (kernel generators, tile lists, `compile`) are shared between the modules and may change in any release.

## Requirements

- WebGPU. OmniVoice (f16 weights) and Whisper (int8 weights) need the `shader-f16` feature; the f32 decoder and encoder don't. In practice: Chrome or Edge 121+ on desktop and Android, Safari 26 on Mac, iOS and iPadOS 26.
- Request the device with the adapter's buffer limits, the large weight tensors need them:

```js
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const lim = adapter.limits;
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'timestamp-query'].filter((f) => adapter.features.has(f)),
  requiredLimits: {
    maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
    maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
  },
});
```

`timestamp-query` is optional: with it, `OmniVoiceEngine` times its tile candidates on the GPU, otherwise by wall clock.

## Weights

Each model is a JSON manifest (configuration and tensor table) and one binary file, hosted on Hugging Face in [`Gigsu/vocoloco-onnx`](https://huggingface.co/Gigsu/vocoloco-onnx/tree/main/engine):

| Model | Files (`https://huggingface.co/Gigsu/vocoloco-onnx/resolve/main/…`) | Size |
|---|---|---|
| OmniVoice main model | `engine/omnivoice-f16.json`, `engine/omnivoice-f16.bin` | 1.1 GB |
| Audio decoder | `engine/decoder-f32.json`, `engine/decoder-f32.bin` | 85 MB |
| Voice encoder | `engine/encoder-f32.json`, `engine/encoder-f32.bin` | 622 MB |
| Whisper large-v3-turbo | `engine/whisper-i8.json`, `engine/whisper-i8.bin` | 936 MB |
| Text tokenizer | `tokenizer.json` | 11 MB |

The engines read the binary file through a *source*, any object with `read(offset, bytes)` returning a `Promise<ArrayBuffer>`. `httpSource(url)` makes one HTTP range request per read, which suits a local server. For the network, download the file once, store it (for example in the Origin Private File System) and read from the stored copy:

```js
const source = { read: (offset, bytes) => file.slice(offset, offset + bytes).arrayBuffer() }; // file: a Blob or File
```

## Install

```sh
npm install @vocoloco/engine
```

Without a bundler, map the package in an import map, or import the files directly (they have no dependencies), for example from `https://cdn.jsdelivr.net/npm/@vocoloco/engine@0.1.0/omnivoice-engine.js`.

## Usage

Text tokenizer (also in Node):

```js
import { Qwen2Tokenizer } from '@vocoloco/engine/qwen2-tokenizer';

const HF = 'https://huggingface.co/Gigsu/vocoloco-onnx/resolve/main';
const tok = new Qwen2Tokenizer(await (await fetch(`${HF}/tokenizer.json`)).json());
tok.encode('<|text_start|>Hello there.<|text_end|>'); // ids; added tokens are matched as written, none are inserted
```

Speech recognition:

```js
import { WhisperEngine } from '@vocoloco/engine/whisper';
import { httpSource } from '@vocoloco/engine/omnivoice';

const manifest = await (await fetch(`${HF}/engine/whisper-i8.json`)).json();
const whisper = await WhisperEngine.load(device, manifest, httpSource(`${HF}/engine/whisper-i8.bin`));
const { text, lang } = await whisper.transcribe(pcm16k); // Float32Array, 16 kHz mono, up to 30 s
```

Voice encoder and audio decoder:

```js
import { AudioEncoder } from '@vocoloco/engine/audio-encoder';
import { AudioDecoder } from '@vocoloco/engine/audio-decoder';

const encoder = await AudioEncoder.load(device, encManifest, encSource);
const { codes, T } = await encoder.encode(pcm24k); // length a multiple of 960 samples; codes: Int32Array [8][T]

const decoder = await AudioDecoder.load(device, decManifest, decSource);
const pcm = await decoder.decode(codes, T);         // Float32Array, T · 960 samples at 24 kHz
```

Text-to-speech: `OmniVoiceEngine` runs the model's steps; the prompt and the unmasking loop are the caller's. One generation with `L` prompt positions, of which the last `T` are the audio to generate:

```js
import { OmniVoiceEngine } from '@vocoloco/engine/omnivoice';

const engine = await OmniVoiceEngine.load(device, manifest, source, { act: 'f16' });
await engine.tuneFor(L, T);                  // first call only: picks this device's fastest tiles (same results)
const plan = engine.plan(L, T);              // working buffers for this shape
await plan.setInputs(condIds, audioStart);   // Int32Array [8][L]: style, text, reference audio tokens, then T mask ids
for (let step = 0; step < steps; step++) {
  plan.setTargets(tokens);                   // [8][T], the mask id where a token isn't decided yet
  await plan.step(pred, scores, { guidanceScale: 2, layerPenalty: 5 }); // per position: best token and its score
  // commit the best-scoring masked positions to `tokens`, by the unmasking schedule
}
plan.destroy();
```

VocoLoco's worker shows the full pipeline: the prompt layout (`prepareInferenceInputs`) and the loop (`generateIterative`) in [`workers/tts-worker.js`](https://github.com/Magkino/vocoloco_tts/blob/main/workers/tts-worker.js), the schedule in [`workers/unmask-schedule.js`](https://github.com/Magkino/vocoloco_tts/blob/main/workers/unmask-schedule.js).

All engines have `destroy()`, which frees their GPU memory.

## Licence

The code in this folder is licensed under the [Apache License 2.0](https://github.com/Magkino/vocoloco_tts/blob/main/LICENSE).

The model weights are **not** covered by that licence. They have their own terms, which apply to anything you build with them:

- **OmniVoice main model** by Xiaomi Corp. (k2-fsa): [CC BY-NC](https://huggingface.co/k2-fsa/OmniVoice#license), non-commercial use only.
- **Audio decoder and voice encoder**: the Higgs Audio v2 tokenizer by Boson AI, under the [Boson Higgs Audio 2 Community License](https://huggingface.co/k2-fsa/OmniVoice/blob/main/audio_tokenizer/LICENSE), based on the Meta Llama 3 Community License. Use must follow the [Llama 3 Acceptable Use Policy](https://www.llama.com/llama3/use-policy); services with more than 100,000 annual active users need an expanded licence from Boson AI.
- **Whisper large-v3-turbo** by OpenAI: MIT.
- **Qwen2 tokenizer** (vocabulary and merges) by the Qwen team: Apache 2.0.

Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Rights Reserved.

This engine is VocoLoco's own implementation of these architectures and is not affiliated with or endorsed by Xiaomi, k2-fsa, Boson AI, Meta or OpenAI.
