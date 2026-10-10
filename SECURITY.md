# Security policy

## Supported versions

Only the live app at [magkino.github.io/vocoloco_tts](https://magkino.github.io/vocoloco_tts/), built from the `main` branch, gets security fixes. Older tags are not patched.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: [**Report a vulnerability**](https://github.com/Magkino/vocoloco_tts/security/advisories/new) (Security tab of this repository). Do not open a public issue, pull request or discussion for a security problem.

Please include:

- what an attacker can do, and under which conditions
- steps to reproduce (a minimal page, input text or audio file if needed)
- browser, operating system and device

You can expect a first reply within a week. Once a fix is live, the advisory is published with credit to you, unless you prefer otherwise.

## Scope

In scope:

- the app in this repository (`index.html`, the scripts and workers, the WebGPU engine in `workers/engine/`)
- anything that sends text, recordings or generated audio off the device, or loads code or models from hosts other than the ones listed under [Third-Party Services](README.md#third-party-services)
- the development setup (`Dockerfile`, `docker-compose.yml`, `scripts/`)

Out of scope:

- the model weights and their behaviour (report to the upstream projects listed in the [README](README.md#license))
- the third-party services themselves (Hugging Face, jsDelivr, GitHub Pages)
- misuse of generated or cloned voices; this is not a vulnerability in the app (see the in-app disclaimer)
