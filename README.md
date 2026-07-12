<p align="center">
  <img src="assets/banner.png" alt="gpu-tts-toolkit banner" width="100%">
</p>

# gpu-tts-toolkit

GPU-based text-to-speech pipelines for converting papers, documents, and text to audio. The supported examples use [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS); legacy Coqui TTS wrappers and an experimental FastSpeech2 prototype are also retained in the repository.

## Features

- **Paper-to-audio pipeline** — search, download, extract text, and synthesize speech from academic papers in one command
- **Qwen3-TTS workflows** — voice cloning, voice design, and built-in CustomVoice speakers
- **GPU-accelerated** — optimized for NVIDIA GPUs including DGX Spark (128GB unified memory)
- **Text preprocessing** — LaTeX extraction, text cleaning, chunked synthesis for long documents
- **MCP integration** — Model Context Protocol server for TTS
- **HPC deployment** — SLURM batch scripts for cluster processing
- **Voice cloning guide** — step-by-step Qwen3-TTS voice cloning workflow

## Quick Start

```bash
# Install on DGX Spark
./install_dgx.sh

# Convert a paper to audio
python pipeline/paper_to_audio.py 'biomimetic concrete' --papers 3

# Simple TTS
python examples/simple_tts.py --text "Hello from Qwen3-TTS"
```

## Project Structure

```
gpu-tts-toolkit/
├── pipeline/              # End-to-end paper-to-audio pipeline
│   ├── paper_search.py    #   Search for papers by query
│   ├── paper_download.py  #   Download PDFs from open access sources
│   ├── pdf_to_text.py     #   Extract and clean text for TTS
│   └── paper_to_audio.py  #   Orchestrate the full pipeline
├── core-engines/          # Low-level synthesis engines
│   └── synthesis/         #   Experimental FastSpeech2 prototype, chunked TTS
├── tts-engines/           # Higher-level TTS wrappers
├── qwen_tts/              # Qwen3-TTS model integration
├── integrations/          # MCP server, external service connectors
├── deployment/            # HPC SLURM scripts
├── finetuning/            # Qwen3-TTS fine-tuning scripts
├── text-preprocessing/    # Text cleaning for TTS input
└── examples/              # Usage examples
```

## TTS Engines

| Engine | Status | Use Case |
|--------|--------|----------|
| **Qwen3-TTS** | Supported | Neural TTS with voice cloning, voice design, and CustomVoice speakers ([upstream repo](https://github.com/QwenLM/Qwen3-TTS)) |
| **Coqui TTS** | Legacy wrapper | Optional multi-speaker experiments; Coqui TTS is not installed by this repository's requirements ([upstream repo](https://github.com/coqui-ai/TTS)) |
| **FastSpeech2** | Experimental prototype | Architecture experiments only; no trained checkpoint or functional vocoder is included |

## Voice Cloning

See [QWEN3_VOICE_CLONING_GUIDE.md](QWEN3_VOICE_CLONING_GUIDE.md) for a detailed walkthrough of voice cloning with Qwen3-TTS.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- See `requirements.txt` and `requirements-qwen3.txt`

## Repository History & Attribution

This toolkit was assembled by merging several independent repositories, one of which was a fork of [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). As a result, the git history and GitHub contributor list include authors from those upstream projects who did not contribute to this toolkit:

- **Xiong Wang** and **Anton Vlasjuk** are [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) authors (Alibaba Qwen team). Their commits are upstream model code carried over from the fork.
- **Claude** (Anthropic) appears due to commit trailers from AI-assisted code generation. Claude is a tool, not a contributor.

The `qwen_tts/` directory (~9,200 lines) contains upstream Qwen3-TTS model and tokenizer code. Everything else — pipelines, TTS engine wrappers, text preprocessing, deployment scripts, MCP integration, and examples (~9,300 lines) — is original work.

## License

See [LICENSE](LICENSE).
