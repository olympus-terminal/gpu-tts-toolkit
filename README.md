<p align="center">
  <img src="assets/banner.png" alt="gpu-tts-toolkit banner" width="100%">
</p>

# gpu-tts-toolkit

GPU-based text-to-speech pipelines for converting papers, documents, and text to audio. Built on [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [Coqui TTS](https://github.com/coqui-ai/TTS), and FastSpeech2.

## Features

- **Paper-to-audio pipeline** — search, download, extract text, and synthesize speech from academic papers in one command
- **Multiple TTS engines** — Qwen3-TTS (voice clone, voice design, custom voice), Coqui TTS (multi-speaker), and FastSpeech2 (GPU-accelerated)
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
python examples/simple_tts.py
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
│   └── synthesis/         #   FastSpeech2 GPU, chunked TTS
├── tts-engines/           # Higher-level TTS wrappers
├── qwen_tts/              # Qwen3-TTS model integration
├── integrations/          # MCP server, external service connectors
├── deployment/            # HPC SLURM scripts
├── finetuning/            # Qwen3-TTS fine-tuning scripts
├── text-preprocessing/    # Text cleaning for TTS input
└── examples/              # Usage examples
```

## TTS Engines

| Engine | Use Case |
|--------|----------|
| **Qwen3-TTS** | High-quality neural TTS with voice cloning and design ([upstream repo](https://github.com/QwenLM/Qwen3-TTS)) |
| **Coqui TTS** | Open-source multi-speaker TTS ([upstream repo](https://github.com/coqui-ai/TTS)) |
| **FastSpeech2** | GPU-accelerated synthesis with TensorRT |

## Voice Cloning

See [QWEN3_VOICE_CLONING_GUIDE.md](QWEN3_VOICE_CLONING_GUIDE.md) for a detailed walkthrough of voice cloning with Qwen3-TTS.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- See `requirements.txt` and `requirements-qwen3.txt`

## License

See [LICENSE](LICENSE).
