# GPU TTS Toolkit

Text-to-speech tools using GPU acceleration. Useful for converting papers and documents to audio.

[![License](https://img.shields.io/github/license/olympus-terminal/gpu-tts-toolkit)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/olympus-terminal/gpu-tts-toolkit?style=social)](https://github.com/olympus-terminal/gpu-tts-toolkit/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/olympus-terminal/gpu-tts-toolkit)](https://github.com/olympus-terminal/gpu-tts-toolkit/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/olympus-terminal/gpu-tts-toolkit)](https://github.com/olympus-terminal/gpu-tts-toolkit/commits/main)
[![Tools](https://img.shields.io/badge/tools-20+-green.svg)](https://github.com/olympus-terminal/gpu-tts-toolkit)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.9+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

Text-to-speech scripts optimized for GPU use. Designed for processing scientific papers, documentation, and other text on local hardware.

## Main tool: `media_to_tts.py`

`media_to_tts.py` is the flagship pipeline of this toolkit. It accepts **any text-bearing file** (PDF, LaTeX, Markdown, plain text, or a mixed bundle), extracts the TTS-suitable prose, synthesizes audio, and runs automatic QC / hallucination screening in a single command.

```
  INPUT                EXTRACT                 SYNTH              QC
  .pdf     ─┐       ┌─ PDFTTSCleaner        ┐
  .tex     ─┤       ├─ LaTeXTTSCleaner      ├─► deep_voice_tts ─► hallucination_report.json
  .md      ─┼──►    ├─ MarkdownTTSCleaner   │      (VCTK VITS)       + _complete.mp3
  .txt     ─┤       ├─ PlainTextTTSCleaner  │
  --multi  ─┘       └─ clean_multi_files()  ┘
```

### Quick usage

```bash
# Single file — extraction + synthesis + screening
python media_to_tts.py paper.pdf --voice p246 --screen
python media_to_tts.py main.tex  --voice p246 --screen

# Multi-file bundle with spoken section headers
python media_to_tts.py --multi reviews.txt responses.md \
    --headers "Peer reviews." "Author responses." \
    --voice p246 --screen

# Extract only (no audio)
python media_to_tts.py paper.pdf -o paper_clean.txt

# Re-screen an existing TTS output directory
python media_to_tts.py --screen-only path/to/<basename>_deep_voice_<stamp>/
```

### Flags

| Flag | Meaning |
|------|---------|
| `input` (positional) | Single-file input (`.pdf` or `.tex`); omit if using `--multi` |
| `-o / --output PATH` | Output `.txt` path (default: auto-timestamped next to input) |
| `--voice ID` | Also synthesize audio with this voice (e.g. `p246`) |
| `--format {mp3,wav}` | Audio format (default: `mp3`) |
| `--screen` | Pre- and post-synthesis hallucination screening |
| `--screen-only DIR` | Analyze an existing TTS output dir (no extraction) |
| `--keep-intermediates` | Keep `chunks/` and temp `.txt` after synthesis |
| `--multi FILE [FILE ...]` | Multi-file mode; accepts `.txt .md .tex .pdf` in order |
| `--headers H1 [H2 ...]` | Spoken headers, one per `--multi` file |

The 10-category hallucination screener (long-number runs, word repeats, bare years, URLs, emails, hex, all-caps runs, punctuation runs, non-word runs, stray equation remnants) writes a structured `hallucination_report.json` alongside the audio output.

See [`docs/CONSOLIDATION.txt`](docs/CONSOLIDATION.txt) for the full architecture, cleaner taxonomy, validation records, and bootstrap procedures.

## Qwen3-TTS (NEW)

The latest addition: **Qwen3-TTS** - Alibaba's state-of-the-art TTS model with voice cloning and custom voice design.

### Quick Install (DGX Spark / High VRAM)

```bash
git clone https://github.com/olympus-terminal/gpu-tts-toolkit.git
cd gpu-tts-toolkit
./install_dgx.sh
```

### Quick Usage

```bash
conda activate qwen3-tts
python deep_voice_tts_v3.py input.txt --voice dylan --model-size 1.7B
```

### Available Voices

| Voice | Description | Type |
|-------|-------------|------|
| `dylan` | Deep American male | Favorite |
| `eric` | Mature male voice | Favorite |
| `ryan` | Clear male voice | Favorite |
| `uncle_fu` | Deep Chinese male | Favorite |
| `aiden` | Young adult male | Standard |
| `vivian` | Clear female | Standard |
| `serena` | Warm female | Standard |

### Model Sizes

| Model | VRAM Required | Quality |
|-------|---------------|---------|
| 0.6B | ~8GB | Good |
| 1.7B | ~16GB | Best |

### Advanced Features

```bash
# Voice cloning
python deep_voice_tts_v3.py input.txt --clone-audio ref.wav --clone-text "Reference transcript"

# Voice design
python deep_voice_tts_v3.py input.txt --design "A deep, authoritative male voice with slight British accent"

# Custom instruction
python deep_voice_tts_v3.py input.txt --voice dylan --instruct "Speak slowly with gravitas"
```

### Key Features

- **GPU-First Architecture**: Native CUDA acceleration for faster synthesis
- **Neural TTS Models**: FastSpeech2 implementation with optimization framework
- **Batch Processing**: Efficient processing of large text datasets
- **HPC Ready**: Designed for SLURM clusters and multi-GPU systems
- **Local Deployment**: No cloud dependencies, runs entirely on your hardware
- **Research Friendly**: Modular design for experimenting with new architectures

## Performance Goals

| Metric | Target | Notes |
|--------|--------|-------|
| GPU Utilization | >80% | During batch synthesis |
| RTF (Real-Time Factor) | <0.1 | Lower is better |
| Memory Efficiency | Optimized | For large batch processing |
| HPC Scaling | Linear | Multi-GPU support planned |

*Performance will vary based on GPU model, batch size, and model architecture*

## Repository Structure

```
gpu-tts-toolkit/
├── media_to_tts.py             # MAIN: any text file → TTS-ready text + audio + QC
├── deep_voice_tts.py           # TTS engine (text → audio), called as a library
├── deep_voice_tts_v2.py        # Older: PDF/LaTeX/TXT → audio in one step
├── deep_voice_tts_v3.py        # Qwen3-TTS variant (voice cloning, design)
├── extract_salient_text.py     # Scientific paper text extractor
├── improved_tts_pipeline.py    # Enhanced pipeline
├── docs/
│   └── CONSOLIDATION.txt       # Full architecture + process documentation
├── core-engines/
│   └── synthesis/              # TTS synthesis scripts
├── deployment/
│   └── hpc/                    # SLURM job scripts
├── integrations/
│   └── mcp/                    # Model Context Protocol
└── examples/                   # Usage examples
```

## Quick Start

### Prerequisites

**System requirements:**
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better recommended)
- CUDA-compatible driver (check with `nvidia-smi`)
- Python 3.10+
- conda or miniconda

### Installation

#### 1. Install System Dependencies

```bash
# Required for text-to-phoneme conversion
sudo apt install espeak-ng

# Optional: for audio playback
sudo apt install portaudio19-dev ffmpeg
```

#### 2. Create Conda Environment

```bash
# Create and activate environment
conda create -n tts-app python=3.10 -y
conda activate tts-app
```

#### 3. Install PyTorch with CUDA

Check your CUDA version first:
```bash
nvidia-smi  # Look for "CUDA Version" in the output
```

Install PyTorch matching your CUDA version:
```bash
# For CUDA 12.x (most modern systems)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Install TTS and Dependencies

```bash
# Clone the repository
git clone https://github.com/olympus-terminal/gpu-tts-toolkit.git
cd gpu-tts-toolkit

# Install Coqui TTS (main TTS engine)
pip install TTS

# Install additional dependencies
pip install pydub librosa soundfile

# Optional: install remaining requirements
pip install -r requirements.txt
```

#### 5. Verify Installation

```bash
# Check PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Check TTS
python -c "from TTS.api import TTS; print('TTS OK')"

# Test the script
python deep_voice_tts.py --help
```

### Basic Usage

```bash
# Convert text file to audio (uses random voice)
python deep_voice_tts.py input.txt

# List available voices
python deep_voice_tts.py --list-voices

# Use specific voice
python deep_voice_tts.py input.txt --voice p240

# Output as MP3
python deep_voice_tts.py input.txt --format mp3

# Force CPU (if GPU issues)
python deep_voice_tts.py input.txt --device cpu
```

### V2: Direct PDF/LaTeX Input

`deep_voice_tts_v2.py` accepts `.pdf`, `.tex`, and `.txt` files directly — no separate extraction step required. It integrates the scientific text extraction from `extract_salient_text.py` and feeds the cleaned output straight to the TTS engine.

```bash
# Plain text (same as v1)
python deep_voice_tts_v2.py paper.txt --voice p240

# LaTeX manuscript → audio (one command)
python deep_voice_tts_v2.py main.tex --voice p240

# PDF manuscript → audio (one command)
python deep_voice_tts_v2.py main.pdf --voice p240

# Preview extracted text without GPU
python deep_voice_tts_v2.py main.tex --preview 500

# Filter sections
python deep_voice_tts_v2.py main.tex --voice p240 --include-sections "SUMMARY,RESULTS"
python deep_voice_tts_v2.py main.tex --voice p240 --exclude-sections "Limitations of the study"

# Strip verbose statistics from scientific text
python deep_voice_tts_v2.py main.tex --voice p240 --strip-heavy-stats
```

| Option | Description |
|--------|-------------|
| `--preview N` | Print first N extracted characters and exit (no GPU needed) |
| `--include-sections` | Comma-separated sections to keep (or `ALL`) |
| `--exclude-sections` | Comma-separated additional sections to exclude |
| `--strip-heavy-stats` | Remove parenthesized statistical details |
| `--keep-figure-refs` | Keep figure/table references in output text |

All standard TTS options (`--voice`, `--format`, `--device`, `--list-voices`, `--output-name`) work the same as v1.

### Batch Processing

```bash
# Process all files in a directory
for f in papers/*.txt; do python deep_voice_tts.py "$f"; done
```

## Paper-to-Audio Pipeline

The `pipeline/` directory contains tools to automatically search, download, and convert academic papers to audio.

### Pipeline Overview

```
Query String → paper_search.py → paper_download.py → pdf_to_text.py → deep_voice_tts.py → MP3 Audio
```

1. **paper_search.py** - Search CrossRef and PubMed APIs for papers matching your query
2. **paper_download.py** - Download PDFs from open access sources (PMC, Unpaywall, Europe PMC)
3. **pdf_to_text.py** - Extract text and clean it for TTS (removes citations, URLs, figure references)
4. **deep_voice_tts.py** - Generate audio from the cleaned text

### Quick Start

```bash
cd pipeline/

# Search for papers and convert to audio
python paper_to_audio.py 'biomimetic concrete' --papers 2

# Specify a voice
python paper_to_audio.py 'CRISPR gene editing' --papers 1 --voice p240

# Keep intermediate files for inspection
python paper_to_audio.py 'bioremediation' --papers 3 --keep-pdfs --keep-text

# Custom output directory
python paper_to_audio.py 'machine learning' --papers 2 --output my_audio/
```

### Pipeline Options

| Option | Description |
|--------|-------------|
| `query` | Search query for academic papers (required) |
| `--papers N` | Number of papers to search for (default: 3) |
| `--voice ID` | Voice profile: speaker ID (p230, p240, etc.) or "random" |
| `--output DIR` | Output directory (default: output/) |
| `--keep-pdfs` | Keep downloaded PDFs after audio generation |
| `--keep-text` | Keep extracted text files after audio generation |

### Individual Scripts

You can also run each pipeline stage independently:

```bash
# Search only - outputs papers.json
python paper_search.py 'your query' --papers 5

# Download PDFs from papers.json
python paper_download.py papers.json --output downloads/

# Extract text from PDFs
python pdf_to_text.py downloads/ texts/
```

### Notes

- Only open access papers can be downloaded (PMC, Unpaywall, Europe PMC)
- Not all papers in search results will have downloadable PDFs
- By default, intermediate files (PDFs, text) are cleaned up after audio generation
- Audio files are saved in timestamped directories within the output folder

## Troubleshooting

### "No espeak backend found"

Install espeak-ng:
```bash
sudo apt install espeak-ng
```

### "No module named 'TTS'"

Install Coqui TTS:
```bash
pip install TTS
```

### CUDA out of memory

Try a smaller batch or force CPU:
```bash
python deep_voice_tts.py input.txt --device cpu
```

### PyTorch not detecting GPU

1. Check driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### First run is slow

The first run downloads model files (~1GB). Subsequent runs are faster.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- New model architectures
- Language support expansion
- Performance optimizations
- Cloud integrations
- Voice quality improvements

## Citations

If you use this toolkit in research, please cite:

```bibtex
@software{gpu_tts_toolkit,
  author = {olympus-terminal},
  title = {GPU-Accelerated TTS Toolkit},
  url = {https://github.com/olympus-terminal/gpu-tts-toolkit},
  year = {2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for CUDA and TensorRT
- Mozilla TTS contributors
- Tacotron2, FastSpeech2, and VITS authors
- Open source TTS community

## Contact

- Issues: [GitHub Issues](https://github.com/olympus-terminal/gpu-tts-toolkit/issues)
- Discussions: [GitHub Discussions](https://github.com/olympus-terminal/gpu-tts-toolkit/discussions)
- Author: [@olympus-terminal](https://github.com/olympus-terminal)