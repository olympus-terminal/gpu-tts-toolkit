# GPU TTS Toolkit

Text-to-speech tools using GPU acceleration. Useful for converting papers and documents to audio.

[![License](https://img.shields.io/github/license/olympus-terminal/gpu-tts-toolkit)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/olympus-terminal/gpu-tts-toolkit?style=social)](https://github.com/olympus-terminal/gpu-tts-toolkit/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/olympus-terminal/gpu-tts-toolkit)](https://github.com/olympus-terminal/gpu-tts-toolkit/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/olympus-terminal/gpu-tts-toolkit)](https://github.com/olympus-terminal/gpu-tts-toolkit/commits/main)
[![Tools](https://img.shields.io/badge/tools-20+-green.svg)](https://github.com/olympus-terminal/gpu-tts-toolkit)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

Text-to-speech scripts optimized for GPU use. Designed for processing scientific papers, documentation, and other text on local hardware.

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
├── deep_voice_tts.py           # Main TTS script
├── improved_tts_pipeline.py    # Enhanced pipeline
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

```bash
# System requirements
- Linux (Ubuntu 20.04+ or similar)
- NVIDIA GPU (GTX 1060 or better recommended)
- CUDA 11.0+ (check with: nvidia-smi)
- Python 3.8+
- 8GB+ GPU memory for batch processing

# Python dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/olympus-terminal/gpu-tts-toolkit.git
cd gpu-tts-toolkit

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Convert text file to audio
python deep_voice_tts.py input.txt output.wav

# Use improved pipeline with preprocessing
python improved_tts_pipeline.py paper.pdf paper_audio.wav
```



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