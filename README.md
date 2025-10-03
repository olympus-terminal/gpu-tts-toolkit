# GPU-Accelerated TTS Toolkit

GPU-accelerated text-to-speech toolkit for Linux and HPC environments. Fast neural speech synthesis on your local hardware.

[![License](https://img.shields.io/github/license/olympus-terminal/gpu-tts-toolkit)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/olympus-terminal/gpu-tts-toolkit?style=social)](https://github.com/olympus-terminal/gpu-tts-toolkit/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/olympus-terminal/gpu-tts-toolkit)](https://github.com/olympus-terminal/gpu-tts-toolkit/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/olympus-terminal/gpu-tts-toolkit)](https://github.com/olympus-terminal/gpu-tts-toolkit/commits/main)
[![Tools](https://img.shields.io/badge/tools-20+-green.svg)](https://github.com/olympus-terminal/gpu-tts-toolkit)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

A high-performance text-to-speech toolkit designed for Linux workstations and HPC environments. Built with GPU acceleration for researchers, developers, and organizations needing fast, high-quality speech synthesis on local infrastructure.

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
├── core-engines/          # Core TTS synthesis engines
│   ├── synthesis/         # Text-to-mel synthesis models
│   ├── vocoding/         # Neural vocoders (WaveGlow, HiFi-GAN)
│   └── voice-cloning/    # Voice cloning and adaptation
├── gpu-acceleration/     # GPU optimization layers
│   ├── cuda/            # Custom CUDA kernels
│   ├── tensorrt/        # TensorRT optimization
│   └── optimization/    # Memory and performance optimization
├── integrations/        # Enterprise integrations
│   ├── api/            # REST and gRPC APIs
│   ├── mcp/            # Model Context Protocol
│   └── streaming/      # WebSocket and real-time streaming
├── voice-models/       # Pre-trained and custom models
│   ├── pretrained/     # Ready-to-use models
│   ├── custom/         # Custom voice models
│   └── fine-tuning/    # Model adaptation tools
├── deployment/         # Local deployment tools
│   ├── hpc/           # SLURM job scripts
│   ├── monitoring/    # Local performance monitoring
│   └── systemd/       # System service configurations
├── workflows/          # Automation and pipelines
│   ├── batch-processing/  # Bulk TTS generation
│   ├── real-time/        # Live synthesis pipelines
│   └── pipelines/        # Custom workflows
└── examples/          # Use case examples
    ├── conversational/   # Chatbots, virtual assistants
    ├── narrative/        # Audiobooks, storytelling
    └── commercial/       # IVR, announcements
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

# Install in development mode
pip install -e .

# Download pre-trained models
python scripts/download_models.py

# Run system check
python scripts/check_gpu.py
```

### Basic Usage

```python
# Simple TTS generation
from gpu_tts import FastTTS

# Initialize with GPU acceleration
tts = FastTTS(device='cuda', model='fastspeech2')

# Generate speech
audio = tts.synthesize("Hello, this is GPU-accelerated speech synthesis!")
audio.save("output.wav")

# Real-time streaming
async for chunk in tts.stream("Streaming text to speech in real-time"):
    play_audio_chunk(chunk)
```

## Core Engines

### FastSpeech2 GPU

Ultra-fast parallel synthesis with consistent quality:

```python
from gpu_tts.engines import FastSpeech2GPU

# Initialize with optimization
engine = FastSpeech2GPU(
    model_path="models/fastspeech2_en.pt",
    use_tensorrt=True,
    precision="fp16"
)

# Batch synthesis
texts = ["First sentence.", "Second sentence.", "Third sentence."]
audios = engine.batch_synthesize(texts, batch_size=32)
```

### Neural Vocoding

High-quality waveform generation:

```python
from gpu_tts.vocoders import HiFiGANGPU

vocoder = HiFiGANGPU(
    checkpoint="models/hifigan_universal.pt",
    use_cuda_graphs=True
)

# Convert mel-spectrogram to audio
waveform = vocoder(mel_spectrogram)
```

### Voice Cloning

Clone any voice with minimal data:

```python
from gpu_tts.cloning import VoiceCloner

cloner = VoiceCloner(device='cuda')

# Clone from audio samples
cloner.adapt(
    reference_audios=["speaker1.wav", "speaker2.wav"],
    transcripts=["Hello, this is...", "Another sample..."]
)

# Generate with cloned voice
cloned_audio = cloner.synthesize("New text in cloned voice")
```

## Enterprise Integration

### REST API Server

Production-ready API with authentication and monitoring:

```bash
# Start the API server
python -m gpu_tts.api.server \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4 \
    --gpu-per-worker 0.25
```

API endpoints:
- `POST /synthesize` - Single text synthesis
- `POST /batch` - Batch processing
- `WebSocket /stream` - Real-time streaming
- `GET /voices` - Available voices
- `POST /clone` - Voice cloning

### MCP Integration

Model Context Protocol for seamless integration:

```python
from gpu_tts.mcp import TTSContextProvider

provider = TTSContextProvider()
provider.register_models(["fastspeech2", "tacotron2", "vits"])
provider.start_server(port=9090)
```

### Streaming with WebSocket

Real-time synthesis for interactive applications:

```javascript
// JavaScript client example
const ws = new WebSocket('ws://localhost:8080/stream');

ws.onopen = () => {
    ws.send(JSON.stringify({
        text: "This will be streamed as it's generated",
        voice: "emma",
        speed: 1.0
    }));
};

ws.onmessage = (event) => {
    // Play audio chunk
    playAudioChunk(event.data);
};
```

## GPU Acceleration

### Custom CUDA Kernels

Optimized operations for maximum performance:

```cpp
// Parallel mel-spectrogram generation
__global__ void generate_mel_kernel(
    float* text_embeddings,
    float* mel_output,
    int batch_size,
    int max_length
) {
    // Optimized CUDA implementation
}
```

### TensorRT Optimization

Deploy with 3-5x additional speedup:

```python
from gpu_tts.optimization import TensorRTOptimizer

optimizer = TensorRTOptimizer()
trt_engine = optimizer.optimize_model(
    model_path="models/fastspeech2.onnx",
    precision="int8",
    workspace_size=2048
)
```

### Memory Management

Efficient GPU memory usage for large batches:

```python
from gpu_tts.optimization import MemoryManager

with MemoryManager(max_memory_gb=8) as mm:
    # Process large batches without OOM
    for batch in large_dataset:
        audio = tts.synthesize_batch(batch)
```

## Monitoring & Analytics

### Prometheus Metrics

Built-in metrics for production monitoring:

```yaml
# Example metrics
tts_synthesis_duration_seconds{model="fastspeech2"} 0.045
tts_batch_size{endpoint="/batch"} 32
tts_gpu_utilization_percent{device="cuda:0"} 87.5
tts_requests_total{status="success"} 15420
```

### Performance Dashboard

Real-time monitoring dashboard included:

```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up
```

Access dashboards:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Deployment

### Docker

Production-ready containers:

```bash
# Build optimized image
docker build -t gpu-tts:latest -f docker/Dockerfile.cuda .

# Run with GPU support
docker run --gpus all -p 8080:8080 gpu-tts:latest
```

### HPC Deployment

SLURM job script example:

```bash
#!/bin/bash
#SBATCH --job-name=gpu-tts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/11.8
module load python/3.10

source venv/bin/activate
python -m gpu_tts.batch_process --input texts.txt --output audio/
```

### Local Server Deployment

Systemd service for persistent TTS server:

```bash
sudo cp deployment/systemd/gpu-tts.service /etc/systemd/system/
sudo systemctl enable gpu-tts
sudo systemctl start gpu-tts
```

## Use Cases

### Conversational AI

```python
# Real-time voice assistant
assistant = GPUVoiceAssistant(
    tts_model="fastspeech2",
    latency_target_ms=100
)

async def handle_conversation(text):
    audio_stream = await assistant.respond(text)
    return audio_stream
```

### Content Creation

```python
# Audiobook generation
narrator = AudiobookNarrator(
    voice="professional_narrator",
    emotion_control=True
)

# Generate with chapter markers
audiobook = narrator.narrate_book(
    text_file="book.txt",
    output_format="m4b",
    chapter_detection=True
)
```

### Research Applications

```python
# Process research datasets
researcher = TTSDatasetProcessor(
    model="fastspeech2",
    output_format="wav",
    sample_rate=22050
)

# Generate speech dataset from transcripts
researcher.process_dataset(
    transcripts="data/transcripts.txt",
    output_dir="data/synthesized/",
    speaker_id=0
)
```

## Advanced Features

### Emotion Control

Fine-grained emotional expression:

```python
tts.synthesize(
    "I'm so excited to see you!",
    emotion="happy",
    intensity=0.8
)
```

### Multi-Speaker Synthesis

Multiple voices in one stream:

```python
dialogue = [
    ("Alice", "Hello Bob, how are you?"),
    ("Bob", "I'm doing great, thanks for asking!"),
]

audio = tts.synthesize_dialogue(dialogue)
```

### Prosody Control

Fine control over speech characteristics:

```python
tts.synthesize(
    text="Important announcement",
    pitch_shift=1.2,
    speed=0.9,
    emphasis_words=["Important"]
)
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