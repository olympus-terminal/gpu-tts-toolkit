#!/bin/bash
# Qwen3-TTS Installation Script for DGX Spark (128GB VRAM)
# This script sets up the full 1.7B model environment

set -e

echo "=== Qwen3-TTS DGX Spark Installation ==="

# Create conda environment
echo "Creating conda environment..."
conda create -n qwen3-tts python=3.10 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3-tts

# Install PyTorch with CUDA 12.9
echo "Installing PyTorch with CUDA 12.9..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install core dependencies
echo "Installing Qwen3-TTS dependencies..."
pip install qwen-tts transformers accelerate soundfile scipy pydub librosa tqdm einops

# Install FlashAttention 2 (takes ~60-90 minutes to compile)
echo "Installing FlashAttention 2 (this will take a while)..."
pip install flash-attn --no-build-isolation

# Install sox (system dependency)
echo "Installing sox..."
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y sox libsox-dev
elif command -v yum &> /dev/null; then
    sudo yum install -y sox sox-devel
fi

# Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import flash_attn
print(f'FlashAttention: {flash_attn.__version__}')

import qwen_tts
print('qwen-tts: installed')
"

echo ""
echo "=== Installation Complete ==="
echo "Usage:"
echo "  conda activate qwen3-tts"
echo "  python deep_voice_tts_v3.py input.txt --voice dylan --model-size 1.7B"
echo ""
echo "Available voices: dylan, eric, ryan, uncle_fu (deep male)"
echo "Model sizes: 0.6B (faster), 1.7B (better quality)"
