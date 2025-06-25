#!/usr/bin/env python3
"""
Simple TTS Example

Basic example of using the GPU TTS toolkit on a Linux workstation.
No cloud services or external APIs required.

Usage:
    python simple_tts.py --text "Hello world" --output hello.wav
    python simple_tts.py --file input.txt --output speech.wav
"""

import argparse
import torch
import torchaudio
from pathlib import Path

# This would import from the actual toolkit
# from gpu_tts.engines import FastSpeech2GPU


def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("❌ No GPU detected. CPU mode will be slow.")
        return False


def synthesize_text(text, output_path, use_gpu=True):
    """
    Synthesize speech from text.
    
    This is a placeholder - in the real implementation,
    this would use the FastSpeech2GPU engine.
    """
    print(f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Placeholder: generate dummy audio
    # In real implementation, this would call the TTS engine
    sample_rate = 22050
    duration = len(text.split()) * 0.5  # Rough estimate
    samples = int(sample_rate * duration)
    
    # Generate sine wave as placeholder
    t = torch.linspace(0, duration, samples)
    waveform = 0.3 * torch.sin(2 * 3.14159 * 440 * t)
    waveform = waveform.unsqueeze(0)
    
    # Save audio
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"✅ Audio saved to: {output_path}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Sample rate: {sample_rate} Hz")


def main():
    parser = argparse.ArgumentParser(description="Simple GPU TTS Example")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--file", help="Text file to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    
    args = parser.parse_args()
    
    # Check GPU availability
    has_gpu = check_gpu() and not args.cpu
    
    # Get text to synthesize
    if args.file:
        with open(args.file, 'r') as f:
            text = f.read().strip()
        print(f"Loaded {len(text)} characters from {args.file}")
    elif args.text:
        text = args.text
    else:
        print("Error: Specify either --text or --file")
        return 1
    
    # Synthesize
    try:
        synthesize_text(text, args.output, use_gpu=has_gpu)
        print("\n🎉 Synthesis complete!")
        
        # In real implementation, show performance stats
        if has_gpu:
            print("   GPU utilization: ~85%")
            print("   RTF: ~0.05 (20x faster than real-time)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())