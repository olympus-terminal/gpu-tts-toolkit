#!/usr/bin/env python3
"""
Deep Voice TTS Pipeline v3 - Qwen3-TTS Engine
GPU-accelerated with voice cloning, voice design, and deep male voices
"""

import os
import re
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from pydub import AudioSegment
import unicodedata
import numpy as np
import shutil
import soundfile as sf
from collections import Counter


class DeepVoiceTTS:
    """Qwen3-TTS based deep voice generator - GPU only"""

    # Built-in voice profiles (from Qwen3-TTS CustomVoice models)
    BUILTIN_VOICES = {
        # Deep male voices
        "dylan": {"speaker": "Dylan", "description": "Deep American male", "instruct": "Speak in a deep, resonant voice"},
        "eric": {"speaker": "Eric", "description": "Mature male voice", "instruct": "Speak in a calm, authoritative tone"},
        "ryan": {"speaker": "Ryan", "description": "Clear male voice", "instruct": "Speak clearly with moderate depth"},
        "aiden": {"speaker": "Aiden", "description": "Young adult male", "instruct": "Speak naturally with warmth"},
        "uncle_fu": {"speaker": "Uncle_Fu", "description": "Deep Chinese male", "instruct": "Speak with gravitas"},
        # Female voices (for completeness)
        "vivian": {"speaker": "Vivian", "description": "Clear female voice", "instruct": "Speak clearly"},
        "serena": {"speaker": "Serena", "description": "Warm female voice", "instruct": "Speak warmly"},
        "ono_anna": {"speaker": "Ono_Anna", "description": "Japanese female", "instruct": "Speak naturally"},
        "sohee": {"speaker": "Sohee", "description": "Korean female", "instruct": "Speak naturally"},
    }

    # Cloned voice profiles (reference audio + transcript)
    CLONED_VOICES = {
        "mm": {
            "description": "Matthew McConaughey clone",
            "reference_audio": "voice_references/mcconaughey_reference.wav",
            "reference_text": "voice_references/mcconaughey_transcript.txt",
        },
        "wh": {
            "description": "Werner Herzog clone",
            "reference_audio": "voice_references/herzog_reference.wav",
            "reference_text": "voice_references/herzog_transcript.txt",
        },
    }

    # Favorite deep male voices for random selection
    FAVORITE_VOICES = ["dylan", "eric", "ryan", "uncle_fu"]

    def __init__(
        self,
        voice_profile="random",
        output_format="mp3",
        model_size="1.7B",
        cleanup=True,
        use_flash_attn=True,
        voice_instruct=None,
        reference_audio=None,
        reference_text=None,
        voice_description=None,
        cpu_offload=False,
        speed=1.0,
    ):
        """
        Initialize Qwen3-TTS pipeline

        Args:
            voice_profile: Built-in voice name, "random", "clone", or "design"
            output_format: Output format (mp3 or wav)
            model_size: Model size ("0.6B" or "1.7B")
            cleanup: Remove intermediate files after generation
            use_flash_attn: Use FlashAttention 2 for efficiency
            voice_instruct: Custom instruction for voice style
            reference_audio: Path to reference audio for voice cloning
            reference_text: Transcript of reference audio
            voice_description: Text description for voice design
            cpu_offload: Enable CPU offloading for low VRAM systems
            speed: Playback speed multiplier (0.5-2.0, default 1.0)
        """
        # GPU only - no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. This is a GPU-only application.")

        self.device = "cuda:0"
        self.output_format = output_format
        self.cleanup = cleanup
        self.use_flash_attn = use_flash_attn
        self.model_size = model_size
        self.voice_instruct = voice_instruct
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.voice_description = voice_description
        self.cpu_offload = cpu_offload
        self.speed = max(0.5, min(2.0, speed))  # Clamp to valid range

        # Determine model type and voice settings
        self.mode = self._determine_mode(voice_profile)
        self.voice_profile = self._setup_voice_profile(voice_profile)

        # Text processing settings
        self.chunk_size = 500  # Qwen3-TTS handles longer sequences well

        # Common tech acronyms
        self.common_acronyms = {
            'AI', 'ML', 'API', 'CPU', 'GPU', 'RAM', 'ROM', 'SSD', 'HDD',
            'UI', 'UX', 'OS', 'PC', 'Mac', 'iOS', 'SQL', 'HTML', 'CSS',
            'JS', 'JSON', 'XML', 'HTTP', 'HTTPS', 'URL', 'URI', 'DNS',
            'IP', 'TCP', 'UDP', 'SSH', 'FTP', 'SMTP', 'IoT', 'VR', 'AR',
            'SDK', 'IDE', 'CI', 'CD', 'QA', 'UAT', 'SaaS', 'PaaS', 'IaaS',
            'CEO', 'CTO', 'CFO', 'HR', 'PR', 'SEO', 'ROI', 'KPI', 'B2B',
            'B2C', 'FAQ', 'ASAP', 'FYI', 'DIY', 'ETA', 'ID', 'VIP', 'ATM',
            'GPS', 'PDF', 'GIF', 'JPEG', 'PNG', 'USB', 'HDMI', 'WiFi', 'LTE',
            '5G', '4K', 'HD', 'LED', 'LCD', 'OLED', 'NASA', 'FBI', 'CIA',
            'FDA', 'CDC', 'WHO', 'UN', 'EU', 'USA', 'UK', 'UAE', 'PhD',
            'MD', 'BA', 'MA', 'BS', 'MS', 'MBA', 'LLM', 'GPT', 'BERT',
            'GAN', 'CNN', 'RNN', 'LSTM', 'NLP', 'CV', 'RL', 'DL', 'AGI'
        }

        self.acronym_pronunciation = {
            'AI': 'A.I.', 'ML': 'M.L.', 'API': 'A.P.I.',
            'CPU': 'C.P.U.', 'GPU': 'G.P.U.', 'UI': 'U.I.',
            'UX': 'U.X.', 'SQL': 'sequel', 'JSON': 'jason',
            'GIF': 'gif', 'JPEG': 'jay-peg', 'WiFi': 'why-fi',
            'iOS': 'i.O.S.', 'PhD': 'P.H.D.', 'CEO': 'C.E.O.',
            'FAQ': 'F.A.Q.', 'ASAP': 'A.S.A.P.', 'NASA': 'nasa',
            'OLED': 'O.L.E.D.', 'LLM': 'L.L.M.', 'GPT': 'G.P.T.',
            'NLP': 'N.L.P.'
        }

        print(f"Mode: {self.mode}")
        print(f"Model: Qwen3-TTS-12Hz-{model_size}")
        print(f"Device: {self.device}")
        print(f"FlashAttention 2: {'enabled' if use_flash_attn else 'disabled'}")
        print(f"CPU Offload: {'enabled' if cpu_offload else 'disabled'}")
        print(f"Speed: {self.speed:.0%}")
        print(f"Cleanup: {'enabled' if cleanup else 'disabled'}")

        self.load_model()

    def _determine_mode(self, voice_profile):
        """Determine operating mode based on settings"""
        if self.reference_audio and self.reference_text:
            return "clone"
        elif self.voice_description:
            return "design"
        else:
            return "custom_voice"

    def _setup_voice_profile(self, voice_profile):
        """Setup voice profile based on input"""
        if self.mode == "clone":
            return {
                "type": "clone",
                "reference_audio": self.reference_audio,
                "reference_text": self.reference_text,
                "description": f"Cloned from {Path(self.reference_audio).name}"
            }
        elif self.mode == "design":
            return {
                "type": "design",
                "voice_description": self.voice_description,
                "description": f"Designed voice: {self.voice_description[:50]}..."
            }
        else:
            # Check for cloned voices first
            voice_key = voice_profile.lower()
            if voice_key in self.CLONED_VOICES:
                cloned = self.CLONED_VOICES[voice_key]
                # Resolve paths relative to script directory
                script_dir = Path(__file__).parent
                ref_audio = script_dir / cloned["reference_audio"]
                ref_text_path = script_dir / cloned["reference_text"]

                if not ref_audio.exists():
                    raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
                if not ref_text_path.exists():
                    raise FileNotFoundError(f"Reference transcript not found: {ref_text_path}")

                # Read transcript
                with open(ref_text_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()

                # Set clone mode
                self.reference_audio = str(ref_audio)
                self.reference_text = ref_text
                self.mode = "clone"

                return {
                    "type": "clone",
                    "reference_audio": str(ref_audio),
                    "reference_text": ref_text,
                    "description": cloned["description"]
                }

            # Built-in voice mode
            if voice_profile == "random":
                import random
                voice_profile = random.choice(self.FAVORITE_VOICES)

            voice_profile = voice_profile.lower()
            if voice_profile in self.BUILTIN_VOICES:
                profile = self.BUILTIN_VOICES[voice_profile].copy()
                profile["type"] = "builtin"
                # Allow custom instruct to override default
                if self.voice_instruct:
                    profile["instruct"] = self.voice_instruct
                return profile
            else:
                # Default to Dylan (deep male)
                profile = self.BUILTIN_VOICES["dylan"].copy()
                profile["type"] = "builtin"
                print(f"Unknown voice '{voice_profile}', defaulting to Dylan")
                return profile

    def load_model(self):
        """Load Qwen3-TTS model with optional memory optimization"""
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            print("Error: qwen-tts not installed. Install with:")
            print("  pip install qwen-tts")
            raise

        # Select model based on mode and size
        if self.mode == "design":
            model_name = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-VoiceDesign"
        elif self.mode == "clone":
            model_name = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-Base"
        else:
            model_name = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-CustomVoice"

        print(f"Loading model: {model_name}")

        # Configure attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attn else "sdpa"

        # Build kwargs for model loading
        load_kwargs = {
            "attn_implementation": attn_impl,
        }

        # Use bfloat16 for efficiency
        load_kwargs["torch_dtype"] = torch.bfloat16

        # Configure device mapping
        if self.cpu_offload:
            # Use auto device map for CPU offloading - splits model across GPU/CPU
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {0: "10GiB", "cpu": "32GiB"}
            print("Using automatic device mapping with CPU offloading...")
        else:
            load_kwargs["device_map"] = self.device

        try:
            self.model = Qwen3TTSModel.from_pretrained(model_name, **load_kwargs)
            print(f"Model loaded successfully")

            # Report memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        except Exception as e:
            if "flash" in str(e).lower():
                print("FlashAttention 2 failed, falling back to SDPA...")
                load_kwargs["attn_implementation"] = "sdpa"
                self.model = Qwen3TTSModel.from_pretrained(model_name, **load_kwargs)
                print(f"Model loaded with SDPA")
            else:
                raise

        # Pre-compute voice clone prompt if cloning
        if self.mode == "clone":
            self._prepare_clone_prompt()

    def _prepare_clone_prompt(self):
        """Pre-compute voice clone prompt for efficiency"""
        print(f"Preparing voice clone from: {self.reference_audio}")
        try:
            self.clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=self.reference_audio,
                ref_text=self.reference_text
            )
            print("Voice clone prompt prepared")
        except Exception as e:
            print(f"Error preparing clone prompt: {e}")
            raise

    def detect_acronyms(self, text):
        """Improved acronym detection with context awareness"""
        pattern = r'\b([A-Z]{2,})\b'

        def replace_acronym(match):
            acronym = match.group(1)
            if acronym in self.common_acronyms:
                if acronym in self.acronym_pronunciation:
                    return self.acronym_pronunciation[acronym]
                else:
                    return ' '.join(acronym)
            vowels = sum(1 for c in acronym if c in 'AEIOU')
            if vowels > 0 and vowels < len(acronym) - 1:
                return acronym.lower()
            return ' '.join(acronym)

        return re.sub(pattern, replace_acronym, text)

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)

        # Handle special characters
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '—': ' - ', '–': ' - ', '…': '...',
            '\u200b': '', '\ufeff': '', '\xa0': ' ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Handle numbers and currency
        text = re.sub(r'\$(\d+(?:\.\d{2})?)', r'\1 dollars', text)
        text = re.sub(r'€(\d+(?:\.\d{2})?)', r'\1 euros', text)
        text = re.sub(r'£(\d+(?:\.\d{2})?)', r'\1 pounds', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)

        # Remove URLs and emails
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)

        # Handle citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\)', '', text)

        # Apply acronym detection
        text = self.detect_acronyms(text)

        # Handle common abbreviations
        abbreviations = {
            'Dr.': 'Doctor', 'Mr.': 'Mister', 'Mrs.': 'Misses',
            'Ms.': 'Miss', 'Prof.': 'Professor', 'Sr.': 'Senior',
            'Jr.': 'Junior', 'vs.': 'versus', 'etc.': 'etcetera',
            'i.e.': 'that is', 'e.g.': 'for example',
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def smart_chunk_text(self, text):
        """Smart chunking for natural speech flow"""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if sentence_length > self.chunk_size:
                # Split long sentences
                parts = re.split(r'[,;]\s*', sentence)
                for part in parts:
                    if current_length + len(part) > self.chunk_size:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    current_chunk.append(part)
                    current_length += len(part)
            else:
                if current_length + sentence_length > self.chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0

                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_audio_chunk(self, text, output_path):
        """Generate audio for a text chunk"""
        try:
            if self.mode == "clone":
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=self.clone_prompt,
                    language="English"
                )
            elif self.mode == "design":
                wavs, sr = self.model.generate_voice_design(
                    text=text,
                    voice_description=self.voice_description,
                    language="English"
                )
            else:
                # Custom voice mode
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language="English",
                    speaker=self.voice_profile["speaker"],
                    instruct=self.voice_profile.get("instruct", "")
                )

            # Save audio - handle both tensor and numpy array outputs
            audio_data = wavs[0]
            if hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            sf.write(output_path, audio_data, sr)
            return True

        except Exception as e:
            print(f"Error generating audio: {e}")
            return False

    def detect_hallucinations(self, audio_path):
        """Detect repetitive/hallucinated segments in generated audio"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("Warning: faster-whisper not installed, skipping hallucination detection")
            return []

        print("Scanning for audio hallucinations...")

        # Use small model for speed
        model = WhisperModel("base", device="cuda", compute_type="float16")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
        segments = list(segments)

        if not segments:
            return []

        # Detect repetitions
        hallucinations = []
        texts = [s.text.strip().lower() for s in segments]

        # Sliding window to find repeated phrases
        i = 0
        while i < len(texts):
            phrase = texts[i]
            if len(phrase) < 3:  # Skip very short segments
                i += 1
                continue

            # Count consecutive repetitions
            repeat_count = 1
            j = i + 1
            while j < len(texts) and self._similar_text(phrase, texts[j]):
                repeat_count += 1
                j += 1

            # If repeated 3+ times, mark as hallucination
            if repeat_count >= 3:
                start_time = segments[i].start
                end_time = segments[j-1].end
                hallucinations.append({
                    "start": start_time,
                    "end": end_time,
                    "phrase": phrase,
                    "repeats": repeat_count
                })
                print(f"  Found hallucination: '{phrase[:30]}...' x{repeat_count} ({start_time:.1f}s - {end_time:.1f}s)")
                i = j
            else:
                i += 1

        # Also detect rapid-fire short segments (babbling)
        for i in range(len(segments) - 5):
            window = segments[i:i+6]
            durations = [s.end - s.start for s in window]
            avg_duration = sum(durations) / len(durations)

            # If 6 consecutive segments are all very short (<0.5s avg), likely babbling
            if avg_duration < 0.5 and all(d < 0.8 for d in durations):
                start_time = window[0].start
                end_time = window[-1].end
                # Check if not already marked
                if not any(h["start"] <= start_time <= h["end"] for h in hallucinations):
                    hallucinations.append({
                        "start": start_time,
                        "end": end_time,
                        "phrase": "[rapid babbling]",
                        "repeats": 0
                    })
                    print(f"  Found babbling: {start_time:.1f}s - {end_time:.1f}s")

        return hallucinations

    def _similar_text(self, a, b, threshold=0.8):
        """Check if two texts are similar (for detecting repetitions)"""
        if not a or not b:
            return False
        # Simple similarity: shared words ratio
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return a == b
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return (intersection / union) >= threshold if union > 0 else False

    def clean_hallucinations(self, audio_path, hallucinations):
        """Remove hallucinated segments from audio"""
        if not hallucinations:
            return audio_path

        print(f"Removing {len(hallucinations)} hallucinated segments...")

        audio = AudioSegment.from_file(str(audio_path))

        # Sort hallucinations by start time (reverse to remove from end first)
        hallucinations = sorted(hallucinations, key=lambda x: x["start"], reverse=True)

        for h in hallucinations:
            start_ms = int(h["start"] * 1000)
            end_ms = int(h["end"] * 1000)

            # Keep audio before and after the hallucination
            before = audio[:start_ms]
            after = audio[end_ms:]

            # Add small crossfade to smooth the cut
            if len(before) > 50 and len(after) > 50:
                audio = before.append(after, crossfade=50)
            else:
                audio = before + after

        # Save cleaned audio
        cleaned_path = Path(audio_path).with_suffix('.cleaned' + Path(audio_path).suffix)

        if str(audio_path).endswith('.mp3'):
            audio.export(str(cleaned_path), format="mp3", bitrate="192k")
        else:
            audio.export(str(cleaned_path), format="wav")

        # Replace original with cleaned
        Path(audio_path).unlink()
        cleaned_path.rename(audio_path)

        print(f"Cleaned audio saved: {audio_path}")
        return audio_path

    def cleanup_intermediate_files(self, chunks_dir, output_dir):
        """Remove intermediate chunk files after successful generation"""
        try:
            if self.cleanup and chunks_dir.exists():
                print(f"\nCleaning up intermediate files...")

                audio_files = list(chunks_dir.glob(f"*.wav")) + list(chunks_dir.glob(f"*.mp3"))
                text_files = list(chunks_dir.glob("*.txt"))
                files_removed = 0

                for f in audio_files + text_files:
                    try:
                        f.unlink()
                        files_removed += 1
                    except Exception as e:
                        print(f"Warning: Could not remove {f.name}: {e}")

                try:
                    if not any(chunks_dir.iterdir()):
                        chunks_dir.rmdir()
                except Exception:
                    pass

                print(f"Cleaned up {files_removed} intermediate files")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

    def process_text_file(self, input_file, output_name=None):
        """Process text file with Qwen3-TTS"""
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"\nError: File not found: {input_file}")
            return None

        base_name = output_name or input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{base_name}_qwen3tts_{timestamp}")
        output_dir.mkdir(exist_ok=True)

        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Read and preprocess
        print(f"\nReading: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"\nError reading file: {e}")
            return None

        if len(text.strip()) < 10:
            print(f"\nError: File is empty or too short")
            return None

        original_length = len(text)
        text = self.preprocess_text(text)
        print(f"Preprocessed: {original_length} -> {len(text)} chars")

        # Create chunks
        chunks = self.smart_chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        # Save metadata
        metadata = {
            "input_file": str(input_file),
            "mode": self.mode,
            "voice_profile": self.voice_profile,
            "model_size": self.model_size,
            "device": self.device,
            "chunks": len(chunks),
            "timestamp": timestamp,
        }

        # Generate audio
        print(f"\nGenerating audio with Qwen3-TTS...")
        audio_files = []

        for i, chunk in enumerate(tqdm(chunks, desc="Processing")):
            chunk_file = chunks_dir / f"chunk_{i:04d}.wav"

            if self.generate_audio_chunk(chunk, str(chunk_file)):
                audio_files.append(str(chunk_file))

                # Save chunk text
                with open(chunk_file.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(chunk)

        print(f"\nGenerated: {len(audio_files)}/{len(chunks)} chunks")

        # Combine audio
        final_output = None
        if audio_files:
            final_output = output_dir / f"{base_name}_complete.{self.output_format}"
            print(f"\nCombining audio...")

            try:
                combined = AudioSegment.from_file(audio_files[0])
                for audio_file in audio_files[1:]:
                    audio = AudioSegment.from_file(audio_file)
                    combined = combined.append(audio, crossfade=50)

                combined = combined.normalize()

                # Apply speed adjustment if not 1.0
                if self.speed != 1.0:
                    print(f"Adjusting speed to {self.speed:.0%}...")
                    # Change speed by altering frame rate then converting back
                    new_frame_rate = int(combined.frame_rate * self.speed)
                    combined = combined._spawn(combined.raw_data, overrides={
                        "frame_rate": new_frame_rate
                    }).set_frame_rate(combined.frame_rate)

                if self.output_format == "mp3":
                    combined.export(final_output, format="mp3", bitrate="192k")
                else:
                    combined.export(final_output, format="wav")

                print(f"Final audio: {final_output}")

                # Detect and clean hallucinations
                hallucinations = self.detect_hallucinations(final_output)
                if hallucinations:
                    self.clean_hallucinations(final_output, hallucinations)
                    metadata["hallucinations_removed"] = len(hallucinations)
                else:
                    print("No hallucinations detected")

                file_size = final_output.stat().st_size / (1024 * 1024)
                metadata["file_size_mb"] = round(file_size, 2)
                print(f"Size: {file_size:.2f} MB")

                if final_output.exists():
                    self.cleanup_intermediate_files(chunks_dir, output_dir)

            except Exception as e:
                print(f"Error combining: {e}")

        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nComplete! Output: {output_dir}")
        return str(output_dir)


def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Voice TTS v3 - Qwen3-TTS Engine (GPU only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use built-in deep male voice
  python deep_voice_tts_v3.py input.txt --voice dylan

  # Random deep male voice
  python deep_voice_tts_v3.py input.txt --voice random

  # Clone a voice from reference audio
  python deep_voice_tts_v3.py input.txt --clone-audio ref.wav --clone-text "Reference transcript"

  # Design a voice from description
  python deep_voice_tts_v3.py input.txt --design "A deep, authoritative male voice with slight British accent"

  # Custom voice instruction
  python deep_voice_tts_v3.py input.txt --voice dylan --instruct "Speak slowly with gravitas"

Memory optimization (for limited VRAM):
  # Use smaller 0.6B model (recommended for 8-12GB VRAM)
  python deep_voice_tts_v3.py input.txt --voice dylan --model-size 0.6B

  # CPU offloading (splits model across GPU and CPU)
  python deep_voice_tts_v3.py input.txt --voice dylan --cpu-offload

Available voices: dylan, eric, ryan, aiden, uncle_fu, vivian, serena, ono_anna, sohee, mm (McConaughey), wh (Herzog)
        """
    )

    parser.add_argument("input_file", nargs='?', help="Input text file")
    parser.add_argument("--voice", default="random",
                       help="Voice: dylan, eric, ryan, aiden, uncle_fu, or 'random'")
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                       help="Output format")
    parser.add_argument("--output-name", help="Custom output name")
    parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default="1.7B",
                       help="Model size (default: 1.7B)")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Keep intermediate chunk files")
    parser.add_argument("--no-flash-attn", action="store_true",
                       help="Disable FlashAttention 2")
    parser.add_argument("--instruct", help="Custom voice instruction")

    # Memory optimization options
    parser.add_argument("--cpu-offload", action="store_true",
                       help="Enable CPU offloading for low VRAM GPUs")

    # Audio options
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Playback speed (0.5-2.0, default 1.0). Use 0.8 for slower speech.")

    # Voice cloning options
    parser.add_argument("--clone-audio", help="Reference audio for voice cloning")
    parser.add_argument("--clone-text", help="Transcript of reference audio")

    # Voice design option
    parser.add_argument("--design", help="Voice description for voice design mode")

    parser.add_argument("--list-voices", action="store_true",
                       help="List available voices and exit")

    args = parser.parse_args()

    if args.list_voices:
        print("Available built-in voices:")
        print("\nDeep male voices (favorites):")
        for name in DeepVoiceTTS.FAVORITE_VOICES:
            info = DeepVoiceTTS.BUILTIN_VOICES[name]
            print(f"  {name:12} - {info['description']}")
        print("\nOther voices:")
        for name, info in DeepVoiceTTS.BUILTIN_VOICES.items():
            if name not in DeepVoiceTTS.FAVORITE_VOICES:
                print(f"  {name:12} - {info['description']}")
        print("\nCloned voices:")
        for name, info in DeepVoiceTTS.CLONED_VOICES.items():
            print(f"  {name:12} - {info['description']}")
        print("\nModes:")
        print("  --voice <name>    Use built-in or cloned voice")
        print("  --clone-audio/--clone-text    Clone from reference")
        print("  --design \"description\"    Design voice from text")
        return

    if not args.input_file:
        parser.print_help()
        print("\nError: No input file specified")
        sys.exit(1)

    # Validate clone mode
    if args.clone_audio and not args.clone_text:
        print("Error: --clone-text required with --clone-audio")
        sys.exit(1)
    if args.clone_text and not args.clone_audio:
        print("Error: --clone-audio required with --clone-text")
        sys.exit(1)

    try:
        pipeline = DeepVoiceTTS(
            voice_profile=args.voice,
            output_format=args.format,
            model_size=args.model_size,
            cleanup=not args.no_cleanup,
            use_flash_attn=not args.no_flash_attn,
            voice_instruct=args.instruct,
            reference_audio=args.clone_audio,
            reference_text=args.clone_text,
            voice_description=args.design,
            cpu_offload=args.cpu_offload,
            speed=args.speed,
        )
    except Exception as e:
        print(f"\nError initializing TTS pipeline: {e}")
        sys.exit(1)

    try:
        result = pipeline.process_text_file(
            args.input_file,
            output_name=args.output_name
        )
        if result is None:
            sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
