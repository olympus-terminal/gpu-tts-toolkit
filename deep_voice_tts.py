#!/usr/bin/env python3
"""
Deep Voice TTS Pipeline - Male voices with accent options
Enhanced acronym detection and voice variety
"""

import os
import re
import torch
from pathlib import Path
from TTS.api import TTS
from tqdm import tqdm
import json
from datetime import datetime
from pydub import AudioSegment
import unicodedata
import numpy as np


class DeepVoiceTTS:
    def __init__(self, voice_profile="random", output_format="mp3", device=None):
        """
        Initialize with voice selection
        Can use speaker IDs directly (p230, p234, etc.) or presets
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_format = output_format
        
        # Favorite deep male voices
        self.favorite_voices = ["p230", "p234", "p240", "p244", "p246", "p248", "p250"]
        
        # Voice profiles with male speakers and potential accents
        self.voice_profiles = {
            "deep_male": {
                "model": "tts_models/en/vctk/vits",
                "speaker": "p245",  # Deep male voice in VCTK
                "description": "Deep British male voice",
                "chunk_size": 400,
                "speed": 0.95  # Slightly slower for deeper effect
            },
            "caribbean_male": {
                "model": "tts_models/en/vctk/vits", 
                "speaker": "p237",  # Male with accent variation
                "description": "Male voice with accent variation",
                "chunk_size": 400,
                "speed": 0.93,
                "pitch_shift": -2  # Lower pitch for deeper sound
            },
            "bass_male": {
                "model": "tts_models/en/vctk/vits",
                "speaker": "p241",  # Another deep male
                "description": "Very deep bass male voice",
                "chunk_size": 400,
                "speed": 0.90,
                "pitch_shift": -3
            },
            "xtts_custom": {
                "model": "tts_models/multilingual/multi-dataset/xtts_v2",
                "speaker": None,  # Will use voice cloning
                "description": "XTTS with custom voice cloning",
                "chunk_size": 350,
                "speed": 0.95,
                "language": "en"
            },
            "sam_male": {
                "model": "tts_models/en/sam/tacotron-DDC",
                "speaker": None,
                "description": "Sam - deep American male voice",
                "chunk_size": 400,
                "speed": 0.95
            }
        }
        
        # Common tech acronyms that should NOT be spelled out
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
        
        # Pronunciation guides for common acronyms
        self.acronym_pronunciation = {
            'AI': 'A.I.',
            'ML': 'M.L.',
            'API': 'A.P.I.',
            'CPU': 'C.P.U.',
            'GPU': 'G.P.U.',
            'UI': 'U.I.',
            'UX': 'U.X.',
            'SQL': 'sequel',
            'JSON': 'jason',
            'GIF': 'gif',
            'JPEG': 'jay-peg',
            'WiFi': 'why-fi',
            'iOS': 'i.O.S.',
            'PhD': 'P.H.D.',
            'CEO': 'C.E.O.',
            'FAQ': 'F.A.Q.',
            'ASAP': 'A.S.A.P.',
            'NASA': 'nasa',  # Pronounced as word
            'LASER': 'laser', # Pronounced as word
            'RADAR': 'radar', # Pronounced as word
            'OLED': 'O.L.E.D.',
            'LLM': 'L.L.M.',
            'GPT': 'G.P.T.',
            'NLP': 'N.L.P.'
        }
        
        # Handle direct speaker IDs (p230, p234, etc.) or special keywords
        if voice_profile == "random":
            # Randomly select from favorites
            import random
            selected_voice = random.choice(self.favorite_voices)
            self.profile = {
                "model": "tts_models/en/vctk/vits",
                "speaker": selected_voice,
                "description": f"Random favorite voice: {selected_voice}",
                "chunk_size": 400,
                "speed": 0.95
            }
        elif voice_profile.startswith("p") and voice_profile[1:].isdigit():
            # Direct speaker ID (e.g., p230, p234)
            self.profile = {
                "model": "tts_models/en/vctk/vits",
                "speaker": voice_profile,
                "description": f"VCTK Speaker {voice_profile}",
                "chunk_size": 400,
                "speed": 0.95
            }
        elif voice_profile in self.voice_profiles:
            # Use predefined profile
            self.profile = self.voice_profiles[voice_profile]
        else:
            # Default to random favorite
            import random
            selected_voice = random.choice(self.favorite_voices)
            self.profile = {
                "model": "tts_models/en/vctk/vits",
                "speaker": selected_voice,
                "description": f"Default to random favorite: {selected_voice}",
                "chunk_size": 400,
                "speed": 0.95
            }
        
        print(f"Voice: {self.profile['description']}")
        print(f"Device: {self.device}")
        
        self.load_model()
    
    def load_model(self):
        """Load the TTS model with speaker support"""
        try:
            print(f"Loading model: {self.profile['model']}")
            self.tts = TTS(self.profile['model'], progress_bar=False).to(self.device)
            
            # Set speaker if multi-speaker model
            if hasattr(self.tts, 'speakers') and self.profile['speaker']:
                if self.profile['speaker'] in self.tts.speakers:
                    print(f"✓ Using speaker: {self.profile['speaker']}")
                else:
                    print(f"⚠️ Speaker {self.profile['speaker']} not found")
                    print(f"Available speakers: {self.tts.speakers[:10]}...")
                    # Use first male speaker as fallback
                    male_speakers = [s for s in self.tts.speakers if s.startswith('p2')]
                    if male_speakers:
                        self.profile['speaker'] = male_speakers[0]
                        print(f"✓ Using fallback speaker: {self.profile['speaker']}")
            
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_acronyms(self, text):
        """
        Improved acronym detection with context awareness
        """
        # Pattern for potential acronyms (2+ capital letters)
        pattern = r'\b([A-Z]{2,})\b'
        
        def replace_acronym(match):
            acronym = match.group(1)
            
            # Check if it's a known common acronym
            if acronym in self.common_acronyms:
                # Use pronunciation guide if available
                if acronym in self.acronym_pronunciation:
                    return self.acronym_pronunciation[acronym]
                else:
                    # Space out letters for unknown tech acronyms
                    return ' '.join(acronym)
            
            # Check if it might be a word in all caps (like HELLO)
            # If it has vowels and consonants mixed, might be a word
            vowels = sum(1 for c in acronym if c in 'AEIOU')
            if vowels > 0 and vowels < len(acronym) - 1:
                # Might be a word, leave as is but lowercase
                return acronym.lower()
            
            # Default: space out the letters
            return ' '.join(acronym)
        
        return re.sub(pattern, replace_acronym, text)
    
    def preprocess_text(self, text):
        """
        Enhanced text preprocessing with better acronym handling
        """
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Handle special characters
        replacements = {
            '"': '"',
            '"': '"', 
            ''': "'",
            ''': "'",
            '—': ' - ',
            '–': ' - ',
            '…': '...',
            '\u200b': '',
            '\ufeff': '',
            '\xa0': ' ',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Handle numbers and currency
        text = re.sub(r'\$(\d+(?:\.\d{2})?)', r'\1 dollars', text)
        text = re.sub(r'€(\d+(?:\.\d{2})?)', r'\1 euros', text)
        text = re.sub(r'£(\d+(?:\.\d{2})?)', r'\1 pounds', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        
        # Remove URLs and emails
        text = re.sub(r'https?://[^\s]+', '[link]', text)
        text = re.sub(r'www\.[^\s]+', '[website]', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[email]', text)
        
        # Handle citations
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        text = re.sub(r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\)', '', text)
        
        # Apply acronym detection
        text = self.detect_acronyms(text)
        
        # Handle common abbreviations
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Misses', 
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'Sr.': 'Senior',
            'Jr.': 'Junior',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'i.e.': 'that is',
            'e.g.': 'for example',
            'ft.': 'feet',
            'in.': 'inches',
            'cm.': 'centimeters',
            'kg.': 'kilograms',
            'lb.': 'pounds',
            'oz.': 'ounces',
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def apply_voice_effects(self, audio_path):
        """
        Apply post-processing effects for deeper/accented voice
        """
        if 'pitch_shift' in self.profile and self.profile['pitch_shift'] != 0:
            try:
                audio = AudioSegment.from_file(audio_path)
                
                # Apply pitch shift (negative values = deeper)
                pitch_shift = self.profile['pitch_shift']
                # Using frame rate adjustment for pitch shifting
                new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))
                pitched_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
                pitched_audio = pitched_audio.set_frame_rate(audio.frame_rate)
                
                # Add slight reverb for depth (optional)
                # This is a simple echo effect
                if 'add_reverb' in self.profile and self.profile['add_reverb']:
                    delay = 50  # ms
                    decay = 0.1  # volume reduction
                    echo = pitched_audio - 10  # reduce volume
                    combined = pitched_audio.overlay(echo, position=delay)
                    pitched_audio = combined
                
                # Save with effects
                pitched_audio.export(audio_path, format=self.output_format)
                
            except Exception as e:
                print(f"Warning: Could not apply voice effects: {e}")
    
    def generate_audio_chunk(self, text, output_path):
        """
        Generate audio with speaker selection and effects
        """
        try:
            # Generate based on model type
            if self.profile['model'] == "tts_models/multilingual/multi-dataset/xtts_v2":
                # XTTS requires language parameter
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=self.profile.get('language', 'en'),
                    speed=self.profile.get('speed', 1.0)
                )
            elif hasattr(self.tts, 'speakers') and self.profile['speaker']:
                # Multi-speaker model
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=self.profile['speaker'],
                    speed=self.profile.get('speed', 1.0)
                )
            else:
                # Single speaker model
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speed=self.profile.get('speed', 1.0)
                )
            
            # Apply post-processing effects if needed
            if 'pitch_shift' in self.profile:
                self.apply_voice_effects(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            return False
    
    def smart_chunk_text(self, text):
        """
        Smart chunking for natural speech flow
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_size = self.profile['chunk_size']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if sentence_length > chunk_size:
                # Split long sentences
                parts = re.split(r'[,;]\s*', sentence)
                for part in parts:
                    if current_length + len(part) > chunk_size:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                    current_chunk.append(part)
                    current_length += len(part)
            else:
                if current_length + sentence_length > chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_text_file(self, input_file, output_name=None):
        """
        Process text file with deep voice generation
        """
        input_path = Path(input_file)
        base_name = output_name or input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{base_name}_deep_voice_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Read and preprocess
        print(f"\n🎙️ Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        original_length = len(text)
        text = self.preprocess_text(text)
        processed_length = len(text)
        
        print(f"📝 Preprocessed: {original_length} → {processed_length} chars")
        print(f"   Acronyms detected and processed")
        
        # Create chunks
        chunks = self.smart_chunk_text(text)
        print(f"📦 Created {len(chunks)} chunks")
        
        # Save metadata
        metadata = {
            "input_file": str(input_file),
            "voice_profile": self.profile,
            "device": self.device,
            "chunks": len(chunks),
            "timestamp": timestamp
        }
        
        # Generate audio
        print(f"\n🎵 Generating deep voice audio...")
        audio_files = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing")):
            chunk_file = chunks_dir / f"chunk_{i:04d}.{self.output_format}"
            
            if self.generate_audio_chunk(chunk, str(chunk_file)):
                audio_files.append(str(chunk_file))
                
                # Save chunk text
                with open(chunk_file.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(chunk)
        
        print(f"\n✓ Generated: {len(audio_files)}/{len(chunks)} chunks")
        
        # Combine audio
        if audio_files:
            final_output = output_dir / f"{base_name}_complete.{self.output_format}"
            print(f"\n🔄 Combining audio...")
            
            try:
                combined = AudioSegment.from_file(audio_files[0])
                for audio_file in audio_files[1:]:
                    audio = AudioSegment.from_file(audio_file)
                    combined = combined.append(audio, crossfade=50)
                
                # Apply final mastering
                combined = combined.normalize()  # Normalize volume
                
                # Export
                if self.output_format == "mp3":
                    combined.export(final_output, format="mp3", bitrate="192k")
                else:
                    combined.export(final_output, format="wav")
                
                print(f"✓ Final audio: {final_output}")
                
                file_size = final_output.stat().st_size / (1024 * 1024)
                metadata["file_size_mb"] = round(file_size, 2)
                print(f"📊 Size: {file_size:.2f} MB")
                
            except Exception as e:
                print(f"Error combining: {e}")
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Complete! Output: {output_dir}")
        return str(output_dir)


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Voice TTS with accent options")
    parser.add_argument("input_file", nargs='?', help="Input text file")
    parser.add_argument("--voice", default="random",
                       help="Voice: speaker ID (p230, p234, etc.), 'random' for random favorite, or preset name")
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                       help="Output format")
    parser.add_argument("--output-name", help="Custom output name")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Force device")
    parser.add_argument("--list-voices", action="store_true", 
                       help="List available favorite voices and exit")
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        print("Favorite deep male voices:")
        favorites = ["p230", "p234", "p240", "p244", "p246", "p248", "p250"]
        for voice in favorites:
            print(f"  {voice}")
        print("\nUsage examples:")
        print("  python deep_voice_tts.py input.txt --voice p230")
        print("  python deep_voice_tts.py input.txt --voice random")
        return
    
    # Create pipeline
    pipeline = DeepVoiceTTS(
        voice_profile=args.voice,
        output_format=args.format,
        device=args.device
    )
    
    # Process
    pipeline.process_text_file(
        args.input_file,
        output_name=args.output_name
    )


if __name__ == "__main__":
    main()