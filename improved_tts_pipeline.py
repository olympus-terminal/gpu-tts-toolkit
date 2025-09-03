#!/usr/bin/env python3
"""
Improved TTS Pipeline - Reduced hallucinations and better quality
Features:
- Advanced text preprocessing
- Multiple model options with quality comparison
- Chunk validation and overlap handling
- Hallucination detection and mitigation
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
import difflib
import unicodedata


class ImprovedTTSPipeline:
    def __init__(self, model_profile="balanced", output_format="mp3", device=None):
        """
        Initialize with different model profiles for different use cases
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_format = output_format
        
        # Model profiles optimized for different needs
        self.model_profiles = {
            "fast": {
                "model": "tts_models/en/ljspeech/glow-tts",
                "vocoder": None,
                "description": "Fast processing, good quality",
                "chunk_size": 450,
                "overlap": 50
            },
            "balanced": {
                "model": "tts_models/en/ljspeech/vits",
                "vocoder": None,
                "description": "Balanced speed and quality, reduced hallucinations",
                "chunk_size": 400,
                "overlap": 40
            },
            "quality": {
                "model": "tts_models/en/vctk/vits",
                "vocoder": None,
                "description": "Higher quality, multi-speaker capable",
                "chunk_size": 350,
                "overlap": 35
            },
            "premium": {
                "model": "tts_models/multilingual/multi-dataset/xtts_v2",
                "vocoder": None,
                "description": "Premium quality, slowest but most accurate",
                "chunk_size": 300,
                "overlap": 30
            }
        }
        
        self.profile = self.model_profiles[model_profile]
        print(f"Using {model_profile} profile: {self.profile['description']}")
        print(f"Device: {self.device}")
        
        self.load_model()
        
    def load_model(self):
        """Load the TTS model"""
        try:
            print(f"Loading model: {self.profile['model']}")
            self.tts = TTS(self.profile['model'], progress_bar=False).to(self.device)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_text(self, text):
        """
        Advanced text preprocessing to reduce hallucinations
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace problematic characters
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '—': ' - ',
            '–': ' - ',
            '…': '...',
            '\u200b': '',  # Zero-width space
            '\ufeff': '',  # BOM
            '\xa0': ' ',   # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Handle special formatting
        text = self.handle_abbreviations(text)
        text = self.handle_numbers(text)
        text = self.handle_acronyms(text)
        
        # Remove URLs more comprehensively
        text = re.sub(r'https?://[^\s]+', '[link]', text)
        text = re.sub(r'www\.[^\s]+', '[website]', text)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[email]', text)
        
        # Handle citations and references
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)  # [1], [1,2,3]
        text = re.sub(r'\([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\)', '', text)
        
        # Clean up mathematical notation
        text = re.sub(r'\$[^\$]+\$', '[formula]', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^\}]*\}', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def handle_abbreviations(self, text):
        """Handle common abbreviations"""
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Misses',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'Sr.': 'Senior',
            'Jr.': 'Junior',
            'Ph.D.': 'PhD',
            'M.D.': 'MD',
            'B.S.': 'BS',
            'M.S.': 'MS',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'i.e.': 'that is',
            'e.g.': 'for example',
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        return text
    
    def handle_numbers(self, text):
        """Convert numbers to more speakable format"""
        # Large numbers with commas
        text = re.sub(r'(\d{1,3}(?:,\d{3})+)', lambda m: m.group().replace(',', ''), text)
        
        # Percentages
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        
        # Currency
        text = re.sub(r'\$(\d+(?:\.\d{2})?)', r'\1 dollars', text)
        text = re.sub(r'€(\d+(?:\.\d{2})?)', r'\1 euros', text)
        text = re.sub(r'£(\d+(?:\.\d{2})?)', r'\1 pounds', text)
        
        return text
    
    def handle_acronyms(self, text):
        """Handle acronyms by adding spaces between letters"""
        # Find sequences of 2+ capital letters
        def space_acronym(match):
            acronym = match.group()
            # Common acronyms that should be pronounced as words
            word_acronyms = {'NASA', 'NATO', 'AIDS', 'COVID', 'LASER', 'RADAR'}
            if acronym in word_acronyms:
                return acronym
            # Otherwise space out the letters
            return ' '.join(acronym)
        
        text = re.sub(r'\b[A-Z]{2,}\b', space_acronym, text)
        return text
    
    def smart_chunk_text(self, text):
        """
        Smart chunking with overlap to reduce cut-off words and improve continuity
        """
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If single sentence is too long, split it
            if sentence_length > self.profile['chunk_size']:
                # Split by commas or semicolons
                parts = re.split(r'[,;]\s*', sentence)
                for part in parts:
                    if current_length + len(part) > self.profile['chunk_size']:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            # Keep last sentence for overlap
                            if len(current_chunk) > 1:
                                current_chunk = [current_chunk[-1]]
                                current_length = len(current_chunk[0])
                            else:
                                current_chunk = []
                                current_length = 0
                    current_chunk.append(part)
                    current_length += len(part)
            else:
                # Normal sentence processing
                if current_length + sentence_length > self.profile['chunk_size']:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Overlap: keep last sentence
                        if self.profile['overlap'] > 0 and len(current_chunk) > 1:
                            current_chunk = [current_chunk[-1]]
                            current_length = len(current_chunk[0])
                        else:
                            current_chunk = []
                            current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def validate_chunk(self, chunk):
        """
        Validate chunk for potential issues that cause hallucinations
        """
        issues = []
        
        # Check for incomplete sentences
        if not chunk.strip().endswith(('.', '!', '?', '"', "'")):
            issues.append("Incomplete sentence")
        
        # Check for unbalanced quotes
        if chunk.count('"') % 2 != 0:
            issues.append("Unbalanced quotes")
        
        # Check for unbalanced parentheses
        if chunk.count('(') != chunk.count(')'):
            issues.append("Unbalanced parentheses")
        
        # Check for very short chunks (might cause repetition)
        if len(chunk.strip()) < 20:
            issues.append("Very short chunk")
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()-]', chunk)) / max(len(chunk), 1)
        if special_char_ratio > 0.1:
            issues.append("High special character ratio")
        
        return len(issues) == 0, issues
    
    def generate_audio_safe(self, text, output_path, max_retries=2):
        """
        Generate audio with error handling and retry logic
        """
        for attempt in range(max_retries):
            try:
                # Validate chunk first
                is_valid, issues = self.validate_chunk(text)
                if not is_valid and attempt == 0:
                    # Try to fix issues
                    if "Incomplete sentence" in issues:
                        text = text.strip() + "."
                    if "Unbalanced quotes" in issues:
                        text = text.replace('"', '')
                
                # Generate audio
                if self.output_format == "mp3":
                    temp_wav = output_path.replace('.mp3', '_temp.wav')
                    self.tts.tts_to_file(text=text, file_path=temp_wav)
                    
                    # Convert to MP3 with good quality settings
                    audio = AudioSegment.from_wav(temp_wav)
                    audio.export(output_path, format="mp3", bitrate="192k", parameters=["-q:a", "2"])
                    
                    # Clean up temp file
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                else:
                    self.tts.tts_to_file(text=text, file_path=output_path)
                
                return True
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt == max_retries - 1:
                    print(f"  Failed to generate audio for chunk after {max_retries} attempts")
                    return False
        
        return False
    
    def combine_audio_files(self, audio_files, output_path):
        """
        Combine multiple audio files with smooth transitions
        """
        if not audio_files:
            return False
        
        try:
            combined = AudioSegment.from_file(audio_files[0])
            
            for audio_file in audio_files[1:]:
                audio = AudioSegment.from_file(audio_file)
                # Add small crossfade for smoother transitions
                combined = combined.append(audio, crossfade=50)
            
            # Export combined audio
            if self.output_format == "mp3":
                combined.export(output_path, format="mp3", bitrate="192k", parameters=["-q:a", "2"])
            else:
                combined.export(output_path, format="wav")
            
            return True
        except Exception as e:
            print(f"Error combining audio files: {e}")
            return False
    
    def process_text_file(self, input_file, output_name=None):
        """
        Main processing function for converting text to audio
        """
        # Setup paths
        input_path = Path(input_file)
        base_name = output_name or input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{base_name}_audio_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Read and preprocess text
        print(f"\n📖 Reading: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        original_length = len(text)
        text = self.preprocess_text(text)
        processed_length = len(text)
        
        print(f"📝 Text preprocessed: {original_length} → {processed_length} characters")
        print(f"   Removed: {original_length - processed_length} characters")
        
        # Create chunks
        chunks = self.smart_chunk_text(text)
        print(f"📦 Created {len(chunks)} chunks")
        
        # Process metadata
        metadata = {
            "input_file": str(input_file),
            "model": self.profile['model'],
            "device": self.device,
            "chunks": len(chunks),
            "timestamp": timestamp,
            "preprocessing": {
                "original_length": original_length,
                "processed_length": processed_length,
                "removed_chars": original_length - processed_length
            }
        }
        
        # Generate audio for each chunk
        print("\n🎵 Generating audio chunks...")
        audio_files = []
        failed_chunks = []
        
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            chunk_file = chunks_dir / f"chunk_{i:04d}.{self.output_format}"
            
            # Validate and generate
            is_valid, issues = self.validate_chunk(chunk)
            if not is_valid:
                print(f"\n  ⚠️ Chunk {i} has issues: {', '.join(issues)}")
            
            if self.generate_audio_safe(chunk, str(chunk_file)):
                audio_files.append(str(chunk_file))
                
                # Save chunk text for debugging
                with open(chunk_file.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                    f.write(chunk)
            else:
                failed_chunks.append(i)
        
        # Report results
        print(f"\n✓ Successfully generated: {len(audio_files)}/{len(chunks)} chunks")
        if failed_chunks:
            print(f"✗ Failed chunks: {failed_chunks}")
            metadata["failed_chunks"] = failed_chunks
        
        # Combine audio files
        if audio_files:
            final_output = output_dir / f"{base_name}_complete.{self.output_format}"
            print(f"\n🔄 Combining audio files...")
            
            if self.combine_audio_files(audio_files, str(final_output)):
                print(f"✓ Final audio saved: {final_output}")
                metadata["final_output"] = str(final_output)
                
                # Get file size
                file_size = final_output.stat().st_size / (1024 * 1024)  # MB
                metadata["file_size_mb"] = round(file_size, 2)
                print(f"📊 File size: {file_size:.2f} MB")
            else:
                print("✗ Failed to combine audio files")
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Processing complete! Output directory: {output_dir}")
        return str(output_dir)


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved TTS Pipeline with reduced hallucinations")
    parser.add_argument("input_file", help="Input text file to convert")
    parser.add_argument("--model", choices=["fast", "balanced", "quality", "premium"],
                       default="balanced", help="Model profile to use")
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                       help="Output audio format")
    parser.add_argument("--output-name", help="Custom output name (without extension)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Force specific device")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ImprovedTTSPipeline(
        model_profile=args.model,
        output_format=args.format,
        device=args.device
    )
    
    # Process file
    pipeline.process_text_file(
        args.input_file,
        output_name=args.output_name
    )


if __name__ == "__main__":
    main()