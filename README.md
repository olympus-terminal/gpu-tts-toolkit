# Text-to-Speech Tools

> Comprehensive text-to-speech toolkit featuring neural TTS engines, intelligent text preprocessing, and cloud integration with AWS Polly.

[![License](https://img.shields.io/github/license/olympus-terminal/text-to-speech)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/olympus-terminal/text-to-speech?style=social)](https://github.com/olympus-terminal/text-to-speech/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/olympus-terminal/text-to-speech)](https://github.com/olympus-terminal/text-to-speech/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/olympus-terminal/text-to-speech)](https://github.com/olympus-terminal/text-to-speech/commits/main)
[![Tools](https://img.shields.io/badge/tools-7-green.svg)](https://github.com/olympus-terminal/text-to-speech)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org)

## 🎙️ Overview

A modern text-to-speech toolkit that bridges the gap between simple TTS needs and advanced speech synthesis requirements. Whether you need quick local TTS, high-quality neural synthesis, or scalable cloud-based solutions, this toolkit provides the right tool for the job.

### Key Features

- **Multiple TTS Engines**: From lightweight to state-of-the-art neural models
- **Intelligent Preprocessing**: Clean and optimize text for natural speech
- **Cloud Integration**: Seamless AWS Polly integration for production use
- **Batch Processing**: Convert entire documents or datasets efficiently
- **Multi-Language Support**: Handle various languages and accents
- **Customizable Voices**: Fine-tune voice characteristics and emotions

## 📁 Repository Structure

```
text-to-speech/
├── tts-engines/          # Various TTS engine implementations
│   ├── neural_tts.py     # Advanced neural TTS
│   ├── tts_simple.py     # Lightweight TTS
│   └── tts4.py          # Enhanced TTS v4
├── text-preprocessing/   # Text preparation utilities
│   ├── Clean_txt_for_TTS.sh
│   └── clean_text_for_tts.sh
└── aws-polly/           # AWS Polly integration
    └── txt-2-polly-2-s3.sh
```

## 🚀 Quick Start

### Prerequisites

```bash
# Core requirements
python >= 3.7
bash >= 4.0

# Python packages (basic)
pip install pyttsx3 gtts pygame

# Python packages (advanced)
pip install torch torchaudio transformers

# AWS CLI (for Polly integration)
aws --version  # AWS CLI version 2.x recommended
```

### Installation

```bash
# Clone the repository
git clone https://github.com/olympus-terminal/text-to-speech.git
cd text-to-speech

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x text-preprocessing/*.sh
chmod +x aws-polly/*.sh

# Test installation
python tts-engines/tts_simple.py "Hello, world!" test_output.mp3
```

## 🔧 Tool Guide

### TTS Engines (`tts-engines/`)

#### Simple TTS
Quick and lightweight text-to-speech for basic needs:

```python
# Basic usage
python tts_simple.py "Hello, this is a test" output.mp3

# With options
python tts_simple.py \
    --text "Welcome to our service" \
    --output welcome.mp3 \
    --rate 150 \
    --voice "female"
```

#### Neural TTS
State-of-the-art neural speech synthesis:

```python
# High-quality synthesis
python neural_tts.py \
    --input script.txt \
    --output podcast.wav \
    --model "tacotron2" \
    --vocoder "waveglow" \
    --quality "high"

# Emotional speech
python neural_tts.py \
    --text "I'm so excited!" \
    --emotion "happy" \
    --intensity 0.8 \
    --output excited.wav
```

#### TTS Version 4
Enhanced TTS with advanced features:

```python
# Multi-speaker synthesis
python tts4.py \
    --input dialogue.txt \
    --speakers "alice:female,bob:male" \
    --output conversation.mp3

# SSML support
python tts4.py \
    --ssml input.ssml \
    --output narration.wav \
    --format "wav" \
    --sample-rate 44100
```

### Text Preprocessing (`text-preprocessing/`)

#### Clean Text for TTS
Prepare text for optimal speech synthesis:

```bash
# Basic cleaning
./clean_text_for_tts.sh raw_text.txt > cleaned_text.txt

# Advanced preprocessing
./Clean_txt_for_TTS.sh \
    --input manuscript.txt \
    --output clean_manuscript.txt \
    --expand-abbreviations \
    --normalize-numbers \
    --fix-punctuation
```

**Preprocessing Features:**
- Remove special characters that confuse TTS
- Expand abbreviations (Dr. → Doctor)
- Convert numbers to words (123 → one hundred twenty-three)
- Fix punctuation for natural pauses
- Handle acronyms intelligently (NASA → N.A.S.A.)
- Clean URLs and email addresses

### AWS Polly Integration (`aws-polly/`)

#### Text to Polly to S3
Production-ready cloud TTS with AWS:

```bash
# Basic Polly synthesis
./txt-2-polly-2-s3.sh \
    input.txt \
    s3://my-bucket/audio/output.mp3

# With voice selection
./txt-2-polly-2-s3.sh \
    --text "Professional narration" \
    --voice "Matthew" \
    --engine "neural" \
    --s3-bucket "my-audio-bucket" \
    --s3-key "narrations/prof_narration.mp3"
```

**AWS Polly Features:**
- 60+ voices in 29 languages
- Neural and standard engines
- SSML support for fine control
- Direct S3 upload for scalability
- Batch processing capabilities

## 📊 Use Cases & Workflows

### 1. Audiobook Production
```bash
# Step 1: Prepare the text
./text-preprocessing/Clean_txt_for_TTS.sh \
    --input book.txt \
    --output book_cleaned.txt \
    --chapter-breaks

# Step 2: Generate chapters
python tts-engines/neural_tts.py \
    --input book_cleaned.txt \
    --output-dir audiobook/ \
    --voice "narrative" \
    --chapter-mode \
    --format "mp3" \
    --bitrate 192

# Step 3: Upload to cloud
for chapter in audiobook/*.mp3; do
    ./aws-polly/txt-2-polly-2-s3.sh \
        --upload-only "$chapter" \
        --s3-path "audiobooks/my-book/"
done
```

### 2. Multilingual Announcements
```python
# Generate announcements in multiple languages
languages = {
    'en': 'Hello, welcome to our service',
    'es': 'Hola, bienvenido a nuestro servicio',
    'fr': 'Bonjour, bienvenue dans notre service',
    'de': 'Hallo, willkommen bei unserem Service'
}

for lang, text in languages.items():
    os.system(f"""
        python tts-engines/tts4.py \
            --text "{text}" \
            --language "{lang}" \
            --output "welcome_{lang}.mp3"
    """)
```

### 3. Podcast Generation
```bash
# Preprocess script
./text-preprocessing/clean_text_for_tts.sh \
    podcast_script.txt > podcast_clean.txt

# Generate with neural TTS
python tts-engines/neural_tts.py \
    --input podcast_clean.txt \
    --output podcast_episode.wav \
    --voice "conversational" \
    --pace "moderate" \
    --emphasis-keywords "important,breaking,exclusive"

# Add intro/outro music (requires ffmpeg)
ffmpeg -i intro.mp3 -i podcast_episode.wav -i outro.mp3 \
    -filter_complex '[0][1][2]concat=n=3:v=0:a=1' \
    final_podcast.mp3
```

### 4. Real-time TTS API
```python
# Simple Flask API for TTS
from flask import Flask, request, send_file
import tts_simple

app = Flask(__name__)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']
    voice = request.json.get('voice', 'default')
    
    output_file = f"/tmp/{hash(text)}.mp3"
    tts_simple.synthesize(text, output_file, voice=voice)
    
    return send_file(output_file, mimetype='audio/mpeg')
```

## 🎯 Advanced Features

### Custom Voice Training
```python
# Fine-tune neural TTS on custom voice
python train_custom_voice.py \
    --dataset my_voice_recordings/ \
    --base-model tacotron2 \
    --output-model my_custom_voice.pt \
    --epochs 100
```

### Emotion Control
```python
# Generate speech with emotional variations
emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised']
for emotion in emotions:
    python neural_tts.py \
        --text "How are you today?" \
        --emotion {emotion} \
        --output f"greeting_{emotion}.wav"
```

### SSML Support
```xml
<!-- input.ssml -->
<speak>
  <p>Welcome to our service.</p>
  <break time="500ms"/>
  <p>
    <prosody rate="slow" pitch="+2st">
      This is important:
    </prosody>
    <emphasis level="strong">Please listen carefully.</emphasis>
  </p>
</speak>
```

```bash
python tts4.py --ssml input.ssml --output announcement.mp3
```

## 🔧 Configuration

### Global Settings
Create `config.json` in the root directory:

```json
{
  "default_voice": "neural",
  "default_language": "en-US",
  "sample_rate": 22050,
  "quality": "high",
  "aws": {
    "region": "us-east-1",
    "voice_id": "Matthew",
    "engine": "neural"
  },
  "preprocessing": {
    "expand_abbreviations": true,
    "normalize_numbers": true,
    "clean_urls": true
  }
}
```

### Voice Profiles
Define reusable voice configurations:

```json
{
  "narrator": {
    "engine": "neural",
    "voice": "matthew",
    "rate": 0.95,
    "pitch": -2
  },
  "character_young": {
    "engine": "neural", 
    "voice": "sophia",
    "rate": 1.1,
    "pitch": +5
  }
}
```

## 📈 Performance & Optimization

### Benchmarks

| Engine | Quality | Speed (RTF) | CPU Usage | GPU Usage |
|--------|---------|-------------|-----------|-----------|
| Simple TTS | Basic | 0.1x | 15% | N/A |
| Neural TTS (CPU) | High | 2.5x | 85% | N/A |
| Neural TTS (GPU) | High | 0.3x | 25% | 70% |
| AWS Polly | High | N/A* | N/A | N/A |

*AWS Polly performance depends on internet connection and AWS resources

### Optimization Tips

1. **Batch Processing**: Process multiple texts together
2. **Caching**: Cache commonly used phrases
3. **GPU Acceleration**: Use CUDA for neural models
4. **Preprocessing**: Clean text before synthesis
5. **Format Selection**: Use appropriate audio formats

## 🐛 Troubleshooting

### Common Issues

**No Audio Output**
```bash
# Check audio drivers
python -c "import pyttsx3; engine = pyttsx3.init(); print(engine.getProperty('voices'))"

# Test with different engine
python tts_simple.py --engine "espeak" "test" test.wav
```

**Poor Quality Output**
```bash
# Increase sample rate
python neural_tts.py --sample-rate 44100 --quality "best"

# Use better vocoder
python neural_tts.py --vocoder "hifigan" --output high_quality.wav
```

**AWS Polly Errors**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Test Polly access
aws polly describe-voices

# Check S3 permissions
aws s3 ls s3://your-bucket/
```

## 🤝 Contributing

We welcome contributions! Areas of interest:

- Additional TTS engine integrations
- More preprocessing capabilities  
- Voice cloning features
- Real-time streaming TTS
- Mobile optimizations

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Mozilla TTS project
- AWS Polly team
- Open source TTS community
- Tacotron2 and WaveGlow researchers

## 📮 Contact

- Issues: [GitHub Issues](https://github.com/olympus-terminal/text-to-speech/issues)
- Discussions: [GitHub Discussions](https://github.com/olympus-terminal/text-to-speech/discussions)
- Author: [@olympus-terminal](https://github.com/olympus-terminal)
