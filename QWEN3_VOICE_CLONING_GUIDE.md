# Qwen3-TTS Voice Cloning — Setup & Usage Guide

A complete guide for Claude agents (or humans) setting up Qwen3-TTS voice cloning on a new machine. This was developed and tested on an Ubuntu system with NVIDIA GPU.

---

## 1. Hardware Requirements

| Model | VRAM Needed | Quality |
|-------|------------|---------|
| 0.6B  | ~8 GB      | Acceptable |
| 1.7B  | ~16 GB     | Best voice fidelity |

CPU offloading is available if VRAM is tight (splits model across GPU + system RAM).

---

## 2. Environment Setup

### Create conda environment

```bash
conda create -n qwen3-tts python=3.10 -y
conda activate qwen3-tts
```

### Install PyTorch with CUDA

Match the CUDA version to your driver. Check with `nvidia-smi`.

```bash
# CUDA 12.9 (adjust cu129 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### Install Qwen3-TTS and dependencies

```bash
pip install qwen-tts>=0.0.5 transformers>=4.57.0 accelerate>=1.12.0
pip install soundfile scipy pydub librosa tqdm einops numpy
```

### Install FlashAttention 2 (strongly recommended, but optional)

This compiles from source and takes **60–90 minutes**. If it fails, the code auto-falls back to SDPA (Scaled Dot-Product Attention), which is slower but works fine.

```bash
pip install flash-attn --no-build-isolation
```

### System dependency (for audio processing)

```bash
sudo apt-get install -y sox libsox-dev    # Debian/Ubuntu
# or
sudo yum install -y sox sox-devel          # RHEL/CentOS
```

### Optional: hallucination detection

```bash
pip install faster-whisper
```

---

## 3. Model Download

Models are hosted on HuggingFace under `Qwen/`. The `qwen-tts` library downloads them automatically on first use, but you can pre-download for offline use:

```python
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",         # 651 MB — shared tokenizer
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",          # 2.4 GB — voice cloning (small)
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",   # 2.4 GB — built-in speakers (small)
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",          # 4.3 GB — voice cloning (large)
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",   # 4.3 GB — built-in speakers (large)
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",   # 4.3 GB — design voice from description
]

for repo in models:
    snapshot_download(repo, local_dir=f"./models/{repo.split('/')[-1]}")
```

**Which model does what:**

| Task | Model variant | Method |
|------|--------------|--------|
| Voice cloning (from reference audio) | `*-Base` | `generate_voice_clone()` |
| Built-in speaker voices (9 speakers) | `*-CustomVoice` | `generate_custom_voice()` |
| Voice design from text description | `*-VoiceDesign` | `generate_voice_design()` |

---

## 4. Minimal Voice Cloning Script

This is the bare-minimum code to clone a voice:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load the Base model (required for voice cloning)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # or "sdpa" if no flash-attn
)

# Step 1: Create voice clone prompt from reference audio
#   - ref_audio: path to a WAV file (3–30 seconds of clean speech)
#   - ref_text: exact transcript of what's said in that audio
clone_prompt = model.create_voice_clone_prompt(
    ref_audio="reference_voice.wav",
    ref_text="This is the exact transcript of the reference audio."
)

# Step 2: Generate speech in the cloned voice
wavs, sr = model.generate_voice_clone(
    text="Any new text you want spoken in the cloned voice.",
    language="English",
    voice_clone_prompt=clone_prompt,
)

# Step 3: Save output
audio_data = wavs[0]
if hasattr(audio_data, 'cpu'):
    audio_data = audio_data.cpu().numpy()
sf.write("output.wav", audio_data, sr)
```

### Key points:

- The `clone_prompt` is reusable — compute it once, generate many utterances
- Both `ref_audio` AND `ref_text` are required (the transcript must match the audio)
- Use `language="English"` explicitly (or `"Auto"` for auto-detect)

---

## 5. Reference Audio Requirements

| Property | Requirement |
|----------|------------|
| Duration | 3+ seconds (tested up to ~30s) |
| Format | WAV (linear PCM, 16-bit) |
| Sample rate | 16 kHz or 22.05 kHz (resampled internally) |
| Quality | Clean speech, no background noise/music |
| Language | Best results when reference matches target language |
| Transcript | Must be accurate and match the audio content |

---

## 6. Full Pipeline (deep_voice_tts_v3.py)

The production script includes text preprocessing, smart chunking, acronym handling, hallucination detection, and audio normalization.

### Built-in voice (CustomVoice mode)

```bash
python deep_voice_tts_v3.py input.txt --voice dylan --model-size 1.7B
```

### Voice cloning from reference audio

```bash
python deep_voice_tts_v3.py input.txt \
    --clone-audio reference.wav \
    --clone-text "Exact transcript of the reference audio" \
    --model-size 1.7B
```

### Voice design from text description

```bash
python deep_voice_tts_v3.py input.txt \
    --design "A deep, authoritative male voice with slight British accent" \
    --model-size 1.7B
```

### Memory-constrained GPU (8–12 GB VRAM)

```bash
python deep_voice_tts_v3.py input.txt --voice dylan --model-size 0.6B --cpu-offload
```

### All CLI options

```
--voice NAME          Built-in voice: dylan, eric, ryan, aiden, uncle_fu,
                      vivian, serena, ono_anna, sohee, or "random"
--clone-audio PATH    Reference audio for voice cloning
--clone-text TEXT     Transcript of reference audio (required with --clone-audio)
--design TEXT         Voice description for voice design mode
--instruct TEXT       Custom speaking style instruction
--model-size SIZE     0.6B or 1.7B (default: 1.7B)
--format FORMAT       mp3 or wav (default: mp3)
--speed FLOAT         Playback speed 0.5–2.0 (default: 1.0)
--cpu-offload         Split model across GPU + CPU RAM
--no-flash-attn       Disable FlashAttention 2 (use SDPA)
--no-cleanup          Keep intermediate chunk files
--output-name NAME    Custom output directory name
--list-voices         List all available voices
```

### Available built-in speakers

| Name | Description |
|------|------------|
| dylan | Deep American male |
| eric | Mature male voice |
| ryan | Clear male voice |
| aiden | Young adult male |
| uncle_fu | Deep Chinese male |
| vivian | Clear female voice |
| serena | Warm female voice |
| ono_anna | Japanese female |
| sohee | Korean female |

---

## 7. Gotchas & Troubleshooting

### FlashAttention 2 won't install
Not critical. The code auto-falls back to SDPA. Just use `--no-flash-attn` or let it fall back.

### "CUDA out of memory"
- Switch to `--model-size 0.6B`
- Add `--cpu-offload` (uses `device_map="auto"` with `max_memory={0: "10GiB", "cpu": "32GiB"}`)
- Both can be combined

### Voice cloning requires BOTH audio and text
You must provide the reference audio AND its accurate transcript. The model uses both for speaker embedding + style extraction. Inaccurate transcripts degrade quality.

### Language mismatch
Set `language="English"` explicitly. Quality drops when reference audio language doesn't match the target synthesis language.

### 1.7B vs 0.6B for cloning
The 1.7B model produces significantly better voice cloning fidelity. Use 0.6B only if VRAM is a hard constraint.

### Hallucination (repetitive babbling)
The pipeline includes automatic detection via `faster-whisper` — it transcribes the output, looks for 3+ consecutive repeated phrases or rapid-fire short segments, and surgically removes them with crossfade. Install `faster-whisper` to enable this.

### Speed adjustment
Use `--speed 0.8` for slower, more deliberate speech. The implementation changes the frame rate before resampling back, which preserves quality better than post-processing.

---

## 8. File Layout

```
gpu-tts-toolkit/
├── deep_voice_tts_v3.py          # Main production script
├── requirements-qwen3.txt        # pip requirements
├── install_dgx.sh                # Full install script (conda + pip + flash-attn)
├── voice_references/             # Reference audio for pre-configured clones
│   ├── mcconaughey_reference.wav
│   ├── mcconaughey_transcript.txt
│   ├── herzog_reference.wav
│   └── herzog_transcript.txt
└── QWEN3_VOICE_CLONING_GUIDE.md  # This file
```

---

## 9. Quick Verification

After setup, verify everything works:

```python
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

try:
    import flash_attn
    print(f'FlashAttention: {flash_attn.__version__}')
except ImportError:
    print('FlashAttention: not installed (will use SDPA fallback)')

import qwen_tts
print('qwen-tts: OK')
"
```
