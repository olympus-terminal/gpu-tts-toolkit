#!/usr/bin/env python3
"""Synthesize real speech with the repository's Qwen3-TTS CustomVoice API.

The first run may download the selected model from Hugging Face. Subsequent
runs can use the local model cache.

Examples:
    python examples/simple_tts.py --text "Hello world"
    python examples/simple_tts.py --file input.txt \
        --output speech_20260712_094331.wav
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Sequence, Union


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_LANGUAGE = "English"


def default_output_path(now: Optional[datetime] = None) -> Path:
    """Return a timestamped WAV path that identifies this example."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(f"simple_tts_qwen3_custom_voice_{timestamp}.wav")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments without loading ML dependencies."""
    parser = argparse.ArgumentParser(
        description="Synthesize speech with Qwen3-TTS CustomVoice."
    )
    text_source = parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text", help="Text to synthesize")
    text_source.add_argument(
        "--file",
        type=Path,
        help="UTF-8 text file containing one short utterance",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output WAV path. Defaults to "
            "simple_tts_qwen3_custom_voice_YYYYMMDD_HHMMSS.wav."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model ID or path")
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER, help="CustomVoice speaker")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Target language")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Transformers device map, for example cuda:0 or cpu",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
        help="Attention backend; FlashAttention 2 requires its optional package",
    )
    return parser.parse_args(argv)


def get_input_text(text: Optional[str], file_path: Optional[Path]) -> str:
    """Read and validate the requested text source."""
    if file_path is not None:
        with file_path.open("r", encoding="utf-8") as handle:
            value = handle.read()
    else:
        value = text or ""

    value = value.strip()
    if not value:
        raise ValueError("Input text is empty.")
    return value


def prepare_output_path(
    output: Optional[Union[str, Path]], now: Optional[datetime] = None
) -> Path:
    """Validate the output path before the expensive model load."""
    output_path = Path(output) if output is not None else default_output_path(now)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output must use the .wav extension.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    return output_path


def reserve_output_path(output_path: Path) -> None:
    """Atomically reserve a new output path so synthesis never overwrites a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb"):
            pass
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}"
        ) from exc


def synthesize_and_save(
    *,
    text: str,
    output_path: Path,
    model_name: str,
    speaker: str,
    language: str,
    device: str,
    attention_implementation: str,
    model_class: Optional[Any] = None,
    torch_module: Optional[Any] = None,
    soundfile_module: Optional[Any] = None,
    timer: Callable[[], float] = perf_counter,
    output_reserver: Callable[[Path], None] = reserve_output_path,
) -> Dict[str, Any]:
    """Load Qwen3-TTS, synthesize speech, save it, and return measured metrics."""
    if torch_module is None:
        import torch as torch_module
    if soundfile_module is None:
        import soundfile as soundfile_module
    if model_class is None:
        from qwen_tts import Qwen3TTSModel

        model_class = Qwen3TTSModel

    uses_cuda = device.lower().startswith("cuda")
    if uses_cuda and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device!r}, but CUDA is unavailable. "
            "Pass --device cpu to run on the CPU."
        )

    dtype = torch_module.bfloat16 if uses_cuda else torch_module.float32
    model = model_class.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
        attn_implementation=attention_implementation,
    )

    if uses_cuda:
        torch_module.cuda.synchronize()
    started_at = timer()
    waveforms, sample_rate = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
    )
    if uses_cuda:
        torch_module.cuda.synchronize()
    elapsed_seconds = timer() - started_at

    if not waveforms:
        raise RuntimeError("Qwen3-TTS returned no waveform.")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise RuntimeError(f"Qwen3-TTS returned an invalid sample rate: {sample_rate}")

    waveform = waveforms[0]
    sample_count = len(waveform)
    if sample_count == 0:
        raise RuntimeError("Qwen3-TTS returned an empty waveform.")

    audio_duration_seconds = sample_count / sample_rate
    real_time_factor = elapsed_seconds / audio_duration_seconds

    output_reserver(output_path)
    soundfile_module.write(str(output_path), waveform, sample_rate)

    return {
        "elapsed_seconds": elapsed_seconds,
        "audio_duration_seconds": audio_duration_seconds,
        "real_time_factor": real_time_factor,
        "sample_count": sample_count,
        "sample_rate": sample_rate,
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "soundfile_version": getattr(soundfile_module, "__version__", "unknown"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line example."""
    args = parse_args(argv)
    try:
        text = get_input_text(args.text, args.file)
        output_path = prepare_output_path(args.output)

        print("Synthesis configuration:")
        print("  Generator: examples/simple_tts.py")
        print(f"  Input source: {args.file if args.file is not None else 'command line'}")
        print(f"  Model: {args.model}")
        print(f"  Speaker: {args.speaker}")
        print(f"  Language: {args.language}")
        print(f"  Device: {args.device}")
        print(f"  Attention: {args.attention_implementation}")
        print(f"  Output: {output_path}")

        metrics = synthesize_and_save(
            text=text,
            output_path=output_path,
            model_name=args.model,
            speaker=args.speaker,
            language=args.language,
            device=args.device,
            attention_implementation=args.attention_implementation,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Audio saved to: {output_path}")
    print(f"Elapsed synthesis time: {metrics['elapsed_seconds']:.3f} seconds")
    print(f"Audio duration: {metrics['audio_duration_seconds']:.3f} seconds")
    print(f"Real-time factor: {metrics['real_time_factor']:.3f}")
    print(f"Sample rate: {metrics['sample_rate']} Hz")
    print(f"PyTorch: {metrics['torch_version']}")
    print(f"SoundFile: {metrics['soundfile_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
