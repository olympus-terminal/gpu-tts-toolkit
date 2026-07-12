"""Offline unit tests for examples/simple_tts.py.

The zero-valued waveform below is deterministic, synthetic test data used only
to verify CLI wiring and metric calculations. It is not speech, a demo output,
or a substitute for evaluating the real Qwen3-TTS model.
"""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "examples" / "simple_tts.py"
SPEC = importlib.util.spec_from_file_location("simple_tts_example", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
simple_tts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simple_tts)

# Deterministic synthetic fixture for unit testing only; never written as demo audio.
TEST_ONLY_ZERO_WAVEFORM = np.zeros(24_000, dtype=np.float32)


class FakeCuda:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeTorch:
    bfloat16 = "bfloat16"
    float32 = "float32"
    __version__ = "test-only"

    def __init__(self) -> None:
        self.cuda = FakeCuda()


class FakeModel:
    def __init__(self) -> None:
        self.generation_calls = []

    def generate_custom_voice(self, **kwargs):
        self.generation_calls.append(kwargs)
        return [TEST_ONLY_ZERO_WAVEFORM], 24_000


class FakeModelClass:
    load_calls = []
    instance = FakeModel()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.load_calls.append((model_name, kwargs))
        return cls.instance


class FakeSoundFile:
    __version__ = "test-only"
    write_calls = []

    @classmethod
    def write(cls, path, waveform, sample_rate):
        cls.write_calls.append((path, waveform, sample_rate))


class SimpleTTSUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeModelClass.load_calls.clear()
        FakeModelClass.instance.generation_calls.clear()
        FakeSoundFile.write_calls.clear()

    def test_parser_requires_exactly_one_text_source(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            simple_tts.parse_args([])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            simple_tts.parse_args(["--text", "hello", "--file", "input.txt"])

        args = simple_tts.parse_args(["--text", "hello"])
        self.assertEqual(args.text, "hello")
        self.assertIsNone(args.file)

    def test_nonempty_text_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            simple_tts.get_input_text(" \n\t", None)

    def test_default_output_is_timestamped(self) -> None:
        now = datetime(2026, 7, 12, 9, 43, 31)
        self.assertEqual(
            simple_tts.default_output_path(now),
            Path("simple_tts_qwen3_custom_voice_20260712_094331.wav"),
        )

    def test_existing_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_test_simple_tts_") as temp_dir:
            output = Path(temp_dir) / "existing_20260712_094331.wav"
            output.touch()
            with self.assertRaises(FileExistsError):
                simple_tts.prepare_output_path(output)

    def test_real_api_wiring_and_measured_rtf(self) -> None:
        fake_torch = FakeTorch()
        reserved_paths = []
        times = iter((100.0, 100.25))
        output = Path("result_20260712_094331.wav")

        metrics = simple_tts.synthesize_and_save(
            text="Hello from an offline unit test.",
            output_path=output,
            model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            speaker="Ryan",
            language="English",
            device="cuda:0",
            attention_implementation="sdpa",
            model_class=FakeModelClass,
            torch_module=fake_torch,
            soundfile_module=FakeSoundFile,
            timer=lambda: next(times),
            output_reserver=reserved_paths.append,
        )

        self.assertEqual(
            FakeModelClass.load_calls,
            [
                (
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    {
                        "device_map": "cuda:0",
                        "dtype": "bfloat16",
                        "attn_implementation": "sdpa",
                    },
                )
            ],
        )
        self.assertEqual(
            FakeModelClass.instance.generation_calls,
            [
                {
                    "text": "Hello from an offline unit test.",
                    "language": "English",
                    "speaker": "Ryan",
                }
            ],
        )
        self.assertEqual(fake_torch.cuda.synchronize_calls, 2)
        self.assertEqual(reserved_paths, [output])
        self.assertEqual(len(FakeSoundFile.write_calls), 1)
        written_path, written_waveform, written_rate = FakeSoundFile.write_calls[0]
        self.assertEqual(written_path, str(output))
        self.assertIs(written_waveform, TEST_ONLY_ZERO_WAVEFORM)
        self.assertEqual(written_rate, 24_000)
        self.assertEqual(metrics["sample_count"], 24_000)
        self.assertEqual(metrics["sample_rate"], 24_000)
        self.assertAlmostEqual(metrics["elapsed_seconds"], 0.25)
        self.assertAlmostEqual(metrics["audio_duration_seconds"], 1.0)
        self.assertAlmostEqual(metrics["real_time_factor"], 0.25)

    def test_real_soundfile_save_uses_new_timestamped_path(self) -> None:
        fake_torch = FakeTorch()
        times = iter((200.0, 200.5))

        with tempfile.TemporaryDirectory(prefix="agent_test_simple_tts_") as temp_dir:
            output = Path(temp_dir) / "saved_20260712_094331.wav"
            metrics = simple_tts.synthesize_and_save(
                text="Real save path test.",
                output_path=output,
                model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                speaker="Ryan",
                language="English",
                device="cpu",
                attention_implementation="sdpa",
                model_class=FakeModelClass,
                torch_module=fake_torch,
                soundfile_module=sf,
                timer=lambda: next(times),
            )

            audio_info = sf.info(output)
            self.assertEqual(audio_info.frames, 24_000)
            self.assertEqual(audio_info.samplerate, 24_000)
            self.assertAlmostEqual(audio_info.duration, 1.0)
            self.assertAlmostEqual(metrics["real_time_factor"], 0.5)


if __name__ == "__main__":
    unittest.main()
