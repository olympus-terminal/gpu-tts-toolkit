#!/usr/bin/env python3
"""
Batch TTS for Tripitaka Suttas — Qwen3-TTS engine.

Loads the model once, then processes all sutta text files in a loop.
Supports resume (skips existing outputs), collection filtering, and
VRAM monitoring to keep a 6 GB GPU stable across 1,200+ files.
"""

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from pydub import AudioSegment
from tqdm import tqdm

# deep_voice_tts_v3.py lives one directory up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deep_voice_tts_v3 import DeepVoiceTTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VRAM_WARN_GB = 5.5  # warn & force-clear above this threshold


def vram_status():
    """Return (allocated_gb, reserved_gb) or (0, 0) if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    resv = torch.cuda.memory_reserved() / 1024 ** 3
    return round(alloc, 2), round(resv, 2)


def force_vram_clear():
    """Aggressively reclaim GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def log(path, msg):
    """Append a timestamped line to the log file and print it."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Single-file TTS (inlined from process_text_file, minus per-file dirs)
# ---------------------------------------------------------------------------

def process_one(tts: DeepVoiceTTS, txt_path: Path, mp3_path: Path,
                log_path: Path) -> bool:
    """
    Read *txt_path*, run TTS, write *mp3_path*.
    Returns True on success, False on skip/error.
    """
    # Read source text
    try:
        text = txt_path.read_text(encoding="utf-8")
    except Exception as e:
        log(log_path, f"  READ ERROR {txt_path.name}: {e}")
        return False

    if len(text.strip()) < 10:
        log(log_path, f"  SKIP (too short) {txt_path.name}")
        return False

    # Preprocess & chunk
    text = tts.preprocess_text(text)
    chunks = tts.smart_chunk_text(text)
    if not chunks:
        log(log_path, f"  SKIP (no chunks) {txt_path.name}")
        return False

    # Temporary directory for intermediate WAVs (next to output)
    chunks_dir = mp3_path.parent / f".chunks_{mp3_path.stem}"
    chunks_dir.mkdir(exist_ok=True)

    # Generate each chunk
    audio_files = []
    for i, chunk in enumerate(chunks):
        chunk_wav = chunks_dir / f"chunk_{i:04d}.wav"
        if tts.generate_audio_chunk(chunk, str(chunk_wav)):
            audio_files.append(chunk_wav)

    if not audio_files:
        log(log_path, f"  FAIL (0 chunks generated) {txt_path.name}")
        _cleanup(chunks_dir)
        return False

    # Combine chunks with crossfade
    try:
        combined = AudioSegment.from_file(str(audio_files[0]))
        for af in audio_files[1:]:
            seg = AudioSegment.from_file(str(af))
            combined = combined.append(seg, crossfade=50)

        combined = combined.normalize()

        # Speed adjustment
        if tts.speed != 1.0:
            new_rate = int(combined.frame_rate * tts.speed)
            combined = combined._spawn(
                combined.raw_data,
                overrides={"frame_rate": new_rate},
            ).set_frame_rate(combined.frame_rate)

        combined.export(str(mp3_path), format="mp3", bitrate="192k")
    except Exception as e:
        log(log_path, f"  COMBINE ERROR {txt_path.name}: {e}")
        _cleanup(chunks_dir)
        return False

    # Hallucination detection & cleanup
    try:
        hallucinations = tts.detect_hallucinations(mp3_path)
        if hallucinations:
            tts.clean_hallucinations(mp3_path, hallucinations)
            log(log_path, f"  removed {len(hallucinations)} hallucination(s)")
    except Exception as e:
        log(log_path, f"  HALLUCINATION-CHECK WARN {txt_path.name}: {e}")

    # Remove intermediate files
    _cleanup(chunks_dir)
    return True


def _cleanup(chunks_dir: Path):
    """Remove the temporary chunks directory."""
    try:
        for f in chunks_dir.iterdir():
            f.unlink(missing_ok=True)
        chunks_dir.rmdir()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def collect_files(input_dir: Path, collections: list[str] | None):
    """
    Return sorted list of .txt paths in *input_dir*, optionally
    filtered to files whose name starts with one of *collections*
    (case-insensitive prefix match).
    """
    all_txt = sorted(input_dir.glob("*.txt"))
    if not collections:
        return all_txt

    prefixes = [c.lower() for c in collections]
    return [p for p in all_txt if any(p.name.lower().startswith(px) for px in prefixes)]


def main():
    parser = argparse.ArgumentParser(
        description="Batch TTS — convert all sutta texts to MP3 with Qwen3-TTS",
    )
    parser.add_argument("--input", default="tripitaka/suttas",
                        help="Directory containing .txt sutta files")
    parser.add_argument("--output", default="tripitaka/audio",
                        help="Directory for output .mp3 files")
    parser.add_argument("--collections", nargs="+", metavar="PREFIX",
                        help="Only process files starting with these prefixes (e.g. dn mn)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be processed, then exit")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0.5-2.0)")
    parser.add_argument("--voice", default="dylan",
                        help="Voice profile (default: dylan)")
    args = parser.parse_args()

    # Resolve paths relative to repo root (one level above this script)
    repo_root = Path(__file__).resolve().parent.parent
    input_dir = (repo_root / args.input).resolve()
    output_dir = (repo_root / args.output).resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    # Collect files to process
    txt_files = collect_files(input_dir, args.collections)
    if not txt_files:
        print("No matching .txt files found.")
        sys.exit(0)

    # Build expected output names and check for existing (resume support)
    todo = []
    skipped = 0
    for txt in txt_files:
        mp3_name = txt.stem + ".mp3"
        mp3_path = output_dir / mp3_name
        if mp3_path.exists():
            skipped += 1
        else:
            todo.append((txt, mp3_path))

    print(f"Files found:   {len(txt_files)}")
    print(f"Already done:  {skipped}")
    print(f"To process:    {len(todo)}")

    if args.dry_run:
        print("\n-- Dry run: files that would be processed --")
        for txt, mp3 in todo:
            print(f"  {txt.name}  ->  {mp3.name}")
        return

    if not todo:
        print("Nothing to do — all files already have audio.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "batch_log.txt"
    log(log_path, f"=== Batch TTS started: {len(todo)} files ===")
    log(log_path, f"Voice: {args.voice}, Speed: {args.speed}")

    # -----------------------------------------------------------------------
    # Load model ONCE
    # -----------------------------------------------------------------------
    log(log_path, "Loading Qwen3-TTS model...")
    tts = DeepVoiceTTS(
        voice_profile=args.voice,
        output_format="mp3",
        speed=args.speed,
    )
    alloc, resv = vram_status()
    log(log_path, f"Model loaded. VRAM: {alloc} GB alloc, {resv} GB reserved")

    # -----------------------------------------------------------------------
    # Process loop
    # -----------------------------------------------------------------------
    succeeded = 0
    failed = 0
    failed_files = []
    start_time = time.time()

    for idx, (txt, mp3) in enumerate(tqdm(todo, desc="Batch TTS", unit="file")):
        file_start = time.time()
        log(log_path, f"[{idx + 1}/{len(todo)}] {txt.name}")

        ok = process_one(tts, txt, mp3, log_path)

        if ok:
            succeeded += 1
            elapsed = time.time() - file_start
            size_mb = mp3.stat().st_size / (1024 * 1024)
            log(log_path, f"  OK  {mp3.name}  ({size_mb:.1f} MB, {elapsed:.0f}s)")
        else:
            failed += 1
            failed_files.append(txt.name)

        # VRAM housekeeping
        force_vram_clear()
        alloc, resv = vram_status()

        if resv > VRAM_WARN_GB:
            log(log_path, f"  VRAM WARNING: {resv} GB reserved (>{VRAM_WARN_GB}). Forcing extra clear.")
            force_vram_clear()
            alloc, resv = vram_status()
            log(log_path, f"  After clear: {alloc} GB alloc, {resv} GB reserved")

        # Log VRAM every 25 files
        if (idx + 1) % 25 == 0:
            log(log_path, f"  -- VRAM checkpoint: {alloc} GB alloc, {resv} GB reserved --")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - start_time
    hours = total_time / 3600
    log(log_path, "")
    log(log_path, f"=== Batch complete ===")
    log(log_path, f"Succeeded: {succeeded}")
    log(log_path, f"Failed:    {failed}")
    log(log_path, f"Time:      {hours:.1f} hours ({total_time:.0f}s)")
    if failed_files:
        log(log_path, f"Failed files:")
        for name in failed_files:
            log(log_path, f"  - {name}")


if __name__ == "__main__":
    main()
