#!/usr/bin/env python3
"""
Paper to Audio - Unified CLI for converting academic papers to audio.

Orchestrates the full pipeline:
1. paper_search.py - Search for papers by query
2. paper_download.py - Download PDFs from open access sources
3. pdf_to_text.py - Extract and clean text for TTS
4. deep_voice_tts.py - Generate audio from text

Usage:
    python paper_to_audio.py 'biomimetic concrete' --papers 3
    python paper_to_audio.py 'bioremediation' --papers 1 --voice p240
    python paper_to_audio.py 'CRISPR gene editing' --papers 2 --keep-pdfs --keep-text
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


class PaperToAudioPipeline:
    """Orchestrate paper search, download, text extraction, and TTS generation."""

    def __init__(self, output_dir: str = "output", voice: str = "random",
                 keep_pdfs: bool = False, keep_text: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.voice = voice
        self.keep_pdfs = keep_pdfs
        self.keep_text = keep_text

        # Intermediate directories
        self.downloads_dir = self.output_dir / "downloads"
        self.texts_dir = self.output_dir / "texts"
        self.papers_json = self.output_dir / "papers.json"

        # Get the directory where this script is located
        self.script_dir = Path(__file__).parent
        self.tts_script = self.script_dir.parent / "deep_voice_tts.py"

    def run_script(self, script_name: str, args: list, description: str) -> bool:
        """Run a Python script as a subprocess."""
        script_path = self.script_dir / script_name

        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}", file=sys.stderr)
            return False

        cmd = [sys.executable, str(script_path)] + args

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"STAGE: {description}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Let output flow to stderr
                text=True
            )

            if result.returncode != 0:
                print(f"[ERROR] {script_name} failed with return code {result.returncode}",
                      file=sys.stderr)
                return False

            return True

        except Exception as e:
            print(f"[ERROR] Failed to run {script_name}: {e}", file=sys.stderr)
            return False

    def run_tts(self, text_file: Path) -> bool:
        """Run the TTS script on a text file."""
        if not self.tts_script.exists():
            print(f"[ERROR] TTS script not found: {self.tts_script}", file=sys.stderr)
            return False

        # Use absolute paths to avoid issues with cwd changes
        abs_text_file = text_file.resolve()
        abs_output_dir = self.output_dir.resolve()

        cmd = [
            sys.executable, str(self.tts_script),
            str(abs_text_file),
            "--voice", self.voice,
            "--format", "mp3"
        ]

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"STAGE: Generating audio for {text_file.name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

        try:
            # Run from the output directory so audio files are created there
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                cwd=str(abs_output_dir)
            )

            if result.returncode != 0:
                print(f"[ERROR] TTS failed for {text_file.name}", file=sys.stderr)
                return False

            return True

        except Exception as e:
            print(f"[ERROR] Failed to run TTS: {e}", file=sys.stderr)
            return False

    def search_papers(self, query: str, num_papers: int) -> bool:
        """Stage 1: Search for papers."""
        return self.run_script(
            "paper_search.py",
            [query, "--papers", str(num_papers), "--output", str(self.papers_json)],
            f"Searching for papers: '{query}'"
        )

    def download_papers(self) -> bool:
        """Stage 2: Download PDFs from open access sources."""
        return self.run_script(
            "paper_download.py",
            [str(self.papers_json), "--output", str(self.downloads_dir)],
            "Downloading PDFs"
        )

    def extract_text(self) -> bool:
        """Stage 3: Extract and clean text from PDFs."""
        return self.run_script(
            "pdf_to_text.py",
            [str(self.downloads_dir), str(self.texts_dir)],
            "Extracting text from PDFs"
        )

    def generate_audio(self) -> bool:
        """Stage 4: Generate audio from text files."""
        text_files = list(self.texts_dir.glob("*.txt"))

        if not text_files:
            print("[WARNING] No text files found for TTS generation", file=sys.stderr)
            return False

        print(f"\n[INFO] Found {len(text_files)} text files for TTS", file=sys.stderr)

        success_count = 0
        for text_file in text_files:
            if self.run_tts(text_file):
                success_count += 1

        print(f"\n[INFO] Generated audio for {success_count}/{len(text_files)} files",
              file=sys.stderr)

        return success_count > 0

    def cleanup(self):
        """Remove intermediate files unless keep flags are set."""
        print(f"\n{'='*60}", file=sys.stderr)
        print("CLEANUP", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        if not self.keep_pdfs and self.downloads_dir.exists():
            print(f"[CLEANUP] Removing PDFs: {self.downloads_dir}", file=sys.stderr)
            shutil.rmtree(self.downloads_dir)
        elif self.keep_pdfs:
            print(f"[KEEP] PDFs preserved: {self.downloads_dir}", file=sys.stderr)

        if not self.keep_text and self.texts_dir.exists():
            print(f"[CLEANUP] Removing text files: {self.texts_dir}", file=sys.stderr)
            shutil.rmtree(self.texts_dir)
        elif self.keep_text:
            print(f"[KEEP] Text files preserved: {self.texts_dir}", file=sys.stderr)

        # Always clean up papers.json unless keeping text
        if not self.keep_text and self.papers_json.exists():
            print(f"[CLEANUP] Removing {self.papers_json}", file=sys.stderr)
            self.papers_json.unlink()

    def run(self, query: str, num_papers: int) -> bool:
        """Run the full pipeline."""
        print(f"\n{'='*60}", file=sys.stderr)
        print("PAPER TO AUDIO PIPELINE", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Query: {query}", file=sys.stderr)
        print(f"Papers: {num_papers}", file=sys.stderr)
        print(f"Voice: {self.voice}", file=sys.stderr)
        print(f"Output: {self.output_dir}", file=sys.stderr)
        print(f"Keep PDFs: {self.keep_pdfs}", file=sys.stderr)
        print(f"Keep text: {self.keep_text}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Stage 1: Search
        if not self.search_papers(query, num_papers):
            print("[FAILED] Paper search failed", file=sys.stderr)
            return False

        # Verify papers were found
        if not self.papers_json.exists():
            print("[FAILED] No papers.json created", file=sys.stderr)
            return False

        with open(self.papers_json) as f:
            papers = json.load(f)

        if not papers:
            print("[FAILED] No papers found for query", file=sys.stderr)
            return False

        print(f"\n[SUCCESS] Found {len(papers)} papers", file=sys.stderr)

        # Stage 2: Download
        if not self.download_papers():
            print("[FAILED] Paper download failed", file=sys.stderr)
            return False

        # Verify PDFs were downloaded
        pdf_files = list(self.downloads_dir.glob("*.pdf"))
        if not pdf_files:
            print("[FAILED] No PDFs downloaded (papers may not be open access)",
                  file=sys.stderr)
            return False

        print(f"\n[SUCCESS] Downloaded {len(pdf_files)} PDFs", file=sys.stderr)

        # Stage 3: Extract text
        if not self.extract_text():
            print("[FAILED] Text extraction failed", file=sys.stderr)
            return False

        # Verify text files were created
        text_files = list(self.texts_dir.glob("*.txt"))
        if not text_files:
            print("[FAILED] No text files created", file=sys.stderr)
            return False

        print(f"\n[SUCCESS] Extracted text from {len(text_files)} PDFs", file=sys.stderr)

        # Stage 4: Generate audio
        if not self.generate_audio():
            print("[FAILED] Audio generation failed", file=sys.stderr)
            return False

        # Stage 5: Cleanup
        self.cleanup()

        # Final summary
        print(f"\n{'='*60}", file=sys.stderr)
        print("PIPELINE COMPLETE", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # List output files
        audio_dirs = [d for d in self.output_dir.iterdir()
                     if d.is_dir() and "deep_voice" in d.name]

        print(f"Audio output directories:", file=sys.stderr)
        for d in audio_dirs:
            mp3_files = list(d.glob("*_complete.mp3"))
            for mp3 in mp3_files:
                size_mb = mp3.stat().st_size / (1024 * 1024)
                print(f"  {mp3} ({size_mb:.1f} MB)", file=sys.stderr)

        print(f"{'='*60}", file=sys.stderr)

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert academic papers to audio via search, download, and TTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python paper_to_audio.py 'biomimetic concrete' --papers 2
    python paper_to_audio.py 'CRISPR' --papers 1 --voice p240
    python paper_to_audio.py 'bioremediation' --papers 3 --keep-pdfs --keep-text
        """
    )
    parser.add_argument('query', help='Search query for papers')
    parser.add_argument('--papers', '-n', type=int, default=3,
                        help='Number of papers to search for (default: 3)')
    parser.add_argument('--voice', '-v', type=str, default='random',
                        help='Voice profile: speaker ID (p230, p240, etc.) or "random" (default: random)')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--keep-pdfs', action='store_true',
                        help='Keep downloaded PDFs after audio generation')
    parser.add_argument('--keep-text', action='store_true',
                        help='Keep extracted text files after audio generation')

    args = parser.parse_args()

    pipeline = PaperToAudioPipeline(
        output_dir=args.output,
        voice=args.voice,
        keep_pdfs=args.keep_pdfs,
        keep_text=args.keep_text
    )

    success = pipeline.run(args.query, args.papers)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
