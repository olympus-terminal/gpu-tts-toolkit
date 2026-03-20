#!/usr/bin/env python3
"""
Deep Voice TTS v2 — accepts .pdf, .tex, and .txt inputs directly.

Subclasses DeepVoiceTTS and integrates the extraction logic from
extract_salient_text.py so a single command goes from manuscript to audio.

Usage:
    python deep_voice_tts_v2.py paper.txt --voice p240
    python deep_voice_tts_v2.py main.tex --voice p240
    python deep_voice_tts_v2.py main.pdf --voice p240
    python deep_voice_tts_v2.py main.tex --preview 500
    python deep_voice_tts_v2.py main.tex --voice p240 --include-sections "SUMMARY,RESULTS"
    python deep_voice_tts_v2.py main.tex --voice p240 --strip-heavy-stats
"""

import sys
from pathlib import Path

from extract_salient_text import (
    LaTeXSalientExtractor,
    PDFSalientExtractor,
    DEFAULT_EXCLUDE,
)

class DeepVoiceTTSv2:
    """Wraps DeepVoiceTTS with automatic .tex/.pdf extraction.

    Model loading is deferred until TTS is actually needed, so preview
    and list-voices work without a GPU.
    """

    def __init__(self, voice_profile="random", output_format="mp3", device=None):
        self._voice_profile = voice_profile
        self._output_format = output_format
        self._device = device
        self._pipeline = None  # lazy-loaded

    def _ensure_pipeline(self):
        """Load the heavy DeepVoiceTTS pipeline on first use."""
        if self._pipeline is None:
            from deep_voice_tts import DeepVoiceTTS
            self._pipeline = DeepVoiceTTS(
                voice_profile=self._voice_profile,
                output_format=self._output_format,
                device=self._device,
            )
        return self._pipeline

    def process_input(self, input_file, *, output_name=None,
                      include_sections=None, exclude_sections=None,
                      strip_heavy_stats=False, keep_figure_refs=False,
                      preview=None):
        """Auto-detect file type, extract if needed, then run TTS.

        For .txt files the extraction step is skipped entirely.
        If *preview* is set, prints the first N characters and returns
        without loading the TTS model.
        """
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: {input_path} not found", file=sys.stderr)
            sys.exit(1)

        suffix = input_path.suffix.lower()

        # --- .txt: pass straight through (extraction args ignored) ---
        if suffix == ".txt":
            if preview is not None:
                text = input_path.read_text(encoding="utf-8")
                self._show_preview(text, preview)
                return None
            pipeline = self._ensure_pipeline()
            return pipeline.process_text_file(str(input_path),
                                              output_name=output_name)

        # --- .tex / .pdf: extract first ---
        if suffix not in (".tex", ".pdf"):
            print(f"ERROR: Unsupported file type: {suffix} "
                  f"(expected .tex, .pdf, or .txt)", file=sys.stderr)
            sys.exit(1)

        # Build section filter lists
        include_all = False
        include = None
        if include_sections:
            if include_sections.upper() == "ALL":
                include_all = True
            else:
                include = [s.strip().upper()
                           for s in include_sections.split(",")]

        exclude = None
        if exclude_sections:
            extra = [s.strip().upper()
                     for s in exclude_sections.split(",")]
            exclude = DEFAULT_EXCLUDE + extra

        if suffix == ".tex":
            print(f"[LaTeX mode] {input_path}", file=sys.stderr)
            tex = input_path.read_text(encoding="utf-8")
            extractor = LaTeXSalientExtractor(
                strip_heavy_stats=strip_heavy_stats,
                keep_figure_refs=keep_figure_refs,
            )
            result, stats = extractor.extract(
                tex, include=include, exclude=exclude,
                include_all=include_all,
            )
        else:  # .pdf
            print(f"[PDF mode] {input_path}", file=sys.stderr)
            extractor = PDFSalientExtractor(
                strip_heavy_stats=strip_heavy_stats,
                keep_figure_refs=keep_figure_refs,
            )
            result, stats = extractor.extract(
                input_path, include=include, exclude=exclude,
                include_all=include_all,
            )

        self._report_stats(stats)

        if preview is not None:
            self._show_preview(result, preview)
            return None

        # Write extracted text to a .tts.txt file in cwd, then run TTS
        tts_txt = Path.cwd() / (input_path.stem + ".tts.txt")
        tts_txt.write_text(result, encoding="utf-8")
        print(f"Extracted text written to: {tts_txt}", file=sys.stderr)

        pipeline = self._ensure_pipeline()
        return pipeline.process_text_file(str(tts_txt),
                                          output_name=output_name or input_path.stem)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _report_stats(stats):
        sep = "=" * 60
        print(f"\n{sep}", file=sys.stderr)
        print("EXTRACTION STATS", file=sys.stderr)
        print(sep, file=sys.stderr)
        print(f"Sections found:    {stats['total_sections']}", file=sys.stderr)
        print(f"Sections kept:     {stats['kept_sections']}", file=sys.stderr)
        print(f"Sections excluded: {stats['excluded_sections']}", file=sys.stderr)
        print(f"Output characters: {stats['output_chars']}", file=sys.stderr)
        print(f"Output words:      ~{stats['output_words']}", file=sys.stderr)
        print(sep, file=sys.stderr)
        print(f"Kept: {', '.join(stats['kept_names'])}", file=sys.stderr)
        excluded_names = [n for n in stats['section_names']
                          if n not in stats['kept_names']]
        if excluded_names:
            print(f"Excluded: {', '.join(excluded_names)}", file=sys.stderr)
        print(sep, file=sys.stderr)

    @staticmethod
    def _show_preview(text, n):
        preview = text[:n]
        if len(text) > n:
            preview += f"\n\n... [{len(text) - n} more characters]"
        print(preview)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Voice TTS v2 — PDF/LaTeX/TXT to audio in one step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s paper.txt --voice p240           # plain text (same as v1)
  %(prog)s main.tex --voice p240            # LaTeX → extract → audio
  %(prog)s main.pdf --voice p240            # PDF   → extract → audio
  %(prog)s main.tex --preview 500           # preview extraction (no GPU)
  %(prog)s main.tex --voice p240 --strip-heavy-stats
  %(prog)s --list-voices                    # show available voices
""",
    )

    # Positional
    parser.add_argument("input_file", nargs="?", help="Input .txt, .tex, or .pdf file")

    # Original TTS args
    parser.add_argument("--voice", default="random",
                        help="Voice: speaker ID (p230, p234, …), 'random', or preset name")
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3",
                        help="Output audio format")
    parser.add_argument("--output-name", help="Custom output name")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Force device")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")

    # Extraction args (only meaningful for .tex/.pdf)
    parser.add_argument("--include-sections",
                        help="Comma-separated sections to include (or ALL)")
    parser.add_argument("--exclude-sections",
                        help="Comma-separated additional sections to exclude")
    parser.add_argument("--strip-heavy-stats", action="store_true",
                        help="Remove parenthesized statistical details")
    parser.add_argument("--keep-figure-refs", action="store_true",
                        help="Keep figure/table/data references in text")

    # Preview mode
    parser.add_argument("--preview", type=int, metavar="N",
                        help="Print first N extracted characters and exit (no GPU needed)")

    args = parser.parse_args()

    if args.list_voices:
        print("Favorite deep male voices:")
        for v in ["p230", "p234", "p240", "p244", "p246", "p248", "p250"]:
            print(f"  {v}")
        print("\nPresets: deep_male, caribbean_male, bass_male, sam_male, xtts_custom")
        print("\nUsage:")
        print("  python deep_voice_tts_v2.py input.tex --voice p230")
        print("  python deep_voice_tts_v2.py input.pdf --voice random")
        return

    if not args.input_file:
        parser.error("input_file is required (unless --list-voices is used)")

    v2 = DeepVoiceTTSv2(
        voice_profile=args.voice,
        output_format=args.format,
        device=args.device,
    )

    v2.process_input(
        args.input_file,
        output_name=args.output_name,
        include_sections=args.include_sections,
        exclude_sections=args.exclude_sections,
        strip_heavy_stats=args.strip_heavy_stats,
        keep_figure_refs=args.keep_figure_refs,
        preview=args.preview,
    )

if __name__ == "__main__":
    main()
