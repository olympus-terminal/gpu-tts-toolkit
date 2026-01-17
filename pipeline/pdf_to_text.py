#!/usr/bin/env python3
"""
PDF to Text - Extract and clean PDF text for TTS readability.

Uses PyMuPDF (fitz) to extract text from PDFs, then applies various cleaning
rules to make the text more suitable for text-to-speech processing.

Usage:
    python pdf_to_text.py downloads/ texts/
    python pdf_to_text.py paper.pdf output.txt

Output:
    Text files cleaned for TTS (one per input PDF)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from tqdm import tqdm

class PDFTextExtractor:
    """Extract and clean PDF text for TTS readability."""

    def __init__(self):
        # Regex patterns for cleaning
        self.patterns = {
            # Numeric citations: [1], [1,2], [1-3], [1, 2, 3], [1-3, 5], etc.
            'numeric_citations': re.compile(r'\[\d+(?:\s*[-–,]\s*\d+)*\]'),

            # Author-year citations: (Smith 2020), (Smith et al., 2020), (Smith & Jones 2020)
            'author_year_citations': re.compile(
                r'\([A-Z][a-zA-Z\-\']+(?:\s+(?:et\s+al\.?|&|and)\s*)?(?:,?\s*\d{4}[a-z]?)\)',
                re.IGNORECASE
            ),

            # More complex author-year: (Smith, 2020; Jones, 2021)
            'multi_author_citations': re.compile(
                r'\([A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?'
                r'(?:\s*;\s*[A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?)*\)'
            ),

            # URLs (http, https, ftp)
            'urls': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),

            # DOIs: 10.xxxx/... or doi:10.xxxx/...
            'dois': re.compile(r'(?:doi:\s*)?10\.\d{4,}/[^\s]+', re.IGNORECASE),

            # Figure references: Figure 1, Fig. 1, Figures 1-3, Fig 1a
            'figure_refs': re.compile(
                r'\b(?:Figure|Fig\.?)\s*\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?',
                re.IGNORECASE
            ),

            # Table references: Table 1, Tables 1-3, Table S1
            'table_refs': re.compile(
                r'\bTables?\s*[S]?\d+[a-z]?(?:\s*[-–]\s*[S]?\d+[a-z]?)?',
                re.IGNORECASE
            ),

            # Supplementary references: Supplementary Fig/Table/Material
            'supplementary_refs': re.compile(
                r'\bSupplementary\s+(?:Figure|Fig\.?|Table|Material|Information|Data)s?\s*[S]?\d*',
                re.IGNORECASE
            ),

            # Page numbers (standalone numbers at line start/end, common patterns)
            'page_numbers': re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE),

            # Email addresses
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),

            # Headers/footers with page info: "Page 1 of 10", "1/10", etc.
            'page_info': re.compile(r'\b(?:Page\s+)?\d+\s*(?:of|/)\s*\d+\b', re.IGNORECASE),

            # Copyright notices
            'copyright': re.compile(
                r'(?:copyright|\u00a9|©)\s*(?:\d{4})?.*?(?:all rights reserved|license)?',
                re.IGNORECASE
            ),

            # Journal article metadata patterns (often in headers)
            'journal_header': re.compile(
                r'^.*(?:journal of|proceedings|volume|vol\.|issue|pp\.).*$',
                re.IGNORECASE | re.MULTILINE
            ),

            # Equation references: Eq. 1, Equation (1), Eqs. 1-3
            'equation_refs': re.compile(
                r'\b(?:Equation|Eq\.?)\s*\(?[\d,\s\-–]+\)?',
                re.IGNORECASE
            ),

            # Section numbers: 1.2.3, 2.1, etc. at line start
            'section_numbers': re.compile(r'^(\d+\.)+\d*\s*', re.MULTILINE),

            # Superscript/subscript notation artifacts: H2O, CO2, etc. handled naturally
            # but things like ^1, _2 from bad PDF extraction
            'sub_super_artifacts': re.compile(r'[\^_]\d+'),

            # Multiple spaces
            'multiple_spaces': re.compile(r' {2,}'),

            # Multiple newlines (more than 2)
            'multiple_newlines': re.compile(r'\n{3,}'),

            # Hyphenation at line breaks (re-join words)
            'hyphenation': re.compile(r'(\w+)-\n(\w+)'),

            # Bullet points and list markers
            'bullets': re.compile(r'^[\u2022\u2023\u25E6\u2043\u2219•●○◦]\s*', re.MULTILINE),
        }

    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        text_parts = []

        for page_num, page in enumerate(doc):
            # Get text, preserving some layout
            text = page.get_text("text")
            text_parts.append(text)

        doc.close()
        return "\n".join(text_parts)

    def clean_for_tts(self, text: str) -> str:
        """Apply all cleaning rules to make text TTS-friendly."""
        # Fix hyphenation at line breaks first
        text = self.patterns['hyphenation'].sub(r'\1\2', text)

        # Remove citations
        text = self.patterns['numeric_citations'].sub('', text)
        text = self.patterns['multi_author_citations'].sub('', text)
        text = self.patterns['author_year_citations'].sub('', text)

        # Remove URLs and DOIs
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['dois'].sub('', text)
        text = self.patterns['emails'].sub('', text)

        # Remove figure/table/equation references
        text = self.patterns['figure_refs'].sub('', text)
        text = self.patterns['table_refs'].sub('', text)
        text = self.patterns['supplementary_refs'].sub('', text)
        text = self.patterns['equation_refs'].sub('', text)

        # Remove page numbers and headers
        text = self.patterns['page_numbers'].sub('', text)
        text = self.patterns['page_info'].sub('', text)

        # Remove copyright and journal headers
        text = self.patterns['copyright'].sub('', text)
        text = self.patterns['journal_header'].sub('', text)

        # Clean up artifacts
        text = self.patterns['sub_super_artifacts'].sub('', text)
        text = self.patterns['bullets'].sub('', text)

        # Normalize whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        text = self.patterns['multiple_newlines'].sub('\n\n', text)

        # Clean up lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines (likely artifacts)
            if len(line) < 3:
                continue
            # Skip lines that are just numbers
            if line.isdigit():
                continue
            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        # Final cleanup: remove orphaned punctuation
        text = re.sub(r'\s+([,;:])', r'\1', text)  # Fix space before punctuation
        text = re.sub(r'([,;:])\s*([,;:])', r'\1', text)  # Remove double punctuation
        text = re.sub(r'\(\s*\)', '', text)  # Remove empty parentheses
        text = re.sub(r'\[\s*\]', '', text)  # Remove empty brackets
        text = re.sub(r'\s+\.', '.', text)  # Fix space before period
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods to single

        return text.strip()

    def process_file(self, pdf_path: Path, output_path: Path) -> bool:
        """Process a single PDF file and save cleaned text."""
        try:
            print(f"[PROCESSING] {pdf_path.name}", file=sys.stderr)

            # Extract text
            raw_text = self.extract_text(pdf_path)
            if not raw_text.strip():
                print(f"  [WARNING] No text extracted from {pdf_path.name}", file=sys.stderr)
                return False

            print(f"  [EXTRACTED] {len(raw_text)} characters", file=sys.stderr)

            # Clean text
            cleaned_text = self.clean_for_tts(raw_text)
            print(f"  [CLEANED] {len(cleaned_text)} characters", file=sys.stderr)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            print(f"  [SAVED] {output_path}", file=sys.stderr)
            return True

        except Exception as e:
            print(f"  [ERROR] Failed to process {pdf_path.name}: {e}", file=sys.stderr)
            return False

    def process_directory(self, input_dir: Path, output_dir: Path) -> List[dict]:
        """Process all PDFs in a directory."""
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"[WARNING] No PDF files found in {input_dir}", file=sys.stderr)
            return []

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"PDF TO TEXT CONVERTER", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Input directory: {input_dir}", file=sys.stderr)
        print(f"Output directory: {output_dir}", file=sys.stderr)
        print(f"PDFs found: {len(pdf_files)}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        results = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs", file=sys.stderr):
            output_path = output_dir / f"{pdf_path.stem}.txt"
            success = self.process_file(pdf_path, output_path)
            results.append({
                'input': str(pdf_path),
                'output': str(output_path) if success else None,
                'success': success
            })

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"SUMMARY", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Total PDFs: {len(pdf_files)}", file=sys.stderr)
        print(f"Successfully converted: {successful}", file=sys.stderr)
        print(f"Failed: {len(pdf_files) - successful}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        return results

def main():
    parser = argparse.ArgumentParser(
        description='Extract and clean PDF text for TTS'
    )
    parser.add_argument('input', help='Input PDF file or directory')
    parser.add_argument('output', help='Output text file or directory')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    extractor = PDFTextExtractor()

    if input_path.is_file():
        # Single file mode
        if not input_path.suffix.lower() == '.pdf':
            print(f"[ERROR] Input file must be a PDF: {input_path}", file=sys.stderr)
            sys.exit(1)

        # If output is a directory, create filename
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"{input_path.stem}.txt"

        success = extractor.process_file(input_path, output_path)
        sys.exit(0 if success else 1)

    elif input_path.is_dir():
        # Directory mode
        results = extractor.process_directory(input_path, output_path)

        if not results:
            sys.exit(1)

        successful = sum(1 for r in results if r['success'])
        sys.exit(0 if successful > 0 else 1)

    else:
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
