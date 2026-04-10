#!/usr/bin/env python3
"""
Media-to-TTS Pipeline  (flagship tool of gpu-tts-toolkit)
=========================================================
Turn any text-bearing file — PDF, LaTeX source, Markdown, or plain text —
into a TTS-ready .txt and, optionally, a narrated audio file with
automatic QC / hallucination screening.

Scope:
  INPUT   : .pdf  .tex  .md  .markdown  .txt  .log
            (mixed bundles supported via --multi)
  EXTRACT : format-specific cleaner dispatched by file extension
  SYNTH   : deep_voice_tts.py (Coqui VCTK VITS, GPU when available)
  QC      : 10-category pre-synthesis pattern screen + post-synthesis
            audio screening, with a JSON report per run

Cleaners (each exposes a clean() method returning TTS-ready text):
  - PDFTTSCleaner        PDFs via pdftotext -layout, then boilerplate strip
  - LaTeXTTSCleaner      .tex sources (citations, math, \\input, macros)
  - MarkdownTTSCleaner   .md / .markdown (headers, emphasis, tables)
  - PlainTextTTSCleaner  .txt / .log (bullets, rules, review headers)
  - clean_multi_files()  dispatcher for multi-file concatenation

Additional features:
  - Greek letter and scientific symbol expansion (_expand_for_tts)
  - Section-aware filtering (keeps prose, drops references, key resources)
  - TTS hallucination screening (text + audio)
  - deep_voice_tts.py integration with output analysis
  - Automatic cleanup of intermediate chunks / temp .txt

Usage:
    python media_to_tts.py paper.pdf [-o output.txt]
    python media_to_tts.py main.tex [-o output.txt]
    python media_to_tts.py main.tex --voice p246           # extract + synthesize
    python media_to_tts.py main.tex --voice p246 --screen  # + hallucination screening
    python media_to_tts.py --multi file1.txt file2.md file3.tex --voice p246
"""

import argparse
import json
import os
import re
import shutil
import string
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Prose detection helpers
# ---------------------------------------------------------------------------
_COMMON = set(
    "the a an in of to and is are was were for on with that this from by at "
    "or as but not be have has had been its it we our their which can may also "
    "than more between both into through across after before during while all "
    "each such under over most when where how these those".split()
)


def _prose_score(text: str) -> float:
    """Return 0-1 score of how 'prose-like' a paragraph is."""
    words = text.split()
    if not words:
        return 0.0
    common = sum(1 for w in words if w.lower().strip(string.punctuation) in _COMMON)
    alpha  = sum(1 for w in words if re.match(r'^[a-zA-Z\-\']+[.,;:!?]?$', w))
    return 0.4 * common / len(words) + 0.6 * alpha / len(words)


def _is_figure_noise(s: str) -> bool:
    """Detect figure axis labels, panel markers, and legend fragments."""
    # Very short numeric-heavy lines
    if len(s) < 30 and re.match(r'^[\d\s,.\-+°%()×]+$', s):
        return True
    # Standalone panel combos: "D Form I E Form I"
    if re.match(r'^([A-Z]\s+Form\s+I{1,2}\s*)+', s):
        return True
    # Axis label strings
    if re.match(r'^[°\w/³²]+(\s+[°\w/³²]+){1,5}$', s) and len(s) < 60:
        return True
    # Taxon + sample count from figure panels
    if re.match(r'^[A-Z][a-z]+(?:phyceae|ales|aceae|phyta|ceae|iniaceae|ida)\s*\(n=\d+\)$', s):
        return True
    # Basin abbreviation bars
    if re.match(r'^Atl\.\s*Pac\.\s*Med\.', s):
        return True
    # Standalone dim-reduction labels
    if s in ('UMAP', 't-SNE', 'RuBisCO Sequences', 'RuBisCO Form'):
        return True
    # Axis-spec fragments like "Temp Chl-a Depth"
    if re.match(r'^(Temp|Chl|Depth|SST|Count|Samples|Density)', s) and len(s) < 80:
        if not any(c in s for c in '.;:'):
            return True
    # "Dataset Composition by Basin ..." or "Data Sources ..." figure internals
    if re.match(r'^(Dataset Composition|Data Sources\s+[A-Z])', s):
        return True
    # "RuBisCO Form Form I (chloroplast)" style legend lines (with mixed taxa + counts)
    if 'RuBisCO Form' in s and 'Form I' in s and len(s) < 300:
        return True
    # Multi-taxon figure panel listings: "Prorocentrales (n=1128) Chromerida (n=398) ..."
    if re.findall(r'\(n=\d+\)', s) and len(s) < 300:
        taxa_count = len(re.findall(r'[A-Z][a-z]+(?:ales|ceae|phyta|ida|iniaceae)\s*\(n=\d+\)', s))
        if taxa_count >= 2:
            return True
    # Tick-label lines: "5 25 100 500 2k 5k Depth (m)"
    if re.match(r'^[\d\s.kKMm%()\-]+\s*(Depth|m|°C|mg|Count|Samples)', s):
        return True
    # Dominated by numbers (>60 % numeric tokens)
    words = s.split()
    if len(words) >= 2:
        nfrac = sum(1 for w in words if re.match(r'^[\d,.\-+−]+$', w)) / len(words)
        if nfrac > 0.6 and len(s) < 120:
            return True
    return False


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------
class PDFTTSCleaner:
    """Extract and clean a PDF document (papers, manuscripts, reports) into TTS-ready text."""

    # Sections to DROP entirely (case-insensitive start-of-paragraph match)
    _DROP_SECTION_STARTS = [
        r'^References\s+1\.',
        r'^SUPPLEMENTAL INFORMATION',
        r'^Key resources table',
        r'^REAGENT or RE',
        r'^Data and code availability',
    ]
    # Bullet points to drop (data availability items)
    _DROP_BULLET = re.compile(
        r'^•\s+(Domain|algaGPT|Pfam|RuBisCO|AlphaEarth|KAN|Novel|DIAMOND|'
        r'All original|Source meta|Any additional)'
    )

    def __init__(self):
        self.stats: dict = {}

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------
    @staticmethod
    def extract_pdf(pdf_path: str) -> str:
        """Run pdftotext -layout and return raw text."""
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True, text=True, check=True,
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Low-level text normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_ligatures(text: str) -> str:
        for old, new in [('ﬁ', 'fi'), ('ﬂ', 'fl'), ('ﬀ', 'ff'), ('ﬃ', 'ffi')]:
            text = text.replace(old, new)
        return text

    @staticmethod
    def _fix_hyphens(text: str) -> str:
        """Rejoin words broken across lines by pdftotext."""
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text

    @staticmethod
    def _strip_line_numbers(text: str) -> str:
        """Strip leading manuscript line numbers (e.g. '123   text')."""
        lines = text.split('\n')
        out = []
        for line in lines:
            s = line.strip()
            if not s:
                out.append('')
                continue
            # Standalone page numbers (1-50)
            if re.match(r'^\d{1,2}$', s):
                continue
            # Leading line numbers
            s = re.sub(r'^\d{1,4}\s{2,}', '', s)
            if s.strip():
                out.append(s)
            else:
                out.append('')
        return '\n'.join(out)

    # ------------------------------------------------------------------
    # Reflow into paragraphs
    # ------------------------------------------------------------------
    @staticmethod
    def _reflow(text: str) -> list[str]:
        """Collapse lines into paragraph strings."""
        text = re.sub(r'\n\s*[A-F]\s*\n', '\n', text)   # panel labels
        text = re.sub(r'\n{3,}', '\n\n', text)
        paras = []
        for block in text.split('\n\n'):
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            if lines:
                joined = ' '.join(lines)
                joined = re.sub(r'  +', ' ', joined)
                paras.append(joined)
        return paras

    # ------------------------------------------------------------------
    # Section-level filtering
    # ------------------------------------------------------------------
    def _filter_sections(self, paragraphs: list[str]) -> list[str]:
        """Keep only prose sections; drop refs, tables, figure noise, etc."""
        keep: list[str] = []
        skip_mode: str | None = None   # 'references', 'resources', 'supplemental', 'data_avail'

        for para in paragraphs:
            s = para.strip()

            # --- toggle skip modes ---
            if re.match(r'^References\s+1\.', s):
                skip_mode = 'references'
                continue
            if s.startswith('STAR METHODS'):
                skip_mode = None
            if s.startswith('Key resources table') or s.startswith('REAGENT or RE'):
                skip_mode = 'resources'
                continue
            if re.match(r'^(Experimental model|Method details)', s):
                if skip_mode == 'resources':
                    skip_mode = None
            if s.startswith('SUPPLEMENTAL INFORMATION'):
                skip_mode = 'supplemental'
                continue
            if skip_mode == 'supplemental' and re.match(r'^(References|STAR METHODS)', s):
                skip_mode = 'references' if s.startswith('References') else None
                if skip_mode:
                    continue
            if s.startswith('Data and code availability'):
                skip_mode = 'data_avail'
                continue
            if skip_mode == 'data_avail':
                if re.match(r'^•\s+', s):
                    continue
                else:
                    skip_mode = None

            if skip_mode:
                continue

            # --- graphical abstract internals ---
            if s.startswith('GRAPHICAL ABSTRACT'):
                keep.append('GRAPHICAL ABSTRACT')
                continue
            if re.match(r'^(EXTRACT|COUPLE|DISCOVER)\b', s):
                continue

            # --- figure noise ---
            if _is_figure_noise(s):
                continue

            # --- strip leading figure panel debris fused with caption ---
            # pdftotext sometimes merges axis labels / taxa lists with the
            # Figure N: caption on the same paragraph.  Strip the prefix.
            fig_cap = re.search(r'(Figure \d+:)', s)
            if fig_cap and fig_cap.start() > 30:
                # Only keep from "Figure N:" onward
                s = s[fig_cap.start():]
                para = s

            # --- prose score ---
            if s.isupper() and len(s) < 100 and len(s.split()) <= 12:
                keep.append(s)        # section header
                continue
            if re.match(r'^Figure \d+:', s):
                keep.append(s)        # figure caption
                continue
            if s.startswith('Graphical Abstract Legend:'):
                keep.append(s)
                continue
            if len(s) < 50:
                continue              # too short to be prose
            if _prose_score(s) < 0.35:
                continue

            # Heavy internal whitespace → table row
            if len(re.findall(r'\S\s{4,}\S', s)) >= 3:
                continue
            # Dominated by PFAM IDs
            pf = len(re.findall(r'PF\d{4,}', s))
            if pf > 5 and pf / max(len(s.split()), 1) > 0.15:
                continue

            keep.append(para)

        return keep

    # ------------------------------------------------------------------
    # TTS-friendly symbol / abbreviation expansion
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_for_tts(text: str) -> str:
        """Expand symbols, abbreviations, and notation for natural TTS."""

        # --- Latin abbreviations ---
        text = re.sub(r'\be\.g\.,\s*', 'for example, ', text)
        text = re.sub(r'\be\.g\.\s', 'for example ', text)
        text = re.sub(r'\bi\.e\.,\s*', 'that is, ', text)
        text = re.sub(r'\bi\.e\.\s', 'that is ', text)
        text = re.sub(r'\bvs\.\s', 'versus ', text)
        text = re.sub(r'\bet al\.', 'and colleagues', text)

        # --- Greek letters (common in scientific text) ---
        greek = {
            'ρ': 'rho', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma',
            'δ': 'delta', 'ε': 'epsilon', 'λ': 'lambda', 'σ': 'sigma',
            'τ': 'tau', 'χ': 'chi', 'Λ': 'Lambda', 'Σ': 'Sigma',
            'Δ': 'Delta', 'Ω': 'Omega',
        }
        for g, name in greek.items():
            text = text.replace(g, name)

        # --- Mathematical / scientific symbols ---
        text = text.replace('∼', 'approximately ')
        text = text.replace('≤', 'less than or equal to ')
        text = text.replace('≥', 'greater than or equal to ')
        text = text.replace('±', ' plus or minus ')
        text = text.replace('↔', ' and ')
        text = text.replace('→', ' to ')
        text = text.replace('−', '-')
        text = text.replace('×', ' times ')
        text = text.replace('R²', 'R-squared')

        # |rho| -> absolute rho (from plain text / markdown)
        text = re.sub(r'\|rho\|', 'absolute rho', text)

        # R2 as a standalone stat token (careful not to touch "CR2" etc.)
        text = re.sub(r'\bR2\s*=', 'R-squared =', text)
        text = re.sub(r'\bR2\s+(?=[a-z><=])', 'R-squared ', text)
        text = re.sub(r'\bR2\)', 'R-squared)', text)
        text = re.sub(r'\bR 2\b', 'R-squared', text)

        # LA4SR pronunciation (use compact form to survive citation stripping;
        # final expansion to spaced form happens at end of this method)
        text = text.replace('LA4 SR', 'LA4SR_PLACEHOLDER')
        text = text.replace('LA4SR', 'LA4SR_PLACEHOLDER')
        text = text.replace('L.A.4.S.R.', 'LA4SR_PLACEHOLDER')

        # p-values
        text = re.sub(r'\bp\s*<\s*', 'p less than ', text)

        # Scientific notation 10^-N  (only when preceded by word boundary)
        text = re.sub(r'\b10(-\d+)\b', lambda m: f'ten to the {m.group(1)}', text)

        # --- Inline citation removal (superscript style) ---
        # These are bare numbers that pdftotext extracts from superscript
        # citations.  ONLY match when the number follows a letter/punctuation
        # (never after a digit) to avoid mangling real numbers like "221.9".

        # ". 42 The"  or  "; 14,15 Environmental"
        text = re.sub(r'(?<=[a-zA-Z\)][.;]) (\d{1,3}(?:,\s*\d{1,3})*) (?=[A-Z])', ' ', text)
        # "basins. 4,8 However"
        text = re.sub(r'(?<=[a-zA-Z][.;]) (\d{1,3}(?:,\d{1,3})+) ', ' ', text)
        # "genome, 22 we"  or  "S.R., 1 a"
        text = re.sub(r'(?<=[a-zA-Z.],) (\d{1,2}) (?=[a-z])', ' ', text)
        # "approaches 9 face" — bare citation between two lowercase words
        text = re.sub(r'(?<=[a-z]) (\d{1,2}) (?=[a-z])', ' ', text)
        # Trailing citation numbers before common English function words
        text = re.sub(
            r'(?<=[a-zA-Z]) (\d{1,2}) (?=and\b|but\b|or\b|the\b|a\b|in\b|'
            r'with\b|from\b|for\b|by\b|to\b|have\b|has\b|was\b|were\b|are\b|'
            r'is\b|that\b|this\b|which\b|while\b)',
            ' ', text
        )
        # "embeddings 12,13 —provide" (citation + em-dash)
        text = re.sub(r' (\d{1,3}(?:,\d{1,3})*) (—)', r' \2', text)
        # "S.R., 1 our" or "hmmsearch 16 against" — number after word + space
        text = re.sub(r'(?<=[a-zA-Z]) (\d{1,2}) (?=[A-Z][a-z])', ' ', text)
        # "space: 3 environmental" — after colon + space + small number + space
        text = re.sub(r'(?<=[a-zA-Z]:) (\d{1,2}) (?=[a-z])', ' ', text)
        # "0.05, 25 342" — number after comma-space that precedes another number
        # (citation between stat value and count)
        text = re.sub(r'(?<=\d,) (\d{1,2}) (?=\d{3})', ' ', text)
        # "genome, 23 RuBisCO" — after comma + space + number + space + capitalized word
        text = re.sub(r'(?<=[a-zA-Z],) (\d{1,2}) (?=[A-Z])', ' ', text)
        # "depth 18 )" — citation before closing paren
        text = re.sub(r'(?<=[a-zA-Z]) (\d{1,2}) (?=\))', ' ', text)
        # "XGBoost 19 /SHAP 20 validation" — citations around slashes
        text = re.sub(r'(?<=[a-zA-Z]) (\d{1,2}) (?=/)', ' ', text)
        # "content 30 causes" — word + space + number + space + word
        text = re.sub(r'(?<=[a-z]) (\d{1,2}) (?=[a-z])', ' ', text)
        # "SHAP 20 validation" or "Pfam-A 5 characterizes" — after non-digit token
        text = re.sub(r'(?<=[A-Z]) (\d{1,2}) (?=[a-z])', ' ', text)
        # "Pfam-A 5 char" — after hyphen-letter
        text = re.sub(r'(?<=[a-zA-Z]) (\d{1,2}) (?=[a-z])', ' ', text)

        # --- Remove URLs, DOIs, RRIDs ---
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\bdoi:\s*\S+', '', text)
        text = re.sub(r'RRID:\S+', '', text)

        # --- Author-line cleanup ---
        text = re.sub(
            r'David Roy Nelson1,3,\*\s*,\s*Maxence Plouviez2\s*,\s*'
            r'and Kourosh Salehi-Ashtiani1',
            'David Roy Nelson, Maxence Plouviez, and Kourosh Salehi-Ashtiani',
            text,
        )
        text = re.sub(r'\b1\s+Division of Science', 'Division of Science', text)
        text = re.sub(r'\b2\s+Cawthron Institute', 'Cawthron Institute', text)
        text = re.sub(r'\b3\s+Lead contact', 'Lead contact', text)
        text = re.sub(r'\*\s*Correspondence:\s*\S+', '', text)

        # Expand LA4SR placeholder to final spoken form (after citation removal)
        text = text.replace('LA4SR_PLACEHOLDER', 'L.A. 4 S.R.')

        # Final whitespace cleanup
        text = re.sub(r'  +', ' ', text)
        return text

    # ------------------------------------------------------------------
    # Final paragraph-level pruning
    # ------------------------------------------------------------------
    @staticmethod
    def _prune_noise_paragraphs(paragraphs: list[str]) -> list[str]:
        out = []
        for p in paragraphs:
            s = p.strip()
            if not s:
                continue
            # CC1 / CC2 standalone noise from CCA figures
            if re.match(r'^CC\d', s) and len(s) < 80:
                continue
            # Short taxon-only lines that slipped through
            if re.match(r'^[A-Z][a-z]+(?:phyceae|ales|aceae|phyta)', s) and len(s) < 100:
                continue
            out.append(s)
        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def clean(self, pdf_path: str) -> str:
        """Full pipeline: PDF → TTS-ready text."""
        # 1. Extract
        raw = self.extract_pdf(pdf_path)

        # 2. Low-level normalisation
        text = self._strip_line_numbers(raw)
        text = self._fix_hyphens(text)
        text = self._fix_ligatures(text)

        # 3. Reflow into paragraphs
        paragraphs = self._reflow(text)

        # 4. Section-level filtering
        paragraphs = self._filter_sections(paragraphs)

        # 5. Reassemble and do TTS expansion
        text = '\n\n'.join(paragraphs)
        text = self._expand_for_tts(text)

        # 6. Final noise pruning
        final = self._prune_noise_paragraphs(text.split('\n\n'))
        text = '\n\n'.join(final)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        # Stats
        words = len(text.split())
        self.stats = {
            'paragraphs': len(final),
            'words': words,
            'est_duration_min': round(words / 150, 1),
        }
        return text


# ---------------------------------------------------------------------------
# LaTeX source cleaner
# ---------------------------------------------------------------------------
class LaTeXTTSCleaner:
    """Extract TTS-ready text directly from LaTeX source (.tex files).

    Advantages over PDF extraction:
      - No pdftotext column-merging artifacts
      - Explicit section/environment boundaries
      - Direct access to math macros for spoken expansion
      - Can reliably strip figures, tables, equations, and bibliography
    """

    # Environments to DROP entirely (content is not prose)
    _DROP_ENVS = {
        'figure', 'table', 'longtable', 'tabular', 'equation', 'align',
        'gather', 'multline', 'eqnarray', 'tikzpicture', 'lstlisting',
        'verbatim', 'thebibliography',
    }

    # Sections to DROP entirely (case-insensitive match)
    _DROP_SECTIONS = [
        'SUPPLEMENTAL INFORMATION',
        'ACKNOWLEDGMENTS',
        'AUTHOR CONTRIBUTIONS',
        'DECLARATION OF INTERESTS',
        'DECLARATION OF GENERATIVE AI',
        'Key resources table',
        'Resource availability',
        'Lead contact',
        'Materials availability',
        'Data and code availability',
        'Additional resources',
        'Quantification and statistical analysis',
    ]

    # Sections where we keep prose but watch for tables/noise
    _KEEP_SECTIONS = [
        'SUMMARY', 'HIGHLIGHTS', 'eTOC BLURB', 'GRAPHICAL ABSTRACT',
        'INTRODUCTION', 'RESULTS', 'DISCUSSION', 'STAR METHODS',
        'Method details', 'Experimental model',
    ]

    def __init__(self):
        self.stats: dict = {}
        self._warnings: list[str] = []

    # ------------------------------------------------------------------
    # LaTeX source reading
    # ------------------------------------------------------------------
    @staticmethod
    def read_tex(tex_path: str) -> str:
        """Read .tex file with encoding detection."""
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(tex_path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode {tex_path}")

    # ------------------------------------------------------------------
    # Environment stripping
    # ------------------------------------------------------------------
    def _strip_environments(self, text: str) -> str:
        """Remove entire environments that should not be spoken."""
        for env in self._DROP_ENVS:
            # Handle \begin{env}...\end{env} including nested braces
            pattern = re.compile(
                r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{' + re.escape(env) + r'\}',
                re.DOTALL
            )
            text = pattern.sub('', text)
        # Also strip standalone \includegraphics lines
        text = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '', text)
        text = re.sub(r'\\includegraphics\{.*?\}', '', text)
        return text

    # ------------------------------------------------------------------
    # Section filtering
    # ------------------------------------------------------------------
    def _filter_sections(self, text: str) -> str:
        """Drop entire sections that should not be read aloud."""
        # Split on section/subsection/subsubsection commands
        # Capture the section title and content until next section
        section_pattern = re.compile(
            r'(\\(?:section|subsection|subsubsection)\*?\{([^}]*)\})',
            re.MULTILINE
        )

        parts = section_pattern.split(text)
        # parts = [pre, full_cmd_1, title_1, content_1, full_cmd_2, title_2, content_2, ...]

        result = []
        skip = False
        i = 0

        # Add preamble content (before first section)
        if parts:
            # First element is text before any section command
            pre = parts[0]
            # Keep document content before first section if it has \begin{document}
            doc_start = pre.find('\\begin{document}')
            if doc_start >= 0:
                pre = pre[doc_start + len('\\begin{document}'):]
            # Strip preamble (everything before \begin{document})
            else:
                pre = ''
            result.append(pre)
            i = 1

        while i < len(parts):
            if i + 2 < len(parts):
                full_cmd = parts[i]
                title = parts[i + 1]
                content = parts[i + 2]
                i += 3

                # Check if this section should be dropped
                skip = False
                for drop in self._DROP_SECTIONS:
                    if drop.lower() in title.lower():
                        skip = True
                        break

                if not skip:
                    # Keep the section header as spoken text
                    result.append(f"\n\n{title}.\n\n")
                    result.append(content)
            else:
                # Trailing content
                result.append(parts[i])
                i += 1

        return ''.join(result)

    # ------------------------------------------------------------------
    # LaTeX command stripping and expansion
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_latex_commands(text: str) -> str:
        """Convert LaTeX commands to spoken equivalents."""

        # --- Document structure commands to remove ---
        text = re.sub(r'\\maketitle\b', '', text)
        text = re.sub(r'\\date\{[^}]*\}', '', text)
        text = re.sub(r'\\label\{[^}]*\}', '', text)
        text = re.sub(r'\\ref\{[^}]*\}', '', text)
        text = re.sub(r'\\pageref\{[^}]*\}', '', text)
        text = re.sub(r'\\noindent\b', '', text)
        text = re.sub(r'\\raggedbottom\b', '', text)
        text = re.sub(r'\\linenumbers\b', '', text)
        text = re.sub(r'\\newpage\b', '', text)
        text = re.sub(r'\\smallskip\b', '', text)
        text = re.sub(r'\\medskip\b', '', text)
        text = re.sub(r'\\bigskip\b', '', text)
        text = re.sub(r'\\hline\b', '', text)
        text = re.sub(r'\\centering\b', '', text)
        text = re.sub(r'\\footnotesize\b', '', text)
        text = re.sub(r'\\small\b', '', text)
        text = re.sub(r'\\normalsize\b', '', text)
        text = re.sub(r'\\large\b', '', text)
        text = re.sub(r'\\Large\b', '', text)
        text = re.sub(r'\\LARGE\b', '', text)
        text = re.sub(r'\\begingroup\b', '', text)
        text = re.sub(r'\\endgroup\b', '', text)
        text = re.sub(r'\\begin\{center\}', '', text)
        text = re.sub(r'\\end\{center\}', '', text)
        text = re.sub(r'\\begin\{flushleft\}', '', text)
        text = re.sub(r'\\end\{flushleft\}', '', text)
        text = re.sub(r'\\begin\{sloppypar\}', '', text)
        text = re.sub(r'\\end\{sloppypar\}', '', text)

        # --- Citations -> remove ---
        text = re.sub(r'\\cite\{[^}]*\}', '', text)
        text = re.sub(r'\\citep?\{[^}]*\}', '', text)
        text = re.sub(r'\\citet?\{[^}]*\}', '', text)

        # --- Cross references -> spoken form ---
        text = re.sub(r'Figure~?(\d+)', r'Figure \1', text)
        text = re.sub(r'Table~?(\d+)', r'Table \1', text)
        text = re.sub(r'Figure~?S(\d+)', r'Supplemental Figure \1', text)
        text = re.sub(r'Table~?S(\d+)', r'Supplemental Table \1', text)
        text = re.sub(r'Data~?S(\d+)', r'Supplemental Data \1', text)

        # --- URLs/DOIs -> remove ---
        text = re.sub(r'\\url\{[^}]*\}', '', text)
        text = re.sub(r'\\href\{[^}]*\}\{([^}]*)\}', r'\1', text)

        # --- Text formatting -> keep content ---
        text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\underline\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textrm\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textsf\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\texttt\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\{\\bf\s+([^}]*)\}', r'\1', text)
        text = re.sub(r'\{\\it\s+([^}]*)\}', r'\1', text)
        text = re.sub(r'\{\\em\s+([^}]*)\}', r'\1', text)

        # --- Captionof -> keep caption text ---
        text = re.sub(r'\\captionof\{[^}]*\}\{', '{', text)
        text = re.sub(r'\\caption\{', '{', text)

        # --- Itemize/enumerate -> convert bullets to prose ---
        text = re.sub(r'\\begin\{itemize\}', '', text)
        text = re.sub(r'\\end\{itemize\}', '', text)
        text = re.sub(r'\\begin\{enumerate\}', '', text)
        text = re.sub(r'\\end\{enumerate\}', '', text)
        text = re.sub(r'\\item\b', '  -', text)

        # --- Special characters ---
        text = text.replace('\\&', 'and')
        text = text.replace('\\%', 'percent')
        text = text.replace('\\$', 'dollars')
        text = text.replace('\\#', 'number')
        text = text.replace('\\\\', ' ')  # line breaks
        text = text.replace('~', ' ')  # non-breaking space
        text = re.sub(r'\\,', ' ', text)  # thin space
        text = re.sub(r'\\ ', ' ', text)  # explicit space
        text = re.sub(r'\\;', ' ', text)  # medium space
        text = re.sub(r'\\:', ' ', text)  # medium space
        text = re.sub(r'\\!', '', text)  # negative space
        text = re.sub(r'\\quad\b', ' ', text)
        text = re.sub(r'\\qquad\b', '  ', text)
        text = re.sub(r'\\allowbreak\b', '', text)

        # --- Accented chars ---
        text = re.sub(r"\\'\\{e}", 'e', text)  # \'{e} -> e
        text = re.sub(r"\\\'\{([a-zA-Z])\}", r'\1', text)
        text = re.sub(r"\\\`\{([a-zA-Z])\}", r'\1', text)
        text = re.sub(r'\\\"\{([a-zA-Z])\}', r'\1', text)

        # --- Remove remaining unknown commands (but keep their braced arg) ---
        # Two-pass: first handle commands with arguments
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        # Then bare commands
        text = re.sub(r'\\[a-zA-Z]+\b', '', text)

        # --- Clean up braces ---
        text = text.replace('{', '')
        text = text.replace('}', '')

        return text

    # ------------------------------------------------------------------
    # Math mode handling
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_math(text: str) -> str:
        """Convert inline math to spoken equivalents, drop display math."""

        # --- Specific known patterns first ---
        # LA$^4$SR -> L.A.4.S.R.  (no spaces between periods/digits to
        # prevent the citation-removal regex from eating the "4")
        text = re.sub(r'LA\$\^4\$SR', 'L.A.4.S.R.', text)
        text = re.sub(r'LA\$\^\{4\}\$SR', 'L.A.4.S.R.', text)
        text = re.sub(r'LA\$\^\{?4\}?\$SR', 'L.A.4.S.R.', text)

        # --- |rho| patterns (absolute value of correlation) ---
        text = re.sub(r'\$\|\\rho\|\$', 'absolute rho', text)
        text = re.sub(r'\$\|\\rho\|\s*=\s*([0-9.]+)\$', r'absolute rho equals \1', text)
        text = re.sub(r'\|\$?\\rho\$?\|', 'absolute rho', text)
        text = re.sub(r'\|rho\|', 'absolute rho', text)
        # General |X| -> absolute X within math
        text = re.sub(r'\$\|([^|$]+)\|\$', r'absolute \1', text)
        # Standalone |\rho| or |ρ|
        text = re.sub(r'\|\\?rho\|', 'absolute rho', text)

        # --- LaTeX math commands outside $ delimiters ---
        text = re.sub(r'\\geq\s*', 'greater than or equal to ', text)
        text = re.sub(r'\\leq\s*', 'less than or equal to ', text)
        text = re.sub(r'\\neq\s*', 'not equal to ', text)
        # Handle subscripts: \lambda_1 -> lambda-1, \lambda_{ij} -> lambda-ij
        text = re.sub(r'\\lambda_\{([^}]*)\}', r'lambda-\1', text)
        text = re.sub(r'\\lambda_(\w)', r'lambda-\1', text)
        text = re.sub(r'\\lambda\b', 'lambda', text)
        text = re.sub(r'\\Lambda\b', 'Lambda', text)
        text = re.sub(r'\\alpha\b', 'alpha', text)
        text = re.sub(r'\\beta\b', 'beta', text)
        text = re.sub(r'\\gamma\b', 'gamma', text)
        text = re.sub(r'\\delta\b', 'delta', text)
        text = re.sub(r'\\sigma\b', 'sigma', text)
        text = re.sub(r'\\rho\b', 'rho', text)
        text = re.sub(r'\\chi\b', 'chi', text)
        text = re.sub(r'\\phi_\{([^}]*)\}', r'phi-\1', text)
        text = re.sub(r'\\phi_(\w)', r'phi-\1', text)
        text = re.sub(r'\\sum_\{([^}]*)\}', r'sum over \1 of', text)
        text = re.sub(r'\\sum_(\w+)', r'sum over \1 of', text)
        text = re.sub(r'\\sum\b', 'sum of', text)
        text = re.sub(r'\\prod\b', 'product of', text)
        text = re.sub(r'\\infty\b', 'infinity', text)
        text = re.sub(r'\\pi\b', 'pi', text)
        text = re.sub(r'\\mu\b', 'mu', text)
        text = re.sub(r'\\phi\b', 'phi', text)
        # Remaining backslash-letter sequences in math residue
        text = re.sub(r'\\mathbf\b', '', text)
        text = re.sub(r'\\text\b', '', text)
        text = re.sub(r'\\hat\b', '', text)
        text = re.sub(r'\\left\b', '', text)
        text = re.sub(r'\\right\b', '', text)
        text = re.sub(r'\\frac\b', '', text)
        text = re.sub(r'\\sqrt\b', '', text)
        text = re.sub(r'\\ln\b', 'natural log of', text)
        text = re.sub(r'\\log\b', 'log of', text)
        text = re.sub(r'\\exp\b', 'exponential of', text)
        text = re.sub(r'\\min\b', 'minimum', text)
        text = re.sub(r'\\max\b', 'maximum', text)

        # --- Em-dashes ---
        text = text.replace('---', ' -- ')
        text = text.replace('--', ' - ')

        # R^2 -> R-squared
        text = re.sub(r'\$R\^2\$', 'R-squared', text)
        text = re.sub(r'\$R\^\{2\}\$', 'R-squared', text)
        text = re.sub(r'R\$\^2\$', 'R-squared', text)
        text = re.sub(r'R\$\^\{2\}\$', 'R-squared', text)
        text = re.sub(r'\$R\^2\s*=\s*([0-9.]+)\$', r'R-squared equals \1', text)
        text = re.sub(r'\$R\^\{2\}\s*=\s*([0-9.]+)\$', r'R-squared equals \1', text)

        # p < value
        text = re.sub(r'\$p\s*<\s*([0-9.]+)\$', r'p less than \1', text)

        # rho = value
        text = re.sub(r'\$\\rho\s*=\s*([0-9.]+)\$', r'rho equals \1', text)
        text = re.sub(r'\$\\?rho\$', 'rho', text)

        # r = value (canonical correlation)
        text = re.sub(r'\$r\s*=\s*([0-9.]+)\$', r'r equals \1', text)

        # n = value (sample size)
        text = re.sub(r'\$n\s*=\s*([0-9,]+)\$', r'n equals \1', text)
        text = re.sub(r'\$n\s*=\s*\\,?([0-9,{}]+)\$', r'n equals \1', text)

        # >= and <=
        text = re.sub(r'\$\\geq\s*([0-9.]+)\$', r'greater than or equal to \1', text)
        text = re.sub(r'\$\\leq\s*([0-9.]+)\$', r'less than or equal to \1', text)
        text = re.sub(r'\$>\s*([0-9.]+)\$', r'greater than \1', text)
        text = re.sub(r'\$<\s*([0-9.]+)\$', r'less than \1', text)

        # Approximate: $\sim$X
        text = re.sub(r'\$\\sim\$\s*', 'approximately ', text)
        text = re.sub(r'\$\\approx\s*([^$]*)\$', r'approximately \1', text)

        # Arrows
        text = re.sub(r'\$\\rightarrow\$', 'to', text)
        text = re.sub(r'\$\\leftarrow\$', 'from', text)
        text = re.sub(r'\$\\leftrightarrow\$', 'and', text)

        # Times: $\times$
        text = re.sub(r'\$\\times\$', 'times', text)

        # Plus-minus
        text = re.sub(r'\$\\pm\s*([^$]*)\$', r'plus or minus \1', text)
        text = re.sub(r'\\pm', 'plus or minus', text)

        # Textdegree
        text = re.sub(r'\\textdegree\s*', ' degrees ', text)

        # Textmu (micrometer)
        text = re.sub(r'\\textmu\{\}m', 'micrometers', text)
        text = re.sub(r'\\textmu\s*m', 'micrometers', text)

        # Chi-squared
        text = re.sub(r'\$\\chi\^2\$', 'chi-squared', text)
        text = re.sub(r'\$\\chi\^\{2\}\$', 'chi-squared', text)

        # Lambda
        text = re.sub(r"\$\\Lambda\$", "Lambda", text)
        text = re.sub(r"\$\\lambda\$", "lambda", text)

        # E-values: $E < 10^{-N}$
        text = re.sub(
            r'\$E\s*<\s*10\^\{?(-?\d+)\}?\$',
            lambda m: f'E less than ten to the {m.group(1)}',
            text
        )

        # Scientific notation: $10^{-N}$ or 10^-N
        text = re.sub(
            r'\$10\^\{?(-?\d+)\}?\$',
            lambda m: f'ten to the {m.group(1)}',
            text
        )

        # Superscripts in general: $^{N}$ -> to the N
        text = re.sub(r'\$\^\{([^}]*)\}\$', r' to the \1', text)
        text = re.sub(r'\$\^(\d)\$', r' to the \1', text)

        # Drop remaining display math blocks
        text = re.sub(r'\$\$.*?\$\$', ' [equation omitted] ', text, flags=re.DOTALL)
        text = re.sub(r'\\\[.*?\\\]', ' [equation omitted] ', text, flags=re.DOTALL)
        text = re.sub(
            r'\\begin\{equation\}.*?\\end\{equation\}',
            ' [equation omitted] ',
            text,
            flags=re.DOTALL
        )
        text = re.sub(
            r'\\begin\{align\}.*?\\end\{align\}',
            ' [equation omitted] ',
            text,
            flags=re.DOTALL
        )

        # Drop remaining inline math delimiters, keep content
        text = re.sub(r'\$([^$]*)\$', r'\1', text)

        return text

    # ------------------------------------------------------------------
    # TTS-hostile content detection
    # ------------------------------------------------------------------
    def _screen_for_tts_hazards(self, paragraphs: list[str]) -> list[str]:
        """Screen and fix paragraphs that would cause TTS problems."""
        clean = []
        for para in paragraphs:
            s = para.strip()
            if not s:
                continue

            # --- Drop equation-omitted placeholders if they're the whole paragraph ---
            if s == '[equation omitted]':
                continue

            # --- Drop paragraphs that are just dashes/hlines ---
            if re.match(r'^[\-=_\s]+$', s):
                continue

            # --- Drop table-like rows (dominated by & separators or pipes) ---
            if s.count('&') > 2 or s.count('|') > 4:
                self._warnings.append(f"DROPPED table row: {s[:80]}...")
                continue

            # --- Drop very short non-prose fragments ---
            if len(s) < 20 and not re.search(r'[a-zA-Z]{3,}', s):
                continue

            # --- Truncate extended bullet lists (>8 items) ---
            if s.count('  -') > 8:
                bullets = s.split('  -')
                kept = bullets[:6]
                self._warnings.append(
                    f"TRUNCATED list from {len(bullets)} to 6 items"
                )
                s = '  -'.join(kept) + '  - and additional items.'

            # --- Flag/fix repeated characters that cause TTS stutter ---
            # e.g., "........" or "--------"
            s = re.sub(r'([.\-=_])\1{4,}', '', s)

            # --- Flag high density of special chars ---
            special = sum(1 for c in s if c in '{}[]\\|@#$%^&*<>/')
            if special / max(len(s), 1) > 0.15 and len(s) > 30:
                self._warnings.append(
                    f"HIGH special-char density ({special}/{len(s)}): {s[:60]}..."
                )
                # Try to clean it
                s = re.sub(r'[{}\\|@#$%^&*<>]', '', s)

            # --- Flag/drop raw LaTeX that survived stripping ---
            if re.search(r'\\[a-zA-Z]{3,}\{', s):
                self._warnings.append(f"RESIDUAL LaTeX: {s[:80]}...")
                s = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', s)
                s = re.sub(r'\\[a-zA-Z]+', '', s)

            if s.strip():
                clean.append(s)

        return clean

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def clean(self, tex_path: str) -> str:
        """Full pipeline: LaTeX source -> TTS-ready text."""
        self._warnings = []

        # 1. Read source
        raw = self.read_tex(tex_path)

        # 2. Strip comments
        raw = re.sub(r'(?<!\\)%.*$', '', raw, flags=re.MULTILINE)

        # 3. Remove preamble (everything before \begin{document})
        doc_match = re.search(r'\\begin\{document\}', raw)
        if doc_match:
            raw = raw[doc_match.end():]
        end_match = re.search(r'\\end\{document\}', raw)
        if end_match:
            raw = raw[:end_match.start()]

        # 4. Strip bibliography
        raw = re.sub(r'\\bibliography\{[^}]*\}', '', raw)
        raw = re.sub(r'\\bibliographystyle\{[^}]*\}', '', raw)

        # 5. Strip figures, tables, equations (full environments)
        raw = self._strip_environments(raw)

        # 6. Handle captionof blocks that may remain
        # Keep caption text but strip the figure wrapper
        raw = re.sub(r'\\captionof\{figure\}\{', '', raw)

        # 7. Filter sections (drop refs, acknowledgments, etc.)
        raw = self._filter_sections(raw)

        # 8. Expand math mode to spoken form
        raw = self._expand_math(raw)

        # 9. Expand/strip LaTeX commands
        raw = self._expand_latex_commands(raw)

        # 10. Apply the same TTS expansions as the PDF cleaner
        raw = PDFTTSCleaner._expand_for_tts(raw)

        # 11. Reflow into paragraphs
        paragraphs = PDFTTSCleaner._reflow(raw)

        # 12. Screen for TTS hazards
        paragraphs = self._screen_for_tts_hazards(paragraphs)

        # 13. Final prose-score filtering
        final = []
        for p in paragraphs:
            s = p.strip()
            if not s:
                continue
            # Keep section headers (short, capitalized)
            if len(s) < 100 and len(s.split()) <= 12:
                final.append(s)
                continue
            # Keep anything with decent prose score
            if _prose_score(s) >= 0.25:
                final.append(s)
            else:
                # Still keep if it's long enough and has enough words
                if len(s) > 80 and len(s.split()) > 10:
                    final.append(s)

        text = '\n\n'.join(final)

        # Final cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        text = text.strip()

        # Stats
        words = len(text.split())
        self.stats = {
            'paragraphs': len(final),
            'words': words,
            'est_duration_min': round(words / 150, 1),
            'warnings': len(self._warnings),
        }
        return text

    def get_warnings(self) -> list[str]:
        """Return warnings generated during cleaning."""
        return list(self._warnings)


# ---------------------------------------------------------------------------
# Plain text / Markdown cleaners for review files
# ---------------------------------------------------------------------------
class PlainTextTTSCleaner:
    """Clean plain-text review files for TTS.

    Handles:
      - Bullet markers (●, -, *)
      - Summary tables (pipe-delimited)
      - Section headers → spoken dividers
      - Numeric references (line 123, lines 456-789)
      - Horizontal rules (---)
    """

    def __init__(self):
        self.stats: dict = {}

    def clean(self, file_path: str, spoken_header: str = '') -> str:
        """Clean a plain text file for TTS."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        lines = raw.split('\n')
        paragraphs: list[str] = []
        current: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()

            # Skip horizontal rules
            if re.match(r'^-{3,}$', stripped):
                # Flush current paragraph
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                continue

            # Skip table rows (pipe-delimited)
            if stripped.startswith('|') and stripped.endswith('|'):
                in_table = True
                continue
            if in_table and not stripped:
                in_table = False
                continue
            if in_table:
                continue

            # Empty line -> paragraph break
            if not stripped:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                continue

            # Strip bullet markers
            stripped = re.sub(r'^[●•]\s*', '', stripped)

            # Convert "REVIEWER N — Field" to spoken form
            m = re.match(r'^REVIEWER\s+(\d+)\s*[—–-]\s*(.*)', stripped)
            if m:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                paragraphs.append(f"Reviewer {m.group(1)}, {m.group(2)}.")
                continue

            # Convert section headers
            if stripped in ('Major Issues:', 'Major Issues', 'Minor Issues:', 'Minor Issues'):
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                paragraphs.append(stripped.rstrip(':') + '.')
                continue

            # Convert "Summary Across Reviewers" and similar
            if re.match(r'^Summary\b', stripped) and len(stripped.split()) <= 6:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                paragraphs.append(stripped + '.')
                continue

            # Strip line references (line 123) -> just remove them for cleaner audio
            stripped = re.sub(r'\(line[s]?\s+\d+[-–]\d+\)', '', stripped)
            stripped = re.sub(r'\(line[s]?\s+\d+\)', '', stripped)
            stripped = re.sub(r'line[s]?\s+\d+[-–]\d+', '', stripped)
            stripped = re.sub(r'line\s+\d+', '', stripped)

            # Clean up numbered issue starts: "1. SNAP gene..." -> "First, SNAP gene..."
            # Leave them as-is for now; TTS handles "1." fine

            current.append(stripped)

        if current:
            paragraphs.append(' '.join(current))

        # Apply shared TTS expansions
        text = '\n\n'.join(paragraphs)
        text = PDFTTSCleaner._expand_for_tts(text)

        # Add spoken header if provided
        if spoken_header:
            text = spoken_header + '\n\n' + text

        # Final cleanup
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        words = len(text.split())
        self.stats = {
            'paragraphs': len(paragraphs),
            'words': words,
            'est_duration_min': round(words / 150, 1),
        }
        return text


class MarkdownTTSCleaner:
    """Clean markdown files (review responses) for TTS.

    Handles:
      - Headers (# ## ###) → spoken section dividers
      - Bold (**text**) → keep content
      - Italic (*text*) → keep content
      - Bullet lists (- item) → spoken prose
      - Backtick code (`code`) → keep content
      - Pipe tables → skip
      - Commit hashes → skip
      - Line/file references → skip
    """

    def __init__(self):
        self.stats: dict = {}

    def clean(self, file_path: str, spoken_header: str = '') -> str:
        """Clean a markdown file for TTS."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        lines = raw.split('\n')
        paragraphs: list[str] = []
        current: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()

            # Skip horizontal rules
            if re.match(r'^-{3,}$', stripped):
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                continue

            # Skip table rows
            if stripped.startswith('|') and '|' in stripped[1:]:
                in_table = True
                continue
            if in_table and not stripped:
                in_table = False
                continue
            if in_table:
                continue

            # Empty line -> paragraph break
            if not stripped:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                continue

            # Convert headers to spoken form
            m = re.match(r'^(#{1,4})\s+(.*)', stripped)
            if m:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                header_text = m.group(2)
                # Clean markdown from header
                header_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', header_text)
                # Convert "REVIEWER N — Field" in headers
                rm = re.match(r'REVIEWER\s+(\d+)\s*[—–-]\s*(.*)', header_text)
                if rm:
                    header_text = f"Reviewer {rm.group(1)}, {rm.group(2)}"
                paragraphs.append(header_text + '.')
                continue

            # Strip bold markers
            stripped = re.sub(r'\*\*([^*]+)\*\*', r'\1', stripped)
            # Strip italic markers
            stripped = re.sub(r'\*([^*]+)\*', r'\1', stripped)
            # Strip backtick code
            stripped = re.sub(r'`([^`]+)`', r'\1', stripped)

            # Convert bullet items to prose
            if stripped.startswith('- '):
                stripped = stripped[2:]

            # Remove commit hashes (e.g., "commit d25efd2")
            stripped = re.sub(r'\(commit[s]?\s+[0-9a-f]{6,}\)', '', stripped)
            stripped = re.sub(r'commit[s]?\s+[0-9a-f]{6,}', '', stripped)

            # Remove line references
            stripped = re.sub(r'main\.tex:\d+[-–]?\d*', '', stripped)
            stripped = re.sub(r'supplemental_information\.tex:\d+', '', stripped)

            # Remove "Where:" prefixes - these are metadata, not content
            if re.match(r'^Where:\s*', stripped):
                stripped = re.sub(r'^Where:\s*', 'Location: ', stripped)

            # Convert "Status: ADDRESSED" etc. to spoken form
            stripped = re.sub(r'Status:\s*ADDRESSED', 'Status: Addressed', stripped)
            stripped = re.sub(r'Status:\s*PARTIALLY ADDRESSED', 'Status: Partially addressed', stripped)
            stripped = re.sub(r'Status:\s*IN PROGRESS', 'Status: In progress', stripped)
            stripped = re.sub(r'Status:\s*Not addressed', 'Status: Not yet addressed', stripped)
            stripped = re.sub(r'Status:\s*Already addressed in manuscript',
                            'Status: Already addressed in the manuscript', stripped)

            # Skip empty after cleanup
            stripped = stripped.strip()
            if not stripped:
                continue

            current.append(stripped)

        if current:
            paragraphs.append(' '.join(current))

        # Apply shared TTS expansions
        text = '\n\n'.join(paragraphs)
        text = PDFTTSCleaner._expand_for_tts(text)

        # Add spoken header if provided
        if spoken_header:
            text = spoken_header + '\n\n' + text

        # Final cleanup
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        words = len(text.split())
        self.stats = {
            'paragraphs': len(paragraphs),
            'words': words,
            'est_duration_min': round(words / 150, 1),
        }
        return text


def clean_multi_files(file_paths: list[str], spoken_headers: list[str] | None = None) -> tuple[str, dict]:
    """Clean and concatenate multiple files (txt/md) into a single TTS-ready text.

    Args:
        file_paths: List of file paths to process in order
        spoken_headers: Optional spoken header for each file (e.g., "Reviews from April 7th")

    Returns:
        (combined_text, stats_dict)
    """
    if spoken_headers is None:
        spoken_headers = ['' for _ in file_paths]

    sections: list[str] = []
    total_words = 0
    total_paragraphs = 0

    for fpath, header in zip(file_paths, spoken_headers):
        ext = Path(fpath).suffix.lower()

        if ext == '.md':
            cleaner = MarkdownTTSCleaner()
            text = cleaner.clean(fpath, spoken_header=header)
        elif ext == '.tex':
            cleaner = LaTeXTTSCleaner()
            text = cleaner.clean(fpath)
            if header:
                text = header + '\n\n' + text
        elif ext == '.pdf':
            cleaner = PDFTTSCleaner()
            text = cleaner.clean(fpath)
            if header:
                text = header + '\n\n' + text
        else:
            # Default to plain text
            cleaner = PlainTextTTSCleaner()
            text = cleaner.clean(fpath, spoken_header=header)

        sections.append(text)
        total_words += cleaner.stats.get('words', 0)
        total_paragraphs += cleaner.stats.get('paragraphs', 0)

    # Join sections with spoken dividers
    combined = '\n\n\n'.join(sections)
    combined = re.sub(r'\n{4,}', '\n\n\n', combined)

    stats = {
        'files': len(file_paths),
        'paragraphs': total_paragraphs,
        'words': total_words,
        'est_duration_min': round(total_words / 150, 1),
    }
    return combined, stats


# ---------------------------------------------------------------------------
# TTS Hallucination Screener
# ---------------------------------------------------------------------------
class TTSHallucinationScreener:
    """Screen TTS text chunks for patterns that cause voice hallucinations.

    Integrates with deep_voice_tts.py to:
      1. Pre-screen text chunks before synthesis
      2. Analyze chunk text files post-synthesis for anomalies
      3. Report problematic chunks for manual review
    """

    # Patterns known to cause TTS hallucinations
    _HALLUCINATION_TRIGGERS = [
        # Bare numbers without context (TTS may repeat or stutter)
        (r'(?<!\w)\d{5,}(?!\w)', 'LONG_NUMBER',
         'Long bare number (>5 digits) -- TTS may hallucinate'),
        # Dense numeric sequences
        (r'(?:\d+[.,]\s*){5,}', 'NUMBER_LIST',
         'Dense numeric list -- TTS stutter risk'),
        # Repeated words (TTS amplifies these)
        (r'\b(\w{3,})\s+\1\b', 'WORD_REPEAT',
         'Repeated word -- TTS may loop'),
        # Parenthetical citation debris
        (r'\(\s*\d{4}[a-z]?\s*\)', 'BARE_YEAR',
         'Bare year in parentheses (citation debris)'),
        # Very long words (chemical names, URLs that survived)
        (r'\b[a-zA-Z]{25,}\b', 'LONG_WORD',
         'Very long word (>25 chars) -- pronunciation risk'),
        # Sequences of abbreviations
        (r'(?:[A-Z]{2,}\s+){4,}', 'ACRONYM_CLUSTER',
         'Dense acronym cluster -- TTS may struggle'),
        # Residual LaTeX
        (r'\\[a-z]+', 'RESIDUAL_LATEX',
         'Residual LaTeX command survived cleaning'),
        # Unbalanced parentheses/brackets (confuses TTS chunking)
        (r'(?:\([^)]{200,}\))', 'LONG_PAREN',
         'Very long parenthetical (>200 chars) -- chunking risk'),
        # Empty sentence fragments
        (r'[.!?]\s*[.!?]', 'EMPTY_SENTENCE',
         'Empty sentence fragment -- TTS may hallucinate filler'),
        # Unicode oddities
        (r'[\x80-\x9f]', 'CONTROL_CHAR',
         'Control character in text -- TTS may produce noise'),
    ]

    def __init__(self):
        self.issues: list[dict] = []

    def screen_text(self, text: str) -> list[dict]:
        """Screen text for hallucination-prone patterns.

        Returns list of dicts with: line, pattern_name, description, match
        """
        self.issues = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            for pattern, name, desc in self._HALLUCINATION_TRIGGERS:
                for m in re.finditer(pattern, line):
                    self.issues.append({
                        'line': line_num,
                        'pattern': name,
                        'description': desc,
                        'match': m.group()[:80],
                        'context': line[max(0, m.start()-20):m.end()+20][:120],
                    })

        return self.issues

    def screen_chunks_dir(self, chunks_dir: str) -> list[dict]:
        """Screen all chunk .txt files in a deep_voice_tts output directory."""
        all_issues = []
        chunk_dir = Path(chunks_dir)

        if not chunk_dir.exists():
            return all_issues

        for txt_file in sorted(chunk_dir.glob('chunk_*.txt')):
            text = txt_file.read_text(encoding='utf-8')
            issues = self.screen_text(text)
            for issue in issues:
                issue['chunk_file'] = txt_file.name
            all_issues.extend(issues)

        return all_issues

    def analyze_tts_output(self, output_dir: str) -> dict:
        """Analyze a deep_voice_tts output directory for quality issues.

        Checks:
          - Chunk text content for hallucination triggers
          - Chunk audio duration anomalies (via metadata)
          - Missing chunks (gaps in numbering)
        """
        out = Path(output_dir)
        report = {
            'output_dir': str(out),
            'text_issues': [],
            'audio_issues': [],
            'summary': {},
        }

        # Check metadata
        meta_file = out / 'metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            report['metadata'] = meta

        # Screen chunk texts
        chunks_dir = out / 'chunks'
        if chunks_dir.exists():
            report['text_issues'] = self.screen_chunks_dir(str(chunks_dir))

            # Check for chunk numbering gaps
            chunk_files = sorted(chunks_dir.glob(f'chunk_*.mp3')) or \
                          sorted(chunks_dir.glob(f'chunk_*.wav'))
            if chunk_files:
                nums = []
                for f in chunk_files:
                    m = re.search(r'chunk_(\d+)', f.name)
                    if m:
                        nums.append(int(m.group(1)))
                if nums:
                    expected = set(range(min(nums), max(nums) + 1))
                    missing = expected - set(nums)
                    if missing:
                        report['audio_issues'].append({
                            'type': 'MISSING_CHUNKS',
                            'description': f'Missing chunk numbers: {sorted(missing)}',
                        })

            # Check for zero-length audio files
            for af in chunk_files:
                if af.stat().st_size < 1000:  # < 1KB is suspicious
                    report['audio_issues'].append({
                        'type': 'TINY_AUDIO',
                        'file': af.name,
                        'size_bytes': af.stat().st_size,
                        'description': f'{af.name} is suspiciously small ({af.stat().st_size} bytes)',
                    })

        report['summary'] = {
            'text_issues_count': len(report['text_issues']),
            'audio_issues_count': len(report['audio_issues']),
            'status': 'CLEAN' if not report['text_issues'] and not report['audio_issues'] else 'REVIEW_NEEDED',
        }

        return report

    def print_report(self, report: dict):
        """Print a human-readable screening report."""
        print("\n" + "=" * 70)
        print("TTS HALLUCINATION SCREENING REPORT")
        print("=" * 70)

        summary = report.get('summary', {})
        status = summary.get('status', 'UNKNOWN')
        print(f"Status: {status}")
        print(f"Text issues: {summary.get('text_issues_count', 0)}")
        print(f"Audio issues: {summary.get('audio_issues_count', 0)}")

        if report.get('text_issues'):
            print(f"\n--- Text Issues ({len(report['text_issues'])}) ---")
            # Group by pattern
            by_pattern: dict[str, list] = {}
            for issue in report['text_issues']:
                by_pattern.setdefault(issue['pattern'], []).append(issue)
            for pat, issues in sorted(by_pattern.items()):
                print(f"\n  [{pat}] ({len(issues)} occurrences)")
                print(f"    {issues[0]['description']}")
                for iss in issues[:3]:  # show first 3
                    loc = iss.get('chunk_file', f"line {iss['line']}")
                    print(f"    @ {loc}: ...{iss['context']}...")
                if len(issues) > 3:
                    print(f"    ... and {len(issues) - 3} more")

        if report.get('audio_issues'):
            print(f"\n--- Audio Issues ({len(report['audio_issues'])}) ---")
            for issue in report['audio_issues']:
                print(f"  [{issue['type']}] {issue['description']}")

        print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog='media_to_tts',
        description=(
            'Media-to-TTS: extract TTS-ready text from any text-bearing file '
            '(PDF, LaTeX, Markdown, plain text) or multi-file bundle, then '
            'optionally synthesize audio with hallucination screening (QC).'
        ),
    )
    parser.add_argument('input', nargs='?', help='Path to input file (.pdf or .tex). Omit if using --multi.')
    parser.add_argument('-o', '--output', help='Output .txt path (default: auto-timestamped)')
    parser.add_argument(
        '--voice', default=None,
        help='If set, also synthesise audio via deep_voice_tts.py with this voice (e.g. p246)',
    )
    parser.add_argument('--format', choices=['mp3', 'wav'], default='mp3')
    parser.add_argument(
        '--screen', action='store_true',
        help='Run TTS hallucination screening on output text and (if --voice) audio chunks',
    )
    parser.add_argument(
        '--screen-only', default=None,
        help='Screen an existing deep_voice_tts output directory (no extraction)',
    )
    parser.add_argument(
        '--keep-intermediates', action='store_true',
        help='Keep intermediate files (chunks/, _tts_ .txt). Default: remove after completion.',
    )
    parser.add_argument(
        '--multi', nargs='+', metavar='FILE',
        help='Process multiple files in order and concatenate into a single TTS output. '
             'Supports .txt, .md, .tex, .pdf files.',
    )
    parser.add_argument(
        '--headers', nargs='+', metavar='HEADER',
        help='Spoken section headers for each --multi file (same count as files).',
    )
    args = parser.parse_args()

    # --- Screen-only mode: analyze existing TTS output ---
    if args.screen_only:
        screener = TTSHallucinationScreener()
        report = screener.analyze_tts_output(args.screen_only)
        screener.print_report(report)
        # Save report JSON
        report_path = Path(args.screen_only) / 'hallucination_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {report_path}")
        sys.exit(0)

    # --- Multi-file mode: concatenate + clean multiple files ---
    if args.multi:
        file_paths = args.multi
        for fp in file_paths:
            if not os.path.isfile(fp):
                print(f"Error: {fp} not found", file=sys.stderr)
                sys.exit(1)

        spoken_headers = args.headers
        if spoken_headers and len(spoken_headers) != len(file_paths):
            print(f"Error: --headers count ({len(spoken_headers)}) must match "
                  f"--multi file count ({len(file_paths)})", file=sys.stderr)
            sys.exit(1)

        print(f"Multi-file mode: {len(file_paths)} files")
        for i, fp in enumerate(file_paths):
            h = spoken_headers[i] if spoken_headers else '(auto)'
            print(f"  {i+1}. {Path(fp).name}  [{Path(fp).suffix}]  header: {h}")

        text, stats = clean_multi_files(file_paths, spoken_headers)

        # Determine output path
        if args.output:
            out_path = args.output
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = str(Path(file_paths[0]).parent / f'combined_reviews_tts_{ts}.txt')

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"\nOutput:     {out_path}")
        print(f"Files:      {stats['files']}")
        print(f"Words:      {stats['words']}")
        print(f"Paragraphs: {stats['paragraphs']}")
        print(f"Est. TTS:   ~{stats['est_duration_min']} min (at 150 wpm)")

        # Pre-synthesis screening
        if args.screen:
            print("\n--- Pre-synthesis Hallucination Screening ---")
            screener = TTSHallucinationScreener()
            issues = screener.screen_text(text)
            if issues:
                print(f"Found {len(issues)} potential hallucination triggers:")
                by_pattern: dict[str, list] = {}
                for issue in issues:
                    by_pattern.setdefault(issue['pattern'], []).append(issue)
                for pat, pat_issues in sorted(by_pattern.items()):
                    print(f"  [{pat}] {len(pat_issues)}x -- {pat_issues[0]['description']}")
                    for iss in pat_issues[:2]:
                        print(f"    line {iss['line']}: ...{iss['context']}...")
            else:
                print("CLEAN -- no hallucination triggers detected.")

        # TTS synthesis
        tts_output_dir = None
        if args.voice:
            try:
                from deep_voice_tts import DeepVoiceTTS
                print(f"\nSynthesising audio with voice={args.voice} ...")
                tts = DeepVoiceTTS(voice_profile=args.voice, output_format=args.format)
                tts_output_dir = tts.process_text_file(out_path)
            except ImportError:
                print("deep_voice_tts.py not found in PATH; skipping synthesis.")
            except Exception as e:
                print(f"TTS synthesis failed: {e}")

        # Post-synthesis screening
        if args.screen and tts_output_dir:
            print("\n--- Post-synthesis Hallucination Screening ---")
            screener = TTSHallucinationScreener()
            report = screener.analyze_tts_output(tts_output_dir)
            screener.print_report(report)
            report_path = Path(tts_output_dir) / 'hallucination_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved: {report_path}")

        # Cleanup
        if not args.keep_intermediates and tts_output_dir:
            tts_out = Path(tts_output_dir)
            print("\n--- Cleaning intermediate files ---")
            chunks_dir = tts_out / 'chunks'
            if chunks_dir.exists():
                n_files = len(list(chunks_dir.iterdir()))
                shutil.rmtree(chunks_dir)
                print(f"  Removed {chunks_dir}/ ({n_files} files)")
            if os.path.isfile(out_path):
                os.remove(out_path)
                print(f"  Removed {out_path}")
            remaining = [f.name for f in tts_out.iterdir()]
            print(f"\nFinal output directory: {tts_out}/")
            for f in sorted(remaining):
                size = (tts_out / f).stat().st_size
                if size > 1024 * 1024:
                    print(f"  {f}  ({size / 1024 / 1024:.1f} MB)")
                else:
                    print(f"  {f}  ({size / 1024:.1f} KB)")

        sys.exit(0)

    # --- Single-file mode ---
    input_path = args.input
    if not input_path:
        parser.error("Must provide an input file or use --multi for multi-file mode.")
    if not os.path.isfile(input_path):
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # --- Detect input type and select cleaner ---
    ext = Path(input_path).suffix.lower()
    warnings = []

    if ext == '.tex':
        print(f"LaTeX source detected: {input_path}")
        cleaner = LaTeXTTSCleaner()
        text = cleaner.clean(input_path)
        warnings = cleaner.get_warnings()
    elif ext == '.pdf':
        print(f"PDF detected: {input_path}")
        cleaner = PDFTTSCleaner()
        text = cleaner.clean(input_path)
    else:
        print(f"Warning: unknown extension '{ext}', attempting as LaTeX...",
              file=sys.stderr)
        cleaner = LaTeXTTSCleaner()
        text = cleaner.clean(input_path)
        warnings = cleaner.get_warnings()

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        stem = Path(input_path).stem
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = str(Path(input_path).parent / f'{stem}_tts_{ts}.txt')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    s = cleaner.stats
    print(f"\nOutput:     {out_path}")
    print(f"Words:      {s['words']}")
    print(f"Paragraphs: {s['paragraphs']}")
    print(f"Est. TTS:   ~{s['est_duration_min']} min (at 150 wpm)")

    if s.get('warnings'):
        print(f"Warnings:   {s['warnings']}")

    # Print cleaning warnings
    if warnings:
        print(f"\n--- Cleaning Warnings ({len(warnings)}) ---")
        for w in warnings[:20]:
            print(f"  {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    # Spot-check key numbers
    probes = ['221.9', '2,357', '85,000', '33,950', '0.82', '201 million', '10,864']
    missing = [p for p in probes if p not in text]
    if missing:
        print(f"\nWARNING: missing key numbers: {missing}")

    # --- Pre-synthesis hallucination screening ---
    tts_output_dir = None
    if args.screen:
        print("\n--- Pre-synthesis Hallucination Screening ---")
        screener = TTSHallucinationScreener()
        issues = screener.screen_text(text)
        if issues:
            print(f"Found {len(issues)} potential hallucination triggers:")
            by_pattern: dict[str, list] = {}
            for issue in issues:
                by_pattern.setdefault(issue['pattern'], []).append(issue)
            for pat, pat_issues in sorted(by_pattern.items()):
                print(f"  [{pat}] {len(pat_issues)}x -- {pat_issues[0]['description']}")
                for iss in pat_issues[:2]:
                    print(f"    line {iss['line']}: ...{iss['context']}...")
        else:
            print("CLEAN -- no hallucination triggers detected.")

    # --- Optional: pipe into deep_voice_tts ---
    if args.voice:
        try:
            from deep_voice_tts import DeepVoiceTTS
            print(f"\nSynthesising audio with voice={args.voice} ...")
            tts = DeepVoiceTTS(voice_profile=args.voice, output_format=args.format)
            tts_output_dir = tts.process_text_file(out_path)
        except ImportError:
            print("deep_voice_tts.py not found in PATH; skipping synthesis.")
        except Exception as e:
            print(f"TTS synthesis failed: {e}")

    # --- Post-synthesis hallucination screening ---
    if args.screen and tts_output_dir:
        print("\n--- Post-synthesis Hallucination Screening ---")
        screener = TTSHallucinationScreener()
        report = screener.analyze_tts_output(tts_output_dir)
        screener.print_report(report)

        # Save report
        report_path = Path(tts_output_dir) / 'hallucination_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {report_path}")

    # --- Cleanup intermediate files ---
    if not args.keep_intermediates and tts_output_dir:
        tts_out = Path(tts_output_dir)
        print("\n--- Cleaning intermediate files ---")

        # 1. Remove chunks/ directory (individual chunk mp3 + txt files)
        chunks_dir = tts_out / 'chunks'
        if chunks_dir.exists():
            n_files = len(list(chunks_dir.iterdir()))
            shutil.rmtree(chunks_dir)
            print(f"  Removed {chunks_dir}/ ({n_files} files)")

        # 2. Remove intermediate _tts_ .txt file (the combined mp3 is the deliverable)
        if os.path.isfile(out_path):
            os.remove(out_path)
            print(f"  Removed {out_path}")

        # 3. Remove the timestamped copy that was also written to the manuscript dir
        tts_copy = Path(input_path).parent / Path(out_path).name
        if tts_copy.exists() and str(tts_copy) != out_path:
            os.remove(tts_copy)
            print(f"  Removed {tts_copy}")

        # Summary of what remains
        remaining = [f.name for f in tts_out.iterdir()]
        print(f"\nFinal output directory: {tts_out}/")
        for f in sorted(remaining):
            size = (tts_out / f).stat().st_size
            if size > 1024 * 1024:
                print(f"  {f}  ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f"  {f}  ({size / 1024:.1f} KB)")


if __name__ == '__main__':
    main()
