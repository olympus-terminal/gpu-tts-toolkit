#!/usr/bin/env python3
"""
Scientific Manuscript Text Extractor for TTS.

Extracts narrative prose from scientific manuscripts (PDF or LaTeX) and converts
it to clean text suitable for text-to-speech processing. Handles heavy scientific
math notation, figure/table environments, section filtering, and domain-specific
patterns found in Cell Press / scientific manuscripts.

Output is directly consumable by deep_voice_tts.py.

Usage:
    python extract_salient_text.py main.tex
    python extract_salient_text.py main.pdf
    python extract_salient_text.py main.tex -o output.tts.txt
    python extract_salient_text.py main.tex --include-sections "SUMMARY,RESULTS"
    python extract_salient_text.py main.tex --exclude-sections "Limitations of the study"
    python extract_salient_text.py main.tex --include-sections ALL
    python extract_salient_text.py main.tex --preview 500
    python extract_salient_text.py main.tex --strip-heavy-stats
    python extract_salient_text.py main.tex --keep-figure-refs
"""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# MathToSpeech — converts LaTeX / Unicode math to spoken English
# ---------------------------------------------------------------------------

class MathToSpeech:
    """Convert inline math expressions (LaTeX or Unicode) to spoken text."""

    GREEK = {
        # LaTeX commands
        r'\alpha': 'alpha', r'\beta': 'beta', r'\gamma': 'gamma',
        r'\delta': 'delta', r'\epsilon': 'epsilon', r'\zeta': 'zeta',
        r'\eta': 'eta', r'\theta': 'theta', r'\iota': 'iota',
        r'\kappa': 'kappa', r'\lambda': 'lambda', r'\mu': 'mu',
        r'\nu': 'nu', r'\xi': 'xi', r'\pi': 'pi', r'\rho': 'rho',
        r'\sigma': 'sigma', r'\tau': 'tau', r'\upsilon': 'upsilon',
        r'\phi': 'phi', r'\chi': 'chi', r'\psi': 'psi', r'\omega': 'omega',
        r'\Lambda': 'Lambda', r'\Gamma': 'Gamma', r'\Delta': 'Delta',
        r'\Theta': 'Theta', r'\Sigma': 'Sigma', r'\Omega': 'Omega',
    }

    OPERATORS = {
        r'\times': 'times', r'\pm': 'plus or minus',
        r'\mp': 'minus or plus', r'\cdot': 'times',
        r'\geq': 'greater than or equal to', r'\ge': 'greater than or equal to',
        r'\leq': 'less than or equal to', r'\le': 'less than or equal to',
        r'\approx': 'approximately', r'\sim': 'approximately',
        r'\neq': 'not equal to', r'\ne': 'not equal to',
        r'\rightarrow': 'leads to', r'\to': 'to',
        r'\leftarrow': 'from', r'\infty': 'infinity',
        r'\propto': 'proportional to', r'\equiv': 'is equivalent to',
        r'\gg': 'much greater than', r'\ll': 'much less than',
    }

    # Unicode equivalents (for PDF-extracted text)
    UNICODE_MAP = {
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b3': 'gamma',
        '\u03b4': 'delta', '\u03b5': 'epsilon', '\u03b6': 'zeta',
        '\u03b7': 'eta', '\u03b8': 'theta', '\u03b9': 'iota',
        '\u03ba': 'kappa', '\u03bb': 'lambda', '\u03bc': 'mu',
        '\u03bd': 'nu', '\u03be': 'xi', '\u03c0': 'pi', '\u03c1': 'rho',
        '\u03c3': 'sigma', '\u03c4': 'tau', '\u03c5': 'upsilon',
        '\u03c6': 'phi', '\u03c7': 'chi', '\u03c8': 'psi', '\u03c9': 'omega',
        '\u00d7': 'times', '\u00b1': 'plus or minus',
        '\u2265': 'greater than or equal to', '\u2264': 'less than or equal to',
        '\u2248': 'approximately', '\u223c': 'approximately',
        '\u2260': 'not equal to', '\u2192': 'leads to',
        '\u221e': 'infinity', '\u00b2': ' squared', '\u00b3': ' cubed',
        '\u2212': '-',  # minus sign
        '\u2013': '-',  # en-dash (used as minus in some PDFs)
    }

    def __init__(self):
        # Pre-compile patterns for inline math conversion
        self._sci_notation = re.compile(
            r'(\d+(?:\.\d+)?)\s*\\times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?')
        self._r_squared = re.compile(r'R\s*\^\s*\{?\s*2\s*\}?')
        self._chi_squared = re.compile(r'\\chi\s*\^\s*\{?\s*2\s*\}?')
        self._superscript_braced = re.compile(r'\^\{([^}]*)\}')
        self._superscript_single = re.compile(r'\^(\w)')
        self._subscript_braced = re.compile(r'_\{([^}]*)\}')
        self._subscript_single = re.compile(r'_(\w)')
        self._fraction = re.compile(r'\\frac\{([^}]*)\}\{([^}]*)\}')
        self._abs_value = re.compile(r'\|([^|]+)\|')
        self._number_comma = re.compile(r'(\d)\{,\}(\d)')

    def convert_latex(self, math_str: str) -> str:
        """Convert a LaTeX math expression to spoken text."""
        s = math_str.strip()

        # Number formatting: 1{,}090 → 1,090
        s = self._number_comma.sub(r'\1,\2', s)

        # R^2
        s = self._r_squared.sub('R squared', s)

        # chi^2
        s = self._chi_squared.sub('chi squared', s)

        # Scientific notation: 10^{-9} → 10 to the minus 9
        s = self._sci_notation.sub(
            lambda m: f"{m.group(1)} times 10 to the {m.group(2)}", s)

        # Fractions
        s = self._fraction.sub(r'\1 over \2', s)

        # Absolute values
        s = self._abs_value.sub(r'absolute value of \1', s)

        # Greek letters (before stripping remaining commands)
        for cmd, name in self.GREEK.items():
            s = s.replace(cmd, f' {name} ')

        # Operators
        for cmd, name in self.OPERATORS.items():
            s = s.replace(cmd, f' {name} ')

        # Superscripts (after R^2 and chi^2 already handled)
        s = self._superscript_braced.sub(r' to the \1', s)
        s = self._superscript_single.sub(r' to the \1', s)

        # Subscripts
        s = self._subscript_braced.sub(r' sub \1', s)
        s = self._subscript_single.sub(r' sub \1', s)

        # Comparisons (bare symbols)
        s = s.replace('<', ' less than ')
        s = s.replace('>', ' greater than ')
        s = s.replace('=', ' equals ')

        # Percent
        s = s.replace(r'\%', ' percent')

        # Strip remaining LaTeX commands and braces
        s = re.sub(r'\\(?:mathrm|text|mathit|mathbf|operatorname)\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\[a-zA-Z]+', '', s)
        s = re.sub(r'[{}]', '', s)

        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def convert_unicode(self, text: str) -> str:
        """Convert Unicode math symbols in PDF-extracted text to spoken form."""
        for char, spoken in self.UNICODE_MAP.items():
            text = text.replace(char, f' {spoken} ')
        # Collapse whitespace introduced by replacements
        text = re.sub(r'\s+', ' ', text)
        return text

# ---------------------------------------------------------------------------
# Section filter — shared between LaTeX and PDF extractors
# ---------------------------------------------------------------------------

# Default sections to include (case-insensitive matching)
DEFAULT_INCLUDE = [
    'SUMMARY', 'ABSTRACT', 'INTRODUCTION', 'RESULTS', 'DISCUSSION',
]

# Default sections to exclude
DEFAULT_EXCLUDE = [
    'KEYWORDS', 'GRAPHICAL ABSTRACT',
    'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS',
    'AUTHOR CONTRIBUTIONS',
    'DECLARATION OF INTERESTS',
    'DECLARATION OF GENERATIVE AI AND AI-ASSISTED TECHNOLOGIES',
    'DECLARATIONS',
    'SUPPLEMENTAL INFORMATION', 'SUPPLEMENTARY INFORMATION',
    'STAR METHODS', 'METHODS', 'MATERIALS AND METHODS',
    'REFERENCES', 'BIBLIOGRAPHY',
    'DATA AVAILABILITY', 'CODE AVAILABILITY',
]

def should_include_section(name: str, include: list, exclude: list,
                           include_all: bool = False) -> bool:
    """Decide whether a section should be kept."""
    if include_all:
        upper = name.upper().strip()
        for ex in exclude:
            if upper == ex or upper.startswith(ex):
                return False
        return True
    upper = name.upper().strip()
    # Explicit exclude overrides include
    for ex in exclude:
        if upper == ex or upper.startswith(ex):
            return False
    for inc in include:
        if upper == inc or upper.startswith(inc):
            return True
    return False

# ---------------------------------------------------------------------------
# LaTeXSalientExtractor — primary path
# ---------------------------------------------------------------------------

class LaTeXSalientExtractor:
    """Extract narrative prose from a LaTeX manuscript for TTS."""

    def __init__(self, *, strip_heavy_stats: bool = False,
                 keep_figure_refs: bool = False):
        self.math = MathToSpeech()
        self.strip_heavy_stats = strip_heavy_stats
        self.keep_figure_refs = keep_figure_refs

    # -- step-by-step pipeline -------------------------------------------

    def extract(self, tex: str, include: list | None = None,
                exclude: list | None = None,
                include_all: bool = False) -> tuple[str, dict]:
        """Return (cleaned_text, stats_dict)."""
        inc = include if include is not None else DEFAULT_INCLUDE
        exc = exclude if exclude is not None else DEFAULT_EXCLUDE

        text = tex

        # 1. Strip preamble
        if r'\begin{document}' in text:
            text = text.split(r'\begin{document}', 1)[1]
        if r'\end{document}' in text:
            text = text.split(r'\end{document}', 1)[0]

        # 2. Remove LaTeX comments
        text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

        # 3. Remove \begingroup...\endgroup figure blocks
        text = re.sub(
            r'\\begingroup\s*\\centering.*?\\endgroup',
            '', text, flags=re.DOTALL)
        # Also handle without \centering
        text = re.sub(
            r'\\begingroup.*?\\captionof\{figure\}.*?\\endgroup',
            '', text, flags=re.DOTALL)

        # 4. Remove standard float environments
        for env in ('figure', 'figure*', 'table', 'table*', 'tabular',
                     'tabular*', 'longtable', 'equation', 'equation*',
                     'align', 'align*', 'gather', 'gather*', 'multline',
                     'multline*', 'eqnarray', 'eqnarray*'):
            pat = re.compile(
                r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{' + re.escape(env) + r'\}',
                re.DOTALL)
            text = pat.sub('', text)

        # 5. Remove \begin{center}...\end{center}, standalone \includegraphics
        text = re.sub(r'\\begin\{center\}.*?\\end\{center\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\includegraphics\[?[^\]]*\]?\{[^}]*\}', '', text)

        # 6. Remove itemize/enumerate (data availability lists)
        for env in ('itemize', 'enumerate', 'description'):
            text = re.sub(
                r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}',
                '', text, flags=re.DOTALL)

        # 7. Remove \begin{small}...\end{small}
        text = re.sub(r'\\begin\{small\}.*?\\end\{small\}', '', text, flags=re.DOTALL)

        # 8. Display math (before section parsing, so it doesn't confuse things)
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)

        # 9. Parse sections into OrderedDict
        sections = self._parse_sections(text)

        # 10. Apply section filter
        filtered = OrderedDict()
        # Track parent section for propagation
        parent_included = False
        parent_name = ''
        for name, body in sections.items():
            level = self._section_level(name)
            if level == 1:
                parent_included = should_include_section(
                    name, inc, exc, include_all)
                parent_name = name
                if parent_included:
                    filtered[name] = body
            else:
                # Subsection: include if parent is included and not explicitly excluded
                if parent_included:
                    upper = name.upper().strip()
                    explicitly_excluded = any(
                        upper == ex or upper.startswith(ex) for ex in exc)
                    if not explicitly_excluded:
                        filtered[name] = body
                    else:
                        # If this subsection is excluded, its children will also be
                        parent_included = False

        stats = {
            'total_sections': len(sections),
            'kept_sections': len(filtered),
            'excluded_sections': len(sections) - len(filtered),
            'section_names': list(sections.keys()),
            'kept_names': list(filtered.keys()),
        }

        # 11. Clean each section body
        cleaned_parts = []
        for name, body in filtered.items():
            heading = name.strip() + '.'
            cleaned_body = self._clean_body(body)
            if cleaned_body.strip():
                cleaned_parts.append(f"\n\n{heading}\n\n{cleaned_body}")

        result = '\n'.join(cleaned_parts)

        # Final global cleanup
        result = self._final_cleanup(result)

        stats['output_chars'] = len(result)
        stats['output_words'] = len(result.split())

        return result.strip(), stats

    # -- section parsing --------------------------------------------------

    def _parse_sections(self, text: str) -> OrderedDict:
        """Parse LaTeX section/subsection/subsubsection into OrderedDict."""
        pattern = re.compile(
            r'\\(section|subsection|subsubsection)\*?\{([^}]*)\}')
        matches = list(pattern.finditer(text))

        sections = OrderedDict()
        if not matches:
            sections['_body'] = text
            return sections

        # Text before first section (title, abstract, etc.) — skip
        for i, m in enumerate(matches):
            name = m.group(2).strip()
            # Clean LaTeX from section name
            name = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', name)
            name = re.sub(r'\\[a-zA-Z]+', '', name)
            name = re.sub(r'[${}]', '', name)
            name = name.strip()

            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end]
            sections[name] = body

        return sections

    def _section_level(self, name: str) -> int:
        """Heuristic: top-level sections are ALL CAPS."""
        stripped = name.strip()
        # If the name is mostly uppercase, treat as level 1
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7:
            return 1
        return 2

    # -- body cleaning ----------------------------------------------------

    def _clean_body(self, body: str) -> str:
        """Clean a section body for TTS."""
        text = body

        # Remove title/author/affiliation/maketitle (if any remain)
        text = re.sub(r'\\title\{[^}]*\}', '', text)
        text = re.sub(r'\\author\[[^\]]*\]\{[^}]*\}', '', text)
        text = re.sub(r'\\author\{[^}]*\}', '', text)
        text = re.sub(r'\\affil\[[^\]]*\]\{[^}]*\}', '', text)
        text = re.sub(r'\\date\{[^}]*\}', '', text)
        text = re.sub(r'\\maketitle', '', text)

        # Remove labels
        text = re.sub(r'\\label\{[^}]*\}', '', text)

        # Remove citations
        text = re.sub(r'~?\\cite[tp]?\{[^}]*\}', '', text)

        # Remove bibliography
        text = re.sub(r'\\bibliography\{[^}]*\}', '', text)
        text = re.sub(r'\\bibliographystyle\{[^}]*\}', '', text)

        # Remove \url{}
        text = re.sub(r'\\url\{[^}]*\}', '', text)

        # Replace ~ (non-breaking space) early so figure refs match cleanly
        text = text.replace('~', ' ')

        # Remove figure/table/data references
        if not self.keep_figure_refs:
            # Parenthesized compound refs: (Figure 2E, H), (Figure 1C -- L)
            text = re.sub(
                r'\((?:Figures?|Fig\.?)\s*\d+[A-Za-z]?'
                r'(?:\s*[-–,]\s*(?:[A-Za-z]|\d+[A-Za-z]?))*'
                r'(?:\s*(?:and|;|,)\s*(?:Figures?|Fig\.?)\s*\d+[A-Za-z]?'
                r'(?:\s*[-–,]\s*(?:[A-Za-z]|\d+[A-Za-z]?))*)*\)',
                '', text, flags=re.IGNORECASE)
            # Parenthesized table/data refs (including mixed refs)
            text = re.sub(
                r'\((?:Tables?|Data)\s*[S]?\d+[A-Za-z]?'
                r'(?:\s*[-–,]\s*[S]?\d+[A-Za-z]?)*'
                r'(?:[;,]\s*(?:(?:Tables?|Data|Figures?|Fig\.?)\s*[S]?\d+[A-Za-z]?'
                r'(?:\s*[-–,]\s*[S]?\d+[A-Za-z]?)*))*\)',
                '', text, flags=re.IGNORECASE)
            # Parenthesized STAR Methods references
            text = re.sub(r'\(STAR\s+Methods\)', '', text)
            # Inline "see STAR Methods" or "(see STAR Methods)"
            text = re.sub(r'\(?\s*see\s+STAR\s+Methods\s*\)?', '', text, flags=re.IGNORECASE)
            # Parenthesized Supplement/Supplemental Text refs
            text = re.sub(
                r'\(Supplement(?:al)?\s+Text(?:\s*;\s*[^)]+)?\)', '', text, flags=re.IGNORECASE)
            # Inline figure/table/data refs (after parenthesized ones)
            text = re.sub(
                r'(?:Figures?|Fig\.?)\s*\d+[A-Za-z]?(?:\s*[-–,]\s*(?:[A-Za-z]|\d+[A-Za-z]?))*',
                '', text, flags=re.IGNORECASE)
            text = re.sub(
                r'(?:Tables?|Data)\s*[S]?\d+[A-Za-z]?(?:\s*[-–,]\s*[S]?\d+[A-Za-z]?)*',
                '', text, flags=re.IGNORECASE)
            # Supplementary/supplemental refs
            text = re.sub(
                r'(?:Supplementary|Supplemental)\s+(?:Figure|Fig\.?|Table|Text|Data|Material|Information)s?'
                r'\s*[S]?\d*[A-Za-z]?(?:\s*[-–,]\s*[S]?\d*[A-Za-z]?)*',
                '', text, flags=re.IGNORECASE)

        # Handle specific patterns: LA$^4$SR → LA4SR
        text = re.sub(r'LA\$\^\{?4\}?\$SR', 'LA4SR', text)

        # LaTeX quotes: ``text'' → "text", `text' → "text"
        text = re.sub(r"``(.*?)''", r'"\1"', text)
        text = re.sub(r"`(.*?)'", r'"\1"', text)

        # Handle \captionof remnants
        text = re.sub(r'\\captionof\{[^}]*\}\{[^}]*\}', '', text, flags=re.DOTALL)

        # Handle $\sim$NUMBER pattern (approximately N)
        text = re.sub(r'\$\\sim\$\s*', 'approximately ', text)

        # Convert inline math via MathToSpeech
        text = re.sub(
            r'\$([^$]+)\$',
            lambda m: self.math.convert_latex(m.group(1)),
            text)

        # Strip heavy stats if requested (parenthesized statistical detail)
        if self.strip_heavy_stats:
            # e.g., (Mann-Whitney p = 3.4 × 10^{-4}), (n = 1,005), (SHAP importance 0.207)
            text = re.sub(
                r'\([^)]*(?:Mann.Whitney|Wilcoxon|ANOVA|FWER|FDR|Bonferroni|'
                r'Jaccard|rank.biserial|SHAP)[^)]*\)',
                '', text, flags=re.IGNORECASE)

        # LaTeX text commands
        text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\texttt\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textsuperscript\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\textsubscript\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\underline\{([^}]*)\}', r'\1', text)

        # Special LaTeX symbols
        text = text.replace(r'\textminus', '-')
        text = text.replace(r'\textdegree', ' degrees')
        text = text.replace(r'\texttimes', ' times ')
        text = text.replace(r'\textbackslash', '\\')
        text = text.replace(r'\&', 'and')
        text = re.sub(r'\\%', ' percent', text)

        # Escaped space (vs.\ ) and inter-sentence spacing
        text = re.sub(r'\\\s', ' ', text)

        # Thin space, comma-in-number
        text = re.sub(r'\\[,;!]', ' ', text)
        text = re.sub(r'(\d)\{,\}(\d)', r'\1,\2', text)  # 1{,}090 → 1,090

        # Dashes
        text = text.replace('---', ' -- ')
        text = text.replace('--', ' -- ')

        # Line breaks
        text = re.sub(r'\\\\', ' ', text)
        text = re.sub(r'\\newline', ' ', text)
        text = re.sub(r'\\par\b', '\n\n', text)
        text = re.sub(r'\\noindent', '', text)
        text = re.sub(r'\\newpage', '', text)
        text = re.sub(r'\\clearpage', '', text)

        # Remove abstract environment markers
        text = re.sub(r'\\begin\{abstract\}', '', text)
        text = re.sub(r'\\end\{abstract\}', '', text)

        # Strip remaining LaTeX commands (catch-all)
        # First: commands with braced arguments (keep content)
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
        # Then: commands without arguments
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        # Stray braces
        text = re.sub(r'[{}]', '', text)

        return text

    def _final_cleanup(self, text: str) -> str:
        """Final whitespace and punctuation cleanup."""
        # Collapse whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)

        # Punctuation fixes
        text = re.sub(r'\s+([.,;:?!])', r'\1', text)
        text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'\(\s*\)', '', text)  # empty parens
        text = re.sub(r'\[\s*\]', '', text)  # empty brackets
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'([.,;:?!])\s*([.,;:?!])', r'\1', text)  # double punctuation
        text = re.sub(r'\.{2,}', '.', text)

        # Orphaned parenthesized remnants like "(; )" or "(, )"
        text = re.sub(r'\(\s*[;,]\s*\)', '', text)
        text = re.sub(r'\(\s*[;,]', '(', text)
        text = re.sub(r'[;,]\s*\)', ')', text)

        # Clean up "e.g.," and "i.e.," that may have lost their referent
        # (these are fine, just ensure no double spaces)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

# ---------------------------------------------------------------------------
# PDFSalientExtractor — fallback path
# ---------------------------------------------------------------------------

class PDFSalientExtractor:
    """Extract narrative prose from a PDF manuscript for TTS."""

    # Known section headings (upper case, for detection in PDF text)
    HEADING_PATTERNS = [
        'SUMMARY', 'ABSTRACT', 'INTRODUCTION', 'RESULTS', 'DISCUSSION',
        'CONCLUSIONS', 'KEYWORDS', 'GRAPHICAL ABSTRACT',
        'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS',
        'AUTHOR CONTRIBUTIONS', 'DECLARATION OF INTERESTS',
        'DECLARATION OF GENERATIVE AI AND AI-ASSISTED TECHNOLOGIES',
        'SUPPLEMENTAL INFORMATION', 'SUPPLEMENTARY INFORMATION',
        'STAR METHODS', 'METHODS', 'MATERIALS AND METHODS',
        'REFERENCES', 'BIBLIOGRAPHY',
        'DATA AVAILABILITY', 'CODE AVAILABILITY',
        'RESOURCE AVAILABILITY',
    ]

    def __init__(self, *, strip_heavy_stats: bool = False,
                 keep_figure_refs: bool = False):
        self.math = MathToSpeech()
        self.strip_heavy_stats = strip_heavy_stats
        self.keep_figure_refs = keep_figure_refs

        # Compiled patterns (reused from pipeline/pdf_to_text.py)
        self._numeric_citations = re.compile(r'\[\d+(?:\s*[-–,]\s*\d+)*\]')
        self._author_year_citations = re.compile(
            r'\([A-Z][a-zA-Z\-\']+(?:\s+(?:et\s+al\.?|&|and)\s*)?(?:,?\s*\d{4}[a-z]?)\)',
            re.IGNORECASE)
        self._multi_author_citations = re.compile(
            r'\([A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?'
            r'(?:\s*;\s*[A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}[a-z]?)*\)')
        self._urls = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self._dois = re.compile(r'(?:doi:\s*)?10\.\d{4,}/[^\s]+', re.IGNORECASE)
        self._emails = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._figure_refs = re.compile(
            r'\b(?:Figure|Fig\.?)\s*\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?', re.IGNORECASE)
        self._table_refs = re.compile(
            r'\bTables?\s*[S]?\d+[a-z]?(?:\s*[-–]\s*[S]?\d+[a-z]?)?', re.IGNORECASE)
        self._supplementary_refs = re.compile(
            r'\bSupplementary\s+(?:Figure|Fig\.?|Table|Material|Information|Data)s?\s*[S]?\d*',
            re.IGNORECASE)
        self._page_numbers = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
        self._copyright = re.compile(
            r'(?:copyright|\u00a9|©)\s*(?:\d{4})?.*?(?:all rights reserved|license)?',
            re.IGNORECASE)
        self._hyphenation = re.compile(r'(\w+)-\n(\w+)')

        # Line number detection: lines starting with bare numbers 1-999
        # that appear sequentially (common in manuscripts with \linenumbers)
        self._line_number_start = re.compile(r'^(\d{1,3})\s+(.+)$', re.MULTILINE)
        self._standalone_number = re.compile(r'^\d{1,4}$')

    def extract(self, pdf_path: Path, include: list | None = None,
                exclude: list | None = None,
                include_all: bool = False) -> tuple[str, dict]:
        """Return (cleaned_text, stats_dict)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("ERROR: PyMuPDF (fitz) required for PDF extraction. "
                  "Install with: pip install PyMuPDF", file=sys.stderr)
            sys.exit(1)

        inc = include if include is not None else DEFAULT_INCLUDE
        exc = exclude if exclude is not None else DEFAULT_EXCLUDE

        # 1. Extract raw text
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        raw = "\n".join(pages)

        stats = {'raw_chars': len(raw)}

        # 2. Strip line numbers
        text = self._strip_line_numbers(raw)

        # 3. Fix hyphenation at line breaks
        text = self._hyphenation.sub(r'\1\2', text)

        # 4. Detect sections
        sections = self._detect_sections(text)

        # 5. Remove figure captions (multi-line blocks starting with "Figure N.")
        cleaned_sections = OrderedDict()
        for name, body in sections.items():
            body = re.sub(
                r'^Figure\s+\d+[A-Za-z]?\.\s.*?(?=\n\n|\Z)',
                '', body, flags=re.MULTILINE | re.DOTALL)
            body = re.sub(
                r'^Table\s+[S]?\d+\.\s.*?(?=\n\n|\Z)',
                '', body, flags=re.MULTILINE | re.DOTALL)
            cleaned_sections[name] = body

        # 6. Apply section filter
        filtered = OrderedDict()
        parent_included = False
        for name, body in cleaned_sections.items():
            level = 1 if name.upper() == name else 2
            if level == 1:
                parent_included = should_include_section(
                    name, inc, exc, include_all)
                if parent_included:
                    filtered[name] = body
            else:
                if parent_included:
                    upper = name.upper().strip()
                    explicitly_excluded = any(
                        upper == ex or upper.startswith(ex) for ex in exc)
                    if not explicitly_excluded:
                        filtered[name] = body

        stats['total_sections'] = len(cleaned_sections)
        stats['kept_sections'] = len(filtered)
        stats['excluded_sections'] = len(cleaned_sections) - len(filtered)
        stats['section_names'] = list(cleaned_sections.keys())
        stats['kept_names'] = list(filtered.keys())

        # 7. Clean each section
        parts = []
        for name, body in filtered.items():
            heading = name.strip() + '.'
            cleaned = self._clean_body(body)
            if cleaned.strip():
                parts.append(f"\n\n{heading}\n\n{cleaned}")

        result = '\n'.join(parts)

        # Final cleanup
        result = self._final_cleanup(result)
        stats['output_chars'] = len(result)
        stats['output_words'] = len(result.split())

        return result.strip(), stats

    def _strip_line_numbers(self, text: str) -> str:
        """Remove sequential line numbers from manuscript PDFs.

        Two-pass approach:
        1. Identify standalone number lines that form a roughly sequential pattern
        2. Remove them
        """
        lines = text.split('\n')

        # Pass 1: find all standalone number lines and check for sequential pattern
        number_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if self._standalone_number.match(stripped):
                number_indices.append((i, int(stripped)))

        # Check if we have a sequential line-number pattern:
        # At least 10 sequential-ish numbers that mostly increment
        if len(number_indices) < 10:
            return text

        # Count how many are roughly sequential (within +/- 3 of expected)
        sequential_count = 0
        for j in range(1, len(number_indices)):
            prev_num = number_indices[j - 1][1]
            curr_num = number_indices[j][1]
            if 0 < curr_num - prev_num <= 3:
                sequential_count += 1

        # If >60% are sequential, treat them all as line numbers
        if sequential_count / max(len(number_indices) - 1, 1) < 0.6:
            return text

        # Pass 2: remove all standalone number lines that fit the pattern
        line_number_set = set(idx for idx, _ in number_indices)
        cleaned = [line for i, line in enumerate(lines) if i not in line_number_set]

        return '\n'.join(cleaned)

    def _detect_sections(self, text: str) -> OrderedDict:
        """Detect section boundaries from known heading patterns."""
        lines = text.split('\n')
        sections = OrderedDict()
        current_section = '_preamble'
        current_lines = []

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            # Check if this line is a section heading
            matched = False
            for heading in self.HEADING_PATTERNS:
                if upper == heading or (upper.startswith(heading) and len(upper) < len(heading) + 5):
                    # Save previous section
                    if current_lines:
                        sections[current_section] = '\n'.join(current_lines)
                    current_section = stripped
                    current_lines = []
                    matched = True
                    break
            # Also detect subsection-like headings (Title Case, short, on own line)
            if not matched and len(stripped) > 3 and len(stripped) < 120:
                # Heuristic: line is a heading if it's title-case-ish and followed
                # by content (we can't look ahead easily, so use pattern matching)
                if (stripped[0].isupper() and
                        not stripped.endswith('.') and
                        not stripped.endswith(',') and
                        stripped.count(' ') < 15 and
                        re.match(r'^[A-Z][A-Za-z0-9\s:,\-–—()]+$', stripped)):
                    # Could be a subsection heading — only if it matches known sub-patterns
                    # or is distinctly heading-like (all caps or title case with few words)
                    words = stripped.split()
                    if len(words) <= 12 and all(c.isupper() for c in stripped if c.isalpha()):
                        if current_lines:
                            sections[current_section] = '\n'.join(current_lines)
                        current_section = stripped
                        current_lines = []
                        matched = True

            if not matched:
                current_lines.append(line)

        # Don't forget the last section
        if current_lines:
            sections[current_section] = '\n'.join(current_lines)

        return sections

    def _clean_body(self, body: str) -> str:
        """Clean a PDF section body for TTS."""
        text = body

        # Remove citations
        text = self._numeric_citations.sub('', text)
        text = self._multi_author_citations.sub('', text)
        text = self._author_year_citations.sub('', text)

        # Remove URLs, DOIs, emails
        text = self._urls.sub('', text)
        text = self._dois.sub('', text)
        text = self._emails.sub('', text)

        # Remove figure/table references
        if not self.keep_figure_refs:
            text = self._figure_refs.sub('', text)
            text = self._table_refs.sub('', text)
            text = self._supplementary_refs.sub('', text)

        # Remove page number lines
        text = self._page_numbers.sub('', text)

        # Remove copyright
        text = self._copyright.sub('', text)

        # Convert Unicode math
        text = self.math.convert_unicode(text)

        # Strip heavy stats if requested
        if self.strip_heavy_stats:
            text = re.sub(
                r'\([^)]*(?:Mann.Whitney|Wilcoxon|ANOVA|FWER|FDR|Bonferroni|'
                r'Jaccard|rank.biserial|SHAP)[^)]*\)',
                '', text, flags=re.IGNORECASE)

        return text

    def _final_cleanup(self, text: str) -> str:
        """Final whitespace and punctuation cleanup."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)

        # Punctuation
        text = re.sub(r'\s+([.,;:?!])', r'\1', text)
        text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\s+\.', '.', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'([.,;:?!])\s*([.,;:?!])', r'\1', text)

        # Orphaned parens
        text = re.sub(r'\(\s*[;,]\s*\)', '', text)
        text = re.sub(r'\(\s*[;,]', '(', text)
        text = re.sub(r'[;,]\s*\)', ')', text)

        # Short lines (artifacts from PDF extraction)
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 3:
                continue
            if stripped.isdigit():
                continue
            cleaned.append(stripped)
        text = '\n'.join(cleaned)

        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract narrative prose from scientific manuscripts for TTS.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s main.tex                          # LaTeX → main.tts.txt
  %(prog)s main.pdf                          # PDF → main.tts.txt
  %(prog)s main.tex -o output.tts.txt        # custom output path
  %(prog)s main.tex --include-sections ALL   # keep all sections
  %(prog)s main.tex --preview 500            # show first 500 chars
  %(prog)s main.tex --strip-heavy-stats      # remove stat parentheticals
  %(prog)s main.tex --keep-figure-refs       # keep figure/table references
""")
    parser.add_argument('input', help='Input .tex or .pdf file')
    parser.add_argument('-o', '--output', help='Output file path (default: {stem}.tts.txt)')
    parser.add_argument('--include-sections',
                        help='Comma-separated sections to include (or ALL)')
    parser.add_argument('--exclude-sections',
                        help='Comma-separated additional sections to exclude')
    parser.add_argument('--preview', type=int, metavar='N',
                        help='Print first N characters to stdout instead of writing file')
    parser.add_argument('--strip-heavy-stats', action='store_true',
                        help='Remove parenthesized statistical details')
    parser.add_argument('--keep-figure-refs', action='store_true',
                        help='Keep figure/table/data references in text')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path.cwd() / (input_path.stem + '.tts.txt')

    # Parse section overrides
    include_all = False
    include = None
    exclude = None

    if args.include_sections:
        if args.include_sections.upper() == 'ALL':
            include_all = True
        else:
            include = [s.strip().upper() for s in args.include_sections.split(',')]

    if args.exclude_sections:
        exclude_extra = [s.strip().upper() for s in args.exclude_sections.split(',')]
        exclude = DEFAULT_EXCLUDE + exclude_extra
    else:
        exclude = None  # use defaults

    # Dispatch based on file type
    suffix = input_path.suffix.lower()
    if suffix == '.tex':
        print(f"[LaTeX mode] {input_path}", file=sys.stderr)
        tex = input_path.read_text(encoding='utf-8')
        extractor = LaTeXSalientExtractor(
            strip_heavy_stats=args.strip_heavy_stats,
            keep_figure_refs=args.keep_figure_refs)
        result, stats = extractor.extract(
            tex, include=include, exclude=exclude, include_all=include_all)

    elif suffix == '.pdf':
        print(f"[PDF mode] {input_path}", file=sys.stderr)
        extractor = PDFSalientExtractor(
            strip_heavy_stats=args.strip_heavy_stats,
            keep_figure_refs=args.keep_figure_refs)
        result, stats = extractor.extract(
            input_path, include=include, exclude=exclude,
            include_all=include_all)

    else:
        print(f"ERROR: Unsupported file type: {suffix} (expected .tex or .pdf)",
              file=sys.stderr)
        sys.exit(1)

    # Report stats
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"EXTRACTION STATS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Sections found:    {stats['total_sections']}", file=sys.stderr)
    print(f"Sections kept:     {stats['kept_sections']}", file=sys.stderr)
    print(f"Sections excluded: {stats['excluded_sections']}", file=sys.stderr)
    print(f"Output characters: {stats['output_chars']}", file=sys.stderr)
    print(f"Output words:      ~{stats['output_words']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Kept: {', '.join(stats['kept_names'])}", file=sys.stderr)
    excluded_names = [n for n in stats['section_names'] if n not in stats['kept_names']]
    if excluded_names:
        print(f"Excluded: {', '.join(excluded_names)}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Output
    if args.preview:
        preview = result[:args.preview]
        if len(result) > args.preview:
            preview += f"\n\n... [{len(result) - args.preview} more characters]"
        print(preview)
    else:
        output_path.write_text(result, encoding='utf-8')
        print(f"\nWritten to: {output_path}", file=sys.stderr)
        print(f"Ready for: python deep_voice_tts.py {output_path}", file=sys.stderr)

if __name__ == '__main__':
    main()
