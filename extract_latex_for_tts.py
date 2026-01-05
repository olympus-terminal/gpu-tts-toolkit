#!/usr/bin/env python3
"""
Extract readable text from LaTeX manuscript for text-to-speech.
Strips figures, tables, equations, references, and bureaucratic content.
"""

import re
import sys
from pathlib import Path

def extract_tts_text(tex_content):
    """Extract readable text from LaTeX content."""

    text = tex_content

    # Remove everything before \begin{document}
    if r'\begin{document}' in text:
        text = text.split(r'\begin{document}', 1)[1]

    # Remove everything after \end{document}
    if r'\end{document}' in text:
        text = text.split(r'\end{document}', 1)[0]

    # Remove bibliography
    text = re.sub(r'\\bibliography\{[^}]*\}', '', text)
    text = re.sub(r'\\bibliographystyle\{[^}]*\}', '', text)

    # Remove figure environments entirely
    text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', text, flags=re.DOTALL)

    # Remove table environments entirely
    text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', text, flags=re.DOTALL)

    # Remove tabular environments
    text = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '', text, flags=re.DOTALL)

    # Remove title, author, affiliation blocks
    text = re.sub(r'\\title\{[^}]*\}', '', text)
    text = re.sub(r'\\author\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'\\author\{[^}]*\}', '', text)
    text = re.sub(r'\\affil\[[^\]]*\]\{[^}]*\}', '', text)
    text = re.sub(r'\\date\{[^}]*\}', '', text)
    text = re.sub(r'\\maketitle', '', text)

    # Remove abstract environment but keep content
    text = re.sub(r'\\begin\{abstract\}', '', text)
    text = re.sub(r'\\end\{abstract\}', '', text)

    # Convert section headers to readable form
    text = re.sub(r'\\section\*?\{([^}]*)\}', r'\n\n\1.\n\n', text)
    text = re.sub(r'\\subsection\*?\{([^}]*)\}', r'\n\n\1.\n\n', text)
    text = re.sub(r'\\subsubsection\*?\{([^}]*)\}', r'\n\n\1.\n\n', text)

    # Remove labels and refs, keeping readable references
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\(Figure~?\\ref\{[^}]*\}\)', '', text)  # Remove figure references in parens
    text = re.sub(r'Figure~?\\ref\{[^}]*\}', 'the figure', text)
    text = re.sub(r'\(Table~?\\ref\{[^}]*\}\)', '', text)
    text = re.sub(r'Table~?\\ref\{[^}]*\}', 'the table', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)

    # Remove citations - they break up the flow for TTS
    text = re.sub(r'~?\\cite\{[^}]*\}', '', text)

    # Convert math to readable form
    # Inline math
    text = re.sub(r'\$([^$]+)\$', lambda m: convert_math(m.group(1)), text)
    # Display math
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', text, flags=re.DOTALL)

    # Convert common LaTeX commands
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textsuperscript\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textsubscript\{([^}]*)\}', r'\1', text)

    # Greek letters
    text = re.sub(r'\\alpha', 'alpha', text)
    text = re.sub(r'\\beta', 'beta', text)
    text = re.sub(r'\\gamma', 'gamma', text)
    text = re.sub(r'\\delta', 'delta', text)
    text = re.sub(r'\\kappa', 'kappa', text)
    text = re.sub(r'\\lambda', 'lambda', text)
    text = re.sub(r'\\mu', 'mu', text)
    text = re.sub(r'\\sigma', 'sigma', text)
    text = re.sub(r'\\omega', 'omega', text)

    # Common symbols
    text = re.sub(r'\\times', ' times ', text)
    text = re.sub(r'\\pm', ' plus or minus ', text)
    text = re.sub(r'\\geq?', ' greater than or equal to ', text)
    text = re.sub(r'\\leq?', ' less than or equal to ', text)
    text = re.sub(r'\\approx', ' approximately ', text)
    text = re.sub(r'\\sim', ' approximately ', text)
    text = re.sub(r'\\%', ' percent', text)

    # Units and formatting
    text = re.sub(r'~', ' ', text)  # Non-breaking spaces
    text = re.sub(r'\\,', ' ', text)  # Thin spaces
    text = re.sub(r'\\;', ' ', text)
    text = re.sub(r'\\!', '', text)
    text = re.sub(r'\\\\', ' ', text)  # Line breaks
    text = re.sub(r'\\newline', ' ', text)
    text = re.sub(r'\\par', '\n\n', text)

    # Remove remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', text)  # Commands with args
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)  # Commands without args
    text = re.sub(r'\{|\}', '', text)  # Stray braces

    # Clean up comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)  # Leading/trailing spaces

    # Clean up punctuation issues
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)  # Space before punctuation
    text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)  # Missing space after punctuation
    text = re.sub(r'\(\s*\)', '', text)  # Empty parentheses
    text = re.sub(r'\s+\)', ')', text)  # Space before closing paren
    text = re.sub(r'\(\s+', '(', text)  # Space after opening paren

    return text.strip()


def convert_math(math_str):
    """Convert simple math expressions to readable text."""
    # Scientific notation
    math_str = re.sub(r'(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?',
                      lambda m: f"{m.group(1)} times 10 to the {m.group(2)}", math_str)

    # Subscripts/superscripts
    math_str = re.sub(r'_\{([^}]*)\}', r' \1', math_str)
    math_str = re.sub(r'_(\w)', r' \1', math_str)
    math_str = re.sub(r'\^\{([^}]*)\}', r' to the \1', math_str)
    math_str = re.sub(r'\^(\w)', r' to the \1', math_str)

    # Fractions
    math_str = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1 over \2', math_str)

    # Comparisons
    math_str = re.sub(r'<', ' less than ', math_str)
    math_str = re.sub(r'>', ' greater than ', math_str)
    math_str = re.sub(r'=', ' equals ', math_str)

    # Clean up
    math_str = re.sub(r'\\[a-zA-Z]+', '', math_str)
    math_str = re.sub(r'[{}]', '', math_str)

    return math_str.strip()


def main():
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = Path(__file__).parent / "manuscript_article.tex"

    if not input_file.exists():
        print(f"Error: {input_file} not found", file=sys.stderr)
        sys.exit(1)

    tex_content = input_file.read_text(encoding='utf-8')
    tts_text = extract_tts_text(tex_content)

    # Output to file
    output_file = input_file.with_suffix('.tts.txt')
    output_file.write_text(tts_text, encoding='utf-8')

    print(f"Extracted {len(tts_text)} characters to {output_file}")
    print(f"Word count: ~{len(tts_text.split())}")


if __name__ == "__main__":
    main()
