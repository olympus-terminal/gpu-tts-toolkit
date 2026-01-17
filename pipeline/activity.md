# Pipeline Development Activity Log

## 2026-01-17

- Created RALPH loop structure for paper-to-audio pipeline
- Defined 10 tasks in plan.md with detailed specs and acceptance criteria
- Created comprehensive PROMPT.md with architecture diagram and API reference
- Installed PyMuPDF 1.26.7 in tts-app conda environment (Task 1 complete)
- Ready to begin implementation of paper_search.py (Task 2)
- Created paper_search.py with CrossRef and PubMed API integration
- Tested successfully: `python paper_search.py 'biomimetic concrete' --papers 3` returns valid JSON
- Features implemented: deduplication by DOI, rate limiting, merged results from both APIs
- Task 2 complete
- Created paper_download.py with PMC/Unpaywall/EuropePMC download support
- Features: PMID-to-PMCID conversion, skip already downloaded, progress tracking, status.json output
- Tested successfully: downloaded 1 PDF from Europe PMC using existing papers.json
- Task 3 complete
- Created pdf_to_text.py with PyMuPDF text extraction and comprehensive TTS cleaning
- Cleaning features: numeric citations, author-year citations, URLs, DOIs, figure/table refs, page numbers, email addresses, copyright notices, equation refs, whitespace normalization, hyphenation rejoining
- Tested successfully: `python pdf_to_text.py downloads/ texts/` converted 1 PDF (52477 chars -> 50601 chars cleaned)
- Task 4 complete
- Created paper_to_audio.py - unified CLI orchestrating search->download->text->TTS
- Features: subprocess calls to each pipeline stage, progress messages, error handling, cleanup logic
- Fixed absolute path issue for TTS stage when running from output directory
- Tested successfully: `python paper_to_audio.py 'SARS-CoV-2 open access' --papers 2 --voice p240` produced 2 MP3 files (30MB + 54MB)
- Intermediate files cleaned up automatically (PDFs, text files, papers.json)
- Task 5 complete

---
