[
  {
    "id": 1,
    "category": "setup",
    "description": "Install PyMuPDF (fitz) for PDF text extraction",
    "passes": true,
    "acceptance": "python -c 'import fitz; print(fitz.version)' runs without error"
  },
  {
    "id": 2,
    "category": "code",
    "description": "Create paper_search.py - query CrossRef and PubMed APIs with plain English query, return ranked list of papers with DOI/PMID/title",
    "passes": true,
    "acceptance": "python paper_search.py 'biomimetic concrete' --papers 3 outputs JSON with paper metadata",
    "spec": {
      "input": "query string, --papers N",
      "output": "papers.json with [{doi, pmid, title, authors, year, source}]",
      "apis": ["CrossRef /works", "PubMed esearch+esummary"],
      "features": ["deduplication by DOI", "rate limiting 1s between requests"],
      "reference": "Adapt API patterns from paper_fetcher_20251101.py search_crossref/search_pubmed"
    }
  },
  {
    "id": 3,
    "category": "code",
    "description": "Create paper_download.py - download PDFs from PMC/Unpaywall/EuropePMC using papers.json",
    "passes": true,
    "acceptance": "python paper_download.py papers.json downloads at least 1 PDF",
    "spec": {
      "input": "papers.json from paper_search.py",
      "output": "downloads/*.pdf + download_status.json",
      "sources": ["PMC (via pmcid)", "Unpaywall (via doi)", "Europe PMC"],
      "features": ["skip already downloaded", "track success/failure per paper"],
      "reference": "Copy download_from_pmc/unpaywall/europepmc from paper_fetcher_20251101.py"
    }
  },
  {
    "id": 4,
    "category": "code",
    "description": "Create pdf_to_text.py - extract and clean PDF text for TTS readability",
    "passes": true,
    "acceptance": "python pdf_to_text.py downloads/ texts/ creates readable .txt files",
    "spec": {
      "input": "directory of PDFs",
      "output": "directory of .txt files",
      "cleaning": [
        "remove citations like [1], [1,2], [1-3]",
        "remove author-year citations like (Smith 2020), (Smith et al., 2020)",
        "remove URLs and DOIs",
        "remove page numbers and headers/footers",
        "collapse multiple whitespace",
        "remove figure/table references like 'Figure 1', 'Table 2'"
      ]
    }
  },
  {
    "id": 5,
    "category": "code",
    "description": "Create paper_to_audio.py - unified CLI that chains search->download->text->TTS",
    "passes": true,
    "acceptance": "python paper_to_audio.py 'test query' --papers 1 produces MP3 output",
    "spec": {
      "usage": "python paper_to_audio.py 'query' --papers N [--voice p240] [--output dir] [--keep-pdfs] [--keep-text]",
      "flow": "paper_search.py -> paper_download.py -> pdf_to_text.py -> deep_voice_tts.py",
      "features": ["subprocess calls to other scripts", "progress messages", "error handling"]
    }
  },
  {
    "id": 6,
    "category": "test",
    "description": "Test paper_search.py with real query and verify JSON output structure",
    "passes": true,
    "acceptance": "Query returns valid JSON with expected fields, no API errors"
  },
  {
    "id": 7,
    "category": "test",
    "description": "Test full pipeline end-to-end with a real query",
    "passes": true,
    "acceptance": "python paper_to_audio.py 'bioremediation' --papers 1 produces working MP3"
  },
  {
    "id": 8,
    "category": "code",
    "description": "Add progress bars using tqdm throughout the pipeline",
    "passes": true,
    "acceptance": "All scripts show progress bars for long operations",
    "spec": {
      "install": "pip install tqdm",
      "locations": ["paper download loop", "PDF processing loop", "TTS chunk processing"]
    }
  },
  {
    "id": 9,
    "category": "code",
    "description": "Add cleanup logic to paper_to_audio.py - remove PDFs and text after audio generated (unless --keep flags)",
    "passes": false,
    "acceptance": "By default, intermediate files are deleted after successful audio generation"
  },
  {
    "id": 10,
    "category": "docs",
    "description": "Update main README with pipeline usage examples",
    "passes": false,
    "acceptance": "README.md has section documenting paper_to_audio.py usage"
  }
]
