# Paper-to-Audio Pipeline - RALPH Loop Instructions

**Working directory:** /home/drn/Documents/projects/tts-app-gpu-toolkit/pipeline/
**Conda environment:** tts-app (activate with: `source /home/drn/miniconda3/etc/profile.d/conda.sh && conda activate tts-app`)
**TTS script:** ../deep_voice_tts.py

## Reference Code (REUSE THIS!)

1. **paper_fetcher_20251101.py** - `/home/drn/Documents/projects/paper-fetcher/paper_fetcher_20251101.py`
   - `search_crossref(author, year, description)` - CrossRef API search
   - `search_pubmed(author, year, description)` - PubMed esearch/esummary
   - `download_from_pmc(pmid)` - PMC PDF download
   - `download_from_unpaywall(doi)` - Unpaywall OA download
   - `download_from_europepmc(pmid, doi)` - Europe PMC download

2. **ris_fetcher_expanded.py** - `/home/drn/Documents/projects/paper-fetcher/ris_fetcher_expanded.py`
   - `fetch_from_arxiv(arxiv_id)` - arXiv API (Atom XML)
   - `fetch_from_biorxiv(doi)` - bioRxiv/medRxiv API
   - `fetch_from_datacite(doi)` - Zenodo/Figshare/datasets
   - `fetch_from_nasa(identifier)` - NASA Technical Reports
   - General patterns for API requests, rate limiting, error handling

## Instructions

1. Read activity.md to understand current state
2. Read plan.md and choose the single highest priority task with `"passes": false`
3. Work on exactly ONE task:
   - Implement the required code/config
   - Test that it works (run python scripts, verify output)
   - Verify no errors
4. After completing:
   - Append dated progress entry to activity.md
   - Update task's `"passes"` to `true` in plan.md
   - Make one git commit with descriptive message

**ONLY WORK ON ONE TASK PER ITERATION.**

When ALL tasks have `"passes": true`, output: `<promise>COMPLETE</promise>`

---

## Architecture Reference

```
Query String (e.g., "biomimetic concrete")
         │
         ▼
┌─────────────────────────────────────────┐
│  paper_search.py                         │
│  - Query CrossRef API with search term   │
│  - Query PubMed API with search term     │
│  - Merge and rank results by relevance   │
│  - Return top N papers with DOI/PMID     │
│  - Output: papers.json                   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  paper_download.py                       │
│  - Read papers.json                      │
│  - For each paper, try download from:    │
│    1. PubMed Central (PMC)               │
│    2. Unpaywall (OA aggregator)          │
│    3. Europe PMC                         │
│  - Save PDFs to downloads/ directory     │
│  - Track which succeeded/failed          │
│  - Output: downloaded PDFs + status.json │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  pdf_to_text.py                          │
│  - Read each PDF with PyMuPDF (fitz)     │
│  - Extract text, skip images/tables      │
│  - Clean for TTS:                        │
│    * Remove citations [1], (Author 2024) │
│    * Remove URLs, DOIs                   │
│    * Remove page numbers, headers        │
│    * Expand common abbreviations         │
│  - Output: text files in texts/          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  paper_to_audio.py (main CLI)            │
│  - Orchestrates all stages               │
│  - Calls: search → download → text → TTS │
│  - Uses ../deep_voice_tts.py for audio   │
│  - Progress bars and status messages     │
│  - Optional cleanup of intermediates     │
│  - Output: MP3 files in output/          │
└─────────────────────────────────────────┘
```

## API Reference

### CrossRef Search
```python
import requests
url = "https://api.crossref.org/works"
params = {"query": "biomimetic concrete", "rows": 10}
response = requests.get(url, params=params)
# Returns: {"message": {"items": [{"DOI": "...", "title": [...], ...}]}}
```

### PubMed Search
```python
# Step 1: ESearch to get PMIDs
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {"db": "pubmed", "term": "biomimetic concrete", "retmax": 10, "retmode": "json"}

# Step 2: ESummary to get metadata
url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
params = {"db": "pubmed", "id": "12345678", "retmode": "json"}
```

### PDF Download Sources (in order of preference)
1. **PMC**: `https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/`
2. **Unpaywall**: `https://api.unpaywall.org/v2/{doi}?email=your@email.com`
3. **Europe PMC**: `https://europepmc.org/articles/{pmcid}?pdf=render`

### Text Extraction
```python
import fitz  # PyMuPDF
doc = fitz.open("paper.pdf")
for page in doc:
    text = page.get_text()
```

### TTS Generation
```bash
python ../deep_voice_tts.py input.txt --voice p240 --output output.mp3
```
