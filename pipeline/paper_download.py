#!/usr/bin/env python3
"""
Paper Download - Download PDFs from open access sources.

Uses papers.json from paper_search.py and attempts to download PDFs from:
1. PubMed Central (PMC) - via PMID lookup
2. Unpaywall - legal open access aggregator
3. Europe PMC - European mirror with additional content

Usage:
    python paper_download.py papers.json
    python paper_download.py papers.json --output downloads/

Output:
    downloads/*.pdf + download_status.json
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path
from typing import Optional, Dict, List
from urllib.parse import quote
import re
from tqdm import tqdm


class PaperDownloader:
    """Download academic paper PDFs from open access sources."""

    def __init__(self, output_dir: str = "downloads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PaperDownloader/1.0 (Academic Research Tool)'
        })

        # Email for Unpaywall API (required)
        self.email = "researcher@example.com"

        # Track download status
        self.status = []

    def download_from_pmc(self, pmid: str) -> Optional[bytes]:
        """
        Try to download PDF from PubMed Central (open access).
        First converts PMID to PMCID, then downloads PDF.
        """
        # Convert PMID to PMCID
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"

        try:
            response = self.session.get(pmc_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = data.get('records', [])
            if records and 'pmcid' in records[0]:
                pmcid = records[0]['pmcid']

                # Download PDF from PMC
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

                pdf_response = self.session.get(pdf_url, timeout=60)
                if pdf_response.status_code == 200:
                    content_type = pdf_response.headers.get('content-type', '')
                    if 'pdf' in content_type.lower():
                        print(f"  [SUCCESS] Downloaded from PMC ({pmcid})", file=sys.stderr)
                        return pdf_response.content

        except Exception as e:
            print(f"  [DEBUG] PMC download failed: {e}", file=sys.stderr)

        return None

    def download_from_unpaywall(self, doi: str) -> Optional[bytes]:
        """
        Try to download PDF from Unpaywall (legal open access aggregator).
        """
        url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()

                # Check for open access PDF
                if data.get('is_oa'):
                    best_oa = data.get('best_oa_location', {})
                    pdf_url = best_oa.get('url_for_pdf')

                    if pdf_url:
                        pdf_response = self.session.get(pdf_url, timeout=60)
                        if pdf_response.status_code == 200:
                            print(f"  [SUCCESS] Downloaded from Unpaywall", file=sys.stderr)
                            return pdf_response.content

        except Exception as e:
            print(f"  [DEBUG] Unpaywall download failed: {e}", file=sys.stderr)

        return None

    def download_from_europepmc(self, pmid: str = None, doi: str = None) -> Optional[bytes]:
        """
        Try to download PDF from Europe PMC.
        """
        if pmid:
            query = f"EXT_ID:{pmid}"
        elif doi:
            query = f"DOI:{doi}"
        else:
            return None

        search_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={quote(query)}&format=json"

        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = data.get('resultList', {}).get('result', [])
            if results and results[0].get('isOpenAccess') == 'Y':
                pmcid = results[0].get('pmcid')
                if pmcid:
                    pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"

                    pdf_response = self.session.get(pdf_url, timeout=60)
                    if pdf_response.status_code == 200:
                        print(f"  [SUCCESS] Downloaded from Europe PMC", file=sys.stderr)
                        return pdf_response.content

        except Exception as e:
            print(f"  [DEBUG] Europe PMC download failed: {e}", file=sys.stderr)

        return None

    def create_filename(self, paper: Dict) -> str:
        """
        Create a safe filename from paper metadata.
        """
        # Use first author if available
        authors = paper.get('authors', [])
        first_author = authors[0] if authors else 'Unknown'
        # Clean author name - get last name only
        first_author = first_author.split()[-1] if first_author else 'Unknown'
        first_author = re.sub(r'[^\w\s-]', '', first_author)[:20]

        # Get year
        year = paper.get('year', 'YYYY')

        # Get title snippet
        title = paper.get('title', 'Untitled')
        title_clean = re.sub(r'[^\w\s-]', '', title)[:40]
        title_clean = title_clean.replace(' ', '_')

        filename = f"{first_author}_{year}_{title_clean}.pdf"
        return filename

    def download_paper(self, paper: Dict) -> Dict:
        """
        Attempt to download a single paper from all available sources.
        Returns status dict with success/failure info.
        """
        doi = paper.get('doi')
        pmid = paper.get('pmid')
        title = paper.get('title', 'Unknown')[:60]

        print(f"\n[PAPER] {title}...", file=sys.stderr)
        if doi:
            print(f"  DOI: {doi}", file=sys.stderr)
        if pmid:
            print(f"  PMID: {pmid}", file=sys.stderr)

        pdf_content = None
        source = None

        # Try PMC first (best quality, definitely open access)
        if pmid:
            print("  [TRYING] PubMed Central...", file=sys.stderr)
            pdf_content = self.download_from_pmc(pmid)
            if pdf_content:
                source = 'pmc'

        # Try Unpaywall (legal open access aggregator)
        if not pdf_content and doi:
            print("  [TRYING] Unpaywall...", file=sys.stderr)
            time.sleep(0.5)  # Rate limit
            pdf_content = self.download_from_unpaywall(doi)
            if pdf_content:
                source = 'unpaywall'

        # Try Europe PMC
        if not pdf_content:
            print("  [TRYING] Europe PMC...", file=sys.stderr)
            time.sleep(0.5)  # Rate limit
            pdf_content = self.download_from_europepmc(pmid=pmid, doi=doi)
            if pdf_content:
                source = 'europepmc'

        # Create status entry
        status_entry = {
            'doi': doi,
            'pmid': pmid,
            'title': paper.get('title'),
            'success': False,
            'source': None,
            'filename': None
        }

        if pdf_content:
            filename = self.create_filename(paper)
            filepath = self.output_dir / filename

            # Handle filename collision
            counter = 1
            while filepath.exists():
                base = filename.rsplit('.', 1)[0]
                filepath = self.output_dir / f"{base}_{counter}.pdf"
                counter += 1

            with open(filepath, 'wb') as f:
                f.write(pdf_content)

            print(f"  [SAVED] {filepath.name}", file=sys.stderr)

            status_entry['success'] = True
            status_entry['source'] = source
            status_entry['filename'] = filepath.name
        else:
            print(f"  [FAILED] Could not download (may not be open access)", file=sys.stderr)

        return status_entry

    def is_already_downloaded(self, paper: Dict) -> Optional[str]:
        """
        Check if paper was already downloaded (by DOI match in status file).
        Returns filename if already downloaded, None otherwise.
        """
        status_file = self.output_dir / 'download_status.json'
        if not status_file.exists():
            return None

        try:
            with open(status_file, 'r') as f:
                existing_status = json.load(f)

            doi = paper.get('doi')
            pmid = paper.get('pmid')

            for entry in existing_status:
                if entry.get('success'):
                    if doi and entry.get('doi') == doi:
                        return entry.get('filename')
                    if pmid and entry.get('pmid') == pmid:
                        return entry.get('filename')
        except:
            pass

        return None

    def download_all(self, papers: List[Dict]) -> List[Dict]:
        """
        Download all papers from the list.
        Skips already downloaded papers.
        """
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"PAPER DOWNLOADER", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Papers to process: {len(papers)}", file=sys.stderr)
        print(f"Output directory: {self.output_dir}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        for paper in tqdm(papers, desc="Downloading papers", file=sys.stderr):
            # Check if already downloaded
            existing = self.is_already_downloaded(paper)
            if existing:
                print(f"\n[SKIP] Already downloaded: {existing}", file=sys.stderr)
                self.status.append({
                    'doi': paper.get('doi'),
                    'pmid': paper.get('pmid'),
                    'title': paper.get('title'),
                    'success': True,
                    'source': 'cached',
                    'filename': existing
                })
                continue

            status = self.download_paper(paper)
            self.status.append(status)

            # Rate limit between papers
            time.sleep(1)

        # Summary
        successful = sum(1 for s in self.status if s['success'])
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"SUMMARY", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Total papers: {len(papers)}", file=sys.stderr)
        print(f"Successfully downloaded: {successful}", file=sys.stderr)
        print(f"Failed (not open access): {len(papers) - successful}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        return self.status


def main():
    parser = argparse.ArgumentParser(
        description='Download PDFs from open access sources'
    )
    parser.add_argument('input', help='Input JSON file from paper_search.py (e.g., papers.json)')
    parser.add_argument('--output', '-o', type=str, default='downloads',
                        help='Output directory for PDFs (default: downloads/)')

    args = parser.parse_args()

    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r') as f:
        papers = json.load(f)

    if not papers:
        print("[ERROR] No papers in input file", file=sys.stderr)
        sys.exit(1)

    # Download
    downloader = PaperDownloader(output_dir=args.output)
    status = downloader.download_all(papers)

    # Write status file
    status_file = Path(args.output) / 'download_status.json'
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    print(f"\n[INFO] Status written to {status_file}", file=sys.stderr)

    # Print status to stdout
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
