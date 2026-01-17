#!/usr/bin/env python3
"""
Paper Search - Query CrossRef and PubMed APIs for academic papers.

Usage:
    python paper_search.py 'biomimetic concrete' --papers 5

Output:
    papers.json with [{doi, pmid, title, authors, year, source}]
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path
from typing import Optional, Dict, List

class PaperSearcher:
    """Search for academic papers using CrossRef and PubMed APIs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PaperSearch/1.0 (Academic Research Tool)'
        })

    def search_crossref(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search CrossRef API with plain text query.
        Returns list of paper metadata dicts.
        """
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': max_results,
            'select': 'DOI,title,author,published-print,published-online'
        }

        results = []
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            for item in data.get('message', {}).get('items', []):
                # Extract year from published-print or published-online
                year = None
                for date_field in ['published-print', 'published-online']:
                    if date_field in item:
                        date_parts = item[date_field].get('date-parts', [[None]])
                        if date_parts and date_parts[0]:
                            year = date_parts[0][0]
                            break

                # Extract authors
                authors = []
                for author in item.get('author', []):
                    name_parts = []
                    if author.get('given'):
                        name_parts.append(author['given'])
                    if author.get('family'):
                        name_parts.append(author['family'])
                    if name_parts:
                        authors.append(' '.join(name_parts))

                # Build result
                result = {
                    'doi': item.get('DOI'),
                    'pmid': None,  # CrossRef doesn't have PMIDs
                    'title': item.get('title', [''])[0] if item.get('title') else '',
                    'authors': authors,
                    'year': year,
                    'source': 'crossref'
                }
                results.append(result)

        except requests.RequestException as e:
            print(f"[WARNING] CrossRef search error: {e}", file=sys.stderr)
        except (KeyError, ValueError) as e:
            print(f"[WARNING] CrossRef parse error: {e}", file=sys.stderr)

        return results

    def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search PubMed API with plain text query.
        Returns list of paper metadata dicts.
        """
        results = []

        # Step 1: ESearch to get PMIDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }

        try:
            response = self.session.get(esearch_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pmids = data.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return results

            # Rate limit between API calls
            time.sleep(0.5)

            # Step 2: ESummary to get metadata for all PMIDs
            esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'json'
            }

            response = self.session.get(esummary_url, params=params, timeout=30)
            response.raise_for_status()
            summary = response.json()

            for pmid in pmids:
                article = summary.get('result', {}).get(pmid, {})
                if not article or 'error' in article:
                    continue

                # Extract DOI from articleids
                doi = None
                for aid in article.get('articleids', []):
                    if isinstance(aid, dict) and aid.get('idtype') == 'doi':
                        doi = aid.get('value')
                        break

                # Extract authors
                authors = []
                for author in article.get('authors', []):
                    if isinstance(author, dict) and author.get('name'):
                        authors.append(author['name'])

                # Extract year from pubdate
                year = None
                pubdate = article.get('pubdate', '')
                if pubdate:
                    # pubdate format is usually "YYYY Mon" or "YYYY"
                    year_match = pubdate.split()[0] if pubdate else None
                    if year_match and year_match.isdigit():
                        year = int(year_match)

                result = {
                    'doi': doi,
                    'pmid': pmid,
                    'title': article.get('title', ''),
                    'authors': authors,
                    'year': year,
                    'source': 'pubmed'
                }
                results.append(result)

        except requests.RequestException as e:
            print(f"[WARNING] PubMed search error: {e}", file=sys.stderr)
        except (KeyError, ValueError) as e:
            print(f"[WARNING] PubMed parse error: {e}", file=sys.stderr)

        return results

    def merge_results(self, crossref_results: List[Dict], pubmed_results: List[Dict]) -> List[Dict]:
        """
        Merge results from CrossRef and PubMed, deduplicating by DOI.
        PubMed results are preferred when there's a DOI collision since they have PMIDs.
        """
        # Index by DOI for deduplication
        by_doi = {}
        no_doi = []

        # Add PubMed results first (preferred since they have PMIDs)
        for paper in pubmed_results:
            doi = paper.get('doi')
            if doi:
                by_doi[doi.lower()] = paper
            else:
                no_doi.append(paper)

        # Add CrossRef results, merging PMID if we have a match
        for paper in crossref_results:
            doi = paper.get('doi')
            if doi:
                doi_lower = doi.lower()
                if doi_lower in by_doi:
                    # Merge: keep PubMed entry but enrich if needed
                    existing = by_doi[doi_lower]
                    if not existing.get('authors') and paper.get('authors'):
                        existing['authors'] = paper['authors']
                else:
                    by_doi[doi_lower] = paper
            else:
                no_doi.append(paper)

        # Combine and return
        merged = list(by_doi.values()) + no_doi
        return merged

    def search(self, query: str, max_papers: int = 5) -> List[Dict]:
        """
        Search both CrossRef and PubMed, merge and rank results.
        """
        print(f"[INFO] Searching for: {query}", file=sys.stderr)

        # Search CrossRef
        print("[INFO] Querying CrossRef...", file=sys.stderr)
        crossref_results = self.search_crossref(query, max_results=max_papers * 2)
        print(f"[INFO] Found {len(crossref_results)} results from CrossRef", file=sys.stderr)

        # Rate limit between API providers
        time.sleep(1)

        # Search PubMed
        print("[INFO] Querying PubMed...", file=sys.stderr)
        pubmed_results = self.search_pubmed(query, max_results=max_papers * 2)
        print(f"[INFO] Found {len(pubmed_results)} results from PubMed", file=sys.stderr)

        # Merge and deduplicate
        merged = self.merge_results(crossref_results, pubmed_results)
        print(f"[INFO] Merged to {len(merged)} unique papers", file=sys.stderr)

        # Return top N
        return merged[:max_papers]

def main():
    parser = argparse.ArgumentParser(
        description='Search for academic papers using CrossRef and PubMed APIs'
    )
    parser.add_argument('query', help='Search query (e.g., "biomimetic concrete")')
    parser.add_argument('--papers', '-n', type=int, default=5,
                        help='Number of papers to return (default: 5)')
    parser.add_argument('--output', '-o', type=str, default='papers.json',
                        help='Output JSON file (default: papers.json)')

    args = parser.parse_args()

    searcher = PaperSearcher()
    results = searcher.search(args.query, args.papers)

    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Wrote {len(results)} papers to {output_path}", file=sys.stderr)

    # Also print to stdout for piping
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
