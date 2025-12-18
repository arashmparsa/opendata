#!/usr/bin/env python3
"""
Visualization-Focused Data Downloader
======================================
Downloads recent (last 6 months) census, demographics, and infrastructure
data from data.gov for creating visualizations.

Minimal footprint: ~10 datasets, 1 CSV each, ~50-200 MB total

Usage:
    python viz_downloader.py

Requirements:
    pip install requests tqdm
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class VizDataDownloader:
    """Minimal downloader for visualization-ready data."""

    BASE_URL = "https://catalog.data.gov/api/3/action"

    def __init__(self, output_dir: str = "./viz_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.stats = {'downloaded': 0, 'failed': 0, 'total_mb': 0}

    def _api_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting."""
        time.sleep(0.5)
        try:
            resp = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get('result', {}) if data.get('success') else {}
        except Exception as e:
            print(f"API error: {e}")
            return {}

    def search_recent(
        self,
        query: str,
        tags: List[str],
        max_results: int = 5,
        months_back: int = 6
    ) -> List[Dict]:
        """Search for recent CSV datasets."""

        date_from = (datetime.now() - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')

        fq_parts = ['res_format:CSV']
        if tags:
            tag_query = ' OR '.join([f'tags:{t}' for t in tags])
            fq_parts.append(f'({tag_query})')
        fq_parts.append(f'metadata_modified:[{date_from}T00:00:00Z TO *]')

        params = {
            'q': query,
            'fq': ' AND '.join(fq_parts),
            'rows': max_results,
            'sort': 'metadata_modified desc'
        }

        result = self._api_request('package_search', params)
        return result.get('results', [])

    def download_file(self, url: str, filepath: Path) -> bool:
        """Download a single file."""
        try:
            resp = self.session.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.stats['downloaded'] += 1
            self.stats['total_mb'] += size_mb
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            self.stats['failed'] += 1
            return False

    def download_datasets(self, datasets: List[Dict]) -> List[Path]:
        """Download first CSV from each dataset."""
        downloaded = []

        for ds in datasets:
            name = ds.get('name', 'unknown')[:50]
            resources = [r for r in ds.get('resources', [])
                        if r.get('format', '').upper() == 'CSV']

            if not resources:
                continue

            url = resources[0].get('url')
            if not url:
                continue

            filename = f"{name}.csv"
            filepath = self.output_dir / filename

            if filepath.exists():
                print(f"  Skipped (exists): {filename}")
                continue

            print(f"  Downloading: {filename}")
            if self.download_file(url, filepath):
                downloaded.append(filepath)

        return downloaded


def main():
    """Download recent census, demographics & infrastructure data."""

    print("=" * 60)
    print("Visualization Data Downloader")
    print("Last 6 months | CSV only | ~50-200 MB total")
    print("=" * 60)

    downloader = VizDataDownloader(output_dir="./viz_data")

    # Define topics for visualizations
    topics = [
        {
            'name': 'Census & Population',
            'query': 'census population statistics',
            'tags': ['census', 'population', 'demographics'],
            'max': 4
        },
        {
            'name': 'Housing & Income',
            'query': 'housing income economic',
            'tags': ['housing', 'income', 'economic'],
            'max': 3
        },
        {
            'name': 'Infrastructure',
            'query': 'infrastructure facilities public',
            'tags': ['infrastructure', 'buildings', 'utilities'],
            'max': 3
        }
    ]

    all_datasets = []

    for topic in topics:
        print(f"\n--- {topic['name']} ---")
        datasets = downloader.search_recent(
            query=topic['query'],
            tags=topic['tags'],
            max_results=topic['max'],
            months_back=6
        )
        print(f"Found {len(datasets)} datasets")
        all_datasets.extend(datasets)

    print(f"\n{'=' * 60}")
    print(f"Total datasets found: {len(all_datasets)}")
    print(f"{'=' * 60}")

    # Save metadata
    meta_path = downloader.output_dir / "datasets_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump([{
            'name': d.get('name'),
            'title': d.get('title'),
            'modified': d.get('metadata_modified'),
            'org': d.get('organization', {}).get('title'),
            'notes': d.get('notes', '')[:200]
        } for d in all_datasets], f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    # Download
    print("\nDownloading files...")
    downloader.download_datasets(all_datasets)

    # Stats
    print(f"\n{'=' * 60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Files downloaded: {downloader.stats['downloaded']}")
    print(f"Files failed:     {downloader.stats['failed']}")
    print(f"Total size:       {downloader.stats['total_mb']:.2f} MB")
    print(f"Output folder:    {downloader.output_dir.absolute()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
