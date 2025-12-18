#!/usr/bin/env python3
"""
Data.gov Bulk Dataset Downloader
================================
A comprehensive script to search, filter, and download datasets from data.gov
using the CKAN API.

Requirements:
    pip install requests tqdm

Usage:
    python datagov_bulk_downloader.py

Author: Generated for bulk data.gov downloading
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


class DataGovDownloader:
    """
    Bulk downloader for data.gov datasets using the CKAN API.
    """
    
    # API Configuration
    BASE_URL = "https://catalog.data.gov/api/3/action"
    
    # Alternative GSA API (requires API key)
    GSA_API_URL = "https://api.gsa.gov/technology/datagov/v3/action"
    
    # Bulk metadata download (updated monthly, ~2.3GB compressed)
    BULK_METADATA_URL = "https://filestore.data.gov/gsa/catalog/jsonl/dataset.jsonl.gz"
    
    def __init__(
        self,
        output_dir: str = "./datagov_downloads",
        api_key: Optional[str] = None,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            api_key: GSA API key (optional, get from https://api.data.gov/signup/)
            rate_limit_delay: Seconds between API requests
            max_retries: Number of retry attempts for failed downloads
            timeout: Request timeout in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Create subdirectories
        self.data_dir = self.output_dir / "datasets"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        for d in [self.data_dir, self.metadata_dir, self.logs_dir]:
            d.mkdir(exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        if api_key:
            self.session.headers['x-api-key'] = api_key
        
        # Download statistics
        self.stats = {
            'searched': 0,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'total_bytes': 0
        }
    
    def _api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make an API request with rate limiting and retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                if data.get('success'):
                    return data.get('result', {})
                else:
                    raise Exception(f"API error: {data.get('error', 'Unknown error')}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait = (attempt + 1) * 2
                    print(f"Request failed, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
        
        return {}
    
    # =========================================================================
    # DISCOVERY METHODS - Find what's available
    # =========================================================================
    
    def get_all_tags(self, limit: int = 500) -> List[str]:
        """Get all available tags/topics."""
        result = self._api_request('tag_list', {'all_fields': False})
        return result if isinstance(result, list) else []
    
    def get_all_organizations(self) -> List[Dict]:
        """Get all publishing organizations."""
        result = self._api_request('organization_list', {'all_fields': True})
        return result if isinstance(result, list) else []
    
    def get_all_groups(self) -> List[Dict]:
        """Get all dataset groups/categories."""
        result = self._api_request('group_list', {'all_fields': True})
        return result if isinstance(result, list) else []
    
    def get_format_counts(self) -> Dict[str, int]:
        """Get counts of datasets by format."""
        formats = {}
        common_formats = ['CSV', 'JSON', 'XML', 'XLS', 'XLSX', 'PDF', 'ZIP', 
                         'GeoJSON', 'KML', 'Shapefile', 'API']
        
        for fmt in common_formats:
            result = self._api_request('package_search', {
                'fq': f'res_format:{fmt}',
                'rows': 0
            })
            formats[fmt] = result.get('count', 0)
        
        return formats
    
    # =========================================================================
    # SEARCH METHODS - Find specific datasets
    # =========================================================================
    
    def search_datasets(
        self,
        query: str = "*:*",
        filters: Dict[str, str] = None,
        formats: List[str] = None,
        organizations: List[str] = None,
        tags: List[str] = None,
        groups: List[str] = None,
        date_from: str = None,
        date_to: str = None,
        rows: int = 100,
        start: int = 0,
        sort: str = "score desc, metadata_modified desc"
    ) -> Dict:
        """
        Search for datasets with flexible filtering.
        
        Args:
            query: Search query (Solr syntax supported)
            filters: Additional filter queries as dict
            formats: List of file formats (e.g., ['CSV', 'JSON'])
            organizations: Filter by publishing org
            tags: Filter by tags/topics
            groups: Filter by groups/categories
            date_from: Filter by modification date (YYYY-MM-DD)
            date_to: Filter by modification date (YYYY-MM-DD)
            rows: Number of results per page (max 1000)
            start: Offset for pagination
            sort: Sort order
            
        Returns:
            Dict with 'count', 'results', and 'facets'
        """
        # Build filter query
        fq_parts = []
        
        if formats:
            format_query = ' OR '.join([f'res_format:{f}' for f in formats])
            fq_parts.append(f'({format_query})')
        
        if organizations:
            org_query = ' OR '.join([f'organization:{o}' for o in organizations])
            fq_parts.append(f'({org_query})')
        
        if tags:
            tag_query = ' OR '.join([f'tags:{t}' for t in tags])
            fq_parts.append(f'({tag_query})')
        
        if groups:
            group_query = ' OR '.join([f'groups:{g}' for g in groups])
            fq_parts.append(f'({group_query})')
        
        if date_from or date_to:
            date_from = date_from or '*'
            date_to = date_to or '*'
            fq_parts.append(f'metadata_modified:[{date_from}T00:00:00Z TO {date_to}T23:59:59Z]')
        
        if filters:
            for key, value in filters.items():
                fq_parts.append(f'{key}:{value}')
        
        params = {
            'q': query,
            'rows': min(rows, 1000),
            'start': start,
            'sort': sort,
            'facet': 'true',
            'facet.field': ['tags', 'organization', 'res_format', 'groups']
        }
        
        if fq_parts:
            params['fq'] = ' AND '.join(fq_parts)
        
        result = self._api_request('package_search', params)
        self.stats['searched'] += result.get('count', 0)
        
        return result
    
    def search_all_datasets(
        self,
        batch_size: int = 1000,
        max_results: int = None,
        **search_kwargs
    ) -> List[Dict]:
        """
        Search and retrieve ALL matching datasets (handles pagination).
        
        Args:
            batch_size: Results per API call (max 1000)
            max_results: Maximum total results to return
            **search_kwargs: Arguments passed to search_datasets()
            
        Returns:
            List of all matching dataset metadata
        """
        all_results = []
        start = 0
        
        # Initial search to get total count
        first_batch = self.search_datasets(rows=batch_size, start=0, **search_kwargs)
        total = first_batch.get('count', 0)
        all_results.extend(first_batch.get('results', []))
        
        if max_results:
            total = min(total, max_results)
        
        print(f"Found {total} matching datasets")
        
        # Continue pagination
        start = batch_size
        iterator = range(start, total, batch_size)
        
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Fetching metadata", unit="batch")
        
        for offset in iterator:
            if max_results and len(all_results) >= max_results:
                break
                
            batch = self.search_datasets(rows=batch_size, start=offset, **search_kwargs)
            results = batch.get('results', [])
            
            if not results:
                break
                
            all_results.extend(results)
        
        return all_results[:max_results] if max_results else all_results
    
    # =========================================================================
    # DOWNLOAD METHODS
    # =========================================================================
    
    def get_dataset_resources(self, dataset_id: str) -> List[Dict]:
        """Get all downloadable resources for a dataset."""
        result = self._api_request('package_show', {'id': dataset_id})
        return result.get('resources', [])
    
    def download_resource(
        self,
        url: str,
        filename: str = None,
        dataset_name: str = "unknown"
    ) -> Optional[Path]:
        """
        Download a single resource file.
        
        Args:
            url: Direct download URL
            filename: Optional filename override
            dataset_name: Dataset name for organizing files
            
        Returns:
            Path to downloaded file, or None if failed
        """
        if not url:
            return None
        
        # Create dataset subdirectory
        dataset_dir = self.data_dir / self._sanitize_filename(dataset_name)
        dataset_dir.mkdir(exist_ok=True)
        
        # Determine filename
        if not filename:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or 'data'
            if not os.path.splitext(filename)[1]:
                filename += '.dat'
        
        filename = self._sanitize_filename(filename)
        filepath = dataset_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            self.stats['skipped'] += 1
            return filepath
        
        # Download with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, 
                    stream=True, 
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Get file size if available
                total_size = int(response.headers.get('content-length', 0))
                
                # Write to file
                with open(filepath, 'wb') as f:
                    if HAS_TQDM and total_size:
                        with tqdm(total=total_size, unit='B', unit_scale=True, 
                                 desc=filename[:30], leave=False) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                self.stats['downloaded'] += 1
                self.stats['total_bytes'] += filepath.stat().st_size
                return filepath
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    print(f"Failed to download {url}: {e}")
                    self.stats['failed'] += 1
                    return None
        
        return None
    
    def download_dataset(
        self,
        dataset: Dict,
        formats: List[str] = None,
        max_resources: int = None
    ) -> List[Path]:
        """
        Download all resources from a dataset.
        
        Args:
            dataset: Dataset metadata dict
            formats: Only download these formats (e.g., ['CSV', 'JSON'])
            max_resources: Maximum resources to download per dataset
            
        Returns:
            List of paths to downloaded files
        """
        downloaded = []
        resources = dataset.get('resources', [])
        
        # Filter by format
        if formats:
            formats_upper = [f.upper() for f in formats]
            resources = [r for r in resources 
                        if r.get('format', '').upper() in formats_upper]
        
        # Limit resources
        if max_resources:
            resources = resources[:max_resources]
        
        dataset_name = dataset.get('name', dataset.get('id', 'unknown'))
        
        # Save metadata
        meta_path = self.metadata_dir / f"{self._sanitize_filename(dataset_name)}.json"
        with open(meta_path, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        # Download each resource
        for resource in resources:
            url = resource.get('url')
            if url:
                filename = resource.get('name') or resource.get('id')
                path = self.download_resource(url, filename, dataset_name)
                if path:
                    downloaded.append(path)
        
        return downloaded
    
    def bulk_download(
        self,
        datasets: List[Dict],
        formats: List[str] = None,
        max_resources_per_dataset: int = 5,
        parallel_downloads: int = 3
    ) -> Dict:
        """
        Bulk download multiple datasets.
        
        Args:
            datasets: List of dataset metadata dicts
            formats: Only download these formats
            max_resources_per_dataset: Limit resources per dataset
            parallel_downloads: Number of concurrent downloads
            
        Returns:
            Download statistics
        """
        all_downloaded = []
        
        iterator = datasets
        if HAS_TQDM:
            iterator = tqdm(datasets, desc="Downloading datasets", unit="dataset")
        
        for dataset in iterator:
            try:
                paths = self.download_dataset(
                    dataset,
                    formats=formats,
                    max_resources=max_resources_per_dataset
                )
                all_downloaded.extend(paths)
            except Exception as e:
                print(f"Error downloading {dataset.get('name')}: {e}")
        
        return self.stats
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _sanitize_filename(self, name: str) -> str:
        """Remove invalid characters from filename."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name[:200]  # Limit length
    
    def save_search_results(self, results: List[Dict], filename: str = "search_results.json"):
        """Save search results to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved {len(results)} results to {filepath}")
        return filepath
    
    def print_stats(self):
        """Print download statistics."""
        print("\n" + "="*50)
        print("Download Statistics")
        print("="*50)
        print(f"Datasets searched: {self.stats['searched']}")
        print(f"Files downloaded:  {self.stats['downloaded']}")
        print(f"Files skipped:     {self.stats['skipped']}")
        print(f"Files failed:      {self.stats['failed']}")
        print(f"Total data:        {self.stats['total_bytes'] / (1024*1024):.2f} MB")
        print("="*50)


# =============================================================================
# PRESET FILTER CONFIGURATIONS
# =============================================================================

TOPIC_PRESETS = {
    'health': {
        'tags': ['health', 'healthcare', 'medical', 'disease', 'hospital', 'medicare', 'medicaid'],
        'organizations': ['hhs-gov', 'cdc-gov', 'nih-gov', 'cms-gov']
    },
    'climate': {
        'tags': ['climate', 'weather', 'environment', 'emissions', 'temperature', 'ocean'],
        'organizations': ['noaa-gov', 'epa-gov', 'nasa-gov', 'doe-gov']
    },
    'finance': {
        'tags': ['finance', 'banking', 'economy', 'budget', 'spending', 'loans', 'treasury'],
        'organizations': ['treasury-gov', 'sec-gov', 'fdic-gov', 'cfpb']
    },
    'transportation': {
        'tags': ['transportation', 'traffic', 'aviation', 'roads', 'transit', 'vehicles'],
        'organizations': ['dot-gov', 'faa-gov', 'nhtsa-gov']
    },
    'energy': {
        'tags': ['energy', 'electricity', 'oil', 'gas', 'renewable', 'nuclear', 'solar'],
        'organizations': ['doe-gov', 'eia-gov', 'nrc-gov']
    },
    'agriculture': {
        'tags': ['agriculture', 'farming', 'food', 'crops', 'livestock'],
        'organizations': ['usda-gov', 'fda-gov']
    },
    'education': {
        'tags': ['education', 'schools', 'students', 'college', 'university'],
        'organizations': ['ed-gov', 'nsf-gov']
    },
    'infrastructure': {
        'tags': ['infrastructure', 'buildings', 'facilities', 'utilities', 'water'],
        'organizations': ['gsa-gov', 'dhs-gov', 'epa-gov']
    },
    'defense': {
        'tags': ['defense', 'military', 'veterans', 'security'],
        'organizations': ['dod-gov', 'va-gov', 'dhs-gov']
    },
    'demographics': {
        'tags': ['census', 'population', 'demographics', 'housing', 'income'],
        'organizations': ['census-gov', 'bls-gov', 'hud-gov']
    }
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """
    Example: Download health-related CSV datasets.
    """
    # Initialize downloader
    downloader = DataGovDownloader(
        output_dir="./datagov_health_data",
        rate_limit_delay=0.5  # Be respectful of the API
    )
    
    # Option 1: Use preset topics
    preset = TOPIC_PRESETS['health']
    
    # Option 2: Custom search
    datasets = downloader.search_all_datasets(
        query="hospital",                    # Search term
        formats=['CSV', 'JSON'],             # Only these formats
        tags=['health', 'healthcare'],       # With these tags
        max_results=50                       # Limit results
    )
    
    print(f"Found {len(datasets)} datasets")
    
    # Save metadata for review before downloading
    downloader.save_search_results(datasets, "health_datasets.json")
    
    # Download the actual data files
    # Uncomment to actually download:
    # downloader.bulk_download(
    #     datasets,
    #     formats=['CSV'],
    #     max_resources_per_dataset=3
    # )
    
    downloader.print_stats()
    
    return datasets


def explore_available_data():
    """
    Explore what's available on data.gov.
    """
    downloader = DataGovDownloader()
    
    print("Fetching available tags...")
    tags = downloader.get_all_tags()
    print(f"Total tags: {len(tags)}")
    print(f"Sample tags: {tags[:20]}")
    
    print("\nFetching organizations...")
    orgs = downloader.get_all_organizations()
    print(f"Total organizations: {len(orgs)}")
    
    print("\nFetching format counts...")
    formats = downloader.get_format_counts()
    for fmt, count in sorted(formats.items(), key=lambda x: -x[1]):
        print(f"  {fmt}: {count:,} datasets")
    
    return {'tags': tags, 'organizations': orgs, 'formats': formats}


if __name__ == "__main__":
    print("Data.gov Bulk Downloader")
    print("========================\n")
    
    # Explore what's available
    # explore_available_data()
    
    # Run example
    example_usage()
