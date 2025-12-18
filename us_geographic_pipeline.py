#!/usr/bin/env python3
"""
US Geographic Data Pipeline
============================
Downloads national datasets with state/county breakdowns from data.gov
Creates choropleth maps and geographic visualizations of US data

Focus: Population, health, economic, demographic data BY GEOGRAPHY
Output: US maps showing densities, counts, rates by state/county
"""

import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path("F:/opendata/geographic_data")
OUTPUT_DIR = Path("C:/Users/Guest2/Personal/Github/opendata/visualizations/maps")
METADATA_DIR = Path("F:/opendata/metadata")

# State FIPS codes for mapping
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
    'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
    'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
    'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
    'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
    'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
    'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
    'WI': '55', 'WY': '56', 'DC': '11', 'PR': '72'
}

FIPS_TO_STATE = {v: k for k, v in STATE_FIPS.items()}

# =============================================================================
# NATIONAL GEOGRAPHIC DATASETS - Full coverage, state/county level
# =============================================================================
GEOGRAPHIC_DATASETS = [
    # POPULATION & DEMOGRAPHICS
    {
        'name': 'state_population',
        'label': 'US Population by State',
        'query': 'population state census estimates total',
        'max': 3,
        'geo_cols': ['state', 'state_fips', 'fips', 'geography']
    },
    {
        'name': 'county_population',
        'label': 'US Population by County',
        'query': 'county population census estimates',
        'max': 2,
        'geo_cols': ['county', 'county_fips', 'fips', 'geography']
    },
    {
        'name': 'demographics_age',
        'label': 'Age Demographics by State',
        'query': 'age distribution demographics state population',
        'max': 2,
        'geo_cols': ['state', 'fips', 'geography']
    },

    # HEALTH BY GEOGRAPHY
    {
        'name': 'health_outcomes',
        'label': 'Health Outcomes by County/State',
        'query': 'health outcomes county state mortality disease',
        'max': 3,
        'geo_cols': ['state', 'county', 'fips', 'location']
    },
    {
        'name': 'covid_by_state',
        'label': 'COVID-19 Cases by State',
        'query': 'COVID-19 cases deaths state county',
        'max': 2,
        'geo_cols': ['state', 'fips', 'jurisdiction']
    },
    {
        'name': 'hospital_capacity',
        'label': 'Hospital Capacity by State',
        'query': 'hospital beds capacity state healthcare',
        'max': 2,
        'geo_cols': ['state', 'fips', 'provider_state']
    },

    # ECONOMIC BY GEOGRAPHY
    {
        'name': 'unemployment_state',
        'label': 'Unemployment Rate by State',
        'query': 'unemployment rate state county labor',
        'max': 3,
        'geo_cols': ['state', 'fips', 'area']
    },
    {
        'name': 'income_state',
        'label': 'Median Income by State/County',
        'query': 'income median household state county',
        'max': 2,
        'geo_cols': ['state', 'county', 'fips', 'geography']
    },
    {
        'name': 'poverty_rate',
        'label': 'Poverty Rate by Geography',
        'query': 'poverty rate state county percentage',
        'max': 2,
        'geo_cols': ['state', 'county', 'fips', 'geography']
    },

    # HOUSING & INFRASTRUCTURE
    {
        'name': 'housing_state',
        'label': 'Housing Statistics by State',
        'query': 'housing units state county occupied vacant',
        'max': 2,
        'geo_cols': ['state', 'fips', 'geography']
    },

    # CRIME & SAFETY
    {
        'name': 'crime_state',
        'label': 'Crime Statistics by State',
        'query': 'crime rate state violent property FBI',
        'max': 2,
        'geo_cols': ['state', 'fips', 'state_abbr']
    },

    # EDUCATION
    {
        'name': 'education_state',
        'label': 'Education Attainment by State',
        'query': 'education attainment state college degree',
        'max': 2,
        'geo_cols': ['state', 'fips', 'geography']
    },
]


class GeoDataDownloader:
    BASE_URL = "https://catalog.data.gov/api/3/action"

    def __init__(self):
        self.session = requests.Session()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.stats = {'found': 0, 'downloaded': 0, 'failed': 0, 'bytes': 0}

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search for datasets - no date limit for comprehensive data."""
        params = {
            'q': query,
            'fq': 'res_format:CSV',
            'rows': max_results,
            'sort': 'score desc, metadata_modified desc'
        }

        time.sleep(0.3)
        try:
            resp = self.session.get(f"{self.BASE_URL}/package_search", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('result', {}).get('results', []) if data.get('success') else []
            self.stats['found'] += len(results)
            return results
        except Exception as e:
            print(f"    Search error: {e}")
            return []

    def download(self, dataset: Dict, category: str, label: str) -> Path:
        """Download full CSV (no row limit)."""
        name = dataset.get('name', 'unknown')[:50]
        title = dataset.get('title', name)
        resources = [r for r in dataset.get('resources', [])
                    if r.get('format', '').upper() == 'CSV']

        if not resources:
            return None

        url = resources[0].get('url')
        if not url:
            return None

        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        filepath = DATA_DIR / f"{category}__{safe_name}.csv"

        if filepath.exists():
            size = filepath.stat().st_size / 1024 / 1024
            print(f"    [exists] {size:.1f} MB")
            return filepath

        try:
            print(f"    Downloading: {title[:50]}...")
            resp = self.session.get(url, stream=True, timeout=600)
            resp.raise_for_status()

            total = int(resp.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=131072):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r    Progress: {pct:.0f}% ({mb:.1f} MB)", end='', flush=True)

            print()
            size = filepath.stat().st_size
            self.stats['downloaded'] += 1
            self.stats['bytes'] += size

            # Save metadata
            meta = METADATA_DIR / f"{category}__{safe_name}.json"
            with open(meta, 'w') as f:
                json.dump({
                    'category': category,
                    'label': label,
                    'title': title,
                    'url': url,
                    'size_mb': size / 1024 / 1024
                }, f, indent=2)

            print(f"    [done] {size / 1024 / 1024:.1f} MB")
            return filepath

        except Exception as e:
            print(f"    [fail] {e}")
            self.stats['failed'] += 1
            if filepath.exists():
                filepath.unlink()
            return None


class USMapVisualizer:
    """Create US choropleth maps from geographic data."""

    def __init__(self):
        self.datasets = {}
        self.state_data = {}

    def load_data(self):
        """Load all geographic CSVs."""
        print("\nLoading geographic datasets...")

        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            try:
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
                        break
                    except:
                        continue

                if len(df) > 0:
                    key = csv_path.stem
                    self.datasets[key] = df
                    print(f"  {key}: {len(df):,} rows, {len(df.columns)} cols")

            except Exception as e:
                print(f"  [error] {csv_path.name}: {e}")

        return self.datasets

    def detect_state_column(self, df: pd.DataFrame) -> str:
        """Find the state identifier column."""
        candidates = ['state', 'state_name', 'state_abbr', 'state_abbreviation',
                     'stname', 'st', 'state_code', 'State', 'STATE', 'jurisdiction',
                     'provider_state', 'location', 'area_name', 'geography']

        for col in candidates:
            if col in df.columns:
                return col

        # Check for FIPS
        for col in ['fips', 'state_fips', 'statefips', 'FIPS', 'state_fips_code']:
            if col in df.columns:
                return col

        # Check column names containing 'state'
        for col in df.columns:
            if 'state' in col.lower():
                return col

        return None

    def detect_value_columns(self, df: pd.DataFrame) -> List[str]:
        """Find numeric columns suitable for mapping."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Filter out ID/code columns
        exclude_patterns = ['fips', 'code', 'id', 'year', 'month', 'day', 'zip', 'index']
        value_cols = [c for c in numeric_cols
                     if not any(p in c.lower() for p in exclude_patterns)]

        return value_cols[:5]  # Top 5 numeric columns

    def aggregate_by_state(self, df: pd.DataFrame, state_col: str,
                           value_col: str) -> pd.DataFrame:
        """Aggregate data to state level."""
        # Standardize state names/codes
        df_copy = df.copy()

        # If FIPS, convert to state abbreviation
        if 'fips' in state_col.lower():
            df_copy['state_abbr'] = df_copy[state_col].astype(str).str[:2].map(FIPS_TO_STATE)
            state_col = 'state_abbr'

        # Group and aggregate
        agg = df_copy.groupby(state_col)[value_col].agg(['mean', 'sum', 'count']).reset_index()
        agg.columns = ['state', 'mean', 'total', 'count']

        return agg

    def create_state_choropleth(self, df: pd.DataFrame, value_col: str,
                                 title: str, filename: str):
        """Create a US state choropleth map."""
        state_col = self.detect_state_column(df)
        if not state_col:
            print(f"    No state column found for {filename}")
            return

        try:
            # Aggregate to state level
            agg = self.aggregate_by_state(df, state_col, value_col)
            agg = agg.dropna(subset=['state', 'mean'])

            if len(agg) < 5:
                print(f"    Not enough state data for {filename}")
                return

            # Create choropleth
            fig = px.choropleth(
                agg,
                locations='state',
                locationmode='USA-states',
                color='mean',
                color_continuous_scale='Blues',
                scope='usa',
                title=f'{title}<br><sub>{value_col.replace("_", " ").title()} by State</sub>',
                labels={'mean': value_col.replace('_', ' ').title()}
            )

            fig.update_layout(
                geo=dict(
                    showlakes=True,
                    lakecolor='rgb(255, 255, 255)'
                ),
                margin=dict(l=0, r=0, t=60, b=0),
                title_x=0.5,
                font=dict(size=12)
            )

            # Save
            output_path = OUTPUT_DIR / f'{filename}.png'
            fig.write_image(str(output_path), width=1200, height=800)
            print(f"    Saved: {filename}.png")

            # Also save interactive HTML
            html_path = OUTPUT_DIR / f'{filename}.html'
            fig.write_html(str(html_path))

        except Exception as e:
            print(f"    Error creating map {filename}: {e}")

    def create_all_maps(self):
        """Generate maps for all loaded datasets."""
        if not self.datasets:
            print("No datasets loaded")
            return

        print("\n[CREATING US MAPS]")
        print("-" * 50)

        for key, df in self.datasets.items():
            category = key.split('__')[0]
            print(f"\n>>> {category}")

            state_col = self.detect_state_column(df)
            if not state_col:
                print(f"    No geographic column found")
                continue

            value_cols = self.detect_value_columns(df)
            if not value_cols:
                print(f"    No suitable value columns")
                continue

            print(f"    State column: {state_col}")
            print(f"    Value columns: {value_cols[:3]}")

            # Create map for top value column
            for i, val_col in enumerate(value_cols[:2]):
                clean_name = f"map_{category}_{val_col[:20]}"
                title = category.replace('_', ' ').title()
                self.create_state_choropleth(df, val_col, title, clean_name)

    def create_summary_dashboard(self):
        """Create a multi-panel summary dashboard."""
        if len(self.datasets) < 2:
            return

        print("\n>>> Creating summary dashboard...")

        # Find datasets with state data
        state_datasets = []
        for key, df in self.datasets.items():
            state_col = self.detect_state_column(df)
            value_cols = self.detect_value_columns(df)
            if state_col and value_cols:
                state_datasets.append((key, df, state_col, value_cols[0]))

        if len(state_datasets) < 2:
            print("    Not enough geographic datasets for dashboard")
            return

        # Create 2x2 dashboard
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "choropleth"}, {"type": "choropleth"}],
                   [{"type": "choropleth"}, {"type": "choropleth"}]],
            subplot_titles=[d[0].split('__')[0].replace('_', ' ').title()
                           for d in state_datasets[:4]]
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for idx, (key, df, state_col, val_col) in enumerate(state_datasets[:4]):
            row, col = positions[idx]

            try:
                agg = self.aggregate_by_state(df, state_col, val_col)
                agg = agg.dropna(subset=['state', 'mean'])

                fig.add_trace(
                    go.Choropleth(
                        locations=agg['state'],
                        locationmode='USA-states',
                        z=agg['mean'],
                        colorscale='Blues',
                        showscale=False,
                        name=key.split('__')[0]
                    ),
                    row=row, col=col
                )
            except:
                continue

        fig.update_geos(
            scope='usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        )

        fig.update_layout(
            title_text='US Geographic Data Dashboard',
            title_x=0.5,
            height=1000,
            width=1400,
            margin=dict(l=20, r=20, t=80, b=20)
        )

        output_path = OUTPUT_DIR / 'dashboard_us_maps.png'
        fig.write_image(str(output_path), width=1400, height=1000)
        print(f"    Saved: dashboard_us_maps.png")

        html_path = OUTPUT_DIR / 'dashboard_us_maps.html'
        fig.write_html(str(html_path))
        print(f"    Saved: dashboard_us_maps.html (interactive)")


def main():
    print("=" * 65)
    print("US GEOGRAPHIC DATA PIPELINE")
    print("National data for state/county choropleth maps")
    print("=" * 65)
    print(f"Data:       {DATA_DIR}")
    print(f"Maps:       {OUTPUT_DIR}")
    print(f"Datasets:   {len(GEOGRAPHIC_DATASETS)} categories")
    print("=" * 65)

    # DOWNLOAD
    print("\n[1] DOWNLOADING NATIONAL GEOGRAPHIC DATA")
    print("-" * 50)

    dl = GeoDataDownloader()

    for config in GEOGRAPHIC_DATASETS:
        print(f"\n>>> {config['label']}")
        datasets = dl.search(config['query'], config['max'])

        if datasets:
            print(f"  Found {len(datasets)} datasets")
            for ds in datasets:
                dl.download(ds, config['name'], config['label'])
        else:
            print("  No data found")

    print(f"\n{'=' * 50}")
    print(f"Downloaded: {dl.stats['downloaded']} files")
    print(f"Total size: {dl.stats['bytes'] / 1024 / 1024 / 1024:.2f} GB")
    print(f"Failed: {dl.stats['failed']}")

    # VISUALIZE
    print("\n[2] CREATING US MAPS")
    print("-" * 50)

    viz = USMapVisualizer()
    viz.load_data()

    if viz.datasets:
        viz.create_all_maps()
        viz.create_summary_dashboard()

    # SUMMARY
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)

    map_files = list(OUTPUT_DIR.glob("*.png"))
    html_files = list(OUTPUT_DIR.glob("*.html"))

    print(f"\nMaps created: {len(map_files)} PNG files")
    print(f"Interactive: {len(html_files)} HTML files")
    print(f"Location: {OUTPUT_DIR}")

    for f in sorted(map_files)[:10]:
        print(f"  - {f.name}")

    print("\nTo push to GitHub:")
    print("  cd /c/Users/Guest2/Personal/Github/opendata")
    print("  git add .")
    print('  git commit -m "Add US geographic maps"')
    print("  git push")
    print("=" * 65)


if __name__ == "__main__":
    main()
