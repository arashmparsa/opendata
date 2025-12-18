#!/usr/bin/env python3
"""
Data Pipeline: Download, Analyze & Visualize
=============================================
Downloads census/demographics/infrastructure data from data.gov to F:/opendata
Runs analysis and creates visualizations in C:/Users/Guest2/Personal/Github/opendata

Data Location:    F:/opendata
Output Location:  C:/Users/Guest2/Personal/Github/opendata/visualizations

Requirements:
    pip install requests pandas matplotlib seaborn tqdm

Usage:
    python data_pipeline.py
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Data analysis & visualization
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False
    print("Install analytics packages: pip install pandas matplotlib seaborn")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("F:/opendata/datasets")           # Where to store downloaded CSVs
OUTPUT_DIR = Path("C:/Users/Guest2/Personal/Github/opendata/visualizations")
METADATA_DIR = Path("F:/opendata/metadata")

# Topics to download
TOPICS = [
    {
        'name': 'Census & Population',
        'query': 'census population county state',
        'tags': ['census', 'population', 'demographics', 'county'],
        'max_datasets': 3
    },
    {
        'name': 'Housing & Economics',
        'query': 'housing income poverty economic',
        'tags': ['housing', 'income', 'poverty', 'economic'],
        'max_datasets': 3
    },
    {
        'name': 'Infrastructure & Utilities',
        'query': 'infrastructure water energy facilities',
        'tags': ['infrastructure', 'utilities', 'energy', 'water'],
        'max_datasets': 2
    },
    {
        'name': 'Health Demographics',
        'query': 'health population statistics county',
        'tags': ['health', 'demographics', 'statistics'],
        'max_datasets': 2
    }
]


# =============================================================================
# DOWNLOADER
# =============================================================================

class DataDownloader:
    """Downloads data from data.gov API."""

    BASE_URL = "https://catalog.data.gov/api/3/action"

    def __init__(self):
        self.session = requests.Session()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, tags: List[str], max_results: int, months_back: int = 6) -> List[Dict]:
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

        time.sleep(0.5)
        try:
            resp = self.session.get(f"{self.BASE_URL}/package_search", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get('result', {}).get('results', []) if data.get('success') else []
        except Exception as e:
            print(f"  Search error: {e}")
            return []

    def download_csv(self, dataset: Dict) -> Path:
        """Download first CSV from dataset."""
        name = dataset.get('name', 'unknown')[:60]
        resources = [r for r in dataset.get('resources', []) if r.get('format', '').upper() == 'CSV']

        if not resources:
            return None

        url = resources[0].get('url')
        if not url:
            return None

        # Clean filename
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        filepath = DATA_DIR / f"{safe_name}.csv"

        if filepath.exists():
            print(f"    [exists] {filepath.name}")
            return filepath

        try:
            print(f"    Downloading: {filepath.name}")
            resp = self.session.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Save metadata
            meta_path = METADATA_DIR / f"{safe_name}.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'name': dataset.get('name'),
                    'title': dataset.get('title'),
                    'org': dataset.get('organization', {}).get('title'),
                    'modified': dataset.get('metadata_modified'),
                    'notes': dataset.get('notes', '')[:500],
                    'url': url
                }, f, indent=2)

            return filepath
        except Exception as e:
            print(f"    [failed] {e}")
            return None


# =============================================================================
# ANALYZER
# =============================================================================

class DataAnalyzer:
    """Analyzes downloaded CSV files and creates visualizations."""

    def __init__(self):
        self.dataframes = {}
        self.summaries = []

    def load_all_csvs(self) -> Dict[str, pd.DataFrame]:
        """Load all CSVs from data directory."""
        csv_files = list(DATA_DIR.glob("*.csv"))
        print(f"\nLoading {len(csv_files)} CSV files...")

        for csv_path in csv_files:
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(csv_path, encoding=encoding, low_memory=False, nrows=50000)
                        break
                    except UnicodeDecodeError:
                        continue

                if len(df) > 0:
                    self.dataframes[csv_path.stem] = df
                    print(f"  Loaded: {csv_path.name} ({len(df)} rows, {len(df.columns)} cols)")
            except Exception as e:
                print(f"  Failed to load {csv_path.name}: {e}")

        return self.dataframes

    def analyze_dataset(self, name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for a dataset."""
        summary = {
            'name': name,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'missing_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

        # Basic stats for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 0:
            summary['numeric_stats'] = numeric_df.describe().to_dict()

        return summary

    def create_overview_viz(self):
        """Create overview visualization of all datasets."""
        if not self.dataframes:
            print("No data loaded for visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data.gov Datasets Overview', fontsize=16, fontweight='bold')

        # 1. Dataset sizes
        ax1 = axes[0, 0]
        sizes = {k: len(v) for k, v in self.dataframes.items()}
        names = [n[:25] + '...' if len(n) > 25 else n for n in sizes.keys()]
        ax1.barh(names, list(sizes.values()), color='steelblue')
        ax1.set_xlabel('Number of Rows')
        ax1.set_title('Dataset Sizes')
        ax1.tick_params(axis='y', labelsize=8)

        # 2. Column counts
        ax2 = axes[0, 1]
        col_counts = {k: len(v.columns) for k, v in self.dataframes.items()}
        ax2.barh(names, list(col_counts.values()), color='coral')
        ax2.set_xlabel('Number of Columns')
        ax2.set_title('Dataset Column Counts')
        ax2.tick_params(axis='y', labelsize=8)

        # 3. Data types distribution (first dataset with most numeric cols)
        ax3 = axes[1, 0]
        best_df_name = max(self.dataframes.keys(),
                          key=lambda k: len(self.dataframes[k].select_dtypes(include=['number']).columns))
        best_df = self.dataframes[best_df_name]
        dtype_counts = best_df.dtypes.astype(str).value_counts()
        ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Data Types: {best_df_name[:30]}')

        # 4. Missing data
        ax4 = axes[1, 1]
        missing = {k: (v.isnull().sum().sum() / (v.shape[0] * v.shape[1]) * 100)
                   for k, v in self.dataframes.items()}
        colors = ['green' if m < 5 else 'orange' if m < 20 else 'red' for m in missing.values()]
        ax4.barh(names, list(missing.values()), color=colors)
        ax4.set_xlabel('Missing Data (%)')
        ax4.set_title('Data Completeness')
        ax4.tick_params(axis='y', labelsize=8)
        ax4.axvline(x=5, color='green', linestyle='--', alpha=0.5)
        ax4.axvline(x=20, color='orange', linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_path = OUTPUT_DIR / 'datasets_overview.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")

    def create_numeric_distributions(self):
        """Create distribution plots for numeric columns."""
        for name, df in self.dataframes.items():
            numeric_cols = df.select_dtypes(include=['number']).columns[:6]  # Max 6 cols

            if len(numeric_cols) == 0:
                continue

            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            fig.suptitle(f'Distributions: {name[:50]}', fontsize=12, fontweight='bold')

            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]

            for idx, col in enumerate(numeric_cols):
                row, col_idx = idx // n_cols, idx % n_cols
                ax = axes[row][col_idx] if n_rows > 1 else axes[0][col_idx]

                try:
                    data = df[col].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
                        ax.set_title(col[:30], fontsize=9)
                        ax.tick_params(labelsize=8)
                except Exception:
                    ax.text(0.5, 0.5, 'Cannot plot', ha='center', va='center')

            # Hide empty subplots
            for idx in range(len(numeric_cols), n_rows * n_cols):
                row, col_idx = idx // n_cols, idx % n_cols
                ax = axes[row][col_idx] if n_rows > 1 else axes[0][col_idx]
                ax.set_visible(False)

            plt.tight_layout()
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
            output_path = OUTPUT_DIR / f'dist_{safe_name[:40]}.png'
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {output_path.name}")

    def create_correlation_heatmaps(self):
        """Create correlation heatmaps for numeric data."""
        for name, df in self.dataframes.items():
            numeric_df = df.select_dtypes(include=['number'])

            if len(numeric_df.columns) < 2:
                continue

            # Limit to 10 columns for readability
            if len(numeric_df.columns) > 10:
                numeric_df = numeric_df.iloc[:, :10]

            try:
                corr = numeric_df.corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                           fmt='.2f', ax=ax, annot_kws={'size': 8})
                ax.set_title(f'Correlation: {name[:50]}', fontsize=12, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)

                plt.tight_layout()
                safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
                output_path = OUTPUT_DIR / f'corr_{safe_name[:40]}.png'
                plt.savefig(output_path, dpi=120, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {output_path.name}")
            except Exception as e:
                print(f"  Skipped correlation for {name}: {e}")

    def generate_report(self) -> str:
        """Generate a text report of all analyses."""
        report = []
        report.append("=" * 70)
        report.append("DATA ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 70)
        report.append(f"\nDatasets analyzed: {len(self.dataframes)}")
        report.append(f"Data location: {DATA_DIR}")
        report.append(f"Visualizations: {OUTPUT_DIR}")

        total_rows = sum(len(df) for df in self.dataframes.values())
        report.append(f"Total rows across all datasets: {total_rows:,}")

        for name, df in self.dataframes.items():
            report.append(f"\n{'-' * 50}")
            report.append(f"DATASET: {name}")
            report.append(f"{'-' * 50}")
            report.append(f"Rows: {len(df):,}")
            report.append(f"Columns: {len(df.columns)}")
            report.append(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                report.append(f"\nNumeric columns ({len(numeric_cols)}):")
                for col in numeric_cols[:10]:
                    stats = df[col].describe()
                    report.append(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                                f"min={stats['min']:.2f}, max={stats['max']:.2f}")

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        report_text = '\n'.join(report)

        # Save report
        report_path = OUTPUT_DIR / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"  Saved: {report_path}")

        return report_text


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete data pipeline."""

    print("=" * 70)
    print("DATA PIPELINE: Download, Analyze & Visualize")
    print("=" * 70)
    print(f"Data storage:      {DATA_DIR}")
    print(f"Visualizations:    {OUTPUT_DIR}")
    print(f"Date filter:       Last 6 months")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PHASE 1: DOWNLOAD
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] DOWNLOADING DATA")
    print("-" * 40)

    downloader = DataDownloader()
    all_datasets = []

    for topic in TOPICS:
        print(f"\n>>> {topic['name']}")
        datasets = downloader.search(
            query=topic['query'],
            tags=topic['tags'],
            max_results=topic['max_datasets'],
            months_back=6
        )
        print(f"  Found {len(datasets)} datasets")

        for ds in datasets:
            path = downloader.download_csv(ds)
            if path:
                all_datasets.append({'name': ds.get('name'), 'path': str(path)})

    print(f"\n>>> Downloaded {len(all_datasets)} CSV files to {DATA_DIR}")

    # -------------------------------------------------------------------------
    # PHASE 2: ANALYZE & VISUALIZE
    # -------------------------------------------------------------------------
    if not HAS_ANALYTICS:
        print("\n[PHASE 2] SKIPPED - Install pandas/matplotlib/seaborn")
        return

    print("\n[PHASE 2] ANALYZING DATA")
    print("-" * 40)

    analyzer = DataAnalyzer()
    analyzer.load_all_csvs()

    if not analyzer.dataframes:
        print("No data to analyze")
        return

    print("\n[PHASE 3] CREATING VISUALIZATIONS")
    print("-" * 40)

    print("\n>>> Overview charts...")
    analyzer.create_overview_viz()

    print("\n>>> Distribution plots...")
    analyzer.create_numeric_distributions()

    print("\n>>> Correlation heatmaps...")
    analyzer.create_correlation_heatmaps()

    print("\n>>> Generating report...")
    analyzer.generate_report()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Data downloaded to:     {DATA_DIR}")
    print(f"Visualizations saved:   {OUTPUT_DIR}")

    # List outputs
    viz_files = list(OUTPUT_DIR.glob("*.png"))
    print(f"\nVisualization files ({len(viz_files)}):")
    for f in viz_files:
        print(f"  - {f.name}")

    print("\nTo push to GitHub:")
    print("  cd /c/Users/Guest2/Personal/Github/opendata")
    print("  git add .")
    print('  git commit -m "Add visualizations"')
    print("  git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
