#!/usr/bin/env python3
"""
Favorite Datasets Pipeline - Based on FAVORITE_DATASETS.md
===========================================================
Downloads specific datasets from data.gov matching your favorites list
Stores data on F:/opendata, creates visualizations in GitHub repo
"""

import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False
    print("Run: pip install pandas matplotlib seaborn")

DATA_DIR = Path("F:/opendata/datasets")
OUTPUT_DIR = Path("C:/Users/Guest2/Personal/Github/opendata/visualizations")
METADATA_DIR = Path("F:/opendata/metadata")

# =============================================================================
# FAVORITE DATASETS from FAVORITE_DATASETS.md
# =============================================================================
FAVORITE_DATASETS = [
    # === DEMOGRAPHICS & CENSUS ===
    {'name': 'ACS_5Year', 'label': 'American Community Survey (ACS) 5-Year',
     'query': 'american community survey 5-year estimates', 'max': 2},

    {'name': 'Decennial_Census', 'label': 'Decennial Census Summary',
     'query': 'decennial census population summary', 'max': 1},

    {'name': 'CPS_Microdata', 'label': 'Current Population Survey (CPS)',
     'query': 'current population survey labor employment', 'max': 1},

    {'name': 'County_Business', 'label': 'County Business Patterns',
     'query': 'county business patterns establishment employment', 'max': 1},

    {'name': 'LEHD_Employment', 'label': 'LEHD Employer-Household Dynamics',
     'query': 'LEHD employment earnings workforce dynamics', 'max': 1},

    # === HEALTH ===
    {'name': 'Medicare_Provider', 'label': 'Medicare Provider Utilization & Payment',
     'query': 'medicare provider utilization payment', 'max': 2},

    {'name': 'Hospital_Compare', 'label': 'Hospital Compare Quality Measures',
     'query': 'hospital compare quality measures ratings', 'max': 1},

    {'name': 'CDC_Mortality', 'label': 'CDC WONDER Mortality Data',
     'query': 'CDC mortality death causes statistics WONDER', 'max': 1},

    {'name': 'FDA_FAERS', 'label': 'FDA Adverse Event Reporting (FAERS)',
     'query': 'FDA adverse event drug reaction FAERS', 'max': 1},

    {'name': 'NHANES_Survey', 'label': 'NHANES Health & Nutrition Survey',
     'query': 'NHANES health nutrition examination survey', 'max': 1},

    {'name': 'COVID19_Cases', 'label': 'COVID-19 Case Surveillance',
     'query': 'COVID-19 case surveillance pandemic coronavirus', 'max': 1},

    {'name': 'Medicaid_Drugs', 'label': 'Medicaid Drug Utilization',
     'query': 'medicaid drug utilization prescription', 'max': 1},

    # === FINANCIAL ===
    {'name': 'HMDA_Mortgage', 'label': 'HMDA Mortgage Disclosure Data',
     'query': 'HMDA mortgage home loan disclosure', 'max': 1},

    {'name': 'SBA_Loans', 'label': 'SBA Loan Data (7a/504)',
     'query': 'SBA small business loan 7a 504', 'max': 1},

    {'name': 'CFPB_Complaints', 'label': 'CFPB Consumer Complaint Database',
     'query': 'consumer complaint CFPB financial', 'max': 1},

    {'name': 'SEC_EDGAR', 'label': 'SEC EDGAR Filings',
     'query': 'SEC EDGAR filings securities', 'max': 1},

    {'name': 'FDIC_Banks', 'label': 'FDIC Failed Banks List',
     'query': 'FDIC failed bank closure list', 'max': 1},

    {'name': 'USASpending', 'label': 'USASpending Federal Contracts',
     'query': 'USASpending federal contracts spending awards', 'max': 1},

    # === ECONOMIC ===
    {'name': 'BLS_Employment', 'label': 'Bureau of Labor Statistics',
     'query': 'BLS employment wages unemployment statistics', 'max': 2},

    {'name': 'BEA_GDP', 'label': 'Bureau of Economic Analysis (GDP)',
     'query': 'BEA GDP economic analysis regional accounts', 'max': 1},

    {'name': 'FRED_Economic', 'label': 'Federal Reserve Economic Data (FRED)',
     'query': 'FRED federal reserve economic data series', 'max': 1},

    {'name': 'Trade_ImEx', 'label': 'Import/Export Trade Data',
     'query': 'import export trade international commerce', 'max': 1},

    {'name': 'Price_Index', 'label': 'Producer/Consumer Price Indices',
     'query': 'CPI PPI price index inflation consumer producer', 'max': 1},

    # === SCIENTIFIC ===
    {'name': 'PubChem', 'label': 'PubChem Chemical Data',
     'query': 'PubChem chemical compound data', 'max': 1},

    {'name': 'USGS_Earthquake', 'label': 'USGS Earthquake Catalog',
     'query': 'USGS earthquake seismic catalog magnitude', 'max': 1},

    {'name': 'NASA_Earth', 'label': 'NASA Earth Observation Data',
     'query': 'NASA earth observation climate satellite environment', 'max': 1},
]


class DataDownloader:
    BASE_URL = "https://catalog.data.gov/api/3/action"

    def __init__(self):
        self.session = requests.Session()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.stats = {'found': 0, 'downloaded': 0, 'failed': 0, 'skipped': 0, 'bytes': 0}

    def search(self, query: str, max_results: int = 2, months: int = 6) -> List[Dict]:
        """Search data.gov for CSV datasets from last N months."""
        date_from = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')

        fq = f'res_format:CSV AND metadata_modified:[{date_from}T00:00:00Z TO *]'

        params = {
            'q': query,
            'fq': fq,
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
        """Download first CSV from dataset."""
        name = dataset.get('name', 'unknown')[:50]
        title = dataset.get('title', name)
        resources = [r for r in dataset.get('resources', []) if r.get('format', '').upper() == 'CSV']

        if not resources:
            return None

        url = resources[0].get('url')
        if not url:
            return None

        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        filepath = DATA_DIR / f"{category}__{safe_name}.csv"

        if filepath.exists():
            print(f"    [skip] exists")
            self.stats['skipped'] += 1
            return filepath

        try:
            print(f"    Downloading: {title[:45]}...")
            resp = self.session.get(url, stream=True, timeout=300)
            resp.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            size = filepath.stat().st_size
            self.stats['downloaded'] += 1
            self.stats['bytes'] += size

            # Save metadata with proper label
            meta = METADATA_DIR / f"{category}__{safe_name}.json"
            with open(meta, 'w') as f:
                json.dump({
                    'category': category,
                    'label': label,  # Human-readable label from FAVORITE_DATASETS.md
                    'title': title,
                    'name': name,
                    'org': dataset.get('organization', {}).get('title', 'Unknown'),
                    'modified': dataset.get('metadata_modified'),
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


class DataAnalyzer:
    def __init__(self):
        self.data = {}
        self.meta = {}

    def load_data(self):
        """Load all CSVs and their metadata."""
        print("\nLoading datasets...")

        for csv_path in sorted(DATA_DIR.glob("*.csv")):
            try:
                meta_path = METADATA_DIR / f"{csv_path.stem}.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.meta[csv_path.stem] = json.load(f)

                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc, low_memory=False, nrows=50000)
                        break
                    except:
                        continue

                if len(df) > 0:
                    self.data[csv_path.stem] = df
                    meta = self.meta.get(csv_path.stem, {})
                    label = meta.get('label', csv_path.stem)[:45]
                    print(f"  {label}: {len(df):,} rows")

            except Exception as e:
                print(f"  [error] {csv_path.name}: {e}")

        return self.data

    def get_label(self, key: str) -> str:
        """Get human-readable label from metadata."""
        meta = self.meta.get(key, {})
        return meta.get('label', meta.get('title', key.split('__')[-1]))[:45]

    def get_category(self, key: str) -> str:
        """Get category from metadata."""
        return self.meta.get(key, {}).get('category', key.split('__')[0])

    def create_overview(self):
        """Create overview with proper labels from FAVORITE_DATASETS.md."""
        if not self.data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Favorite Datasets Overview\n(Based on FAVORITE_DATASETS.md)', fontsize=14, fontweight='bold')

        # Group by category
        categories = {}
        for key in self.data:
            cat = self.get_category(key)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(key)

        # 1. Datasets by category
        ax1 = axes[0, 0]
        cats = list(categories.keys())
        counts = [len(categories[c]) for c in cats]
        colors = plt.cm.Set2(range(len(cats)))
        bars = ax1.barh(cats, counts, color=colors)
        ax1.set_xlabel('Number of Datasets')
        ax1.set_title('Datasets by Category')
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontsize=10)

        # 2. Data volume by category
        ax2 = axes[0, 1]
        rows_by_cat = {cat: sum(len(self.data[k]) for k in keys)
                       for cat, keys in categories.items()}
        ax2.barh(list(rows_by_cat.keys()), list(rows_by_cat.values()), color=colors)
        ax2.set_xlabel('Total Rows')
        ax2.set_title('Data Volume by Category')
        ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        # 3. Individual datasets with proper labels
        ax3 = axes[1, 0]
        labels = [self.get_label(k) for k in self.data.keys()]
        sizes = [len(df) for df in self.data.values()]
        y_pos = range(len(labels))
        ax3.barh(y_pos, sizes, color='steelblue')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels, fontsize=7)
        ax3.set_xlabel('Number of Rows')
        ax3.set_title('Individual Dataset Sizes')

        # 4. Category pie
        ax4 = axes[1, 1]
        ax4.pie(counts, labels=cats, autopct='%1.0f%%', colors=colors, startangle=90)
        ax4.set_title('Distribution by Category')

        plt.tight_layout()
        path = OUTPUT_DIR / 'overview_favorite_datasets.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path.name}")

    def create_distributions(self):
        """Create distribution plots with labels from FAVORITE_DATASETS.md."""
        for key, df in list(self.data.items())[:10]:
            numeric = df.select_dtypes(include=['number']).columns[:6]
            if len(numeric) < 2:
                continue

            label = self.get_label(key)
            category = self.get_category(key)

            n_cols = min(3, len(numeric))
            n_rows = (len(numeric) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            fig.suptitle(f'{label}\nCategory: {category}', fontsize=11, fontweight='bold')

            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

            for idx, col in enumerate(numeric):
                ax = axes_flat[idx]
                try:
                    data = df[col].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
                        clean_col = col.replace('_', ' ').title()[:30]
                        ax.set_title(clean_col, fontsize=9)
                        ax.tick_params(labelsize=7)
                except:
                    pass

            for idx in range(len(numeric), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            plt.tight_layout()
            safe = key[:45]
            path = OUTPUT_DIR / f'dist_{safe}.png'
            plt.savefig(path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path.name}")

    def create_correlations(self):
        """Create correlation heatmaps with proper labels."""
        for key, df in list(self.data.items())[:8]:
            numeric = df.select_dtypes(include=['number'])
            if len(numeric.columns) < 3:
                continue

            if len(numeric.columns) > 10:
                numeric = numeric.iloc[:, :10]

            label = self.get_label(key)
            category = self.get_category(key)

            try:
                corr = numeric.corr()
                clean_cols = [c.replace('_', ' ').title()[:18] for c in corr.columns]
                corr.columns = clean_cols
                corr.index = clean_cols

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                           fmt='.2f', ax=ax, annot_kws={'size': 7})
                ax.set_title(f'{label}\nCorrelation Matrix ({category})', fontsize=11, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)

                plt.tight_layout()
                path = OUTPUT_DIR / f'corr_{key[:45]}.png'
                plt.savefig(path, dpi=120, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {path.name}")
            except Exception as e:
                print(f"  Skip correlation for {key}: {e}")

    def generate_report(self, stats: dict):
        """Generate report with proper dataset labels."""
        lines = [
            "=" * 65,
            "FAVORITE DATASETS ANALYSIS REPORT",
            "Based on FAVORITE_DATASETS.md",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 65,
            "",
            "DOWNLOAD SUMMARY",
            f"  Found:      {stats.get('found', 0)}",
            f"  Downloaded: {stats.get('downloaded', 0)}",
            f"  Skipped:    {stats.get('skipped', 0)}",
            f"  Failed:     {stats.get('failed', 0)}",
            f"  Total Size: {stats.get('bytes', 0) / 1024 / 1024:.1f} MB",
            "",
            "DATASETS LOADED",
        ]

        for key, df in self.data.items():
            meta = self.meta.get(key, {})
            label = meta.get('label', key)
            lines.append(f"\n--- {label} ---")
            lines.append(f"Category: {meta.get('category', 'Unknown')}")
            lines.append(f"Source: {meta.get('org', 'Unknown')}")
            lines.append(f"Rows: {len(df):,}")
            lines.append(f"Columns: {len(df.columns)}")

            numeric = df.select_dtypes(include=['number']).columns
            if len(numeric) > 0:
                lines.append(f"Numeric columns: {len(numeric)}")
                for col in numeric[:3]:
                    try:
                        s = df[col].describe()
                        lines.append(f"  {col}: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")
                    except:
                        pass

        lines.append("\n" + "=" * 65)

        report_path = OUTPUT_DIR / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Saved: {report_path.name}")


def main():
    print("=" * 65)
    print("FAVORITE DATASETS PIPELINE")
    print("Based on FAVORITE_DATASETS.md")
    print("=" * 65)
    print(f"Data:    {DATA_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Targets: {len(FAVORITE_DATASETS)} datasets")
    print("=" * 65)

    # DOWNLOAD
    print("\n[1] DOWNLOADING FROM data.gov")
    print("-" * 45)

    dl = DataDownloader()

    for config in FAVORITE_DATASETS:
        print(f"\n>>> {config['label']}")
        datasets = dl.search(config['query'], config['max'], months=6)

        if datasets:
            print(f"  Found {len(datasets)} matches")
            for ds in datasets:
                dl.download(ds, config['name'], config['label'])
        else:
            print("  No recent data found")

    print(f"\n{'=' * 45}")
    print(f"Downloaded: {dl.stats['downloaded']} files ({dl.stats['bytes']/1024/1024:.1f} MB)")
    print(f"Skipped: {dl.stats['skipped']}, Failed: {dl.stats['failed']}")

    if not HAS_ANALYTICS:
        print("\nSkipping analysis - install pandas/matplotlib/seaborn")
        return

    # ANALYZE
    print("\n[2] ANALYZING DATA")
    print("-" * 45)

    analyzer = DataAnalyzer()
    analyzer.load_data()

    if not analyzer.data:
        print("No data to analyze")
        return

    # VISUALIZE
    print("\n[3] CREATING VISUALIZATIONS")
    print("-" * 45)

    print("\nOverview chart...")
    analyzer.create_overview()

    print("\nDistribution plots...")
    analyzer.create_distributions()

    print("\nCorrelation heatmaps...")
    analyzer.create_correlations()

    print("\nAnalysis report...")
    analyzer.generate_report(dl.stats)

    # DONE
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)

    viz_files = list(OUTPUT_DIR.glob("*.png"))
    print(f"\nVisualizations: {len(viz_files)} files")
    print(f"Location: {OUTPUT_DIR}")

    print("\nTo push to GitHub:")
    print("  cd /c/Users/Guest2/Personal/Github/opendata")
    print("  git add .")
    print('  git commit -m "Add favorite datasets analysis"')
    print("  git push")
    print("=" * 65)


if __name__ == "__main__":
    main()
