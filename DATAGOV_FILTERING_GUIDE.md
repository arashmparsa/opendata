# Data.gov Dataset Filtering Guide
## ~300,000+ Datasets - How to Find What You Need

---

## Quick Start

```python
from datagov_bulk_downloader import DataGovDownloader, TOPIC_PRESETS

downloader = DataGovDownloader(output_dir="./my_data")

# Use a preset topic
datasets = downloader.search_all_datasets(
    tags=TOPIC_PRESETS['health']['tags'],
    formats=['CSV'],
    max_results=100
)
```

---

## Dataset Categories on Data.gov

### By Topic/Domain

| Category | Estimated Datasets | Key Tags | Major Publishers |
|----------|-------------------|----------|------------------|
| **Health & Medical** | ~50,000 | health, healthcare, disease, hospital, medicare | HHS, CDC, NIH, CMS |
| **Climate & Environment** | ~40,000 | climate, weather, emissions, ocean, pollution | NOAA, EPA, NASA |
| **Finance & Economy** | ~30,000 | finance, budget, spending, economy, loans | Treasury, SEC, FDIC |
| **Transportation** | ~25,000 | transportation, traffic, aviation, roads | DOT, FAA, NHTSA |
| **Energy** | ~20,000 | energy, electricity, oil, gas, renewable | DOE, EIA |
| **Agriculture & Food** | ~15,000 | agriculture, food, crops, livestock | USDA, FDA |
| **Demographics & Census** | ~15,000 | census, population, demographics, housing | Census Bureau, BLS |
| **Education** | ~10,000 | education, schools, students, college | Dept. of Education |
| **Defense & Security** | ~8,000 | defense, military, veterans, security | DoD, DHS, VA |
| **Geospatial** | ~50,000 | geographic, mapping, boundaries, coordinates | USGS, Census |

### By File Format

| Format | Count | Best For |
|--------|-------|----------|
| **CSV** | ~100,000 | Tabular data, easy to process |
| **JSON** | ~50,000 | APIs, structured data |
| **XML** | ~30,000 | Legacy systems, complex hierarchies |
| **GeoJSON/KML/Shapefile** | ~40,000 | Geographic/mapping data |
| **PDF** | ~20,000 | Reports (harder to process) |
| **API** | ~10,000 | Live/streaming data |
| **XLS/XLSX** | ~15,000 | Spreadsheets with formatting |

---

## Filtering Strategies

### 1. By Sensitivity Level (Your Original Interest)

**Potentially Sensitive Categories:**

```python
# Infrastructure & Critical Systems
sensitive_infrastructure = downloader.search_all_datasets(
    query="infrastructure OR facilities OR utilities",
    tags=['infrastructure', 'facilities', 'buildings', 'power grid'],
    formats=['CSV', 'JSON'],
    max_results=500
)

# Geographic/Location Data
location_data = downloader.search_all_datasets(
    query="coordinates OR latitude OR longitude OR address",
    formats=['CSV', 'GeoJSON', 'KML'],
    max_results=500
)

# Demographic + Geographic Combinations
demographic_geo = downloader.search_all_datasets(
    tags=['census', 'demographics', 'population'],
    formats=['CSV'],
    max_results=500
)

# Health Disparities by Region
health_regional = downloader.search_all_datasets(
    query="health disparity OR mortality rate OR disease prevalence",
    tags=['health', 'demographics'],
    max_results=500
)

# Emergency Response Data
emergency_data = downloader.search_all_datasets(
    query="emergency response OR first responder OR 911",
    tags=['emergency', 'public safety'],
    max_results=200
)

# Federal Facility Locations
federal_facilities = downloader.search_all_datasets(
    query="federal facility OR government building OR military base",
    formats=['CSV', 'GeoJSON'],
    max_results=300
)
```

### 2. By Data Freshness

```python
from datetime import datetime, timedelta

# Recent data (last 6 months)
recent_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

recent_datasets = downloader.search_all_datasets(
    date_from=recent_date,
    formats=['CSV'],
    max_results=1000
)

# Historical data (specific year)
historical = downloader.search_all_datasets(
    date_from='2020-01-01',
    date_to='2020-12-31',
    max_results=500
)
```

### 3. By Publishing Organization

```python
# Major data publishers
MAJOR_PUBLISHERS = {
    'hhs-gov': 'Health & Human Services',
    'noaa-gov': 'Weather/Ocean/Climate',
    'census-gov': 'Demographics/Population',
    'epa-gov': 'Environmental',
    'dot-gov': 'Transportation',
    'treasury-gov': 'Financial',
    'usda-gov': 'Agriculture',
    'doe-gov': 'Energy',
    'dod-gov': 'Defense',
    'nasa-gov': 'Space/Earth Science',
    'fda-gov': 'Food/Drug Safety',
    'cdc-gov': 'Disease Control',
    'sec-gov': 'Securities/Finance',
    'dhs-gov': 'Homeland Security',
    'va-gov': 'Veterans Affairs',
}

# Get all data from specific agency
epa_data = downloader.search_all_datasets(
    organizations=['epa-gov'],
    formats=['CSV'],
    max_results=1000
)
```

### 4. By Use Case

```python
# For Machine Learning Training
ml_friendly = downloader.search_all_datasets(
    formats=['CSV', 'JSON'],  # Easy to parse
    query="dataset",
    max_results=5000
)

# For Geographic Analysis
geo_data = downloader.search_all_datasets(
    formats=['GeoJSON', 'KML', 'Shapefile'],
    max_results=2000
)

# For Time Series Analysis
timeseries = downloader.search_all_datasets(
    query="annual OR monthly OR daily OR time series OR historical",
    formats=['CSV'],
    max_results=2000
)

# For NLP/Text Analysis
text_data = downloader.search_all_datasets(
    query="text OR document OR report OR survey responses",
    formats=['CSV', 'JSON', 'XML'],
    max_results=1000
)
```

---

## High-Value Dataset Categories

### For AI/ML Training (Potentially Concerning Uses)

| Category | Why Sensitive | Example Datasets |
|----------|--------------|------------------|
| **Facial Recognition Training** | Privacy, surveillance | Mugshots, ID photos |
| **Predictive Policing** | Bias, civil liberties | Crime statistics by location |
| **Credit Scoring** | Discrimination | Income by demographics |
| **Health Risk Prediction** | Insurance discrimination | Disease rates by region |
| **Infrastructure Targeting** | Security | Facility locations, vulnerabilities |
| **Behavioral Profiling** | Privacy | Survey responses, social patterns |

### Search Queries for These Categories

```python
# Crime/Law Enforcement
crime_data = downloader.search_all_datasets(
    query="crime OR arrest OR incident OR enforcement",
    formats=['CSV'],
    max_results=500
)

# Financial/Credit Related
financial_data = downloader.search_all_datasets(
    query="loan OR credit OR mortgage OR income",
    formats=['CSV'],
    max_results=500
)

# Behavioral/Survey Data
survey_data = downloader.search_all_datasets(
    query="survey OR behavior OR response OR questionnaire",
    formats=['CSV'],
    max_results=500
)

# Vulnerability Assessments
vulnerability_data = downloader.search_all_datasets(
    query="vulnerability OR risk assessment OR hazard",
    formats=['CSV', 'JSON'],
    max_results=300
)
```

---

## Bulk Download Strategies

### Strategy 1: Metadata First, Download Later

```python
# Step 1: Search and save metadata (fast)
downloader = DataGovDownloader(output_dir="./data_audit")

all_csv_datasets = downloader.search_all_datasets(
    formats=['CSV'],
    max_results=10000
)

# Save for review
downloader.save_search_results(all_csv_datasets, "all_csv_metadata.json")

# Step 2: Review the JSON file manually
# Step 3: Filter and download only what you need
```

### Strategy 2: Download by Topic Batches

```python
for topic, config in TOPIC_PRESETS.items():
    print(f"\nProcessing: {topic}")
    
    datasets = downloader.search_all_datasets(
        tags=config['tags'],
        formats=['CSV'],
        max_results=100
    )
    
    downloader.save_search_results(
        datasets, 
        f"{topic}_datasets.json"
    )
```

### Strategy 3: Full Catalog Download

```python
# Download the bulk metadata file (2.3GB compressed)
# Contains ALL federal dataset metadata
import requests
import gzip

BULK_URL = "https://filestore.data.gov/gsa/catalog/jsonl/dataset.jsonl.gz"

# Download
response = requests.get(BULK_URL, stream=True)
with open('all_metadata.jsonl.gz', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Process line by line
with gzip.open('all_metadata.jsonl.gz', 'rt') as f:
    for line in f:
        dataset = json.loads(line)
        # Filter and process as needed
```

---

## Rate Limiting & Best Practices

```python
# Respectful downloading
downloader = DataGovDownloader(
    rate_limit_delay=1.0,      # 1 second between API calls
    max_retries=3,             # Retry failed downloads
    timeout=120                # 2 minute timeout for large files
)

# For large downloads, consider:
# - Running overnight
# - Using a VPN/different IP if rate limited
# - Breaking into smaller batches
# - Downloading metadata first, data second
```

---

## API Key (Optional but Recommended)

For heavy usage, get a free API key:
1. Go to: https://api.data.gov/signup/
2. Register with your email
3. Use the key:

```python
downloader = DataGovDownloader(
    api_key="YOUR_API_KEY_HERE"
)
```

---

## Sample Workflow

```python
#!/usr/bin/env python3
"""
Complete workflow to audit and download data.gov datasets
"""

from datagov_bulk_downloader import DataGovDownloader, TOPIC_PRESETS
import json

def main():
    # Initialize
    downloader = DataGovDownloader(
        output_dir="./datagov_analysis",
        rate_limit_delay=0.5
    )
    
    # 1. Explore available data
    print("Exploring available formats...")
    formats = downloader.get_format_counts()
    print(json.dumps(formats, indent=2))
    
    # 2. Search for specific topic
    print("\nSearching for infrastructure data...")
    infra_datasets = downloader.search_all_datasets(
        query="infrastructure",
        tags=['infrastructure', 'facilities'],
        formats=['CSV', 'JSON'],
        max_results=200
    )
    
    # 3. Save metadata for review
    downloader.save_search_results(infra_datasets, "infrastructure_audit.json")
    
    # 4. Print summary
    print(f"\nFound {len(infra_datasets)} datasets")
    print("\nTop 10 datasets:")
    for ds in infra_datasets[:10]:
        print(f"  - {ds.get('title', 'Unknown')[:60]}...")
        print(f"    Org: {ds.get('organization', {}).get('title', 'Unknown')}")
        print(f"    Resources: {len(ds.get('resources', []))}")
    
    # 5. Optionally download
    # downloader.bulk_download(infra_datasets[:10], formats=['CSV'])
    
    downloader.print_stats()

if __name__ == "__main__":
    main()
```

---

## Notes on "Sensitive" Public Data

Even though all data on data.gov is public, consider:

1. **Aggregation creates new information** - Combining datasets can reveal patterns not visible in individual datasets

2. **Context matters** - Hospital locations are public, but combined with demographic data and response times, they become a targeting map

3. **Dual-use potential** - Infrastructure data serves legitimate planning purposes but could inform attacks

4. **Re-identification risk** - Anonymous datasets can often be de-anonymized when combined with other data

5. **Model encoding** - ML models trained on this data may "memorize" sensitive correlations even if individual records aren't exposed

---

## Resources

- **Data.gov Homepage**: https://data.gov
- **CKAN API Docs**: https://docs.ckan.org/en/latest/api/
- **GSA API Portal**: https://open.gsa.gov/api/datadotgov/
- **Bulk Metadata**: https://filestore.data.gov/gsa/catalog/jsonl/dataset.jsonl.gz
