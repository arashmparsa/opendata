# Data.gov: What's Actually Useful?
## Filtering 370,000 Datasets Down to What Matters

---

## The Reality: 90%+ Is Noise

Most data.gov datasets are **not useful** for ML training because:

| Problem | % of Datasets | Examples |
|---------|---------------|----------|
| **Broken/dead links** | ~15-20% | URLs pointing to defunct servers |
| **PDFs only** | ~20% | Scanned documents, reports (not machine-readable) |
| **Tiny datasets** | ~30% | <100 rows, single tables |
| **Duplicate/redundant** | ~10% | Same data published by multiple agencies |
| **API-only (no bulk)** | ~10% | Requires authenticated access |
| **Outdated** | ~15% | Data from 2010 that's never updated |

**Realistically useful**: ~5-15% of the catalog (~20,000-50,000 datasets)

---

## High-Value Categories for ML Training

### Tier 1: Gold Standard (Large, Clean, Updated)

#### 1. **Census & Demographics** (~5,000 datasets)
- American Community Survey (ACS)
- Decennial Census microdata
- Population estimates by geography
- **Why valuable**: Labeled, structured, massive scale, updated regularly
- **ML uses**: Demographic prediction, geographic modeling, economic forecasting
- **Sensitivity concern**: Profiling, redlining, discriminatory targeting

#### 2. **Weather & Climate** (~20,000 datasets)
- NOAA daily observations (billions of records)
- Historical climate data (100+ years)
- Satellite imagery
- **Why valuable**: Time series, spatial data, high volume
- **ML uses**: Forecasting, anomaly detection, climate modeling
- **Size**: Petabytes (largest single category)

#### 3. **Health & Epidemiology** (~15,000 datasets)
- Medicare/Medicaid claims (aggregated)
- Disease surveillance (CDC)
- Hospital quality metrics
- Drug adverse events (FDA)
- **Why valuable**: Labeled outcomes, large populations
- **ML uses**: Risk prediction, drug discovery, resource allocation
- **Sensitivity concern**: Health discrimination, insurance profiling

#### 4. **Transportation & Infrastructure** (~8,000 datasets)
- Flight data (every US flight since 1987)
- Traffic counts and patterns
- Bridge/road condition assessments
- **Why valuable**: Complete coverage, time series
- **ML uses**: Optimization, predictive maintenance, routing
- **Sensitivity concern**: Infrastructure vulnerability mapping

#### 5. **Financial & Economic** (~10,000 datasets)
- SEC filings (structured)
- Small business loans (SBA)
- Mortgage data (HMDA)
- Consumer complaints (CFPB)
- **Why valuable**: Labeled outcomes, diverse features
- **ML uses**: Credit scoring, fraud detection, market prediction
- **Sensitivity concern**: Discriminatory lending models

---

### Tier 2: Specialized Value

#### 6. **Geospatial/GIS** (~50,000 datasets)
- Land use/cover
- Building footprints
- Boundaries (political, zip codes)
- Elevation models
- **Why valuable**: Foundation for spatial ML
- **ML uses**: Computer vision, routing, urban planning

#### 7. **Scientific Research** (~30,000 datasets)
- NIH-funded study data
- EPA environmental monitoring
- USGS geological surveys
- **Why valuable**: High quality, peer-reviewed
- **ML uses**: Domain-specific models

#### 8. **Education** (~5,000 datasets)
- School performance metrics
- College scorecards
- Student loan outcomes
- **Why valuable**: Outcome labels, longitudinal
- **ML uses**: Prediction, policy analysis

---

### Tier 3: Niche but Valuable

| Category | Key Datasets | ML Application |
|----------|--------------|----------------|
| **Agriculture** | Crop yields, soil surveys | Yield prediction |
| **Energy** | Power plant emissions, grid data | Demand forecasting |
| **Crime** | FBI UCR, local incident data | Pattern detection |
| **Patents** | USPTO full-text patents | NLP, innovation analysis |
| **Contracts** | Federal procurement (FPDS) | Pricing models |

---

## What Makes a Dataset Actually Useful?

### Quality Indicators (in metadata)

```python
def is_useful_dataset(dataset):
    """
    Score a dataset's usefulness based on metadata.
    """
    score = 0
    
    # Format (most important)
    formats = [r.get('format', '').upper() for r in dataset.get('resources', [])]
    if 'CSV' in formats or 'JSON' in formats:
        score += 30
    elif 'XML' in formats:
        score += 15
    elif 'API' in formats:
        score += 10
    # PDF/HTML only = basically useless
    if formats and all(f in ['PDF', 'HTML', 'DOC'] for f in formats):
        return 0
    
    # Recency
    modified = dataset.get('metadata_modified', '')
    if '2024' in modified or '2025' in modified:
        score += 20
    elif '2023' in modified:
        score += 10
    
    # Has actual downloadable resources
    resources = dataset.get('resources', [])
    downloadable = [r for r in resources if r.get('url', '').startswith('http')]
    if len(downloadable) > 0:
        score += 15
    
    # Organization reputation
    good_orgs = ['census-gov', 'noaa-gov', 'cdc-gov', 'cms-gov', 'bls-gov', 
                 'sec-gov', 'dot-gov', 'epa-gov', 'nih-gov', 'fda-gov']
    org = dataset.get('organization', {}).get('name', '')
    if org in good_orgs:
        score += 15
    
    # Has description (indicates curation)
    if len(dataset.get('notes', '')) > 100:
        score += 10
    
    # Tags indicate ML-readiness
    tags = [t.get('name', '').lower() for t in dataset.get('tags', [])]
    ml_tags = ['time-series', 'statistics', 'survey', 'census', 'records']
    if any(t in tags for t in ml_tags):
        score += 10
    
    return score
```

### Red Flags (Skip These)

```python
def has_red_flags(dataset):
    """
    Identify datasets to skip.
    """
    red_flags = []
    
    # No actual data links
    resources = dataset.get('resources', [])
    if not resources:
        red_flags.append('no_resources')
    
    # Only PDFs
    formats = [r.get('format', '').upper() for r in resources]
    if formats and all(f in ['PDF', 'DOC', 'DOCX'] for f in formats):
        red_flags.append('pdf_only')
    
    # Very old
    modified = dataset.get('metadata_modified', '')
    if modified and modified < '2018':
        red_flags.append('outdated')
    
    # Known bad patterns in URLs
    for r in resources:
        url = r.get('url', '')
        if 'arcgis' in url.lower() and 'services' in url.lower():
            red_flags.append('arcgis_service')  # Often requires auth
        if 'ftp://' in url:
            red_flags.append('ftp_link')  # Often broken
    
    # Suspiciously small
    title = dataset.get('title', '').lower()
    if 'annual report' in title or 'fact sheet' in title:
        red_flags.append('likely_report')
    
    return red_flags
```

---

## Recommended Download Strategy

### Phase 1: High-Value Targets (~1-5 TB)

```python
# Priority downloads
PRIORITY_QUERIES = [
    # Census (demographics)
    {'tags': ['census', 'demographics'], 'formats': ['CSV'], 'orgs': ['census-gov']},
    
    # Health outcomes
    {'tags': ['health', 'medicare', 'disease'], 'formats': ['CSV'], 'orgs': ['cdc-gov', 'cms-gov']},
    
    # Financial
    {'tags': ['loans', 'mortgage', 'financial'], 'formats': ['CSV'], 'orgs': ['cfpb', 'sec-gov']},
    
    # Transportation
    {'tags': ['aviation', 'traffic', 'transportation'], 'formats': ['CSV'], 'orgs': ['dot-gov', 'faa-gov']},
]
```

### Phase 2: Broad Coverage (~10-50 TB)

```python
# Secondary targets
SECONDARY_QUERIES = [
    # All CSV files from top agencies
    {'formats': ['CSV', 'JSON'], 'orgs': GOOD_ORGS, 'min_score': 50},
    
    # Time series data
    {'query': 'annual monthly daily time series', 'formats': ['CSV']},
    
    # Labeled outcomes
    {'query': 'outcome results performance score', 'formats': ['CSV']},
]
```

### Phase 3: Specialized (~50-200 TB)

```python
# Domain-specific
SPECIALIZED = [
    # Geospatial (large)
    {'formats': ['GeoJSON', 'Shapefile'], 'tags': ['geographic', 'boundaries']},
    
    # Scientific
    {'orgs': ['nasa-gov', 'noaa-gov', 'usgs-gov'], 'formats': ['CSV', 'NetCDF']},
]
```

---

## Top 50 Most Valuable Specific Datasets

### Demographics & Census
1. American Community Survey (ACS) 5-Year Estimates
2. Decennial Census Summary Files
3. Current Population Survey (CPS) Microdata
4. County Business Patterns
5. Longitudinal Employer-Household Dynamics (LEHD)

### Health
6. Medicare Provider Utilization and Payment Data
7. Hospital Compare Quality Measures
8. CDC WONDER Mortality Data
9. FDA Adverse Event Reporting System (FAERS)
10. NHANES Survey Data
11. COVID-19 Case Surveillance
12. Medicaid Drug Utilization

### Financial
13. HMDA Mortgage Data (billions of records)
14. SBA Loan Data (7(a) and 504 programs)
15. CFPB Consumer Complaint Database
16. SEC EDGAR Filings (structured)
17. FDIC Failed Banks List
18. USASpending Federal Contracts

### Transportation
19. Bureau of Transportation Statistics (all flights since 1987)
20. NHTSA Traffic Fatality Data (FARS)
21. National Bridge Inventory
22. Highway Performance Monitoring System
23. Transit Ridership Data

### Climate & Environment
24. NOAA Global Historical Climatology Network (GHCN)
25. EPA Air Quality System (AQS)
26. USGS Water Data
27. National Land Cover Database
28. Toxic Release Inventory (TRI)

### Economic
29. Bureau of Labor Statistics (all series)
30. Bureau of Economic Analysis (GDP, trade)
31. Federal Reserve Economic Data (FRED)
32. Import/Export Trade Data
33. Producer/Consumer Price Indices

### Education
34. College Scorecard
35. IPEDS (Integrated Postsecondary Education)
36. Civil Rights Data Collection
37. National Assessment of Educational Progress (NAEP)

### Crime & Safety
38. FBI Uniform Crime Reports
39. NICS Firearm Background Checks
40. OSHA Inspection Data
41. FAA Wildlife Strike Database

### Geospatial
42. TIGER/Line Shapefiles (Census)
43. National Address Database
44. Building Footprints
45. Protected Areas Database

### Scientific
46. PubChem Chemical Data
47. GenBank Sequences
48. Landsat Satellite Imagery
49. USGS Earthquake Catalog
50. NASA Earth Observation Data

---

## Datasets to AVOID for Ethical ML

### High Risk for Discriminatory Outcomes
- Crime data by demographics → biased policing models
- Loan denial data by zip code → redlining
- Health outcomes by race → insurance discrimination
- School discipline data → profiling students

### High Risk for Privacy Violations
- Voter registration + demographics → political targeting
- Health facility + demographics → medical profiling
- Income estimates + location → financial surveillance

### High Risk for Security Concerns
- Critical infrastructure locations
- Emergency response times by area
- Federal facility details
- Cybersecurity incident data

---

## Quick Filter Script

```python
#!/usr/bin/env python3
"""
Filter data.gov bulk metadata for useful datasets.
"""

import json
import gzip
from collections import Counter

# Download first: https://filestore.data.gov/gsa/catalog/jsonl/dataset.jsonl.gz

GOOD_FORMATS = {'CSV', 'JSON', 'XML', 'GEOJSON', 'XLSX', 'XLS'}
GOOD_ORGS = {
    'census-gov', 'noaa-gov', 'cdc-gov', 'cms-gov', 'bls-gov',
    'sec-gov', 'dot-gov', 'epa-gov', 'nih-gov', 'fda-gov',
    'usda-gov', 'treasury-gov', 'dol-gov', 'hud-gov', 'ed-gov'
}

def filter_useful_datasets(input_file='dataset.jsonl.gz', output_file='useful_datasets.jsonl'):
    """
    Filter bulk metadata file for useful datasets.
    """
    stats = Counter()
    useful = []
    
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                ds = json.loads(line)
                stats['total'] += 1
                
                # Get formats
                resources = ds.get('resources', [])
                formats = {r.get('format', '').upper() for r in resources}
                
                # Skip if no good formats
                if not formats & GOOD_FORMATS:
                    stats['bad_format'] += 1
                    continue
                
                # Skip if no downloadable URLs
                urls = [r.get('url', '') for r in resources if r.get('url', '').startswith('http')]
                if not urls:
                    stats['no_urls'] += 1
                    continue
                
                # Prefer good organizations
                org = ds.get('organization', {}).get('name', '')
                
                # Score it
                score = 0
                if 'CSV' in formats: score += 30
                if 'JSON' in formats: score += 20
                if org in GOOD_ORGS: score += 20
                if len(ds.get('notes', '')) > 200: score += 10
                
                # Check recency
                modified = ds.get('metadata_modified', '')
                if '2024' in modified or '2025' in modified: score += 20
                elif '2023' in modified: score += 10
                
                if score >= 40:
                    ds['_usefulness_score'] = score
                    useful.append(ds)
                    stats['useful'] += 1
                else:
                    stats['low_score'] += 1
                    
            except json.JSONDecodeError:
                stats['parse_error'] += 1
    
    # Save useful datasets
    with open(output_file, 'w') as f:
        for ds in sorted(useful, key=lambda x: -x['_usefulness_score']):
            f.write(json.dumps(ds) + '\n')
    
    print(f"Filtering complete:")
    print(f"  Total datasets: {stats['total']:,}")
    print(f"  Bad format: {stats['bad_format']:,}")
    print(f"  No URLs: {stats['no_urls']:,}")
    print(f"  Low score: {stats['low_score']:,}")
    print(f"  Useful datasets: {stats['useful']:,}")
    print(f"  Saved to: {output_file}")
    
    return useful

if __name__ == '__main__':
    filter_useful_datasets()
```

---

## Summary

| Category | Est. Useful Datasets | Est. Size | ML Value |
|----------|---------------------|-----------|----------|
| Census/Demographics | 2,000 | 500 GB | ⭐⭐⭐⭐⭐ |
| Health/Medical | 5,000 | 2 TB | ⭐⭐⭐⭐⭐ |
| Weather/Climate | 10,000 | 50+ TB | ⭐⭐⭐⭐ |
| Financial | 3,000 | 1 TB | ⭐⭐⭐⭐⭐ |
| Transportation | 2,000 | 500 GB | ⭐⭐⭐⭐ |
| Geospatial | 10,000 | 20 TB | ⭐⭐⭐⭐ |
| Education | 1,000 | 100 GB | ⭐⭐⭐ |
| Scientific | 5,000 | 100+ TB | ⭐⭐⭐⭐ |
| **TOTAL USEFUL** | **~30,000-40,000** | **~50-200 TB** | - |

Out of 370,000 datasets, roughly **30,000-40,000 are actually useful** for ML training, totaling approximately **50-200 terabytes** of downloadable, structured data.
