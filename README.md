# Aviation Accident Analysis

**ECE 143 - Programming for Data Analysis**

A comprehensive data analysis project examining historical aviation accidents from 1921 to present. This project scrapes, cleans, and visualizes data from [PlaneCrashInfo.com](https://www.planecrashinfo.com/) to uncover patterns and insights about aviation safety trends.

---

## Project Overview

This project performs end-to-end data analysis on aviation accident records:

1. **Data Scraping** - Automated collection of accident records from online database
2. **Data Cleaning** - Standardization, parsing, and feature extraction
3. **Visualization** - 25+ static plots and interactive maps
4. **Analysis** - Temporal, geographical, and categorical insights

---

## File Structure

```
ECE-143/
├── data_scraping.py              # Web scraper for planecrashinfo.com
├── data_cleaning.py              # Data processing and feature engineering
├── visualization.py              # Static plots (matplotlib/seaborn)
├── map.py                        # Interactive maps (plotly)
├── analysis_notebook.ipynb       # Jupyter notebook with all visualizations
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── planecrashinfo_accidents.csv  # Raw scraped data
├── planecrashinfo_clean.csv      # Cleaned dataset
├── plots/                        # Generated visualization images (26 plots)
│   ├── yearly_crashes_fatalities.png
│   ├── top_countries_accidents.png
│   ├── survival_rate_by_decade.png
│   ├── cumulative_fatalities.png
│   ├── ... (and more)
└── maps/                         # Interactive HTML maps
    ├── accidents_by_country.html
    ├── accidents_by_country_decade.html
    └── fatality_ratio_by_country.html
```

### Python Modules (.py files)

| File | Description |
|------|-------------|
| `data_scraping.py` | Web scraper that collects accident data from planecrashinfo.com |
| `data_cleaning.py` | Cleans raw data, parses fields, and extracts features |
| `visualization.py` | Contains 25+ plotting functions for static visualizations |
| `map.py` | Generates interactive choropleth maps using Plotly |

### Jupyter Notebook

| File | Description |
|------|-------------|
| `analysis_notebook.ipynb` | Complete presentation notebook with all visualizations |

---

## 📦 Third-Party Modules

This project uses the following third-party Python packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.5.0 | Data manipulation and analysis |
| `numpy` | >=1.23.0 | Numerical computing |
| `matplotlib` | >=3.6.0 | Static visualizations |
| `seaborn` | >=0.12.0 | Statistical data visualization |
| `plotly` | >=5.10.0 | Interactive maps and charts |
| `requests` | >=2.28.0 | HTTP requests for web scraping |
| `beautifulsoup4` | >=4.11.0 | HTML parsing for web scraping |
| `lxml` | >=4.9.0 | XML/HTML parser |


## Visualizations

### Static Plots (25+ visualizations)

| Category | Plots |
|----------|-------|
| **Temporal Analysis** | Yearly crashes & fatalities, Survival rate by decade, Cumulative fatalities, Top deadly years, Monthly patterns |
| **Geographic Analysis** | Top countries by accidents |
| **Operator Analysis** | Top operators, Operator safety comparison |
| **Aircraft Analysis** | Category trends, Decade proportions, Fatality ratios by type, Category-decade heatmap |
| **Severity Analysis** | Fatality ratio distributions, Aboard vs fatalities scatter, Crew vs passenger fatalities |
| **Time-of-Day Analysis** | Hourly accident histogram, Hourly severity, Stacked severity by hour |
| **Weather & Phase** | Weather condition counts, Weather vs fatality ratio, Flight phase analysis, Phase-fatality heatmap |
| **Ground Impact** | Ground fatalities analysis |

### Interactive Maps

1. **accidents_by_country.html** - Choropleth of total accidents per country
2. **accidents_by_country_decade.html** - Animated choropleth showing accidents over time
3. **fatality_ratio_by_country.html** - Country-level severity visualization

---

## Dataset Schema

### Raw Fields (from scraping)
| Field | Description |
|-------|-------------|
| `date` | Accident date |
| `time` | Time of accident |
| `location` | Crash location |
| `operator` | Aircraft operator |
| `flight_no` | Flight number |
| `route` | Flight route |
| `aircraft_type` | Aircraft make/model |
| `registration` | Aircraft registration |
| `cn_ln` | Construction/line number |
| `aboard` | People aboard |
| `fatalities` | Fatality information |
| `ground_fatalities` | Ground casualties |
| `summary` | Accident description |

### Derived Fields (from cleaning)
| Field | Description |
|-------|-------------|
| `date_parsed` | Parsed datetime |
| `time_hhmm` | Normalized time (HH:MM) |
| `fatalities_total` | Total fatalities |
| `fatalities_passengers` | Passenger fatalities |
| `fatalities_crew` | Crew fatalities |
| `location_city` | Extracted city |
| `location_state` | Extracted state |
| `location_country` | Extracted country |
| `aircraft_category` | Categorized aircraft type |
| `phase_clean` | Flight phase at accident |
| `weather_condition` | Inferred weather |
| `weather_adverse` | Boolean adverse weather flag |

---

## Key Insights

### Temporal Trends
- **Peak decade:** 1970s-1980s saw the highest number of accidents
- **Improving safety:** Fatality ratios have generally decreased over time
- **Seasonal patterns:** Slight variations in accident frequency by month

### Geographic Distribution
- **United States** has the highest number of recorded accidents (likely due to aviation activity volume)
- **Fatality ratios** vary significantly by country

### Aircraft & Operator Patterns
- **Jet aircraft** involvement increased dramatically from 1960s onward
- **Piston/Prop aircraft** dominated early aviation history
- Significant variation in safety records across operators

### Time-of-Day Analysis
- Accidents occur throughout the day with some hourly variations
- Fatality severity shows patterns related to visibility conditions

### Weather & Flight Phase
- **Approach and landing** phases show significant accident concentration
- Adverse weather conditions correlate with higher fatality ratios
