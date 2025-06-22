# Turtle Games Dataset Metadata

## Overview
This document contains metadata for the combined Turtle Games dataset, consisting of two primary CSV files that provide customer review data and global sales data for video games.

**Source Files:**
- `turtle_reviews.csv` - Customer demographics and review data
- `turtle_sales.csv` - Global video game sales data

---

## Dataset 1: turtle_reviews.csv

### Description
Customer review and demographic data from Turtle Games' web platform.

### Schema

| Column | Data Type | Sample Value | Description |
|--------|-----------|--------------|-------------|
| `gender` | string | "male", "female" | Customer gender |
| `age` | integer | 25 | Customer age in years |
| `remuneration` | float | 45.5 | Annual customer income in thousands of pounds (k£) |
| `spending_score` | integer (1-100) | 67 | Turtle Games proprietary spending behavior score |
| `loyalty_points` | integer | 1250 | Points based on purchase value and customer actions |
| `education` | string | "graduate" | Education level: Diploma, Graduate, Postgraduate, PhD |
| `language` | string | "EN" | Review language (all English) |
| `platform` | string | "Web" | Review collection platform (all Web) |
| `product` | integer | 12345 | Unique product identifier |
| `review` | text | "When it comes to a DM's screen..." | Full customer review text |
| `summary` | text | "The fact that 50% of this..." | Review summary |

### Data Quality Notes
- All reviews collected from Turtle Games website
- All reviews in English language
- Spending score is proprietary algorithm (1-100 scale)
- Loyalty points convert monetary value to point system

---

## Dataset 2: turtle_sales.csv

### Description
Global video game sales data with platform and regional breakdowns.

### Schema

| Column | Data Type | Sample Value | Description |
|--------|-----------|--------------|-------------|
| `Ranking` | integer | 1 | Global sales ranking |
| `Product` | integer | 12345 | Unique product identifier (links to reviews) |
| `Platform` | string | "Wii" | Gaming console/platform |
| `Year` | integer | 2008 | Initial release year |
| `Genre` | string | "Sports" | Game genre classification |
| `Publisher` | string | "Nintendo" | Publishing company |
| `NA_Sales` | float | 15.75 | North America sales (millions of pounds) |
| `EU_Sales` | float | 8.89 | Europe sales (millions of pounds) |
| `Global_Sales` | float | 82.74 | Total worldwide sales (millions of pounds) |

### Data Quality Notes
- Sales figures in millions of pounds
- Global_Sales = NA_Sales + EU_Sales + Other_Sales
- Year represents initial release, not all sales years

---

## Reference Data: Gaming Platforms

### Platform Metadata

| Code | Name | Manufacturer | Production Years | Generation | Type | Status |
|------|------|--------------|------------------|------------|------|---------|
| GB | Game Boy | Nintendo | 1989-2003 | Gen 0 | Handheld | In Dataset |
| GBA | Game Boy Advance | Nintendo | 2001-2007 | Gen 1 | Handheld | In Dataset |
| NES | Nintendo Entertainment System | Nintendo | 1983-2003 | Gen 0 | Console | In Dataset |
| SNES | Super Nintendo Entertainment System | Nintendo | 1990-2003 | Gen 1 | Console | In Dataset |
| N64 | Nintendo 64 | Nintendo | 1996-2002 | Gen 2 | Console | In Dataset |
| DS | Nintendo DS | Nintendo | 2004-2013 | Gen 2 | Handheld | In Dataset |
| GC | Nintendo GameCube | Nintendo | 2001-2007 | Gen 3 | Console | In Dataset |
| Wii | Nintendo Wii | Nintendo | 2006-2013 | Gen 3 | Console | In Dataset |
| 3DS | Nintendo 3DS | Nintendo | 2011-2017 | Gen 3 | Handheld | In Dataset |
| WiiU | Nintendo Wii U | Nintendo | 2012-2017 | Gen 4 | Console | In Dataset |
| XB | Xbox | Microsoft | 2001-2016 | Gen 2 | Console | In Dataset |
| X360 | Xbox 360 | Microsoft | 2005-2016 | Gen 2 | Console | In Dataset |
| XOne | Xbox One | Microsoft | 2013-2024 | Gen 3 | Console | In Dataset |
| PS | PlayStation | Sony | 1994-2006 | Gen 1 | Console | In Dataset |
| PS2 | PlayStation 2 | Sony | 2000-2013 | Gen 2 | Console | In Dataset |
| PS3 | PlayStation 3 | Sony | 2006-2016 | Gen 3 | Console | In Dataset |
| PS4 | PlayStation 4 | Sony | 2013-2019 | Gen 4 | Console | In Dataset |
| PSP | PlayStation Portable | Sony | 2004-2014 | Gen 3 | Handheld | In Dataset |
| PSV | PlayStation Vita | Sony | 2011-2016 | Gen 4 | Handheld | In Dataset |
| PC | Personal Computer | Various | 1996-Present | N/A | Computer | In Dataset |
| GENS | Sega Genesis | Sega | 1988-1997 | Gen 0 | Console | In Dataset |
| DC | Dreamcast | Sega | 1998-2001 | Gen 2 | Console | Reference Only |
| ATA | Atari 2600 | Atari | 1977-1990 | Gen 0 | Console | In Dataset |

### Console Generation Timeline

| Generation | Years | Key Characteristics |
|------------|-------|-------------------|
| Gen 0 | 1980-1994 | Early gaming era, basic graphics |
| Gen 1 | 1994-1999 | 3D graphics introduction |
| Gen 2 | 2000-2005 | DVD support, online capabilities |
| Gen 3 | 2006-2012 | HD graphics, motion controls |
| Gen 4 | 2013-2019 | 4K support, digital distribution |
| Gen 5 | 2020-2024 | Ray tracing, SSD storage |

---

## Data Relationships

### Primary Keys
- `turtle_reviews.csv`: No explicit primary key (row-based)
- `turtle_sales.csv`: `Product` + `Platform` combination

### Foreign Keys
- `Product` field links both datasets
- Reviews can be joined to sales data via Product ID

### Data Lineage
- Original university project data (LSE, 2021)
- Enhanced with industry expert consultation
- Platform metadata added for analytical depth

---

## Analysis Considerations

### Potential Data Issues
- Currency inconsistencies (all values in pounds, but verify)
- Missing "Other Sales" component in regional breakdown
- Platform generation classifications are approximate
- Some platforms have overlapping production years

### Recommended Preprocessing
1. Validate Product ID consistency between datasets
2. Handle missing values in review summaries
3. Standardize education level categories
4. Consider inflation adjustment for sales figures across years

### Analysis Opportunities
- Customer segmentation by demographics and spending patterns
- Platform lifecycle analysis using generation data
- Genre popularity trends over time
- Regional sales pattern analysis

### Data Processing History
| Field | Description | LLM Tool Benefit |
|-------|-------------|------------------|
| `last_validated` | ISO timestamp of last data validation | Tools can check freshness |
| `validation_rules` | JSON array of validation constraints | Automated quality checks |
| `missing_value_strategy` | How nulls/NAs should be handled per column | Preprocessing guidance |
| `outlier_detection_method` | Statistical method used for outlier detection | Consistent analysis approach |
| `data_lineage_hash` | Hash of source data for change detection | Version control for datasets |

### Column-Level Statistical Metadata
| Field | Description | Example |
|-------|-------------|---------|
| `min_value` | Minimum observed value | `age: 18` |
| `max_value` | Maximum observed value | `age: 69` |
| `value_distribution` | Distribution type (normal, skewed, etc.) | `spending_score: uniform` |
| `unique_value_count` | Number of distinct values | `gender: 2` |
| `null_percentage` | Percentage of missing values | `summary: 0.05` |
| `cardinality_level` | High/Medium/Low cardinality classification | `product: high` |

### Semantic Classifications
| Column | Semantic_Type | Analysis_Role | Tool_Hints |
|--------|---------------|---------------|------------|
| `gender` | categorical_demographic | segmentation_variable | ["groupby_candidate", "bias_check_required"] |
| `age` | numerical_demographic | continuous_predictor | ["binning_candidate", "lifecycle_analysis"] |
| `review` | unstructured_text | sentiment_source | ["nlp_processing", "topic_modeling", "embedding_candidate"] |
| `product` | identifier | join_key | ["foreign_key", "dimension_table_lookup"] |
| `spending_score` | derived_metric | target_variable | ["model_target", "segmentation_basis"] |
| `Global_Sales` | business_metric | kpi | ["aggregation_target", "trend_analysis"] |

### Text Processing Metadata
| Field | Description | Tool Application |
|-------|-------------|------------------|
| `text_preprocessing_required` | Cleaning steps needed | Auto-preprocessing |
| `expected_language` | Language for NLP processing | Model selection |
| `text_complexity_level` | Reading level/complexity | Processing strategy |
| `sentiment_polarity_expected` | Expected sentiment range | Validation checks |
| `topic_domains` | Subject matter domains present | Domain-specific processing |
### Business Context
| Field | Description | Value |
|-------|-------------|-------|
| `analysis_objectives` | Primary research questions | ["customer_segmentation", "sales_forecasting", "platform_lifecycle"] |
| `regulatory_constraints` | Data usage limitations | ["gdpr_compliant", "no_individual_identification"] |
| `business_definitions` | Domain-specific term meanings | {"loyalty_points": "proprietary_scoring_system"} |
| `temporal_context` | Time-based analysis considerations | {"sales_seasonality": "holiday_peaks", "platform_lifecycle": "generational_shifts"} |

### Model Readiness Indicators
| Column | Feature_Type | Encoding_Strategy | Missing_Strategy |
|--------|--------------|-------------------|------------------|
| `gender` | categorical | one_hot | mode_imputation |
| `age` | numerical | standardization | median_imputation |
| `platform` | high_cardinality_categorical | target_encoding | separate_category |
| `review` | text | tfidf_or_embeddings | empty_string |
| `year` | temporal | cyclical_encoding | interpolation |
### Tool-Specific Configurations
```json
{
  "visualization_hints": {
    "spending_score": {"chart_type": "histogram", "bins": 20},
    "sales_by_platform": {"chart_type": "stacked_bar", "group_by": "generation"},
    "review_sentiment": {"chart_type": "violin_plot", "split_by": "product_category"}
  },
  "aggregation_preferences": {
    "sales_data": ["sum", "mean", "median"],
    "demographic_data": ["mode", "distribution"],
    "text_data": ["count", "sentiment_score", "topic_distribution"]
  },
  "join_strategies": {
    "reviews_to_sales": {
      "key": "product",
      "type": "left_join",
      "validation": "one_to_many"
    }
  }
}
```

### **Small Model Optimization Fields**

### Prompt Engineering Metadata
| Field | Purpose | Example |
|-------|---------|---------|
| `column_aliases` | Human-readable names for prompts | `remuneration: "annual_income_thousands_gbp"` |
| `business_logic_rules` | Domain rules for validation | `Global_Sales >= NA_Sales + EU_Sales` |
| `common_analysis_patterns` | Frequent query types | ["platform_comparison", "genre_trends", "customer_lifetime_value"] |
| `contextual_explanations` | Background for non-domain experts | `loyalty_points: "Proprietary metric combining purchase value and engagement"` |

### Complexity Indicators
| Aspect | Level | Reasoning | Small_Model_Strategy |
|--------|-------|-----------|---------------------|
| `join_complexity` | Medium | Multiple datasets with hierarchical relationships | Pre-compute common joins |
| `temporal_complexity` | High | Multi-generational platform lifecycle analysis | Segment by time periods |
| `text_analysis_complexity` | High | Sentiment + topic modeling required | Use specialized text models |
| `statistical_complexity` | Medium | Customer segmentation + forecasting | Break into simpler sub-problems |

