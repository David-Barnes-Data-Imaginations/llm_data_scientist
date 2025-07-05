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

### Validation Schema

| Column | Valid Data Type | Sample Value                        | Description |
|--------|-----------------|-------------------------------------|-------------|
| `gender` | string          | "male", "female"                    | Customer gender |
| `age` | integer         | integer (10-100)                    | Customer age in years |
| `remuneration` | float           | 45.5                                | Annual customer income in thousands of pounds (k£) |
| `spending_score` | integer (1-100) | 67                                  | Turtle Games proprietary spending behavior score |
| `loyalty_points` | integer         | 1250                                | Points based on purchase value and customer actions |
| `education` | string          | "graduate"                          | Education level: Diploma, Graduate, Postgraduate, PhD |
| `language` | string          | "EN"                                | Review language (all English) |
| `platform` | string          | "Web"                               | Review collection platform (all Web) |
| `product` | integer         | 12345                               | Unique product identifier |
| `review` | text            | "When it comes to a DM's screen..." | Full customer review text |
| `summary` | text            | "The fact that 50% of this..."      | Review summary |

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

Analysis


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

