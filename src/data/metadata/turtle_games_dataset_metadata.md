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
