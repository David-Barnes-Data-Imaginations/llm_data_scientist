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

# Notes on Tools:

---
### DataFrame Melt

The melt function is a powerful tool used to transform dataframes from a "wide" format to a "long" format. 
This process is often called "unpivoting" or "flattening" a dataframe.
Imagine you have data where different measurements or categories are spread across multiple columns. melt allows you to gather these columns into a single column, creating new rows for each original measurement. This is particularly useful for:
Tidying Data: It brings data into a "tidy" format, which is often preferred for analysis and visualization, as each variable forms a column, each observation forms a row, and each type of observational unit forms a table.
Database Normalization: Similar to normalization principles in databases, melt can help structure your data more efficiently.
Preparing data for plotting libraries: Many plotting libraries (like Seaborn or Matplotlib) expect data in a long format.

How melt Works (Core Concepts)
The melt function essentially takes two main types of arguments:

id_vars (identifier variables): These are the columns you want to keep as they are. 
Their values will be repeated for each "melted" row. Think of them as the columns that uniquely identify an observation before unpivoting.
value_vars (value variables): These are the columns you want to "unpivot" or "melt." The values from these columns will be stacked into a single new column.
When you apply melt, it creates two new columns by default:
variable (or var_name if specified): This column will contain the names of the original value_vars columns.
value (or value_name if specified): This column will contain the values from the original value_vars columns.

A Simple Example
Let's illustrate with an example.

Imagine you have sales data like this:

```
data = {'Region': ['North', 'South'],
'Product_A_Sales': [100, 150],
'Product_B_Sales': [200, 250],
'Product_C_Sales': [50, 75]}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
```
Original DataFrame:

```
Region  Product_A_Sales  Product_B_Sales  Product_C_Sales
0  North              100              200               50
1  South              150              250               75
```

Here, Product_A_Sales, Product_B_Sales, and Product_C_Sales are our "wide" columns representing different sales figures. 
We want to transform this so that we have a single "Product" column and a single "Sales" column.

Now, let's melt it:

Python
```
melted_df = pd.melt(df, id_vars=['Region'],
value_vars=['Product_A_Sales', 'Product_B_Sales', 'Product_C_Sales'],
var_name='Product',
value_name='Sales')

print("\nMelted DataFrame:")
print(melted_df)
Melted DataFrame:
```

Region          Product  Sales
0  US  Product_A_Sales    100
1  EU  Product_A_Sales    150
2  US  Product_B_Sales    200
3  EU  Product_B_Sales    250
4  US  Product_C_Sales     50
5  EU  Product_C_Sales     75
Explanation of the melt arguments used:
df: This is the DataFrame we want to melt.

id_vars=['Region']: We want to keep the 'Region' column as an identifier. Its values will be repeated for each product's sales.
value_vars=['Product_A_Sales', 'Product_B_Sales', 'Product_C_Sales']: These are the columns whose values we want to "unpivot" into a single column.
var_name='Product': This renames the default 'variable' column to 'Product'. This column now tells us which product's sales are represented.
value_name='Sales': This renames the default 'value' column to 'Sales'. This column now contains the actual sales figures.

Key melt Parameters:
frame (DataFrame, required): The DataFrame to unpivot.
id_vars (scalar, list-like, or None, optional): Column(s) to use as identifier variables. If None (default), all columns not specified in value_vars will be used as id_vars.
value_vars (scalar, list-like, or None, optional): Column(s) to unpivot. If None (default), all columns not specified in id_vars will be used as value_vars.
var_name (scalar, optional): Name to use for the 'variable' column. Defaults to 'variable'.
value_name (scalar, optional): Name to use for the 'value' column. Defaults to 'value'.
col_level (int or str, optional): If columns are a MultiIndex, then use this level to melt.

When to Use melt:
When your data has columns that represent different categories of the same variable (like Product_A_Sales, Product_B_Sales where "Sales" is the variable and "Product A", "Product B" are categories).
When preparing data for statistical analysis or machine learning models that expect a long format (e.g., when you want to treat "Product" as a categorical feature).
When creating visualizations where you want to group or differentiate based on the "unpivoted" categories.
In summary, Pandas melt is a fundamental function for reshaping your data, making it more amenable to a wide range of analytical and visualization tasks.
---