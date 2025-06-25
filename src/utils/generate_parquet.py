import pandas as pd
from pandas import DataFrame
from pathlib import Path
from typing import Final

# Define paths as constants
DATA_DIR: Final = Path("../../data")
REVIEWS_CSV: Final = DATA_DIR / "turtle_reviews.csv"
SALES_CSV: Final = DATA_DIR / "turtle_sales.csv"
REVIEWS_PARQUET: Final = DATA_DIR / "turtle_reviews.parquet"
SALES_PARQUET: Final = DATA_DIR / "turtle_sales.parquet"

def convert_to_parquet() -> None:
    try:
        # Ensure data directory exists
        DATA_DIR.mkdir(exist_ok=True)
        
        # Read the CSV files
        df_reviews: DataFrame = pd.read_csv(REVIEWS_CSV)
        df_sales: DataFrame = pd.read_csv(SALES_CSV)
        
        # Save as Parquet
        df_reviews.to_parquet(REVIEWS_PARQUET, index=False)
        df_sales.to_parquet(SALES_PARQUET, index=False)
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except PermissionError as e:
        print(f"Error: Permission denied - {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    convert_to_parquet()