import pandas as pd
from numpy import append
from pandas import DataFrame
from pydantic import ValidationError
from smolagents import Tool
import numpy as np
from src.data.validate_schema import DataValidator

# tool under consideration as it is possibly less accurate than normal analysis
# Possibly ask llm to create 2 DF's and use both methods
class ValidateData(Tool):
    name = "validate_data"
    description = "Validates data against a specified schema and returns a cleaned DataFrame."
    inputs = {
        "chunk": {"type": "object", "description": "DataFrame to validate"},
        "valid_rows": {"type": "list", "description": "List to store valid rows", "optional": True},
        "errors": {"type": "list", "description": "List to store validation errors", "optional": True}
    }
    output_type = "tuple"  # Returns tuple of DataFrame and errors

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.valid_rows = []
        self.errors = []
        self.cleaning_stats = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates()
        self.cleaning_stats['duplicates_removed'] = initial_rows - len(df)

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('Unknown')
        return df

    def forward(self, chunk: DataFrame, valid_rows= None, errors=None) -> tuple[DataFrame, {append, name}]:
        df = pd.DataFrame(chunk)
        for idx, row in df.iterrows():
            try:
                validated_row = DataValidator(**row.to_dict())
                valid_rows.append(validated_row.model_dump())
            except ValidationError as e:
                errors.append({'row': idx, 'errors': str(e)})

            self.cleaning_stats['validation_errors'] = len(errors)
        return pd.DataFrame(valid_rows), errors

    def process(self, df: pd.DataFrame) -> dict:
        cleaned_df = self.clean_data(df.copy())
        validated_df, validation_errors = self.forward(cleaned_df, valid_rows=[], errors=[])

        return {
            'cleaned_data': validated_df,
            'validation_errors': validation_errors,
            'stats': self.cleaning_stats
        }



class AnalyzePatterns(Tool):
    name = "analyze_patterns"
    description = "Analyzes specific patterns in the data chunk based on the specified analysis type."
    inputs = {
        "chunk": {"type": "list", "description": "List of dictionaries containing the data"},
        "analysis_type": {"type": "string", "description": "Type of analysis to perform (demographic, review_sentiment, spending_patterns, platform_specific)"}
    }
    output_type = "dict"  # Returns dictionary of analysis results

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, chunk: list[dict], analysis_type: str) -> dict:
        """
        Args:
            chunk (list[dict]): List of dictionaries containing the data
            analysis_type (str): Type of analysis to perform:
                - 'demographic': Age/gender/education patterns
                - 'review_sentiment': Analysis of review text patterns
                - 'spending_patterns': Relationship between spending and other factors
                - 'platform_specific': Platform-based patterns

        Returns:
            dict: Analysis results with patterns found, including
                - Statistical distributions
                - Correlations
                - Aggregated metrics by category

        Example input:
        chunk = [
            {'age': 35, 'gender': 'Male', 'education': 'Bachelor', 'spending_score (1-100)': 85},
            {'age': 28, 'gender': 'Female', 'education': 'Master', 'spending_score (1-100)': 92}
        ]
        analysis_type = 'demographic'
        """
        df = pd.DataFrame(chunk)
        patterns = {}

        if analysis_type == 'demographic':
            patterns['age_distribution'] = df['age'].describe()
            patterns['education_by_age'] = df.groupby('education')['age'].agg(['mean', 'std'])
            patterns['spending_by_gender'] = df.groupby('gender')['spending_score (1-100)'].mean()

        elif analysis_type == 'review_sentiment':
            # Basic sentiment patterns in reviews
            patterns['avg_review_length'] = df['review'].str.len().mean()
            patterns['common_words'] = df['review'].str.split().explode().value_counts().head(10)

        elif analysis_type == 'spending_patterns':
            patterns['spending_by_education'] = df.groupby('education')['spending_score (1-100)'].mean()
            patterns['loyalty_spending_corr'] = df['spending_score (1-100)'].corr(df['loyalty_points'])

        elif analysis_type == 'platform_specific':
            patterns['platform_demographics'] = df.groupby('platform').agg({
                'age': 'mean',
                'spending_score (1-100)': 'mean',
                'loyalty_points': 'mean'
            })

        return patterns



class CheckDataframe(Tool):
    name = "check_dataframe"
    description = "Inspects a pandas DataFrame for any non-numeric, NaN, or infinite values."
    inputs = {
        "chunk": {"type": "list", "description": "List of dictionaries to be converted to DataFrame and checked"}
    }
    output_type = "string"  # Returns success message or raises ValueError

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, chunk: list[dict]) -> str:
        """
        Args:
            chunk (list[dict]): List of dictionaries to be converted to DataFrame and checked.

        Returns:
            str: Success message if no issues are found.

        Raises:
            ValueError: If the DataFrame contains any non-numeric, NaN, or infinite values.

        Example input:
        chunk = [
            {'feature1': 10, 'feature2': 20},
            {'feature1': 30, 'feature2': 40}
        ]
        """
        # Convert a list of dictionaries to DataFrame
        df = pd.DataFrame(chunk)

        # Ensure it contains only numeric data
        if not df.select_dtypes(include=['number']).shape[1] == df.shape[1]:
            raise ValueError("DataFrame contains non-numeric data. Consider encoding these columns.")

        # Check for NaN values
        if df.isnull().any().any():
            raise ValueError("DataFrame contains NaN values. Consider filling or dropping these columns.")

        # Check for Inf values
        if np.isinf(df.values).any():
            raise ValueError("DataFrame contains Inf values. Consider handling these columns.")

        return "DataFrame validation passed successfully"

class InspectDataframe(Tool):
    name = "inspect_dataframe"
    description = "Inspects and provides a comprehensive overview of a pandas DataFrame."
    inputs = {
        "df": {"type": "object", "description": "The DataFrame to inspect and analyze"}
    }
    output_type = "object"  # Returns DataFrame with descriptive statistics

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df):
        """
        Args:
            df (pandas.DataFrame): The DataFrame to inspect and analyze.

        Returns:
            pandas.DataFrame: A DataFrame containing descriptive statistics for all columns,
                             generated by pandas' describe() method with include='all'.

        Example input:
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        """
        print(df.head())
        print("\nShape:")
        print(df.shape)
        print("\nColumns:")
        print(df.columns)
        print("\nDescriptive Statistics:")
        # describe() only analyses numeric data by default.
        # To include categorical columns
        # in the summary statistics, an argument can be added to the describe() method.
        return df.describe(include='all')
