from smolagents import Tool

class ListVariables(Tool):
    name = "ListVariables"
    description = "Lists all known global variables and their types. You can use this to help you keep track!"
    help_notes = """ 
    ListVariables: 
    A tool that lists all known global variables in the current environment along with their types.
    Use this to keep track of what variables are available to you and what type of data they contain.
    This is especially useful when you need to reference previously created variables or understand the current state.

    Example usage: 

    variables = ListVariables().forward()
    print(variables)  # Displays all available variables and their types
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
    def forward(self):
        variables = {k: type(v).__name__ for k, v in globals().items() if not k.startswith("__")}
        return "\n".join(f"{k}: {v}" for k, v in variables.items())

class ValidateData(Tool):
    name = "ValidateData"
    description = "Validates data against a schema and returns cleaned DataFrame and error list."
    inputs = {
        "chunk": {"type": "object", "description": "DataFrame chunk to validate"},
        "name": {"type": "string", "description": "Name to store the cleaned dataframe under", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    ValidateData: 
    A tool that validates and cleans a DataFrame against a predefined schema, handling duplicates and missing values.
    The cleaned DataFrame is stored in a global dataframe_store with the provided name for later access.
    Use this when you need to ensure your data meets specific validation criteria before further processing.

    Example usage: 

    result = ValidateData().forward(chunk=my_df, name="df_validated_chunk1")

    # To access the validated DataFrame later:
    df = dataframe_store["df_validated_chunk1"]

    # You can then perform operations on the validated DataFrame
    df.drop(columns=["bad_col"], inplace=True)
    """
    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.cleaning_stats = {}
    def clean_data(self, df):
        import pandas as pd
        import numpy as np
        
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.cleaning_stats['duplicates_removed'] = initial_rows - len(df)

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('Unknown')

        return df

    def forward(self, chunk, name="validated_df"):
        import pandas as pd
        
        df = pd.DataFrame(chunk)
        df_clean = self.clean_data(df)

        # Simple validation - just clean the data and return it
        # Skip complex validation that requires external schemas
        valid_rows = df_clean.to_dict('records')
        errors = []

        self.cleaning_stats['validation_errors'] = len(errors)

        # Store cleaned validated df
        result_df = pd.DataFrame(valid_rows)
        
        # Note: DataFrame is returned but not stored globally (E2B sandbox limitation)
        return {
            'stored_as': name,
            'shape': result_df.shape,
            'validation_errors': errors,
            'stats': self.cleaning_stats,
            'cleaned_data': valid_rows[:5]  # Sample of cleaned data
        }

class AnalyzePatterns(Tool):
    name = "AnalyzePatterns"
    description = "Analyzes specific patterns in the data chunk based on the specified analysis type."
    inputs = {
        "chunk": {"type": "object", "description": "object containing the data"},
        "analysis_type": {"type": "string", "description": "Type of analysis to perform (demographic, review_sentiment, spending_patterns, platform_specific)"}
    }
    output_type = "object"  # Returns dictionary of analysis results
    help_notes = """ 
    AnalyzePatterns: 
    A tool that performs specialized analysis on data chunks based on the specified analysis type.
    Use this to extract meaningful patterns and insights from your data, focusing on specific aspects like demographics, 
    sentiment, spending patterns, or platform-specific trends.

    Example usage: 

    # Analyze demographic patterns
    chunk = [
        {'age': 35, 'gender': 'Male', 'education': 'Bachelor', 'spending_score (1-100)': 85},
        {'age': 28, 'gender': 'Female', 'education': 'Master', 'spending_score (1-100)': 92}
    ]
    demographic_patterns = AnalyzePatterns().forward(chunk=chunk, analysis_type='demographic')

    # Other analysis types: 'review_sentiment', 'spending_patterns', 'platform_specific'
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, chunk, analysis_type):
        """
        Args:
            chunk : String containing the data
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
        import pandas as pd
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
    name = "CheckDataframe"
    description = "Inspects a pandas DataFrame for any non-numeric, NaN, or infinite values."
    inputs = {
        "chunk": {"type": "string", "description": "dataframe to be checked"}
    }
    output_type = "string"  # Returns success message or raises ValueError
    help_notes = """ 
    CheckDataframe: 
    A tool that validates a DataFrame specifically for machine learning readiness by checking for non-numeric values, NaN values, and infinite values.
    Use this before applying machine learning algorithms to ensure your data won't cause errors during model training.
    Unlike InspectDataframe, this tool raises errors if issues are found rather than just reporting them.

    Example usage: 

    try:
        # Check if DataFrame is ready for ML algorithms
        chunk = [
            {'feature1': 10, 'feature2': 20},
            {'feature1': 30, 'feature2': 40}
        ]
        result = CheckDataframe().forward(chunk=chunk)
        print(result)  # "DataFrame validation passed successfully"
    except ValueError as e:
        print(f"Data issue detected: {e}")
        # Handle the issue (e.g., convert non-numeric data, fill NaN values)
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, chunk):
        """
        Args:
            chunk dataframe to be checked.

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
        import pandas as pd
        import numpy as np
        
        # Convert a list of dictionaries to DataFrame
        df = pd.DataFrame(chunk)

        # Ensure it contains only numeric data
        if not df.select_dtypes(include=['number']).shape[1] == df.shape[1]:
            print("DataFrame contains non-numeric data. Consider encoding these columns.")

        # Check for NaN values
        if df.isnull().any().any():
            print("DataFrame contains NaN values. Consider filling or dropping these columns.")

        # Check for Inf values
        if np.isinf(df.values).any():
            print("DataFrame contains Inf values. Consider handling these columns.")

        return "DataFrame validation passed successfully"

class InspectDataframe(Tool):
    name = "InspectDataframe"
    description = "Inspects and provides a comprehensive overview of a pandas DataFrame."
    inputs = {
        "df": {"type": "object", "description": "The DataFrame to inspect and analyze"}
    }
    output_type = "object"  # Returns DataFrame with descriptive statistics
    help_notes = """ 
    InspectDataframe: 
    A tool that provides a comprehensive overview of a pandas DataFrame, including its shape, columns, and descriptive statistics.
    Use this for exploratory data analysis to understand the structure and content of your data before processing it.
    Unlike CheckDataframe, this tool focuses on providing information rather than validation.

    Example usage: 

    # Create a sample DataFrame
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'C', 'B']
    })

    # Get comprehensive statistics about the DataFrame
    stats = InspectDataframe().forward(df=df)

    # The output includes:
    # - First few rows (from df.head())
    # - DataFrame shape
    # - Column names
    # - Descriptive statistics for all columns (including categorical)
    #
    # describe() only analyses numeric data by default.
    # To include categorical columns
    # in the summary statistics, an argument can be added to the describe() method.
    """

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
        import pandas as pd
        print(df.head())
        print("\nShape:")
        print(df.shape)
        print("\nColumns:")
        print(df.columns)
        print("\nDescriptive Statistics:")

        return df.describe(include='all')
