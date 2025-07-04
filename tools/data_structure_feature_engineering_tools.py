from smolagents import Tool

class CalculateSparsity(Tool):
    name = "CalculateSparsity"
    description = "Calculate the sparsity of the given data (proportion of elements that are zero)."
    inputs = {
        "data": {"type": "object", "description": "Input array (can be any shape)"}
    }
    output_type = "object"
    help_notes = """ 
    CalculateSparsity: 
    A tool that calculates the sparsity of a dataset, which is the proportion of elements that are zero.
    Use this to assess how sparse your data is, which can be important for choosing appropriate algorithms or storage formats.
    High sparsity (close to 1.0) indicates that most values are zeros.

    Example usage: 

    import numpy as np

    # Create a sparse array where 80% of elements are zero
    data = np.array([0, 1, 0, 0, 2, 0, 0, 0, 3, 0])

    # Calculate sparsity
    sparsity = CalculateSparsity().forward(data=data)  # Should return 0.8
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
    def forward(self, data):
        """
        Calculate and return the sparsity of the given 'data'.

        Sparsity is defined as the proportion of elements that are zero.

        Args:
            data (np.ndarray): Input array (can be any shape).

        Returns:
            float: Sparsity as a proportion of zero elements (0 to 1).
        """
        import numpy as np

        if isinstance(data, np.ndarray):
            total_elements = data.size

            if total_elements == 0:  # Prevent division by zero
                return 0.0

            num_zeros = np.count_nonzero(data == 0)
            sparsity = num_zeros / total_elements
            return sparsity
        else:
            return 0.0


class HandleMissingValues(Tool):
    name = "HandleMissingValues"
    description = "Handle Missing Values in a pandas DataFrame using interpolation or various imputation strategies."
    inputs = {
        "df": {"type": "object", "description": "Input DataFrame containing data with missing values"},
        "method": {"type": "string", "description": "Interpolation method (default: 'linear')", "optional": True, "nullable": True},
        "axis": {"type": "integer", "description": "Axis to interpolate along (default: 0)", "optional": True, "nullable": True},
        "fill_strategy": {"type": "string", "description": "Imputation strategy ('mean', 'median', 'mode', or scalar)", "optional": True, "nullable": True},
        "inplace": {"type": "boolean", "description": "Whether to modify DataFrame in place (default: False)", "optional": True, "nullable": True}
    }
    output_type = "object"  # Returns DataFrame
    help_notes = """ 
    HandleMissingValues: 
    A tool that handles missing values (NaN, None) in pandas DataFrames using either interpolation or imputation strategies.
    Use this during data cleaning to address missing data, which is a common issue in real-world datasets.

    Example usage: 

    # Using interpolation (fills missing values based on surrounding data points)
    df_cleaned = HandleMissingValues().forward(df=df, method='linear', axis=0)

    # Using mean imputation (replaces missing values with column means)
    df_cleaned = HandleMissingValues().forward(df=df, fill_strategy='mean')

    # Using a constant value for imputation
    df_cleaned = HandleMissingValues().forward(df=df, fill_strategy=0)  # Replace NaNs with 0
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, method='linear', axis=0, fill_strategy=None, inplace=False):
        """
        Args:
            df (pd.DataFrame): Input DataFrame containing data with missing values.
            method (str): Interpolation method. Default is 'linear'. Options include:
                'linear', 'time', 'index', 'values', 'nearest', etc.
            axis (int): Axis to interpolate along. Use 0 for rows and 1 for columns.
            fill_strategy (str or None): Imputation strategy. If not None, this overrides interpolation. Options:
                - 'mean': Replace missing values with column mean.
                - 'median': Replace missing values with column median.
                - 'mode': Replace missing values with column mode.
                - Any scalar value to directly use as a replacement.
            inplace (bool): If True, modifies the input DataFrame directly. Default is False.

        Returns:
            pd.DataFrame: A DataFrame with missing values handled (if `inplace=False`),
                or None if modified in place.

        Usage Examples:
        Using Interpolation:
        df_copy = handle_missing_values(df_copy, method='linear', axis=0)

        Using Imputation with Mean:
        df_copy = handle_missing_values(df_copy, fill_strategy='mean')

        Replacing with a Scalar:
        df_copy = handle_missing_values(df_copy, fill_strategy=0)  # Replace NaNs with 0
        """
        try:
            import pandas as pd

            if not inplace:
                df = df.copy()  # Avoid modifying the original DataFrame

            if fill_strategy is not None:
                # Handle imputation based on the provided strategy
                if fill_strategy == 'mean':
                    df.fillna(df.mean(), inplace=True)
                elif fill_strategy == 'median':
                    df.fillna(df.median(), inplace=True)
                elif fill_strategy == 'mode':
                    for col in df.columns:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
                else:
                    # Assume fill_strategy is a scalar value
                    df.fillna(fill_strategy, inplace=True)
            else:
                # Use interpolation to handle missing values
                df.interpolate(method=method, axis=axis, inplace=True)

            return df

        except Exception as e:
            raise ValueError(f"Error handling missing values: {e}")
