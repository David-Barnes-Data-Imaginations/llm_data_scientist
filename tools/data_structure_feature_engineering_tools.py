from smolagents import Tool

# functions from my pip library to be turned into tools depending on which functions i choose:
class OneHotEncode(Tool):
    name = "one_hot_encode"
    description = "One-hot encodes columns in a DataFrame or categories in a NumPy array."
    inputs = {
        "data": {"type": "object", "description": "DataFrame or array-like data to encode"},
        "column": {"type": "string", "description": "Name of the column to encode, if input is a DataFrame", "optional": True}
    }
    output_type = "object"  # Returns DataFrame or np.ndarray

def __init__(self, sandbox=None):
    super().__init__()
    self.sandbox = sandbox

def forward(self, data, column=None):

        """
        Args:
            data (DataFrame or array-like): Input data to encode.
            column (str, optional): Name of the column to encode, if input is a DataFrame.

        Returns:
            DataFrame or np.ndarray: Encoded data with one-hot features.

        Example input:
        # For DataFrame
        data = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
        column = 'category'

        # For NumPy array
        data = np.array(['A', 'B', 'A', 'C'])
        """
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd
        import numpy as np

        if isinstance(data, pd.DataFrame):
            if column is None:
                raise ValueError("Column name must be specified for a pandas DataFrame.")
            return pd.get_dummies(data, columns=[column])

        elif isinstance(data, (np.ndarray, list)):
            encoder = OneHotEncoder(sparse_output=False)  # Non-sparse array
            data = np.array(data).reshape(-1, 1)  # Ensures data is 2D
            return encoder.fit_transform(data)

        else:
            raise ValueError("Input must be either a pandas DataFrame or a NumPy array.")
        pass

class ApplyFeatureHashing(Tool):
    name = "apply_feature_hashing"
    description = "Apply feature hashing to the input data."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, data, n_features=10):
        """
        Args:
            data (iterable): An iterable object such as a list of lists, or a pandas DataFrame/Series,
                           where each row represents features.
            n_features (int): Number of output features (columns) for the hash space.

        Returns:
            scipy.sparse.csr_matrix: Transformed data with hashed features.

        Example input:
        # For DataFrame
        data = pd.DataFrame({'feature1': ['A', 'B', 'C'], 'feature2': [1, 2, 3]})
        n_features = 8

        # For a list of lists
        data = [['A', 1], ['B', 2], ['C', 3]]
        n_features = 8
        """
        from sklearn.feature_extraction import FeatureHasher

        # Convert data into a list of dictionaries
        # Works for both DataFrame and list of lists
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict(orient="records")
        elif isinstance(data, list):
            data_dict = [
                {f"feature_{i}": val for i, val in enumerate(row)}
                for row in data
            ]
        else:
            raise ValueError("Input data must be a pandas DataFrame or a list of lists.")

        # Initialize the FeatureHasher
        hasher = FeatureHasher(n_features=n_features, input_type="dict")

        # Transform data to a hashed feature space
        hashed_features = hasher.transform(data_dict)

        # Return the sparse matrix result
        return hashed_features


class SmoteBalance(Tool):
    name = "smote_balance"
    description = "Applies SMOTE (Synthetic Minority Over-sampling Technique) to oversample imbalanced datasets."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, X, y, test_size=0.3, random_state=42):
        """
        Args:
            X (DataFrame or array-like): Input features.
            y (Series or array-like): Target values.
            test_size (float): Proportion of the dataset for testing.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Balanced X_train, y_train, and original X_test, y_test containing:
                - X_resampled: Training features after SMOTE balancing
                - y_resampled: Training target values after SMOTE balancing
                - X_test: Testing features (unchanged)
                - y_test: Testing target values (unchanged)

        Example input:
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6], 'feature2': [7, 8, 9, 10, 11, 12]})
        y = pd.Series([0, 0, 0, 0, 1, 1])
        # Imbalanced classes (4 samples of class 0, 2 samples of class 1)
        """
        # Modular imports
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split

        smote = SMOTE(random_state=random_state)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Apply SMOTE to the training set
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Convert outputs into pandas DataFrame if X is a DataFrame
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.DataFrame(data=X_resampled, columns=X.columns)
            y_resampled = pd.Series(data=y_resampled, name=y.name if hasattr(y, "name") else "target")

        return X_resampled, y_resampled, X_test, y_test


def CalculateSparsity(Tool) -> float:
    name = "CalculateSparsity"
    description = "Applies SMOTE (Synthetic Minority Over-sampling Technique) to oversample imbalanced datasets."

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, data: object) -> float:
        """
  ** Calculate and return the sparsity of the given 'data'.
  **
  ** Sparsity is defined as the proportion of elements that are zero.
  **
  ** Parameters:
  ** data (np.ndarray): Input array (can be any shape).
  **
  ** Returns:
  **    float: Sparsity as a proportion of zero elements (0 to 1).
  **    """
    import numpy as np
    if isinstance(data, np.ndarray):
        total_elements = data.size
    if total_elements == 0:  # Prevent division by zero
        return 0.0
    num_zeros = np.count_nonzero(data == 0)
    sparsity: float = num_zeros / total_elements
    return sparsity

class HandleMissingValues(Tool):
    name = "HandleMissingValues"
    description = "Handle Missing Values in a pandas DataFrame using interpolation or various imputation strategies."
    inputs = {
        "df": {"type": "object", "description": "Input DataFrame containing data with missing values"},
        "method": {"type": "string", "description": "Interpolation method (default: 'linear')", "optional": True},
        "axis": {"type": "integer", "description": "Axis to interpolate along (default: 0)", "optional": True},
        "fill_strategy": {"type": "string", "description": "Imputation strategy ('mean', 'median', 'mode', or scalar)", "optional": True},
        "inplace": {"type": "boolean", "description": "Whether to modify DataFrame in place (default: False)", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

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
        if not inplace:
            df = df.copy()  # Avoid modifying the original DataFrame

        try:
            if fill_strategy is not None:
                # Handle imputation based on the provided strategy
                if fill_strategy == 'mean':
                    df.fillna(df.mean(), inplace=True)
                elif fill_strategy == 'median':
                    df.fillna(df.median(), inplace=True)
                elif fill_strategy == 'mode':
                    for col in df.columns:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    # Assume fill_strategy is a scalar value
                    df.fillna(fill_strategy, inplace=True)
            else:
                # Use interpolation to handle missing values
                df.interpolate(method=method, axis=axis, inplace=True)
        except Exception as e:
            raise ValueError(f"Error handling missing values: {e}")

        return df
    pass