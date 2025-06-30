from smolagents import Tool
from src.client.telemetry import TelemetryManager

class OneHotEncode(Tool):
    name = "one_hot_encode"
    description = "One-hot encodes columns in a DataFrame or categories in a NumPy array."
    inputs = {
        "data": {"type": "object", "description": "DataFrame or array-like data to encode"},
        "column": {"type": "string", "description": "Name of the column to encode, if input is a DataFrame", "optional": True}
    }
    output_type = "object"  # Returns DataFrame or np.ndarray
    help_notes = """ 
    OneHotEncode: 
    A tool that converts categorical variables into a binary matrix representation (one-hot encoding).
    Use this when you need to transform categorical features into a format suitable for machine learning models that require numerical input.

    Example usage: 

    # For DataFrame
    df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
    encoded_df = OneHotEncode().forward(data=df, column='category')

    # For NumPy array
    data = np.array(['A', 'B', 'A', 'C'])
    encoded_array = OneHotEncode().forward(data=data)
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.telemetry = TelemetryManager()

    def forward(self, data, column=None):
        """
        Args:
            data (DataFrame or array-like): Input data to encode.
            column (str, optional): Name of the column to encode, if input is a DataFrame.

        Returns:
            DataFrame or np.ndarray: Encoded data with one-hot features.
        """
        # Start telemetry trace
        trace = self.telemetry.start_trace("one_hot_encode", {
            "data_type": str(type(data).__name__),
            "column": column
        })

        try:
            from sklearn.preprocessing import OneHotEncoder
            import pandas as pd
            import numpy as np

            result = None

            if isinstance(data, pd.DataFrame):
                if column is None:
                    raise ValueError("Column name must be specified for a pandas DataFrame.")

                self.telemetry.log_event(trace, "processing", {
                    "step": "dataframe_encoding",
                    "column": column,
                    "shape": data.shape
                })

                result = pd.get_dummies(data, columns=[column])

            elif isinstance(data, (np.ndarray, list)):
                self.telemetry.log_event(trace, "processing", {
                    "step": "array_encoding",
                    "data_length": len(data)
                })

                encoder = OneHotEncoder(sparse_output=False)
                data_array = np.array(data).reshape(-1, 1)
                result = encoder.fit_transform(data_array)

            else:
                raise ValueError("Input must be either a pandas DataFrame or a NumPy array.")

            # Log success
            self.telemetry.log_event(trace, "success", {
                "output_shape": str(result.shape) if hasattr(result, 'shape') else str(len(result))
            })

            return result

        except Exception as e:
            # Log error
            self.telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            # Always finish the trace
            self.telemetry.finish_trace(trace)
class ApplyFeatureHashing(Tool):
    name = "apply_feature_hashing"
    description = "Apply feature hashing to the input data."
    inputs = {
        "data": {"type": "object", "description": "An iterable object such as a list of lists, or a pandas DataFrame/Series"},
        "n_features": {"type": "integer", "description": "Number of output features (columns) for the hash space", "optional": True}
    }
    output_type = "object"  # Returns scipy.sparse.csr_matrix
    help_notes = """ 
    ApplyFeatureHashing: 
    A tool that applies the feature hashing technique (the hashing trick) to convert high-dimensional categorical features into a fixed-size feature space.
    Use this when dealing with high-cardinality categorical features or text data to reduce dimensionality while preserving information.

    Example usage: 

    # For DataFrame
    df = pd.DataFrame({'feature1': ['A', 'B', 'C'], 'feature2': [1, 2, 3]})
    hashed_features = ApplyFeatureHashing().forward(data=df, n_features=8)

    # For a list of lists
    data = [['A', 1], ['B', 2], ['C', 3]]
    hashed_features = ApplyFeatureHashing().forward(data=data, n_features=8)
    """

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
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("apply_feature_hashing", {
            "data_type": str(type(data).__name__),
            "n_features": n_features
        })

        try:
            from sklearn.feature_extraction import FeatureHasher
            import pandas as pd

            # Convert data into a list of dictionaries
            # Works for both DataFrame and list of lists
            if isinstance(data, pd.DataFrame):
                telemetry.log_event(trace, "processing", {
                    "step": "dataframe_conversion",
                    "shape": data.shape
                })
                data_dict = data.to_dict(orient="records")

            elif isinstance(data, list):
                telemetry.log_event(trace, "processing", {
                    "step": "list_conversion",
                    "length": len(data)
                })
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

            # Log success
            telemetry.log_event(trace, "success", {
                "output_shape": str(hashed_features.shape) if hasattr(hashed_features, 'shape') else "unknown"
            })

            # Return the sparse matrix result
            return hashed_features

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            # Always finish the trace
            telemetry.finish_trace(trace)

class SmoteBalance(Tool):
    name = "smote_balance"
    description = "Applies SMOTE (Synthetic Minority Over-sampling Technique) to oversample imbalanced datasets."
    inputs = {
        "X": {"type": "object", "description": "Input features (DataFrame or array-like)"},
        "y": {"type": "object", "description": "Target values (Series or array-like)"},
        "test_size": {"type": "float", "description": "Proportion of the dataset for testing", "optional": True},
        "random_state": {"type": "integer", "description": "Random seed for reproducibility", "optional": True}
    }
    output_type = "tuple"  # Returns tuple of balanced datasets
    help_notes = """ 
    SmoteBalance: 
    A tool that applies the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to address class imbalance in your dataset.
    Use this when your target variable has imbalanced classes and you want to generate synthetic samples for the minority class to improve model performance.

    Example usage: 

    # With an imbalanced dataset (4 samples of class 0, 2 samples of class 1)
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6], 'feature2': [7, 8, 9, 10, 11, 12]})
    y = pd.Series([0, 0, 0, 0, 1, 1])

    # Apply SMOTE to balance the classes
    X_resampled, y_resampled, X_test, y_test = SmoteBalance().forward(X=X, y=y, test_size=0.3, random_state=42)
    """

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
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("smote_balance", {
            "X_type": str(type(X).__name__),
            "y_type": str(type(y).__name__),
            "test_size": test_size,
            "random_state": random_state
        })

        try:
            # Modular imports
            from imblearn.over_sampling import SMOTE
            from sklearn.model_selection import train_test_split
            import pandas as pd

            telemetry.log_event(trace, "processing", {
                "step": "data_splitting",
                "X_shape": str(X.shape) if hasattr(X, 'shape') else "unknown"
            })

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            telemetry.log_event(trace, "processing", {
                "step": "applying_smote",
                "X_train_shape": str(X_train.shape) if hasattr(X_train, 'shape') else "unknown"
            })

            # Initialize SMOTE
            smote = SMOTE(random_state=random_state)

            # Apply SMOTE to the training set
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            # Convert outputs into pandas DataFrame if X is a DataFrame
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(data=X_resampled, columns=X.columns)
                y_resampled = pd.Series(data=y_resampled, name=y.name if hasattr(y, "name") else "target")

            # Log success
            telemetry.log_event(trace, "success", {
                "X_resampled_shape": str(X_resampled.shape) if hasattr(X_resampled, 'shape') else "unknown",
                "y_resampled_length": len(y_resampled) if hasattr(y_resampled, '__len__') else "unknown"
            })

            return X_resampled, y_resampled, X_test, y_test

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            # Always finish the trace
            telemetry.finish_trace(trace)


class CalculateSparsity(Tool):
    name = "calculate_sparsity"
    description = "Calculate the sparsity of the given data (proportion of elements that are zero)."
    inputs = {
        "data": {"type": "object", "description": "Input array (can be any shape)"}
    }
    output_type = "float"
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

    def forward(self, data: object) -> float:
        """
        Calculate and return the sparsity of the given 'data'.

        Sparsity is defined as the proportion of elements that are zero.

        Args:
            data (np.ndarray): Input array (can be any shape).

        Returns:
            float: Sparsity as a proportion of zero elements (0 to 1).
        """
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("calculate_sparsity", {
            "data_type": str(type(data).__name__)
        })

        try:
            import numpy as np

            if isinstance(data, np.ndarray):
                total_elements = data.size

                telemetry.log_event(trace, "processing", {
                    "step": "calculating_sparsity",
                    "total_elements": total_elements
                })

                if total_elements == 0:  # Prevent division by zero
                    telemetry.log_event(trace, "warning", {
                        "message": "Empty array detected, returning 0.0"
                    })
                    return 0.0

                num_zeros = np.count_nonzero(data == 0)
                sparsity = num_zeros / total_elements

                # Log success
                telemetry.log_event(trace, "success", {
                    "sparsity": sparsity,
                    "num_zeros": num_zeros,
                    "total_elements": total_elements
                })

                return sparsity
            else:
                telemetry.log_event(trace, "warning", {
                    "message": "Input is not a numpy array, returning 0.0"
                })
                return 0.0

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            # Always finish the trace
            telemetry.finish_trace(trace)

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
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("handle_missing_values", {
            "method": method,
            "axis": axis,
            "fill_strategy": str(fill_strategy),
            "inplace": inplace
        })

        try:
            import pandas as pd

            # Log initial state
            telemetry.log_event(trace, "processing", {
                "step": "initial_state",
                "df_shape": str(df.shape) if hasattr(df, 'shape') else "unknown",
                "missing_values_count": str(df.isna().sum().sum()) if hasattr(df, 'isna') else "unknown"
            })

            if not inplace:
                df = df.copy()  # Avoid modifying the original DataFrame

            if fill_strategy is not None:
                telemetry.log_event(trace, "processing", {
                    "step": "imputation",
                    "strategy": str(fill_strategy)
                })

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
                telemetry.log_event(trace, "processing", {
                    "step": "interpolation",
                    "method": method,
                    "axis": axis
                })

                # Use interpolation to handle missing values
                df.interpolate(method=method, axis=axis, inplace=True)

            # Log final state
            telemetry.log_event(trace, "success", {
                "remaining_missing_values": str(df.isna().sum().sum()) if hasattr(df, 'isna') else "unknown"
            })

            return df

        except Exception as e:
            # Log error
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise ValueError(f"Error handling missing values: {e}")
        finally:
            # Always finish the trace
            telemetry.finish_trace(trace)
