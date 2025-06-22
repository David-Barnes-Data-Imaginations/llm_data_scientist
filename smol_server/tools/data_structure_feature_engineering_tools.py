

# functions from my pip library to be turned into tools depending on which functions i choose:

def one_hot_encode(data, column=None):
    """
    One-hot encodes columns in a DataFrame or categories in a NumPy array.

    Parameters:
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
    import pd as pd
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


# Helper function to save me having to type incorrectly
def split_data(X, y, test_size=0.30, random_state=1, stratify=None):
    """
    Splits dataset into training and testing subsets.

    Parameters:
    X (array-like or DataFrame): Input features.
    y (array-like or Series): Target values.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    stratify (array-like, optional): If not None, stratifies splits according to class distribution.

    Returns:
    tuple: (X_train, X_test, y_train, y_test) containing:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training target values
        - y_test: Testing target values

    Example input:
    X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1])
    test_size = 0.25
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def apply_feature_hashing(data, n_features=10):
    """
    Apply feature hashing to the input data.

    Parameters:
    data (iterable): An iterable object such as a list of lists, or a pandas DataFrame/Series,
                     where each row represents features.
    n_features (int): Number of output features (columns) for the hash space.

    Returns:
    scipy.sparse.csr_matrix: Transformed data with hashed features.

    Example input:
    # For DataFrame
    data = pd.DataFrame({'feature1': ['A', 'B', 'C'], 'feature2': [1, 2, 3]})
    n_features = 8

    # For list of lists
    data = [['A', 1], ['B', 2], ['C', 3]]
    n_features = 8
    """
    from sklearn.feature_extraction import FeatureHasher

    # Convert data into a list of dictionaries
    # Works for both DataFrame and list of lists
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


# Validation for splitting helper functions
def validate_split_data(X, y):
    """
    Helper function to validate inputs for splitting data.

    Parameters:
    X (array-like or DataFrame): Input features.
    y (array-like or Series): Target values.

    Returns:
    None: Function raises ValueError if validation fails, otherwise returns None.

    Raises:
    ValueError: If X and y have incompatible shapes or invalid inputs.

    Example input:
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if len(X) == 0:
        raise ValueError("X and y must not be empty.")


def smote_balance(X, y, test_size=0.3, random_state=42):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to oversample imbalanced datasets.

    Parameters:
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
    y = pd.Series([0, 0, 0, 0, 1, 1])  # Imbalanced classes (4 samples of class 0, 2 samples of class 1)
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
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(data=X_resampled, columns=X.columns)
        y_resampled = pd.Series(data=y_resampled, name=y.name if hasattr(y, "name") else "target")

    return X_resampled, y_resampled, X_test, y_test
