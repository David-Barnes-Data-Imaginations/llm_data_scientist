from typing import Any

import pandas as pd


def calculate_sparsity(data: object) -> float:
    """
  **    Calculate and return the sparsity of the given 'data'.
  **
  **    Sparsity is defined as the proportion of elements that are zero.
  **
  **    Parameters:
  **    data (np.ndarray): Input array (can be any shape).
  **
  **    Returns:
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


def handle_missing_values(df, method: object = 'linear', axis=0, fill_strategy=None, inplace=False):
    """
    Handle Missing Values in a pandas DataFrame using interpolation or various imputation strategies.

1. **Supports Both Interpolation and Imputation:**
    - If `fill_strategy` is provided, the function handles imputation (e.g., mean, median, mode).
    - Otherwise, it defaults to interpolation using pandas' robust `interpolate()`.

2. **Flexible Value Replacement:*
    - Users can directly provide a scalar value for replacement instead of strategies.
    - Example:
    'df_copy = handle_missing_values(df_copy, fill_strategy=0)  # Replace NaNs with 0'
3. **In-Place Modification:**
    - It supports both in-place modification (`inplace=True`) and returning a copy (`inplace=False`).
### **Usage Examples:**
#### **Using Interpolation:**
df_copy = handle_missing_values(df_copy, method='linear', axis=0)

#### **Using Imputation with Mean:**
df_copy = handle_missing_values(df_copy, fill_strategy='mean')

#### **Replacing with a Scalar:**
df_copy = handle_missing_values(df_copy, fill_strategy=0)  # Replace NaNs with 0

              **    Handle missing values in a pandas DataFrame using interpolation or various imputation strategies.
              **
              **    Parameters:
              **    df_copy (pd.DataFrame): Input DataFrame containing data with missing values.
              **    method (str): Interpolation method. Default is 'linear'. Options include:
              **                  'linear', 'time', 'index', 'values', 'nearest', etc.
              **    axis (int): Axis to interpolate along. Use 0 for rows and 1 for columns.
              **    fill_strategy (str or None): Imputation strategy. If not None, this overrides interpolation. Options:
              **                                 - 'mean': Replace missing values with column mean.
              **                                 - 'median': Replace missing values with column median.
              **                                 - 'mode': Replace missing values with column mode.
              **                                 - Any scalar value to directly use as replacement.
              **    inplace (bool): If True, modifies the input DataFrame directly. Default is False.
              **
              **    Returns:
              **    pd.DataFrame: A DataFrame with missing values handled (if `inplace=False`),
              **                  or None if modified in place.

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


def numpy_pca(data: object, n_components: object) -> Any:
    """
    **    Perform Principal Component Analysis (PCA) on the given numerical data.
    **
    **    Parameters:
    **    data (array-like): Input data for PCA (e.g., NumPy array or pandas DataFrame).
    **                       Must only contain numerical values.
    **    n_components (int): Number of principal components to extract.
    **
    **    Returns:
    **    np.ndarray: Transformed data with reduced dimensionality.
    **
    **    Raises:
    **    ValueError: If input data is not numerical or if `n_components` is invalid.
    :type data: object
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd

    # Convert pandas DataFrame to NumPy array (if applicable)
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Validate input data
    if not isinstance(data, (np.ndarray, list)):
        raise ValueError("Input data must be a NumPy array, a pandas DataFrame, or a list.")

    # Ensure data is a NumPy array
    data = np.array(data)

    # Check for valid n_components
    if n_components <= 0 or n_components > min(data.shape):
        raise ValueError(
            f"Invalid n_components={n_components}. It must be > 0 and ≤ the smallest dimension of the data."
        )

    # Perform PCA
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def apply_feature_hashing(data, n_features=10):
    """
    Apply feature hashing to the input data.

    Parameters:
    data (iterable): An iterable object such as a list of lists, or a pandas DataFrame/Series,
                     where each row represents features.
    n_features (int): Number of output features (columns) for the hash space.

    Returns:
    scipy.sparse.csr_matrix: Transformed data with hashed features.
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

    Raises:
    ValueError: If X and y have incompatible shapes or invalid inputs.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if len(X) == 0:
        raise ValueError("X and y must not be empty.")


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
    tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def smote_balance(X, y, test_size=0.3, random_state=42):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to oversample imbalanced datasets.

    Parameters:
    X (DataFrame or array-like): Input features.
    y (Series or array-like): Target values.
    test_size (float): Proportion of the dataset for testing.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Balanced X_train, y_train, and original X_test, y_test.
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


def perform_tsne(data, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
    """
    Performs t-SNE (t-Distributed Stochastic Neighbor Embedding) on a dataset.

    Parameters:
    data (array-like or DataFrame): Input data for dimensional reduction.
    n_components (int): Dimension of the embedded space.
    perplexity (float): Perplexity for nearest-neighbor estimation.
    learning_rate (float): Learning rate for gradient descent in t-SNE optimization.
    n_iter (int): Maximum number of iterations for optimization.

    Returns:
    np.ndarray: Transformed data with reduced dimensions.
    """
    # Modular Imports
    from sklearn.manifold import TSNE
    import pd as pd
    import numpy as np

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("Input data must be a pandas DataFrame or NumPy array.")

    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    return tsne.fit_transform(data)


def perform_umap(data, n_neighbors=5, min_dist=0.3, n_components=2):
    """
    Performs UMAP (Uniform Manifold Approximation and Projection) on a dataset.

    Parameters:
    data (array-like or DataFrame): Input data for dimensional reduction.
    n_neighbors (int): Size of the local neighborhood.
    min_dist (float): Minimum distance between points in the low-dimensional projection.
    n_components (int): Dimension of the embedded space.

    Returns:
    np.ndarray: Transformed data with reduced dimensions.
    """
    # Modular imports
    from umap import UMAP
    import pandas as pd
    import numpy as np

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("Input data must be a pandas DataFrame or NumPy array.")

    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    return umap.fit_transform(data)


def one_hot_encode(data, column=None):
    """
    One-hot encodes columns in a DataFrame or categories in a NumPy array.

    Parameters:
    data (DataFrame or array-like): Input data to encode.
    column (str, optional): Name of the column to encode, if input is a DataFrame.

    Returns:
    DataFrame or np.ndarray: Encoded data with one-hot features.
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


def one_hot_encode_inplace(data, column):
    """
    One-hot encodes a column in a pandas DataFrame directly, modifying the original DataFrame.

    Parameters:
    data (DataFrame): Input DataFrame with the column to encode.
    column (str): Name of the column to encode.

    Returns:
    None: DataFrame is modified in place.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    dummies = pd.get_dummies(data[column], prefix=column)
    data.drop(columns=[column], inplace=True)
    for col in dummies.columns:
        data[col] = dummies[col]

# ************************************************************
