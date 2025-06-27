import pandas as pd
from smolagents import Tool

class DataframeMelt(Tool):
    name = "dataframe_melt"
    description = "Melt a DataFrame into a long-format DataFrame."
    inputs = {
        "frame": {"type": "object", "description": "DataFrame to melt"},
        "id_vars": {"type": "list", "description": "Column(s) to use as identifier variables", "optional": True},
        "value_vars": {"type": "list", "description": "Column(s) to unpivot", "optional": True},
        "var_name": {"type": "string", "description": "Name to use for the 'variable' column", "optional": True},
        "value_name": {"type": "string", "description": "Name to use for the 'value' column", "optional": True},
        "col_level": {"type": "integer", "description": "If columns are a MultiIndex, level to melt", "optional": True},
        "ignore_index": {"type": "boolean", "description": "If True, reset index in result", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, frame, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True):
        """
        Args:
            frame (DataFrame): DataFrame to melt
            id_vars (list, optional): Column(s) to use as identifier variables
            value_vars (list, optional): Column(s) to unpivot
            var_name (str, optional): Name to use for the 'variable' column
            value_name (str, optional): Name to use for the 'value' column
            col_level (int, optional): If columns are a MultiIndex, level to melt
            ignore_index (bool, optional): If True, reset index in result

        Returns:
            DataFrame: Melted DataFrame
        """
        import pandas as pd
        return pd.melt(frame, id_vars=id_vars, value_vars=value_vars, var_name=var_name, 
                      value_name=value_name, col_level=col_level, ignore_index=ignore_index)

class DataframeConcat(Tool):
    name = "dataframe_concat"
    description = "Concatenate DataFrames along a specified axis."
    inputs = {
        "objs": {"type": "list", "description": "List of DataFrames to concatenate"},
        "axis": {"type": "integer", "description": "Axis to concatenate along (0 for rows, 1 for columns)", "optional": True},
        "join": {"type": "string", "description": "How to handle indexes on other axis ('inner' or 'outer')", "optional": True},
        "ignore_index": {"type": "boolean", "description": "If True, do not use the index values", "optional": True},
        "keys": {"type": "list", "description": "Construct hierarchical index", "optional": True},
        "levels": {"type": "list", "description": "Specific levels to use", "optional": True},
        "names": {"type": "list", "description": "Names for the levels in the resulting hierarchical index", "optional": True},
        "verify_integrity": {"type": "boolean", "description": "Check whether the new concatenated axis contains duplicates", "optional": True},
        "sort": {"type": "boolean", "description": "Sort non-concatenation axis if it is not already aligned", "optional": True},
        "copy": {"type": "boolean", "description": "If False, do not copy data unnecessarily", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None):
        """
        Args:
            objs (list): List of DataFrames to concatenate
            axis (int, optional): Axis to concatenate along (0 for rows, 1 for columns)
            join (str, optional): How to handle indexes on other axis ('inner' or 'outer')
            ignore_index (bool, optional): If True, do not use the index values
            keys (list, optional): Construct hierarchical index
            levels (list, optional): Specific levels to use
            names (list, optional): Names for the levels in the resulting hierarchical index
            verify_integrity (bool, optional): Check whether the new concatenated axis contains duplicates
            sort (bool, optional): Sort non-concatenation axis if it is not already aligned
            copy (bool, optional): If False, do not copy data unnecessarily

        Returns:
            DataFrame: Concatenated DataFrame
        """
        import pandas as pd
        return pd.concat(objs, axis=axis, join=join, ignore_index=ignore_index, keys=keys, 
                        levels=levels, names=names, verify_integrity=verify_integrity, 
                        sort=sort, copy=copy)

class DataframeDrop(Tool):
    name = "dataframe_drop"
    description = "Drop rows or columns from a DataFrame."
    inputs = {
        "df": {"type": "object", "description": "DataFrame to modify"},
        "labels": {"type": "list", "description": "Index or column labels to drop", "optional": True},
        "axis": {"type": "integer", "description": "Whether to drop labels from index (0) or columns (1)", "optional": True},
        "index": {"type": "list", "description": "Alternative to labels for dropping by index", "optional": True},
        "columns": {"type": "list", "description": "Alternative to labels for dropping by columns", "optional": True},
        "level": {"type": "integer", "description": "For MultiIndex, level from which to drop labels", "optional": True},
        "inplace": {"type": "boolean", "description": "If True, do operation inplace and return None", "optional": True},
        "errors": {"type": "string", "description": "If 'ignore', suppress error if labels not in axis", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        """
        Args:
            df (DataFrame): DataFrame to modify
            labels (list, optional): Index or column labels to drop
            axis (int, optional): Whether to drop labels from index (0) or columns (1)
            index (list, optional): Alternative to labels for dropping by index
            columns (list, optional): Alternative to labels for dropping by columns
            level (int, optional): For MultiIndex, level from which to drop labels
            inplace (bool, optional): If True, do operation inplace and return None
            errors (str, optional): If 'ignore', suppress error if labels not in axis

        Returns:
            DataFrame: DataFrame with specified labels removed
        """
        import pandas as pd
        if inplace:
            df.drop(labels=labels, axis=axis, index=index, columns=columns, 
                   level=level, inplace=True, errors=errors)
            return df
        else:
            return df.drop(labels=labels, axis=axis, index=index, columns=columns, 
                          level=level, inplace=False, errors=errors)

class DataframeFill(Tool):
    name = "dataframe_fill"
    description = "Fill missing values in a DataFrame."
    inputs = {
        "df": {"type": "object", "description": "DataFrame to fill missing values in"},
        "value": {"type": "object", "description": "Value to use to fill holes", "optional": True},
        "method": {"type": "string", "description": "Method to use for filling holes ('ffill', 'bfill', etc.)", "optional": True},
        "axis": {"type": "integer", "description": "Axis along which to fill missing values", "optional": True},
        "inplace": {"type": "boolean", "description": "If True, fill in-place", "optional": True},
        "limit": {"type": "integer", "description": "Maximum number of consecutive NaN values to fill", "optional": True},
        "downcast": {"type": "string", "description": "Downcast dtypes if possible", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        """
        Args:
            df (DataFrame): DataFrame to fill missing values in
            value (object, optional): Value to use to fill holes
            method (str, optional): Method to use for filling holes ('ffill', 'bfill', etc.)
            axis (int, optional): Axis along which to fill missing values
            inplace (bool, optional): If True, fill in-place
            limit (int, optional): Maximum number of consecutive NaN values to fill
            downcast (str, optional): Downcast dtypes if possible

        Returns:
            DataFrame: DataFrame with missing values filled
        """
        import pandas as pd
        if inplace:
            df.fillna(value=value, method=method, axis=axis, inplace=True, 
                     limit=limit, downcast=downcast)
            return df
        else:
            return df.fillna(value=value, method=method, axis=axis, inplace=False, 
                            limit=limit, downcast=downcast)

class DataframeMerge(Tool):
    name = "dataframe_merge"
    description = "Merge DataFrames along an axis with optional filling logic."
    inputs = {
        "left": {"type": "object", "description": "Left DataFrame"},
        "right": {"type": "object", "description": "Right DataFrame"},
        "how": {"type": "string", "description": "Type of merge to be performed ('left', 'right', 'outer', 'inner')", "optional": True},
        "on": {"type": "list", "description": "Column names to join on", "optional": True},
        "left_on": {"type": "list", "description": "Column names from left DataFrame to join on", "optional": True},
        "right_on": {"type": "list", "description": "Column names from right DataFrame to join on", "optional": True},
        "left_index": {"type": "boolean", "description": "Use the index from left DataFrame as join key", "optional": True},
        "right_index": {"type": "boolean", "description": "Use the index from right DataFrame as join key", "optional": True},
        "sort": {"type": "boolean", "description": "Sort the join keys lexicographically", "optional": True},
        "suffixes": {"type": "list", "description": "Suffix to apply to overlapping column names", "optional": True},
        "copy": {"type": "boolean", "description": "If False, avoid copying data", "optional": True},
        "indicator": {"type": "boolean", "description": "Add a column to output showing merge source", "optional": True},
        "validate": {"type": "string", "description": "Validate merge keys ('one_to_one', 'one_to_many', etc.)", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        """
        Args:
            left (DataFrame): Left DataFrame
            right (DataFrame): Right DataFrame
            how (str, optional): Type of merge to be performed ('left', 'right', 'outer', 'inner')
            on (list, optional): Column names to join on
            left_on (list, optional): Column names from left DataFrame to join on
            right_on (list, optional): Column names from right DataFrame to join on
            left_index (bool, optional): Use the index from left DataFrame as join key
            right_index (bool, optional): Use the index from right DataFrame as join key
            sort (bool, optional): Sort the join keys lexicographically
            suffixes (tuple, optional): Suffix to apply to overlapping column names
            copy (bool, optional): If False, avoid copying data
            indicator (bool, optional): Add a column to output showing merge source
            validate (str, optional): Validate merge keys ('one_to_one', 'one_to_many', etc.)

        Returns:
            DataFrame: Merged DataFrame
        """
        import pandas as pd
        return pd.merge(left, right, how=how, on=on, left_on=left_on, right_on=right_on, 
                       left_index=left_index, right_index=right_index, sort=sort, 
                       suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)

class DataframeToNumeric(Tool):
    name = "dataframe_to_numeric"
    description = "Convert values in a DataFrame to numeric data."
    inputs = {
        "df": {"type": "object", "description": "DataFrame to modify"},
        "column": {"type": "string", "description": "Column to convert to numeric"},
        "errors": {"type": "string", "description": "How to handle errors ('ignore', 'raise', 'coerce')", "optional": True},
        "downcast": {"type": "string", "description": "Type to downcast to if possible", "optional": True}
    }
    output_type = "object"  # Returns DataFrame

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, column, errors='coerce', downcast=None):
        """
        Args:
            df (DataFrame): DataFrame to modify
            column (str): Column to convert to numeric
            errors (str, optional): How to handle errors ('ignore', 'raise', 'coerce')
            downcast (str, optional): Type to downcast to if possible

        Returns:
            DataFrame: DataFrame with specified column converted to numeric
        """
        import pandas as pd
        df_clean = df.copy()
        # Handle currency symbols and other non-numeric characters
        if df_clean[column].dtype == 'object':
            df_clean[column] = df_clean[column].astype(str).str.replace('$', '').str.replace(',', '')

        # Convert to numeric
        df_clean[column] = pd.to_numeric(df_clean[column], errors=errors, downcast=downcast)
        return df_clean
