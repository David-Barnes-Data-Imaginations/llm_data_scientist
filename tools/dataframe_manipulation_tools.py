import pandas as pd
from smolagents import Tool
import numpy as np
from src.client.telemetry import TelemetryManager
from langfuse import observe, get_client

# =====================================
# MELT
# =====================================


class DataframeMelt(Tool):
    name = "DataframeMelt"
    description = "Melt a DataFrame into a long-format DataFrame."
    inputs = {
        "frame": {"type": "object", "description": "DataFrame to melt"},
        "id_vars": {"type": "object", "description": "Identifier columns", "optional": True, "nullable": True},
        "value_vars": {"type": "object", "description": "Columns to unpivot", "optional": True, "nullable": True},
        "var_name": {"type": "string", "description": "Name of variable column", "optional": True, "nullable": True},
        "value_name": {"type": "string", "description": "Name of value column", "optional": True, "nullable": True},
        "col_level": {"type": "integer", "description": "MultiIndex column level", "optional": True, "nullable": True},
        "ignore_index": {"type": "boolean", "description": "Reset index in result", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    DataframeMelt: 
    A tool that transforms a DataFrame from wide format to long format (also called "unpivoting").
    Use this when you need to convert columns into rows, which is often required for visualization or certain types of analysis.
    This is the opposite of pivot operations.

    Example usage: 

    # Lets say we have a DataFrame:
    df = pd.DataFrame({
        'Name': ['John', 'Mary'],
        'Math': [90, 95],
        'Science': [85, 92],
        'History': [75, 88]
    })

    # Melt the DataFrame to convert subject columns to rows
    melted_df = DataframeMelt(
        frame=df,
        id_vars=['Name'],
        value_vars=['Math', 'Science', 'History'],
        var_name='Subject',
        value_name='Score'
    )

    # Result will have columns: 'Name', 'Subject', 'Score'
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
    @observe(name="DataframeMelt")
    def forward(self, frame: object, id_vars: object = None, value_vars: object = None, 
                var_name: str = None, value_name: str = None, col_level: int = None, 
                ignore_index: bool = None):
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("dataframe_melt", {
            "frame_type": str(type(frame).__name__),
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "melting_dataframe",
                "frame_shape": str(frame.shape) if hasattr(frame, 'shape') else "unknown"
            })

            # Build the melt parameters dict, filtering out None values
            melt_params = {
                "frame": frame,
            }

            if id_vars is not None:
                melt_params["id_vars"] = id_vars
            if value_vars is not None:
                melt_params["value_vars"] = value_vars
            if var_name is not None:
                melt_params["var_name"] = var_name
            if value_name is not None:
                melt_params["value_name"] = value_name
            if col_level is not None:
                melt_params["col_level"] = col_level
            if ignore_index is not None:
                melt_params["ignore_index"] = ignore_index

            # Call pd.melt with the correct parameters
            result = pd.melt(**melt_params)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return result

        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)
            pass



# =====================================
# CONCAT
# =====================================
class DataframeConcat(Tool):
    name = "DataframeConcat"
    description = "Concatenate DataFrames along a specified axis."
    inputs = {
        "objs": {"type": "object", "description": "First DataFrame to concatenate"},
        "frame": {"type": "object", "description": "Second DataFrames to concatenate"},
        "axis": {"type": "integer", "description": "the axis to concatenate on", "optional": True, "nullable": True},
        "ignore_index": {"type": "boolean", "description": "defaults to True, whether to ignore the index", "optional": True, "nullable": True},

    }
    output_type = "object"
    help_notes = """ 
    DataframeConcat: 
    A tool that combines multiple DataFrames into a single DataFrame, either by stacking them vertically (axis=0) or horizontally (axis=1).
    Use this when you need to combine separate DataFrames that have similar structure.

    Example usage: 

    # Lets say we have two dataframes:
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

    # Concatenate vertically (stack rows)
    vertical_concat = DataframeConcat(
        objs=[df1, df2],
        axis=0,
        ignore_index=True
    )

    # Concatenate horizontally (join columns)
    df3 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})
    horizontal_concat = DataframeConcat(
        objs=[df1, df3],
        axis=1
    )
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    @observe(name="DataframeConcat")
    def forward(self, objs: object, frame: object, axis: int = 0, ignore_index: bool = False, ):
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("dataframe_concat", {
            "objs": objs, "frame": frame, "axis": axis, "ignore_index": ignore_index,
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "concatenating_dataframes",
                "objs_types": str([type(objs).__name__]) if hasattr(objs, '__iter__') else "unknown"
            })
            # Build the melt parameters dict, filtering out None values
            concat_params = {
                "frame": frame,
                "objs": objs,
            }

            if axis is not None:
                concat_params["axis"] = axis

            concat_params["axis"] = axis

            result = pd.concat(**concat_params)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)



# =====================================
# DROP
# =====================================
class DataframeDrop(Tool):
    name = "DataframeDrop"
    description = "Drop rows or columns from a DataFrame."
    inputs = {
        "df": {"type": "object", "description": "DataFrame to modify"},
        "labels": {"type": "object", "description": "This is the label or list of labels (row index or column name(s)) that you want to drop.", "optional": True, "nullable": True},
        "axis": {"type": "integer", "description": "This parameter determines whether you're dropping rows or columns. 0=rows, 1=columns", "optional": True, "nullable": True},
        "inplace": {"type": "boolean", "description": "determines whether the operation modifies the DataFrame directly or returns a new DataFrame.", "optional": True, "nullable": True},
        "errors": {"type": "string", "description": "This parameter controls how drop() handles situations where a specified label is not found in the DataFrame. Default is 'raise', optional is 'ignore'", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    DataframeDrop: 
    DataframeDrop is used to specify what you want to remove and how you want to remove it. 
    You can drop rows or columns based on their labels (names).
    To drop a single row/column, provide a single label.
    To drop multiple rows/columns, provide a list of labels.

    df_no_city = DataframeDrop(labels='City', axis=1)
    print("\nDataFrame after dropping 'City' column (new DataFrame):")
    print(df_no_city)

    print("\nOriginal DataFrame (unchanged):")
    print(df)
    """
    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    @observe(name="DataframeDrop")
    def forward(self, df, labels=None, axis=None, inplace=False, errors='raise'):
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("dataframe_drop", {
            "df_type": str(type(df).__name__),
            "labels": str(labels),
            "axis": axis,
            "inplace": inplace,
            "errors": errors
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "dropping_from_dataframe",
                "df_shape_before": str(df.shape) if hasattr(df, 'shape') else "unknown"
            })

            drop_params = {}
            if labels is not None:
                drop_params["labels"] = labels
            if axis is not None:
                drop_params["axis"] = axis
            if errors is not None:
                drop_params["errors"] = errors

            if inplace:
                df.drop(inplace=True, **drop_params)
                result = df
            else:
                result = df.drop(**drop_params)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })

            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)


# =====================================
# FILL
# =====================================
class DataframeFill(Tool):
    name = "DataframeFill"
    description = "Fill missing values in a DataFrame."
    inputs = {
        "df": {"type": "object", "description": "DataFrame to modify"},
        "value": {"type": "object", "description": "Value to use to fill holes (e.g. 0), or dict/Series of values specifying which value to use for each column", "optional": True, "nullable": True},
        "method": {"type": "string", "description": "Method to use for filling holes: 'backfill'/'bfill', 'pad'/'ffill', None. Default is None", "optional": True, "nullable": True},
        "axis": {"type": "integer", "description": "Axis along which to fill missing values. 0 for index (rows), 1 for columns", "optional": True, "nullable": True},
        "inplace": {"type": "boolean", "description": "If True, fill in-place. Note: this will modify any other views on this object", "optional": True, "nullable": True},
        "limit": {"type": "integer", "description": "Maximum number of consecutive NaN values to forward/backward fill", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    DataframeFill: 
    A tool that fills missing values (NaN) in a DataFrame using various methods. This is useful for data cleaning and preparation before analysis.
    Use this when you need to handle missing data in your dataset.

    Example usage: 

    # Let's say we have a dataframe with missing values:
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan],
        'B': [np.nan, 5, np.nan, 7]
    })

    # Fill all missing values with a specific value
    filled_df = DataframeFill(
        df=df,
        value=0
    )

    # Fill missing values using forward fill method
    ffill_df = DataframeFill(
        df=df,
        method='ffill'
    )
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    @observe(name="DataframeFill")
    def forward(self, df, value=None, method=None, axis=None, inplace=False, limit=None):
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("dataframe_fill", {
            "df_type": str(type(df).__name__),
            "value": str(value),
            "method": method,
            "axis": axis,
            "inplace": inplace,
            "limit": limit
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "filling_dataframe",
                "df_shape": str(df.shape) if hasattr(df, 'shape') else "unknown",
                "missing_values_before": str(df.isna().sum().sum()) if hasattr(df, 'isna') else "unknown"
            })

            fill_params = {}
            if value is not None:
                fill_params["value"] = value
            if method is not None:
                fill_params["method"] = method
            if axis is not None:
                fill_params["axis"] = axis
            if limit is not None:
                fill_params["limit"] = limit

            if inplace:
                df.fillna(inplace=True, **fill_params)
                result = df
            else:
                result = df.fillna(**fill_params)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown",
                "missing_values_after": str(result.isna().sum().sum()) if hasattr(result, 'isna') else "unknown"
            })

            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            telemetry.finish_trace(trace)


# =====================================
# MERGE
# =====================================
class DataframeMerge(Tool):
    name = "DataframeMerge"
    description = "Merge two DataFrames."
    inputs = {
        "left": {"type": "object", "description": "First DataFrame to merge"},
        "right": {"type": "object", "description": "Second DataFrame to merge"},
        "how": {"type": "string", "description": "Type of merge to perform: 'left', 'right', 'outer', 'inner', 'cross'. Default is 'inner'", "optional": True, "nullable": True},
        "on": {"type": "object", "description": "Column name(s) to join on. Must be found in both DataFrames", "optional": True, "nullable": True},
        "left_on": {"type": "object", "description": "Column name(s) to join on in left DataFrame", "optional": True, "nullable": True},
        "right_on": {"type": "object", "description": "Column name(s) to join on in right DataFrame", "optional": True, "nullable": True},
        "left_index": {"type": "boolean", "description": "Use the index from the left DataFrame as the join key", "optional": True, "nullable": True},
        "right_index": {"type": "boolean", "description": "Use the index from the right DataFrame as the join key", "optional": True, "nullable": True},
        "sort": {"type": "boolean", "description": "Sort the join keys lexicographically in the result DataFrame", "optional": True, "nullable": True},
        "suffixes": {"type": "object", "description": "Tuple of suffixes to add to overlapping column names. Default is ('_x', '_y')", "optional": True, "nullable": True},
        "copy": {"type": "boolean", "description": "If False, avoid copying data when possible", "optional": True, "nullable": True},
        "indicator": {"type": "boolean", "description": "If True, adds a column '_merge' indicating the source of each row", "optional": True, "nullable": True},
        "validate": {"type": "string", "description": "Verify merge type: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    DataframeMerge: 
    A tool that combines two DataFrames based on common columns or indices, similar to SQL joins.
    Use this when you need to combine data from different sources that share common identifiers.
    The merge operation allows different types of joins to handle various data relationships.

    Example usage: 

    # Let's say we have two DataFrames:
    df_customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'city': ['NYC', 'LA', 'Chicago', 'Miami']
    })

    df_orders = pd.DataFrame({
        'order_id': [101, 102, 103],
        'customer_id': [1, 2, 1],
        'amount': [250, 150, 300]
    })

    # Inner merge (only matching records)
    merged_df = DataframeMerge(
        left=df_customers,
        right=df_orders,
        on='customer_id',
        how='inner'
    )

    # Left merge (all customers, even those without orders)
    all_customers = DataframeMerge(
        left=df_customers,
        right=df_orders,
        on='customer_id',
        how='left'
    )
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    @observe(name="DataframeMerge")
    def forward(self, left, right, how='inner', on=None, left_on=None, right_on=None, 
                left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), 
                copy=True, indicator=False, validate=None):
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("dataframe_merge", {
            "left_type": str(type(left).__name__),
            "right_type": str(type(right).__name__),
            "how": how,
            "on": str(on),
            "left_on": str(left_on),
            "right_on": str(right_on)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "merging_dataframes",
                "left_shape": str(left.shape) if hasattr(left, 'shape') else "unknown",
                "right_shape": str(right.shape) if hasattr(right, 'shape') else "unknown"
            })

            merge_params = {"left": left, "right": right}
            if how is not None:
                merge_params["how"] = how
            if on is not None:
                merge_params["on"] = on
            if left_on is not None:
                merge_params["left_on"] = left_on
            if right_on is not None:
                merge_params["right_on"] = right_on
            if left_index is not None:
                merge_params["left_index"] = left_index
            if right_index is not None:
                merge_params["right_index"] = right_index
            if sort is not None:
                merge_params["sort"] = sort
            if suffixes is not None:
                merge_params["suffixes"] = suffixes
            if copy is not None:
                merge_params["copy"] = copy
            if indicator is not None:
                merge_params["indicator"] = indicator
            if validate is not None:
                merge_params["validate"] = validate

            result = pd.merge(**merge_params)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return result

        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")


# =====================================
# TO NUMERIC
# =====================================
class DataframeToNumeric(Tool):
    name = "DataframeToNumeric"
    description = "Convert DataFrame column to numeric values."
    inputs = {
        "df": {"type": "object", "description": "DataFrame containing the column to convert"},
        "column": {"type": "string", "description": "Name of the column to convert to numeric values"},
        "errors": {"type": "string", "description": "How to handle conversion errors: 'raise' (default), 'coerce' (invalid values become NaN), or 'ignore' (return original)", "optional": True, "nullable": True},
        "downcast": {"type": "string", "description": "Downcast integer, signed or unsigned: 'integer', 'signed', 'unsigned', 'float'", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    DataframeToNumeric: 
    A tool that converts DataFrame columns containing text representations of numbers into actual numeric data types.
    Use this when you have columns with numbers stored as strings (often from CSV imports) that need to be converted 
    for mathematical operations or analysis. It automatically handles common formatting like dollar signs and commas.

    Example usage: 

    # Let's say we have a DataFrame with string numbers:
    df = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'price': ['$1,200', '$850', '$2,400'],
        'quantity': ['10', '25', '15']
    })

    # Convert price column to numeric (removes $ and commas automatically)
    df_numeric = DataframeToNumeric(
        df=df,
        column='price',
        errors='coerce'
    )

    # Convert quantity column to numeric
    df_final = DataframeToNumeric(
        df=df_numeric,
        column='quantity',
        errors='coerce'
    )

    # Now you can perform calculations:
    # df_final['total_value'] = df_final['price'] * df_final['quantity']
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
    @observe(name="DataframeToNumeric")
    def forward(self, df, column, errors='coerce', downcast=None):
        telemetry = TelemetryManager()
        langfuse = get_client()
        trace = telemetry.start_trace("dataframe_to_numeric", {
            "df_type": str(type(df).__name__),
            "column": column,
            "errors": errors,
            "downcast": str(downcast)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "converting_to_numeric",
                "df_shape": str(df.shape) if hasattr(df, 'shape') else "unknown",
                "column_dtype_before": str(df[column].dtype) if hasattr(df, '__getitem__') else "unknown"
            })

            df_clean = df.copy()

            if df_clean[column].dtype == 'object':
                telemetry.log_event(trace, "processing", {
                    "step": "cleaning_string_values",
                    "column": column
                })
                df_clean[column] = df_clean[column].astype(str).str.replace('$', '').str.replace(',', '')

            telemetry.log_event(trace, "processing", {
                "step": "converting_to_numeric",
                "column": column
            })
            
            # Convert to numeric using pd.to_numeric
            to_numeric_params = {"arg": df_clean[column]}
            if errors is not None:
                to_numeric_params["errors"] = errors
            if downcast is not None:
                to_numeric_params["downcast"] = downcast
                
            df_clean[column] = pd.to_numeric(**to_numeric_params)

            telemetry.log_event(trace, "success", {
                "result_shape": str(df_clean.shape) if hasattr(df_clean, 'shape') else "unknown",
                "column_dtype_after": str(df_clean[column].dtype) if hasattr(df_clean, '__getitem__') else "unknown"
            })
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            return df_clean

        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)
            langfuse.update_current_trace(user_id="cmc1u2sny0176ad07fpb9il4b")
            pass
