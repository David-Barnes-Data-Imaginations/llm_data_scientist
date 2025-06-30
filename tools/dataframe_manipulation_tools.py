import pandas as pd
from smolagents import Tool
import numpy as np
from src.client.telemetry import TelemetryManager

# =====================================
# ✅ GENERAL NOTES (applies to all classes)
# =====================================
# ✔ Good modularity
# ✔ Inputs and outputs are correctly specified
# ✔ Proper handling of optional/inplace arguments
# 🟡 Could improve: Error handling, input validation, and memory persistence

# =====================================
# MELT
# =====================================
class DataframeMelt(Tool):
    name = "dataframe_melt"
    description = "Melt a DataFrame into a long-format DataFrame."
    inputs = {
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

    # Create a sample wide-format DataFrame
    df = pd.DataFrame({
        'Name': ['John', 'Mary'],
        'Math': [90, 95],
        'Science': [85, 92],
        'History': [75, 88]
    })

    # Melt the DataFrame to convert subject columns to rows
    melted_df = DataframeMelt().forward(
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

    def forward(self, frame, **kwargs):
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("dataframe_melt", {
            "frame_type": str(type(frame).__name__),
            "kwargs": str(kwargs)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "melting_dataframe",
                "frame_shape": str(frame.shape) if hasattr(frame, 'shape') else "unknown"
            })

            result = pd.melt(frame, **kwargs)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })

            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)


# =====================================
# CONCAT
# =====================================
class DataframeConcat(Tool):
    name = "dataframe_concat"
    description = "Concatenate DataFrames along a specified axis."
    inputs = {
        "objs": {"type": "object", "description": "List of DataFrames to concatenate"},
        "axis": {"type": "integer", "optional": True, "nullable": True},
        "join": {"type": "string", "optional": True, "nullable": True},
        "ignore_index": {"type": "boolean", "optional": True, "nullable": True},
        "keys": {"type": "object", "optional": True, "nullable": True},
        "levels": {"type": "object", "optional": True, "nullable": True},
        "names": {"type": "object", "optional": True, "nullable": True},
        "verify_integrity": {"type": "boolean", "optional": True, "nullable": True},
        "sort": {"type": "boolean", "optional": True, "nullable": True},
        "copy": {"type": "boolean", "optional": True, "nullable": True},
    }
    output_type = "object"
    help_notes = """ 
    DataframeConcat: 
    A tool that combines multiple DataFrames into a single DataFrame, either by stacking them vertically (axis=0) or horizontally (axis=1).
    Use this when you need to combine separate DataFrames that have similar structure.

    Example usage: 

    # Create sample DataFrames
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

    # Concatenate vertically (stack rows)
    vertical_concat = DataframeConcat().forward(
        objs=[df1, df2],
        axis=0,
        ignore_index=True
    )

    # Concatenate horizontally (join columns)
    df3 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})
    horizontal_concat = DataframeConcat().forward(
        objs=[df1, df3],
        axis=1
    )
    """

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, objs, **kwargs):
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("dataframe_concat", {
            "objs_count": len(objs) if hasattr(objs, '__len__') else "unknown",
            "kwargs": str(kwargs)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "concatenating_dataframes",
                "objs_types": str([type(obj).__name__ for obj in objs]) if hasattr(objs, '__iter__') else "unknown"
            })

            result = pd.concat(objs, **kwargs)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })

            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)


# =====================================
# DROP
# =====================================
class DataframeDrop(Tool):
    name = "dataframe_drop"
    description = "Drop rows or columns from a DataFrame."
    inputs = {
        "df": {"type": "object", "description": "DataFrame to modify"},
        "labels": {"type": "object", "optional": True, "nullable": True},
        "axis": {"type": "integer", "optional": True, "nullable": True},
        "index": {"type": "object", "optional": True, "nullable": True},
        "columns": {"type": "object", "optional": True, "nullable": True},
        "level": {"type": "integer", "optional": True, "nullable": True},
        "inplace": {"type": "boolean", "optional": True, "nullable": True},
        "errors": {"type": "string", "optional": True, "nullable": True},
    }
    output_type = "object"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, inplace=False, **kwargs):
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("dataframe_drop", {
            "df_type": str(type(df).__name__),
            "inplace": inplace,
            "kwargs": str(kwargs)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "dropping_from_dataframe",
                "df_shape_before": str(df.shape) if hasattr(df, 'shape') else "unknown"
            })

            if inplace:
                df.drop(inplace=True, **kwargs)
                result = df
            else:
                result = df.drop(**kwargs)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })

            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)


# =====================================
# FILL
# =====================================
class DataframeFill(Tool):
    name = "dataframe_fill"
    description = "Fill missing values in a DataFrame."
    inputs = {
        "df": {"type": "object"},
        "value": {"type": "object", "optional": True, "nullable": True},
        "method": {"type": "string", "optional": True, "nullable": True},
        "axis": {"type": "integer", "optional": True, "nullable": True},
        "inplace": {"type": "boolean", "optional": True, "nullable": True},
        "limit": {"type": "integer", "optional": True, "nullable": True},
        "downcast": {"type": "string", "optional": True, "nullable": True},
    }
    output_type = "object"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, inplace=False, **kwargs):
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("dataframe_fill", {
            "df_type": str(type(df).__name__),
            "inplace": inplace,
            "kwargs": str(kwargs)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "filling_dataframe",
                "df_shape": str(df.shape) if hasattr(df, 'shape') else "unknown",
                "missing_values_before": str(df.isna().sum().sum()) if hasattr(df, 'isna') else "unknown"
            })

            if inplace:
                df.fillna(inplace=True, **kwargs)
                result = df
            else:
                result = df.fillna(**kwargs)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown",
                "missing_values_after": str(result.isna().sum().sum()) if hasattr(result, 'isna') else "unknown"
            })

            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)


# =====================================
# MERGE
# =====================================
class DataframeMerge(Tool):
    name = "dataframe_merge"
    description = "Merge two DataFrames."
    inputs = {
        "left": {"type": "object"},
        "right": {"type": "object"},
        "how": {"type": "string", "optional": True, "nullable": True},
        "on": {"type": "object", "optional": True, "nullable": True},
        "left_on": {"type": "object", "optional": True, "nullable": True},
        "right_on": {"type": "object", "optional": True, "nullable": True},
        "left_index": {"type": "boolean", "optional": True, "nullable": True},
        "right_index": {"type": "boolean", "optional": True, "nullable": True},
        "sort": {"type": "boolean", "optional": True, "nullable": True},
        "suffixes": {"type": "object", "optional": True, "nullable": True},
        "copy": {"type": "boolean", "optional": True, "nullable": True},
        "indicator": {"type": "boolean", "optional": True, "nullable": True},
        "validate": {"type": "string", "optional": True, "nullable": True},
    }
    output_type = "object"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, left, right, **kwargs):
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("dataframe_merge", {
            "left_type": str(type(left).__name__),
            "right_type": str(type(right).__name__),
            "kwargs": str(kwargs)
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "merging_dataframes",
                "left_shape": str(left.shape) if hasattr(left, 'shape') else "unknown",
                "right_shape": str(right.shape) if hasattr(right, 'shape') else "unknown"
            })

            result = pd.merge(left, right, **kwargs)

            telemetry.log_event(trace, "success", {
                "result_shape": str(result.shape) if hasattr(result, 'shape') else "unknown"
            })

            return result
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)


# =====================================
# TO NUMERIC
# =====================================
class DataframeToNumeric(Tool):
    name = "dataframe_to_numeric"
    description = "Convert DataFrame column to numeric values."
    inputs = {
        "df": {"type": "object"},
        "column": {"type": "string"},
        "errors": {"type": "string", "optional": True, "nullable": True},
        "downcast": {"type": "string", "optional": True, "nullable": True},
    }
    output_type = "object"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, column, errors='coerce', downcast=None):
        telemetry = TelemetryManager()
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

            df_clean[column] = pd.to_numeric(df_clean[column], errors=errors, downcast=downcast)

            telemetry.log_event(trace, "success", {
                "result_shape": str(df_clean.shape) if hasattr(df_clean, 'shape') else "unknown",
                "column_dtype_after": str(df_clean[column].dtype) if hasattr(df_clean, '__getitem__') else "unknown"
            })

            return df_clean
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)
