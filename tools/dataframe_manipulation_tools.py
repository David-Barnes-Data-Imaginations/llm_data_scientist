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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("dataframe_melt")
        self.trace.add_input("frame", "DataFrame to melt")
        self.trace.add_input("id_vars", "Identifier columns")
        self.trace.add_input("value_vars", "Columns to unpivot")
        self.trace.add_input("var_name", "Name of variable column")
        self.trace.add_input("value_name", "Name of value column")
        self.trace.add_input("col_level", "MultiIndex column level")
        self.trace.add_input("ignore_index", "Reset index in result")
        self.trace.add_output("melted_df", "Melted DataFrame")
        self.trace.end()

    def forward(self, frame, **kwargs):
        return pd.melt(frame, **kwargs)


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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("dataframe_concat")
        self.trace.add_input("objs", "List of DataFrames to concatenate")
        self.trace.add_input("axis", "Axis to concatenate along")
        self.trace.add_input("join", "How to handle indexes")
        self.trace.add_input("ignore_index", "Whether to ignore index values")
        self.trace.add_output("concatenated_df", "Concatenated DataFrame")
        self.trace.end()

    def forward(self, objs, **kwargs):
        return pd.concat(objs, **kwargs)


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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("dataframe_drop")
        self.trace.add_input("df", "DataFrame to modify")
        self.trace.add_input("labels", "Labels to drop")
        self.trace.add_input("axis", "Axis to drop from")
        self.trace.add_input("index", "Index labels to drop")
        self.trace.add_input("columns", "Column labels to drop")
        self.trace.add_input("inplace", "Whether to modify DataFrame in place")
        self.trace.add_output("modified_df", "Modified DataFrame")
        self.trace.end()

    def forward(self, df, inplace=False, **kwargs):
        if inplace:
            df.drop(inplace=True, **kwargs)
            return df
        return df.drop(**kwargs)


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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("dataframe_fill")
        self.trace.add_input("df", "DataFrame to fill")
        self.trace.add_input("value", "Value to fill with")
        self.trace.add_input("method", "Method to use for filling")
        self.trace.add_input("axis", "Axis to fill along")
        self.trace.add_input("inplace", "Whether to modify DataFrame in place")
        self.trace.add_input("limit", "Maximum number of consecutive NaNs to fill")
        self.trace.add_output("filled_df", "DataFrame with filled values")
        self.trace.end()

    def forward(self, df, inplace=False, **kwargs):
        if inplace:
            df.fillna(inplace=True, **kwargs)
            return df
        return df.fillna(**kwargs)


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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("dataframe_merge")
        self.trace.add_input("left", "Left DataFrame")
        self.trace.add_input("right", "Right DataFrame")
        self.trace.add_input("how", "Type of merge to perform")
        self.trace.add_input("on", "Column names to join on")
        self.trace.add_input("left_on", "Column names from left DataFrame to join on")
        self.trace.add_input("right_on", "Column names from right DataFrame to join on")
        self.trace.add_output("merged_df", "Merged DataFrame")
        self.trace.end()

    def forward(self, left, right, **kwargs):
        return pd.merge(left, right, **kwargs)


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

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("dataframe_to_numeric")
        self.trace.add_input("df", "DataFrame to modify")
        self.trace.add_input("column", "Column to convert to numeric")
        self.trace.add_input("errors", "How to handle errors")
        self.trace.add_input("downcast", "Type to downcast to")
        self.trace.add_output("converted_df", "DataFrame with numeric column")
        self.trace.end()

    def forward(self, df, column, errors='coerce', downcast=None):
        df_clean = df.copy()
        if df_clean[column].dtype == 'object':
            df_clean[column] = df_clean[column].astype(str).str.replace('$', '').str.replace(',', '')
        df_clean[column] = pd.to_numeric(df_clean[column], errors=errors, downcast=downcast)
        return df_clean
