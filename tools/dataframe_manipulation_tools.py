import pandas as pd
from smolagents import Tool
import numpy as np

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

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

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

    def forward(self, df, column, errors='coerce', downcast=None):
        df_clean = df.copy()
        if df_clean[column].dtype == 'object':
            df_clean[column] = df_clean[column].astype(str).str.replace('$', '').str.replace(',', '')
        df_clean[column] = pd.to_numeric(df_clean[column], errors=errors, downcast=downcast)
        return df_clean