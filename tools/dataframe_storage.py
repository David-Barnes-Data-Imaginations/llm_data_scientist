import pandas as pd
import numpy as np
from smolagents import Tool
import pandas as pd
from main import dataframe_store
from src.client.telemetry import TelemetryManager
telemetry = TelemetryManager()
from main import chunk_number

class CreateDataframe(Tool):
    name = "create_dataframe"
    description = "Creates and stores a DataFrame from a list of dicts."
    inputs = {
        "data": {"type": "object", "description": "List of dictionaries (rows of data)"},
        "name": {"type": "string", "description": "Name to store the DataFrame under"}
    }
    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("create_dataframe")
        self.trace.add_input("data", "List of dictionaries (rows of data)")
        self.trace.add_input("name", "Name to store the DataFrame under")
        self.trace.add_output("success_message", "success message if no issues are found")
        self.trace.end()

    def forward(self, data: list, name: str = "df") -> str:
        df = pd.DataFrame(data)
        dataframe_store[name] = df
        return f"✅ Created DataFrame '{name}' with shape {df.shape}"


class CopyDataframe(Tool):
    name = "copy_dataframe"
    description = "Copies a DataFrame from one key to another."
    inputs = {
        "source_dataframe": {"type": "string", "description": "Name of the source DataFrame"},
        "copy_name": {"type": "string", "description": "New name for the copied DataFrame"}
    }
    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox
        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("copy_dataframe")
        self.trace.add_input("source_dataframe", "Name of the source DataFrame")
        self.trace.add_input("copy_name", "New name for the copied DataFrame")
        self.trace.add_output("success_message", "success message if no issues are found")
        self.trace.end()

    def forward(self, source_dataframe: str, copy_name: str) -> str:
        if source_dataframe not in dataframe_store:
            raise ValueError(f"❌ Source DataFrame '{source_dataframe}' not found.")
        dataframe_store[copy_name] = dataframe_store[source_dataframe].copy()
        return f"📄 DataFrame '{source_dataframe}' copied to '{copy_name}'"

class SaveCleanedDataframe(Tool):
    name = "save_cleaned_dataframe"
    description = "Saves the cleaned DataFrame to a CSV in the sandbox."
    inputs = {
        "df": {"type": "object", "description": "The cleaned DataFrame"},
        "filename": {"type": "string", "description": "File name for the CSV output", "optional": True, "nullable": True}
    }
    output_type = "string"  # Returns confirmation message

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

        self.telemetry = TelemetryManager()
        self.trace = self.telemetry.start_trace("save_cleaned_dataframe")
        self.trace.add_input("df", "The cleaned DataFrame")
        self.trace.add_input("filename", "File name for the CSV output")
        self.trace.add_output("confirmation", "Confirmation message")
        self.trace.end()

    def forward(self, df: pd.DataFrame, filename: str = "tg_reviews_cleaned.csv") -> str:
        """
        Args:
            df (pd.DataFrame): The cleaned DataFrame
            filename (str): File name for the CSV output

        Returns:
            str: Confirmation message
        """
        csv_bytes = df.to_csv(index=False).encode()

        if self.sandbox:
            self.sandbox.files.write(filename, csv_bytes)
            return f"Saved cleaned DataFrame to sandbox file: {filename}"
        else:
            with open(filename, "wb") as f:
                f.write(csv_bytes)
            return f"Saved cleaned DataFrame locally: {filename}"
