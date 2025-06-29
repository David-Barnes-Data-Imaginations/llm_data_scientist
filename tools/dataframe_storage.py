import pandas as pd
import numpy as np
from smolagents import Tool
import pandas as pd
from main import dataframe_store
from src.client.telemetry import TelemetryManager
telemetry = TelemetryManager()

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

