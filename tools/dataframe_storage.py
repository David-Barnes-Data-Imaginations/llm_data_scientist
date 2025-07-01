from smolagents import Tool
import pandas as pd
from src.states.shared_state import dataframe_store
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

    def forward(self, data: list, name: str = "df") -> str:
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("create_dataframe", {
            "data_length": len(data) if hasattr(data, '__len__') else "unknown",
            "name": name
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "creating_dataframe",
                "data_sample": str(data[:2]) if hasattr(data, '__getitem__') and len(data) > 0 else "empty"
            })

            df = pd.DataFrame(data)
            dataframe_store[name] = df

            telemetry.log_event(trace, "success", {
                "df_shape": str(df.shape) if hasattr(df, 'shape') else "unknown",
                "df_columns": str(list(df.columns)) if hasattr(df, 'columns') else "unknown"
            })

            return f"✅ Created DataFrame '{name}' with shape {df.shape}"
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)


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

    def forward(self, source_dataframe: str, copy_name: str) -> str:
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("copy_dataframe", {
            "source_dataframe": source_dataframe,
            "copy_name": copy_name
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "checking_source_dataframe",
                "available_dataframes": str(list(dataframe_store.keys()))
            })

            if source_dataframe not in dataframe_store:
                telemetry.log_event(trace, "error", {
                    "error_type": "ValueError",
                    "error_message": f"Source DataFrame '{source_dataframe}' not found."
                })
                raise ValueError(f"❌ Source DataFrame '{source_dataframe}' not found.")

            telemetry.log_event(trace, "processing", {
                "step": "copying_dataframe",
                "source_shape": str(dataframe_store[source_dataframe].shape) if hasattr(dataframe_store[source_dataframe], 'shape') else "unknown"
            })

            dataframe_store[copy_name] = dataframe_store[source_dataframe].copy()

            telemetry.log_event(trace, "success", {
                "copy_shape": str(dataframe_store[copy_name].shape) if hasattr(dataframe_store[copy_name], 'shape') else "unknown"
            })

            return f"📄 DataFrame '{source_dataframe}' copied to '{copy_name}'"
        except Exception as e:
            telemetry.log_event(trace, "error", {
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            })
            raise
        finally:
            telemetry.finish_trace(trace)

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

    def forward(self, df: pd.DataFrame, filename: str = "tg_reviews_cleaned.csv") -> str:
        """
        Args:
            df (pd.DataFrame): The cleaned DataFrame
            filename (str): File name for the CSV output

        Returns:
            str: Confirmation message
        """
        telemetry = TelemetryManager()
        trace = telemetry.start_trace("save_cleaned_dataframe", {
            "df_type": str(type(df).__name__),
            "filename": filename
        })

        try:
            telemetry.log_event(trace, "processing", {
                "step": "preparing_csv",
                "df_shape": str(df.shape) if hasattr(df, 'shape') else "unknown",
                "df_columns": str(list(df.columns)) if hasattr(df, 'columns') else "unknown"
            })

            csv_bytes = df.to_csv(index=False).encode()

            telemetry.log_event(trace, "processing", {
                "step": "saving_file",
                "csv_size_bytes": len(csv_bytes),
                "destination": "sandbox" if self.sandbox else "local"
            })

            if self.sandbox:
                self.sandbox.files.write(filename, csv_bytes)
                result = f"Saved cleaned DataFrame to sandbox file: {filename}"
            else:
                with open(filename, "wb") as f:
                    f.write(csv_bytes)
                result = f"Saved cleaned DataFrame locally: {filename}"

            telemetry.log_event(trace, "success", {
                "message": result
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
