from smolagents import Tool



class CreateDataframe(Tool):
    name = "CreateDataframe"
    description = "Creates and stores a DataFrame from a list of dicts."
    inputs = {
        "data": {"type": "object", "description": "List of dictionaries (rows of data)"},
        "name": {"type": "string", "description": "Name to store the DataFrame under", "nullable": True}
    }
    output_type = "string"

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, data, name="df"):
        import pandas as pd

        df = pd.DataFrame(data)

        return f"✅ Created DataFrame '{name}' with shape {df.shape}"


class SaveCleanedDataframe(Tool):
    name = "SaveCleanedDataframe"
    description = "Saves the cleaned DataFrame to a CSV in the sandbox."
    inputs = {
        "df": {"type": "object", "description": "The cleaned DataFrame"},
        "filename": {"type": "string", "description": "File name for the CSV output", "optional": True, "nullable": True}
    }
    output_type = "string"  # Returns confirmation message

    def __init__(self, sandbox=None):
        super().__init__()
        self.sandbox = sandbox

    def forward(self, df, filename="tg_reviews_cleaned.csv"):
        """
        Args:
            df (pd.DataFrame): The cleaned DataFrame
            filename (str): File name for the CSV output

        Returns:
            str: Confirmation message
        """
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            
        csv_bytes = df.to_csv(index=False).encode()

        if self.sandbox:
            self.sandbox.files.write(filename, csv_bytes)
            result = f"Saved cleaned DataFrame to sandbox file: {filename}"
        else:
            with open(filename, "wb") as f:
                f.write(csv_bytes)
            result = f"Saved cleaned DataFrame locally: {filename}"

        return result
