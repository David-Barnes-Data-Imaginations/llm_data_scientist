# src/client/telemetry.py
from opentelemetry import trace
from langfuse import Langfuse
from opentelemetry.trace import format_trace_id

class TelemetryManager:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.langfuse = Langfuse()

    def start_trace(self, operation_name: str):
        return self.tracer.start_as_current_span(operation_name)

    def log_interaction(self, trace_id: str, input_data: str, output_data: str):
        self.langfuse.trace(
            id=trace_id,
            input=input_data,
            output=output_data
        )

    def log_feedback(self, trace_id: str, is_positive: bool):
        self.langfuse.score(
            value=1 if is_positive else 0,
            name="user-feedback",
            trace_id=trace_id
        )

# Convenience functions
def log_user_feedback(trace_id: str, is_positive: bool):
    TelemetryManager().log_feedback(trace_id, is_positive)