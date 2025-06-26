from opentelemetry import trace
import os, base64
from langfuse import Langfuse
from opentelemetry.trace import format_trace_id
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

LANGFUSE_PUBLIC_KEY: os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY: os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Evals functionality via opentelemetry and langfuse

class TelemetryManager:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.langfuse = Langfuse()
        # from HF docs, to review at runtime
        self.trace_provider = TracerProvider()
        self.trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        SmolagentsInstrumentor().instrument(tracer_provider=self.trace_provider)


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

# Helper functions
def log_user_feedback(trace_id: str, is_positive: bool):
    TelemetryManager().log_feedback(trace_id, is_positive)