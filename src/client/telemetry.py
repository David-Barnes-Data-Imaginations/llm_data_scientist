from opentelemetry import trace
import os, base64
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class TelemetryManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not TelemetryManager._initialized:
            self.tracer = trace.get_tracer(__name__)
            self.langfuse = Langfuse()
            
            # Initialize trace provider only once
            self.trace_provider = TracerProvider()
            self.trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
            
            # Only instrument once
            try:
                SmolagentsInstrumentor().instrument(tracer_provider=self.trace_provider)
            except Exception as e:
                print(f"Instrumentation warning: {e}")
            
            TelemetryManager._initialized = True

    def start_trace(self, name: str, input_data: dict = None):
        """Start a new Langfuse trace"""
        if input_data is None:
            input_data = {}
        return self.langfuse.trace(name=name, input=input_data)

    def log_event(self, trace, name: str, metadata: dict):
        """Log an event to the trace"""
        if hasattr(trace, 'event'):
            trace.event(name=name, metadata=metadata)

    def log_observation(self, trace, key: str, value):
        """Log an observation to the trace"""
        if hasattr(trace, 'observation'):
            trace.observation(key, value=value)

    def finish_trace(self, trace):
        """Finish the trace"""
        if hasattr(trace, 'end'):
            trace.end()


# Helper functions
def log_user_feedback(trace_id: str, is_positive: bool):
    TelemetryManager().log_feedback(trace_id, is_positive)