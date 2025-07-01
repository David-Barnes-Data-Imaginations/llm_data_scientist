from opentelemetry import trace
import os, base64
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
LANGFUSE_PUBLIC_KEY= os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_SECRET_KEY= os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

# Global flag to prevent multiple instrumentation calls
_instrumentation_done = False

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
        
        # Use the correct Langfuse method - it should be trace() but let's be more defensive
        try:
            # Try the standard method first
            return self.langfuse.trace(
                name=name,
                input=input_data
            )
        except AttributeError:
            # If that doesn't work, try alternative methods
            try:
                return self.langfuse.create_trace(
                    name=name,
                    input=input_data
                )
            except AttributeError:
                # Fallback - create a simple mock trace object
                print(f"Warning: Langfuse trace creation failed, using mock trace for {name}")
                return MockTrace(name, input_data)

    def log_event(self, trace, name: str, metadata: dict):
        """Log an event to the trace"""
        try:
            if hasattr(trace, 'event'):
                trace.event(name=name, metadata=metadata)
            elif hasattr(trace, 'log'):
                trace.log(name=name, metadata=metadata)
        except Exception as e:
            print(f"Warning: Failed to log event {name}: {e}")

    def log_observation(self, trace, key: str, value):
        """Log an observation to the trace"""
        try:
            if hasattr(trace, 'observation'):
                trace.observation(key, value=value)
        except Exception as e:
            print(f"Warning: Failed to log observation {key}: {e}")

    def finish_trace(self, trace):
        """Finish the trace"""
        try:
            if hasattr(trace, 'end'):
                trace.end()
            elif hasattr(trace, 'finish'):
                trace.finish()
        except Exception as e:
            print(f"Warning: Failed to finish trace: {e}")


class MockTrace:
    """Complete mock trace object for fallback when Langfuse API is unavailable"""
    def __init__(self, name, input_data):
        self.name = name
        self.input_data = input_data
        self._inputs = {}
        self._outputs = {}
        print(f"Mock trace started: {name}")
    
    def event(self, name, metadata=None):
        print(f"Mock event: {name} - {metadata}")
    
    def observation(self, key, value=None):
        print(f"Mock observation: {key} = {value}")
    
    def add_input(self, key, value):
        """Add input to mock trace"""
        self._inputs[key] = value
        print(f"Mock input: {key} = {value}")
    
    def add_output(self, key, value):
        """Add output to mock trace"""
        self._outputs[key] = value
        print(f"Mock output: {key} = {value}")
    
    def log(self, name, metadata=None):
        print(f"Mock log: {name} - {metadata}")
    
    def end(self):
        print(f"Mock trace ended: {self.name}")
    
    def finish(self):
        print(f"Mock trace finished: {self.name}")


# Helper functions
def log_user_feedback(trace_id: str, is_positive: bool):
    telemetry = TelemetryManager()
    telemetry.log_feedback(trace_id, is_positive)