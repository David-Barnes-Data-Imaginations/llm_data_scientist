import pytest
from unittest.mock import patch, MagicMock
from tools.database_tools import DatabaseConnect

@pytest.fixture
def db_connect():
    """Setup for DatabaseConnect tests."""
    db_tool = DatabaseConnect()
    yield db_tool
    # Teardown can be added here if necessary

def test_forward_success(db_connect):
    """Test successful database connection."""
    with patch('tools.database_tools.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value.execute.return_value = None
        
        result = db_connect.forward()
        
        assert result == "Successfully connected to database: sqlite:///data/tg_database.db"

def test_forward_failure(db_connect):
    """Test failed database connection due to an exception."""
    with patch('tools.database_tools.create_engine', side_effect=Exception("Connection error")):
        result = db_connect.forward()
        
        assert result == "Failed to connect to database: Connection error"

def test_forward_logging_success(db_connect):
    """Test logging success event on successful connection."""
    with patch('tools.database_tools.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value.execute.return_value = None
        
        with patch('tools.database_tools.TelemetryManager') as mock_telemetry:
            mock_telemetry_instance = mock_telemetry.return_value
            db_connect.forward()
            
            mock_telemetry_instance.log_event.assert_called_with(mock_telemetry_instance.start_trace.return_value, "success", {
                "message": "Successfully connected to database"
            })

def test_forward_logging_error(db_connect):
    """Test logging error event on failed connection."""
    with patch('tools.database_tools.create_engine', side_effect=Exception("Connection error")):
        with patch('tools.database_tools.TelemetryManager') as mock_telemetry:
            mock_telemetry_instance = mock_telemetry.return_value
            db_connect.forward()
            
            mock_telemetry_instance.log_event.assert_called_with(mock_telemetry_instance.start_trace.return_value, "error", {
                "error_type": "Exception",
                "error_message": "Connection error"
            })