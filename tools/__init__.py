"""
Tools module for the data scientist project.

This module provides various tools for data structure inspection, feature engineering,
database operations, and documentation.
"""

# Import from data_structure_inspection_tools.py
from .data_structure_inspection_tools import AnalyzePatterns, CheckDataframe, InspectDataframe, ValidateData
# Import from data_structure_feature_engineering_tools.py
from .data_structure_feature_engineering_tools import HandleMissingValues, CalculateSparsity


# Import from database_tools.py
from .database_tools import DatabaseConnect, QuerySales, QueryReviews
from .dataframe_manipulation_tools import DataframeMelt, DataframeConcat, DataframeDrop, DataframeFill, DataframeMerge, DataframeToNumeric
from .dataframe_storage import CreateDataframe, CopyDataframe
from .help_tools import GetToolHelp
from .code_tools import RunCodeRaiseErrors, RunSQL
# Import from documentation_tools.py
from .documentation_tools import (
    DocumentLearningInsights,
    RetrieveSimilarChunks,
    ValidateCleaningResults,
    SaveCleanedDataframe, RetrieveMetadata
)

__all__ = [
    # Data structure inspection tools
    'ValidateData',
    'AnalyzePatterns',
    'CheckDataframe',
    'InspectDataframe',

    # Help Tools
    'GetToolHelp',

    # Code Tools
    'RunCodeRaiseErrors',
    'RunSQL',

    # Data structure feature engineering tools
    'HandleMissingValues',
    'CalculateSparsity',
    
    # Database tools
    'DatabaseConnect',
    'QuerySales',
    'QueryReviews',
    
    # Documentation tools
    'DocumentLearningInsights',
    'RetrieveSimilarChunks',
    'ValidateCleaningResults',
    'SaveCleanedDataframe',
    'RetrieveMetadata',
    'GetToolHelp',

    # Dataframe Manipulation Tools
    'DataframeMelt',
    'DataframeConcat',
    'DataframeDrop',
    'DataframeFill',
    'DataframeMerge',
    'DataframeToNumeric',

    # Dataframe Storage Tools
    'CreateDataframe',
    'CopyDataframe',
]