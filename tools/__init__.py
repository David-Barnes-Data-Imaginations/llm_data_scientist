"""
Tools module for the data scientist project.

This module provides various tools for data structure inspection, feature engineering,
database operations, and documentation.
"""

# Import from data_structure_inspection_tools.py
from .data_structure_inspection_tools import AnalyzePatterns, CheckDataframe, InspectDataframe
# Import from data_structure_feature_engineering_tools.py
from .data_structure_feature_engineering_tools import OneHotEncode, ApplyFeatureHashing, SmoteBalance, HandleMissingValues, CalculateSparsity


# Import from database_tools.py
from .database_tools import DatabaseConnect, QuerySales, QueryReviews

# Import from documentation_tools.py
from .documentation_tools import (
    DocumentLearningInsights,
    EmbedAndStore,
    RetrieveSimilarChunks,
    ValidateCleaningResults,
    SaveCleanedDataframe
)

__all__ = [
    # Data structure inspection tools
    'AnalyzePatterns',
    'CheckDataframe',
    'InspectDataframe',
    
    # Data structure feature engineering tools
    'OneHotEncode',
    'ApplyFeatureHashing',
    'SmoteBalance',
    'HandleMissingValues',
    'CalculateSparsity',
    
    # Database tools
    'DatabaseConnect',
    'QuerySales',
    'QueryReviews',
    
    # Documentation tools
    'DocumentLearningInsights',
    'EmbedAndStore',
    'RetrieveSimilarChunks',
    'ValidateCleaningResults',
    'SaveCleanedDataframe'
]