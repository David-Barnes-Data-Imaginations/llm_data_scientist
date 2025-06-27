"""
Tools module for the data scientist project.

This module provides various tools for data structure inspection, feature engineering,
database operations, and documentation.
"""

# Import from data_structure_inspection_tools.py
from .data_structure_inspection_tools import analyze_data_patterns, check_dataframe, inspect_dataframe

# Import from data_structure_feature_engineering_tools.py
from .data_structure_feature_engineering_tools import (
    one_hot_encode, 
    apply_feature_hashing, 
    smote_balance, 
    handle_missing_values,
    calculate_sparsity
)

# Import from database_tools.py
from .database_tools import get_db_connection, query_sales, query_reviews

# Import from documentation_tools.py
from .documentation_tools import (
    DocumentLearningInsights,
    embed_and_store,
    retrieve_similar_chunks,
    validate_cleaning_results,
    save_cleaned_dataframe
)

__all__ = [
    # Data structure inspection tools
    'analyze_data_patterns',
    'check_dataframe',
    'inspect_dataframe',
    
    # Data structure feature engineering tools
    'one_hot_encode',
    'apply_feature_hashing',
    'smote_balance',
    'handle_missing_values',
    'calculate_sparsity',
    
    # Database tools
    'get_db_connection',
    'query_sales',
    'query_reviews',
    
    # Documentation tools
    'DocumentLearningInsights',
    'embed_and_store',
    'retrieve_similar_chunks',
    'validate_cleaning_results',
    'save_cleaned_dataframe'
]