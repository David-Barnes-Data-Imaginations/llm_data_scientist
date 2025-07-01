# File for shared variable states
from typing import Dict

import pandas as pd

chunk_number = 0
dataframe_store: Dict[str, pd.DataFrame] = {}