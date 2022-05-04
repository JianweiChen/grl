
import pandas as pd
# from grl import monkey
from .base_util import monkey

@monkey(pd.DataFrame, "assign_by")
def _pandas_assign_by(df, key, **kwargv)->pd.DataFrame:
    pass