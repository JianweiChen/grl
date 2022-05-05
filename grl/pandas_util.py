
import pandas as pd
# from grl import monkey
from .base_util import monkey
import time

class PandasIndexContext(object):
    def __init__(self, df, index_key):
        self.df = df
        self.index_key = index_key
        self.old_index_key = None
    def __enter__(self):
        df = self.df
        old_index_key = self.df.index.name
        if not old_index_key:
            old_index_key = "index"
        self.old_index_key = old_index_key
        df = df \
            .reset_index() \
            .set_index(self.index_key)
        self.df = df
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        df = self.df
        df = df \
            .reset_index() \
            .set_index(self.old_index_key)
        self.df = df

@monkey(pd.DataFrame, "index_context")
def _pandas_index_context(df, index_key):
    return PandasIndexContext(df, index_key)


@monkey(pd.DataFrame, "assign_by")
def _pandas_assign_by(df, key, **kwargv)->pd.DataFrame:
    with df.index_context(key) as ctx:
        df = ctx.df
        df = df.assign(**kwargv)
        ctx.df = df
    return ctx.df

@monkey(pd.DataFrame, "assign_join_col")
def _pandas_assign_join_col(df, **kwargv)->pd.DataFrame:
    import functools
    for output_col, input_cols in kwargv.items():
        series_list = [
            getattr(df, input_col).astype(str) for input_col in input_cols
        ]
        output_series = functools.reduce(lambda x1, x2: x1+'_'+x2, series_list)
        df = df.assign(**{output_col: output_series})
    return df
