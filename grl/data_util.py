
import pathlib
import pandas as pd
import dill
import igraph as ig
import logging
from .np_util import *

DATA_PATH = "/Users/didi/Desktop/data"
def data_path_(p):
    return str((pathlib.Path(DATA_PATH) / p).absolute())

def desc_city():
    r = pd.read_csv(data_path_("geo_city.csv")).sort_values("plid_count").iloc[30:60]
    return r