
import pathlib

DATA_PATH = "/Users/didi/Desktop/data"
def data_path_(p):
    return str((pathlib.Path(DATA_PATH) / p).absolute())

