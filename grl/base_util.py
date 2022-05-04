import pandas as pd
import networkx as nx

def iterdir(obj):
    df = pd.DataFrame(dir(obj), columns=['member'])
    df = df.query("not member.str.startswith('_')")
    lines = df \
        .assign(first_letter=df.member.apply(lambda x: str(x)[0])) \
        .groupby('first_letter') \
        .member.unique() \
        .apply(lambda xs: ", ".join(xs)) \
        .to_numpy() \
        .tolist()
    for line in lines:
        yield line

def printdir(obj):
    for line in iterdir(obj):
        print("-", line)

def head(jter, n=10):
    more = False
    for i, line in enumerate(jter):
        print(line)
        if i >= n:
            more = True
            break
    if more:
        print("... for head")

def monkey(_class, method_name=None):
    def _decofunc(func):
        if not method_name:
            _method_name = func.__name__
            if _method_name.startswith('_'):
                _method_name = _method_name[1:]
        else:
            _method_name = method_name
        setattr(_class, _method_name, func)
        return func
    return _decofunc