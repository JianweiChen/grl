
# def ej():
#     import pathlib
#     exec(pathlib.Path("/Users/didi/Desktop/repos/grl/dev.py").open('r').read(), globals())

import numpy as np
import pandas as pd
import pathlib
import plotly
import ipyleaflet
import ipyvuetify as ipyv
import ipywidgets
import igraph as ig
import torch

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

def take(n):
    def func(xs):
        if n >= xs.__len__():
            return None
        return xs[n]
    return func

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

class GrlReport(object):
    def __init__(self, prefix):
        self.prefix = prefix
        self.tc = None
    def __call__(self, msg):
        logging.warning(f"{self.prefix} {msg}")
    def __enter__(self):
        self.tc = time.time()
        self("doing...")
    def __exit__(self, exc_type, exc_val, exc_tb):
        cost = time.time() - self.tc
        msg = f"done, cost {cost} sec"
        self(msg)

"""可以考虑将公交图分割成若干块
只有将这种管理上的连续化做好 事情才能更容易的做下去

N- Sign
P- Platform
E- Entrance
T- Track
B- Bus
这种设计T和B的关键区别定要名正言顺 关键是T要有进站操作 进站要买票

站外走路 Nn, Ne, En, Ee
站内走路 Pp, Pe, Ep
坐车 Bb, Tt
上下车 Bn, Nb, Tp, Pt
特别的Ee表示火车站出来进地铁或者机场线这种 Pp转Ee操作

line_group: 6hao#9879987
cluster_group: beijing#1
stop_name: handianhuangzhuang#123213219
sequence_no: 1,2,3,4...
direction: 0, 1
lng
lat
platform_type: 0, 1
loop_type: 0, 1, 2
schedules: 1#10:00, 2#10:10, 3#10:20, 5#10:30... 11:15
tripcount: 0
entrances: Akou#12314213

"""
class BusGraph(object):
    def __init__(self):
        pass
