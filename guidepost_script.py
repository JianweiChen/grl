# author: chenjianwei
## Write only functions, not classes if necessary
# my: 2731289611338
# def ej():
#     import pathlib
#     exec(pathlib.Path("/Users/didi/Desktop/repos/grl/guidepost.py").open('r').read(), globals())

import collections
import functools
import itertools
import pathlib
import random
import time

import dill
import igraph
import igraph as ig
import ipyleaflet
import ipyvuetify as ipyv
import ipywidgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import pypinyin
import redis
import torch
import torch.nn as nn
import traitlets
from scipy.spatial import KDTree

STATIC_LOCATION_BUAA                    = [116.34762, 39.97692]
STATIC_LOCATION_SHANGHAI_SUZHOU_MIDDLE  = [120.95164, 31.28841]
STATIC_LOCATION_SHENYANG                = [123.38, 41.8]

DATA_PATH = "/Users/didi/Desktop/data"

def data_path_(p):
    return str((pathlib.Path(DATA_PATH) / p).absolute())

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
for i in range(10):
    globals()[f'take_{i}'] = take(i)

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


def _np_distance_args_preprocess(*args):
    assert len(args) in (1, 4)
    if args.__len__() == 4:
        lng1, lat1, lng2, lat2 = args
    if args.__len__() == 1:
        gps_arr = args[0]
        lng1, lat1, lng2, lat2 = [gps_arr[:, i] for i in range(4)]
    return lng1, lat1, lng2, lat2

def np_distance(*args):
    lng1, lat1, lng2, lat2 = _np_distance_args_preprocess(*args)
    radius = 6371
    dlng = np.radians(lng2-lng1)
    dlat = np.radians(lat2-lat1)
    a = np.sin(dlat/2)*np.sin(dlat/2) \
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2) * np.sin(dlng/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d
def np_coords(*args):
    lng1, lat1, lng2, lat2 = _np_distance_args_preprocess(*args)
    sx = np.copysign(1, lng1-lng2)
    sy = np.copysign(1, lat1-lat2)
    x = np_distance(lng1, lat1, lng2, lat1)
    y = np_distance(lng1, lat1, lng1, lat2)
    return np.stack([sx*x, sy*y], axis=0).T
def np_mht(*args):
    r = np_coords(*args)
    y = np.abs(r[:, 0]) + np.abs(r[:, 1])
    return y
def torch_mht(x):
    return torch.tensor(np_mht(x.numpy()))

import torch
import numpy as np
import pathlib
from common import *

K1 = 64

def tensor_to_shape_with_zero(x, shape):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x_zero = torch.zeros(shape)
    x = x[:shape[0], :shape[1]]
    x = torch.cat([
        x,
        x_zero[..., :x.shape[1]]
    ], axis=0)
    x = x[:shape[0], :shape[1]]
    x = torch.cat([
        x,
        x_zero[:x.shape[0], ...]
    ], axis=1)
    x = x[:shape[0], :shape[1]]
    return x

def to_line_group_feature(x):
    K2 = K1 - 1
    seqcount, _ = x.shape
    if seqcount <= K2:
        x = tensor_to_shape_with_zero(x, (K1, 2))
        return x
    n_output = seqcount // K2 + 1
    h1 = seqcount // n_output
    hs = [h1]*n_output
    h2 = sum(hs)
    h3 = seqcount - h2
    i = 0
    while h3>0:
        hs[i] += 1
        h3 -= 1
        i += 1
    hcum = 0
    output_tensor_list = []
    for h in hs:
        ka, kb = hcum-1, hcum+h
        ka = max(0, ka)
        xpart = x[ka: kb, ...]
        xpart = tensor_to_shape_with_zero(xpart, (K1, 2))
        output_tensor_list.append(xpart)
        hcum += h
    output_tensor = torch.cat(output_tensor_list, axis=1)
    return output_tensor

def save_guidepost_material_tensor():
    df = pd.read_csv(data_path_("history/geo_stop_all_country.csv"))
    df = df.assign(location=lambda xdf: xdf.apply(axis=1, func=lambda row: (row.lng, row.lat)))
    df = df.groupby("lineid").location.unique()
    material_tensor = torch.cat(
        df.apply(lambda x: torch.tensor(x.tolist())).apply(to_line_group_feature).tolist(), axis=1)
    material_tensor = material_tensor.reshape((K1, -1, 2)).permute(1, 0, 2)
    torch.save(material_tensor, pathlib.Path(data_path_("material_tensor.pt")).open('wb'))

def load_guodepost_material_tensor():
    material_tensor = torch.load(
        pathlib.Path(data_path_("material_tensor.pt")).open('rb'))
    return material_tensor

def filter_material_tensor(x_tensor, center, km_a):
    _temp = material_tensor.to_sparse(2)
    kdtree = KDTree(_temp.values().numpy())
    x_idxs = torch.tensor(kdtree.query_ball_point(center, km_a/110))
    ls_tensor = _temp.indices().T.index_select(0, x_idxs)[:, 0].unique()
    output_tensor = material_tensor.index_select(0, ls_tensor)
    return output_tensor

@monkey(ig.Graph)
def dijk(g, src, tgts):
    kwargs = dict(mode='out')
    if 'weight' in g.edge_attributes():
        kwargs['weights'] = 'weight'
    paths = g.get_shortest_paths(src, tgts, **kwargs)
    return paths

def save_guidepost_tensor(name, material_tensor):
    """
    >>> material_tensor = load_guodepost_material_tensor()
    >>> center, km = [120.0, 30.0], 50
    >>> material_tensor = filter_material_tensor(material_tensor, center, km_a)
    >>> save_guidepost_tensor("area_near_shanyang", material_tensor)
    """
    material_tensor_sparse_view = material_tensor.to_sparse(2)
    x0 = material_tensor
    x1 = material_tensor_sparse_view.values()
    x2 = material_tensor_sparse_view.indices().T
    _temp = torch.cat([x2, x2.roll(-1, 0)], axis=1)
    _temp = 1-(_temp[:, 0] - _temp[:, 2]).type(torch.BoolTensor).type(torch.IntTensor)
    _temp = _temp.to_sparse().indices().squeeze()
    x3 = torch.stack([_temp, _temp+1]).T
    _temp = pd.Series(KDTree(location_tensor.numpy()).query_ball_point(location_tensor.numpy(), 0.8/110)) \
        .explode().reset_index() \
        .to_numpy() \
        .astype(np.int32)
    _temp = torch.tensor(_temp)
    x4 = _temp.index_select(0, (
        _temp[:, 0]-_temp[:, 1]).type(torch.BoolTensor).type(torch.IntTensor).to_sparse().indices().squeeze())

    _temp = torch.cat([
        location_tensor.index_select(0, x3[..., 0]),
        location_tensor.index_select(0, x3[..., 1])
    ], axis=1)
    x5 = torch_mht(_temp)
    _temp = torch.cat([
        location_tensor.index_select(0, x4[..., 0]),
        location_tensor.index_select(0, x4[..., 1])
    ], axis=1)
    x6 = torch_mht(_temp)
    x7 = torch.cat([x3, x4], axis=0)
    x8 = torch.stack([
        torch.concat([x5, x6], axis=0),
        torch.concat([x5 / 45, x6/4.5+5/60], axis=0)
    ], axis=1)

    _temp = torch.stack([
        x2[..., 0].index_select(0, x7[..., 0]),
        x2[..., 0].index_select(0, x7[..., 1]),
    ], axis=1)
    _temp2 = (_temp[..., 0] - _temp[..., 1]).type(torch.BoolTensor).type(torch.IntTensor).to_sparse().indices().squeeze()
    _temp = _temp.index_select(0, _temp2)

    _temp = pd.DataFrame(_temp, columns=['src', 'tgt']) \
        .groupby(['src', 'tgt']).size().reset_index()[['src', 'tgt']] \
        .to_numpy()
    x9 = torch.tensor(_temp)
    dirpath = pathlib.Path(data_path_(f"tensor_{name}"))
    dirpath.mkdir(parents=True, exist_ok=True)
    output_tensor_ids = [0, 1, 2, 7, 8, 9]
    for _id in output_tensor_ids:
        xx = locals()[f"x{_id}"]
        torch.save(xx, (dirpath/f'x{_id}.pt').open('wb'))

def load_guidepost_tensor(name):
    import os, re
    dirpath = pathlib.Path(data_path_(f"tensor_{name}"))
    assert dirpath.exists()
    tensor_map = dict()
    for filename in os.listdir(dirpath.absolute()):
        _rs = re.findall('(x[0-9]+).pt', filename)
        if len(_rs) == 1:
            varname = _rs[0]
            tensor_map[varname] = torch.load((dirpath/filename).open('rb'))
    return tensor_map

class Gpctx(object):
    """
    >>> gpctx_name = "shanghai_suzhou"
    >>> tensor_map = load_guidepost_tensor(gpctx_name)
    >>> gpctx = Gpctx(tensor_map)
    """
    def __init__(self, tensor_map, name=None):
        self.name = name
        self.g1 = None
        self.g2 = None
        self.kdtree = None
        self.lf = None
        print("Gpctx构建中...")
        self.tensor_map_to_gpctx(tensor_map)
        print("Gpctx构建完成")
    
    def tensor_map_to_gpctx(self, tensor_map):
        _dfe = pd.concat([
            pd.DataFrame(x7, columns=['src', 'tgt']),
            pd.DataFrame(x8, columns=['mht', 'weight'])
        ], axis=1)
        _dfv = pd.concat([
            pd.DataFrame(x2, columns=['l', 'seq']),
            pd.DataFrame(x1, columns=['lng', 'lat']),
        ], axis=1).reset_index()
        self.g1 = ig.Graph.DataFrame(pd.DataFrame(x9.numpy()))
        self.g2 = ig.Graph.DataFrame(_dfe, directed=True, vertices=_dfv)
        self.kdtree = KDTree(x1.numpy())
        self.lf = x0
