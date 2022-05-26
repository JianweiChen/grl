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

class StepReport(object):
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
GrlReport = StepReport

@monkey(ig.Graph, 'draw')
def _ig_graph_draw(g):
    nx.draw(g.to_networkx(), with_labels=True)


@monkey(pd.Series, 'swap')
def _swap_index_value(sr):
    index_name = sr.index.name
    value_name = sr.name
    if not index_name:
        index_name = "index"
    if not value_name:
        value_name = "value"
    sr = sr.rename(value_name)
    sr = sr.rename_axis(index_name)
    sr = sr.to_frame().reset_index().set_index(value_name)[index_name]
    return sr

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


@monkey(pd.DataFrame, 'assign_gps')
def _assign_gps(df):
    df = df.assign_join_col(gps=('lng', 'lat'))
    return df
@monkey(pd.DataFrame)
def assign_lineseq(df):
    df = df.assign(lineseq=lambda xdf: xdf.lineid.astype('str')+'_'+xdf.seq.astype('str'))
    return df

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

def reset_seq(seqs, k1):
    n = (len(seqs)-1) // k1
    n += 1
    p = len(seqs) // n + 1
    output_seqs = []
    for i in range(n):
        output_seqs.extend([(_, i, _-p*i) for _ in seqs[i*p:(i+1)*p]])
    return output_seqs

def check_seqs(seqs):
    return tuple(seqs) == tuple(range(1, len(seqs)+1))

def get_raw_df_stop():
    df_stop = pd.read_csv(data_path_("history/geo_stop_all_country.csv")) \
        .assign(subway=lambda xdf: (xdf.type==0).astype('int')) \
        [['lineid', 'sequenceno', 'lng', 'lat', 'stopname', 'linename', 'subway']] \
        .assign(lineid=lambda xdf: xdf.lineid.astype('str').apply(lambda x: x[:-4]).astype('int')) \
        .groupby(["lineid", 'sequenceno'], as_index=False).first() \
        .rename(dict(sequenceno='seq'), axis=1)
    return df_stop

def get_guidepost_input_stop_dataframe(df_stop, config):
    need_columns = [
        'lineid', 'seq',
        'lng', 'lat', 'subway',
        'stopname', 'linename'
    ]
    for need_column in need_columns:
        assert need_column in df_stop.columns

    k1 = config.get("k1", 32)
    dfl = df_stop.groupby("lineid").seq.unique().rename('seqs').to_frame()
    assert dfl.assign(seqs_normal=dfl.seqs.apply(check_seqs)).seqs_normal.all()
    dfl2 = dfl.assign(new_seqs=dfl.seqs.apply(lambda seqs: reset_seq(seqs, k1))) \
        .drop("seqs", axis=1) \
        .explode("new_seqs").rename(dict(new_seqs='seq_tp'), axis=1) \
        .assign(old_seq=lambda xdf: xdf.seq_tp.apply(take_0)) \
        .assign(sub_lineid=lambda xdf: xdf.seq_tp.apply(take_1)) \
        .assign(seq=lambda xdf: xdf.seq_tp.apply(take_2)) \
        .drop("seq_tp", axis=1) \
        .reset_index() \
        .assign(stopkey=lambda xdf: xdf.lineid.astype('str')+'_'+xdf.old_seq.astype('str')) \
        .assign(new_lineid=lambda xdf: xdf.lineid.astype("str")+'_'+xdf.sub_lineid.astype("str")) \
        .drop("lineid", axis=1) \
        .rename({"new_lineid": 'lineid'}, axis=1) \
        .set_index("stopkey")[['lineid', 'seq']]
    df_stop = df_stop.assign(stopkey=lambda xdf: xdf.lineid.astype("str")+'_'+xdf.seq.astype('str')) \
        .drop('lineid', axis=1).drop("seq", axis=1) \
        .set_index("stopkey") \
        .assign(lineid=dfl2.lineid, seq=dfl2.seq) \
        .reset_index(drop=True) \
        [['lineid', 'seq', 'lng', 'lat', 'subway', 'stopname', 'linename']] \
        .assign(lng=lambda xdf: round(xdf.lng, 5)) \
        .assign(lat=lambda xdf: round(xdf.lat, 5)) 
    return df_stop

   
"""
A machine learning solution for generating demand for bus transfer routes; as a whole, it is a recommendation system problem that recommends routes based on the current origin and destination conditions;
On the basis of known bus routes, determine the stops of getting on and off the bus according to some rules, please note that route recommendation solves almost 90% of the functions in this scenario;

TODO: The simplest version of the feature composition should be made into rnn variable length in the future
feature shape:
    k1 * (2 + k_history + k_lvp + k_candidate + 2)
context...
"""

def get_default_guidepost_config():
    config = {
        "center": [116.34762, 39.97692],
        "r_km": 50,
        "transfer_km": 0.8,
        "k1": 32,
        "k_history": 32,
        "k_lvp": 512,
        "k_candidate": 256,
        "max_loop": 10,
    }
    return config

def get_location_ls(context, location):
    """A location is represented by several nearby lines. 
    The order of these lines should be insensitive to the model.
    """
    _, ri_list = context['location_kdtree'].query(location, 10)
    lineids = [context['m_locationid_to_lineid'].get(_) for _ in ri_list]
    vertex_ids = [context['m_lineid_to_vertexid'].get(_) for _ in lineids]
    new_vertex_ids = []
    for vertex_id in vertex_ids:
        if vertex_id in new_vertex_ids:
            continue
        new_vertex_ids.append(vertex_id)
    return new_vertex_ids

def guidepost_get_item_feature(ctx, cls):
    """The actual serving method is the inner product between the output vector of the position subgraph and the output vector of the item subgraph, that is, the form of double towers.
    This is beneficial for serving in scenarios with an indeterminate number of candidate lines
    """
    g = context['g']
    return torch.stack(g.vs[cls]['lf'])

#deprecated: see robot_user_make_req
def make_extract_situation_feature_demo_req(ctx):
    """Given a guidepost_context, use its center position as the starting point, 10 kilometers northward as the ending point to imitate a user behavior, and use a random line near the starting point as the first item of history
    Just for testing function `extract_situation_feature`
    """
    config = ctx['config']
    center = config['center']
    ols = get_location_ls(ctx, origin)
    dls = get_location_ls(ctx, destination)
    first_l = random.choice(ols)
    history = [first_l]
    g = ctx['g']
    candidate_ls = g.neighbors(first_l, mode='out')
    req = dict(
        origin=center,
        destination=[center[0], center[1]+0.08],
        ols=ols,
        dls=dls,
        candidate_ls=candidate_ls,
        history=history,
    )
    return req

def extract_situation_feature(context, req):
    """
    feat history refers to the line selected in history; 
    feat lvp is the shortest path of the line nodes obtained by the Cartesian product of the start and end points; 
    feat_candidate refers to the line that can be selected in the current situation; 
    any situation that does not meet the length will be completed with 0 to obtain A regularized shape; 
    in particular, the lvp feature uses a k1*2 all-zero tensor as a separator between every two shortest paths of line nodes
    """

    origin = req['origin']
    destination = req['destination']
    ols = req['ols']
    dls = req['dls']
    candidate_ls = req['candidate_ls']
    
    g = context['g']
    config = context['config']
    k1 = config['k1']
    k_candidate = config['k_candidate']
    k_history = config['k_history']
    k_lvp = config['k_lvp']

    placeholder_vertex_id = g.vcount()-1

    history = req['history']
    feat_dst = np.broadcast_to(np.array([destination]), (k1, 2))
    feat_org = np.broadcast_to(np.array([origin]), (k1, 2))
    # dst,ln,..,l3,l2,l1,org
    feat_history_list = [
        feat_dst, *g.vs[history[::-1]]['line_feature'], feat_org
    ]
    feat_history = np.concatenate(feat_history_list, axis=1)
    feat_history = completion_arr(feat_history, (k1, k_history))
    
    if history:
        from_list = [history[-1]]
    else:
        from_list = ols
    tps = []
    for ol in ols:
        rs = context['g'].get_shortest_paths(ol, dls, mode='out')
        for r in rs:
            tps.append(tuple(r)+(placeholder_vertex_id,))
    lvids = functools.reduce(lambda x, y: x+y, tps, ())
    feat_lvp = np.concatenate(context['g'].vs[lvids]['line_feature'], axis=1)
    feat_lvp = completion_arr(feat_lvp, (k1, k_lvp))
    
    feat_candidate = np.concatenate(g.vs[candidate_ls]['line_feature'], axis=1)
    feat_candidate = completion_arr(feat_candidate, (k1, k_candidate))
    
    
    feat = np.concatenate([
        feat_history,
        feat_lvp,
        feat_candidate,
    ], axis=1)
    
    rsp2 = dict(
        feat_history=feat_history,
        feat_lvp = feat_lvp,
        feat_candidate=feat_candidate,
    )
    rsp = dict(
        feat=feat
    )
    return rsp

def set_default_config(config):
    if 'center' not in config:
        config['center'] = [116.35366, 39.98274]
    if 'k1' not in config:
        config['k1'] = 32
    if 'k_history' not in config:
        config['k_history'] = 16
    if 'k_candidate' not in config:
        config['k_candidate'] = 512
    if 'k_lvp' not in config:
        config['k_lvp'] = 256
    if 'transfer_km' not in config:
        config['transfer_km'] = 0.8
    if 'r_km' not in config:
        config['r_km'] = 50
    config['has_default'] = True

def get_df_stop_specific_area(df_stop, config):
    center = config['center']
    r_km = config['r_km']
    all_location_arr = df_stop[['lng', 'lat']].to_numpy()
    all_location_kdtree = KDTree(all_location_arr)
    
    ri_list = all_location_kdtree.query_ball_point(center, r_km/110)
    lineids = df_stop.loc[ri_list].lineid.unique().tolist()
    df_stop = df_stop.set_index("lineid").loc[lineids].reset_index()
    df_stop = df_stop.sort_values(['lineid', 'seq']).reset_index(drop=True) \
        .rename_axis("locationid")
    return df_stop

def make_g_sn(df_stop, config):
    transfer_km = config["transfer_km"]
    df = df_stop_part.assign_gps().assign_lineseq().sort_values(["lineid", 'seq'])
    df_stop_vertex_n = df[['gps', 'lng', 'lat']] \
        .rename(dict(gps='nodename'), axis=1) \
        .assign(nodetype='n')
    df_stop_vertex_s = df[['lineseq', 'lng', 'lat']] \
        .rename(dict(lineseq='nodename'), axis=1) \
        .assign(nodetype='s')
    df_stop_vertex = pd.concat([df_stop_vertex_n, df_stop_vertex_s], axis=0) \
        .groupby("nodename", as_index=False).first() \
        .sort_values("nodetype", ascending=False)
    df_s_src = df_stop_vertex_s[['nodename']] \
        .assign(src_tp=lambda xdf: xdf.nodename.apply(lambda x: tuple(x.split('_')))) \
        .sort_values("src_tp") \
        .rename(dict(nodename='src'), axis=1).reset_index(drop=True)
    df_s_tgt = df_s_src.rename(dict(src='tgt', src_tp='tgt_tp'), axis=1).reset_index(drop=True)
    df_ss_edge = pd.concat([df_s_src, df_s_tgt.shift(-1)], axis=1) \
            .query("src==src and tgt==tgt") \
            .reset_index(drop=True) \
            .assign(src_lineid=lambda xdf: xdf.src_tp.apply(take_0)) \
            .assign(tgt_lineid=lambda xdf: xdf.tgt_tp.apply(take_0)) \
            .query("src_lineid==tgt_lineid") \
            [['src', 'tgt']] \
            .assign(edgetype='ss')
    df_sn_edge = df[['gps', 'lineseq']] \
        .rename(dict(gps='src', lineseq='tgt'), axis=1) \
        [['src', 'tgt']] \
        .assign(edgetype='sn')
    df_ns_edge = df_sn_edge \
        .rename(dict(src='tgt', tgt='src'), axis=1) \
        [['src', 'tgt']] \
        .assign(edgetype='ns')
    arr = df_stop_vertex_n[['lng', 'lat']].to_numpy()
    kdtree = KDTree(arr)
    ri_list = kdtree.query_ball_point(arr, transfer_km/110)
    
    df_nn_edge = pd.Series(ri_list).explode() \
        .rename_axis('src_idx').rename('tgt_idx').to_frame().reset_index() \
        .astype(dict(src_idx='int', tgt_idx='int')) \
        .assign_by("src_idx", src=df_stop_vertex_n.nodename) \
        .assign_by("tgt_idx", tgt=df_stop_vertex_n.nodename) \
        .query("src!=tgt") \
        .reset_index(drop=True) \
        [['src', 'tgt']].assign(edgetype='nn')
    
    sr_vertex_lng = df_stop_vertex.set_index('nodename').lng
    sr_vertex_lat = df_stop_vertex.set_index('nodename').lat
    
    df_stop_edge = pd.concat([
        df_ss_edge,
        df_sn_edge,
        df_ns_edge,
        df_nn_edge,
    ], axis=0) \
    .reset_index(drop=True) \
    .groupby(['src', 'tgt'], as_index=False).first()
    edge_arr = df_stop_edge \
        .assign_by("src", srclng=sr_vertex_lng, srclat=sr_vertex_lat) \
        .assign_by('tgt', tgtlng=sr_vertex_lng, tgtlat=sr_vertex_lat) \
        [['srclng', 'srclat', 'tgtlng', 'tgtlat']] \
        .to_numpy()
    df_mht = pd.DataFrame(np_mht(edge_arr), columns=['mht'])
    df_stop_edge = df_stop_edge.assign(mht=df_mht.mht)
    df_k = pd.DataFrame([
        ("nn", 0.0, 1.0, 4.5),
        ("ns", 5/60, 0.0, 1.0),
        ("sn", 0.0, 0.0, 1.0),
        ('ss', 0.0, 1.0, 45),
    ], columns=['edgetype', 'k0', 'k1', 'k2']) \
    .set_index("edgetype")
    df_stop_edge = df_stop_edge.assign_by('edgetype', k0=df_k.k0, k1=df_k.k1, k2=df_k.k2) \
        .assign(weight=lambda xdf: xdf.k0 + xdf.mht * xdf.k1 / xdf.k2) \
        [['src', 'tgt', 'edgetype', 'mht', 'weight']].sort_values("weight", ascending=False)
    g_sn = ig.Graph.DataFrame(df_stop_edge, directed=True, vertices=df_stop_vertex)
    return g_sn

def build_guidepost_context(df_stop, config):
    if not config.get("has_default"):
        set_default_config(config)
    k1 = config['k1']

    center = config['center']
    r_km = config['r_km']
    transfer_km = config['transfer_km']

    df_stop = get_df_stop_specific_area(df_stop, guidepost_config) \
            .assign(location=lambda xdf: xdf.apply(axis=1, func=lambda row: (row.lng, row.lat)))

    location_arr = df_stop[['lng', 'lat']].to_numpy()
    location_kdtree = KDTree(location_arr)

    df_line_vertex = df_stop \
        .groupby("lineid").location.agg(list).apply(np.array).apply(lambda x: completion_arr(x, (k1, 2))) \
        .to_frame() \
        .rename({"location": "line_feature"}, axis=1) \
        .reset_index()
    sr_lineid = df_stop.lineid
    
    df_line_edge = pd.Series(location_kdtree.query_ball_point(location_arr, transfer_km/110)) \
        .explode() \
        .rename("tgt").rename_axis('src') \
        .to_frame() \
        .reset_index() \
        .astype(dict(src='int', tgt='int')) \
        .assign_by("src", srclid=sr_lineid) \
        .assign_by("tgt", tgtlid=sr_lineid) \
        .query("srclid!=tgtlid") \
        .groupby(['srclid', 'tgtlid']) \
        .size() \
        .rename("transfer_count") \
        .to_frame() \
        .reset_index() \
        [['srclid', 'tgtlid', 'transfer_count']]
    g = ig.Graph.DataFrame(df_line_edge, directed=True, vertices=df_line_vertex)
    g.add_vertex('placeholder_0', lf=np.zeros((k1, 2)))
    m_locationid_to_lineid = dict(df_stop[['lineid']].reset_index().to_numpy().tolist())
    m_lineid_to_vertexid = dict(zip(g.vs['name'], g.vs.indices))
    m_desc = dict(
        df_stop \
        .assign(key=lambda xdf: xdf.lineid+"_"+xdf.seq.astype("str")) \
            .assign(val=lambda xdf: xdf[['linename', 'stopname', 'lng', 'lat', 'subway']].apply(axis=1, func=dict)) \
            [['key', 'val']].to_numpy().tolist()
    )
    g_sn = make_g_sn(df_stop, config)
    
    ctx = dict(
        g=g,
        g_sn=g_sn,
        location_kdtree=location_kdtree,
        m_locationid_to_lineid=m_locationid_to_lineid,
        m_lineid_to_vertexid=m_lineid_to_vertexid,
        m_desc=m_desc,
        config=config,
    )
    return ctx

class GuidepostSituationModel(nn.Module):
    def __init__(self):
        super(GuidepostSituationModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        n_hidden_list = [
            1024,
            256,
            64
        ]
        ns = [
            k1*k2,
            *n_hidden_list
        ]
        sequential_list = []
        for i in range(len(ns)-1):
            sequential_list.append((f'linear_{i}', nn.Linear(ns[i], ns[i+1])))
            sequential_list.append((f'relu_{i}', nn.ReLU()))
        self.sequential = nn.Sequential(collections.OrderedDict(sequential_list))
    def forward(self, x):
        x = self.flatten(x)
        x = self.sequential(x)
        return x
class GuidepostItemModel(nn.Module):
    def __init__(self):
        super(GuidepostItemModel, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        n_hidden_list = [
            256,
            64
        ]
        ns = [
            k1*2,
            *n_hidden_list
        ]
        sequential_list = []
        for i in range(len(ns)-1):
            sequential_list.append((f'linear_{i}', nn.Linear(ns[i], ns[i+1])))
            sequential_list.append((f'relu_{i}', nn.ReLU()))
        self.sequential = nn.Sequential(collections.OrderedDict(sequential_list))
    def forward(self, x):
        x = self.flatten(x)
        x = self.sequential(x)
        return x
        
def load_situation_model():
    return GuidepostSituationModel()
def load_item_model():
    return GuidepostItemModel()

def generate_od(context, req):
    """用于根据context图中心的位置随机生成od对
    """
    rsp = dict()
    return rsp

def generate_instances(ctx, req):
    """生成一个batch的样本
    """
    rsp = dict()
    return rsp

def completion_arr(arr, shape):
    """The array becomes the specified shape
    Essentially different from reshape, it adopts the method of cropping and 0-completion
    """
    if shape[0] >= 0:
        extra_row_shape = (shape[0]-arr.shape[0], arr.shape[1])
        arr = arr[:shape[0], ...]
        if extra_row_shape[0] > 0:
            arr = np.concatenate([
                arr,
                np.broadcast_to(np.zeros_like(arr[-1:, ...]), extra_row_shape)
            ], axis=0)
    if shape[1] >= 0:
        arr = arr[..., :shape[1]]
        extra_column_shape = (arr.shape[0], shape[1]-arr.shape[1])
        if extra_column_shape[1] > 0:

            arr = np.concatenate([
                arr,
                np.broadcast_to(np.zeros_like(arr[..., -1:]), extra_column_shape)
            ], axis=1)
    return arr

def robot_user_make_req(context, pre_req):
    origin = random_location(context)
    destination = random_location(context)
    #TODO origin or destination use a GAN network to generate
    guidepost_req = dict(
        origin=origin,
        destination=destination,
    )
    return guidepost_req

def robot_user_do_label(context, guidepost_req, history_list):
    """
    :context dict:  line-vertex graph and some related data structure
    :guidepost_req dict:    including o/d and some pre calc data
    :history_list list: history represents the solution of line-vertex graph
    """
    pass

### 

def reward_furthest_from_org():
    pass
def reward_nearest_from_dst():
    pass
def reward_less_same_line():
    pass
def reward_shortest_lvp():
    pass

GrlReport = StepReport
