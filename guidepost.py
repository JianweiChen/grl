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

def get_raw_df_stop():
    df_stop = pd.read_csv(data_path_("history/geo_stop_all_country.csv")) \
        .assign(subway=lambda xdf: (xdf.type==0).astype('int')) \
        [['lineid', 'sequenceno', 'lng', 'lat', 'stopname', 'linename', 'subway']] \
        .assign(lineid=lambda xdf: xdf.lineid.astype('str').apply(lambda x: x[:-4]).astype('int')) \
        .groupby(["lineid", 'sequenceno'], as_index=False).first() \
        .rename(dict(sequenceno='seq'), axis=1)
    return df_stop

def get_default_guidepost_config():
    config = {
        # "center": [116.34762, 39.97692],
        "center": [120.95164, 31.28841],
        "km_area": 50,
        "km_transfer": 0.8,
        "k1": 32,
        "k_solution": 32,
        "k_dijk": 512,
        "k_candidate": 256,
        "k_sibling": 256,
    }
    return config

def set_default_config(config):
    if 'center' not in config:
        config['center'] = [116.35366, 39.98274]
    if 'k1' not in config:
        config['k1'] = 32
    if 'k_solution' not in config:
        config['k_solution'] = 16
    if 'k_candidate' not in config:
        config['k_candidate'] = 512
    if 'k_djik' not in config:
        config['k_djik'] = 256
    if 'k_sibling' not in config:
        config['k_sibling'] = 256
    if 'km_transfer' not in config:
        config['km_transfer'] = 0.8
    if 'km_area' not in config:
        config['km_area'] = 50
    config['has_default'] = True

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

def get_df_stop_specific_area(df_stop, config):
    center = config['center']
    km_area = config['km_area']
    all_location_arr = df_stop[['lng', 'lat']].to_numpy()
    all_location_kdtree = KDTree(all_location_arr)
    
    ri_list = all_location_kdtree.query_ball_point(center, km_area/110)
    lineids = df_stop.loc[ri_list].lineid.unique().tolist()
    df_stop = df_stop.set_index("lineid").loc[lineids].reset_index()
    df_stop = df_stop.sort_values(['lineid', 'seq']).reset_index(drop=True) \
        .rename_axis("locationid")
    return df_stop

def make_g_sn(df_stop, config):
    km_transfer = config["km_transfer"]
    df = df_stop.assign_gps().assign_lineseq().sort_values(["lineid", 'seq'])
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
    ri_list = kdtree.query_ball_point(arr, km_transfer/110)
    
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
    g_sn.vs['lineid'] = [_.rsplit('_', 1)[0] for _ in g_sn.vs['name']]
    return g_sn
def guidepost_create_location(ctx):
    config = ctx['config']
    km_area = config['km_area']
    center = config['center']
    x_theta = torch.rand(1,) * 2 * torch.pi
    x_abs = torch.rand(1,) * km_area
    x_coord = torch.polar(x_abs, x_theta)
    return torch.tensor([x_coord.real, x_coord.imag]).numpy()

def guidepost_feature_to_shape(x, shape):
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
def guidepost_feature_remove_zero(x):
    first_column = x[..., 0].tolist()
    first_row = x[0, ...].tolist()
    a1 = x.shape[0] if 0 not in first_column else first_column.index(0)
    a2 = x.shape[1] if 0 not in first_row else first_row.index(0)
    return x[:a1, :a2]

class LocationLsParser(object):
    def __init__(
        self,
        location_kdtree,
        m_locationid_to_lineid,
        m_lineid_to_vertexid
    ):
        self.location_kdtree = location_kdtree
        self.m_locationid_to_lineid = m_locationid_to_lineid
        self.m_lineid_to_vertexid = m_lineid_to_vertexid
    def get_ls(self, location):
        if isinstance(location, torch.Tensor):
            location = location.detach().numpy()
        if isinstance(location, list):
            location = np.array(location)
        _, ri_list = self.location_kdtree.query(location, 10)
        lineids = [self.m_locationid_to_lineid.get(_) for _ in ri_list]
        ls = [self.m_lineid_to_vertexid.get(_) for _ in lineids]
        output_ls = []
        for l in ls:
            if l in output_ls: continue
            output_ls.append(l)
        return output_ls

def mget(m, keys):
    return [
        m.get(key) for key in keys
    ]

def get_xs_ys(ctx, l):
    g = ctx['g']
    assert l < g.vcount(), f"{l} not in g in `get_xs_ys`"
    x_lf = g.vs[l]['lf']
    x_lf = guidepost_feature_remove_zero(x_lf)
    xs, ys = zip(*x_lf.tolist())
    return (xs, ys)

def plot_ls(ctx, ls, marker='-'):
    """
    >>> plot_ls(context, solutions[0], 'o')
    """
    g = ctx['g']
    args = []
    for l in ls:
        xs, ys = get_xs_ys(ctx, l)
        if not xs:
            continue
        args.append(xs)
        args.append(ys)
        args.append(marker)
    plt.plot(*args)

def guidepost_rand_location(ctx):
    config = ctx['config']
    km_area = config['km_area']
    center = config['center']
    x_theta = torch.rand(1,) * 2 * torch.pi
    x_abs = torch.rand(1,) * km_area / 110
    x_coord = torch.polar(x_abs, x_theta)
    return torch.tensor([center[0]+x_coord.real, center[1]+x_coord.imag]).numpy()

def guidepost_rand_query_req(ctx):
    origin = guidepost_rand_location(ctx)
    destination = guidepost_rand_location(ctx)
    query_req = dict(
        origin = origin,
        destination = destination,
        sibling=[],
        g_update=[],
        g_sn_update=[]
    )
    return query_req
    

def guidepost_init_query_context_from_query_req(ctx, query_req):
    origin = query_req['origin']
    destination = query_req['destination']
    g = ctx['g']
    lm = ctx['lm']
    ols = lm.get_ls(origin)
    dls = lm.get_ls(destination)
    cls = [] #TODO
    sls = []
    dijk = []
    for ol in ols:
        dijk.extend(g.get_shortest_paths(ol, dls, mode='out'))
    query_ctx = dict(
        origin=origin,
        destination=destination,
        ols=ols,
        dls=dls,
        cls=cls,
        sls=sls,
        dijk=dijk
    )
    return query_ctx

def update_query_context_by_sl(ctx, query_ctx, sl):
    g = ctx['g']
    dls = query_ctx['dls']
    query_ctx['sls'] = [*query_ctx['sls'], sl]
    query_ctx['cls'] = g.neighbors(sl, mode='out')
    query_ctx['dijk'] = g.get_shortest_paths(sl, dls, mode='out') # maybe use ols, too
    return query_ctx

def guidepost_get_feature(ctx, query_ctx):
    config = ctx['config']
    k1 = config.get("k1")
    k_solution = config.get("k_solution")
    k_dijk = config.get('k_dijk')
    k_candidate = config.get("k_candidate")
    k_sibling = config.get("k_sibling")
    g = context['g']
    
    origin = query_ctx['origin']
    destination = query_ctx['destination']
    ols = query_ctx['ols']
    dls = query_ctx['dls']
    cls = query_ctx['cls']
    sls = query_ctx['sls']
    
    ol_feat = torch.broadcast_to(torch.Tensor(origin), (k1, 2))
    dl_feat = torch.broadcast_to(torch.Tensor(destination), (k1, 2))
    solution_feat_list = [
        dl_feat, *g.vs[sls[::-1]]['lf'], ol_feat
    ]
    solution_feat = torch.cat(solution_feat_list, axis=1)
    solution_feat = guidepost_feature_to_shape(solution_feat, (k1, k_solution))
    
    dijk = query_ctx['dijk']
    dijk_feat_list = []
    for dijk_ls in dijk:
        dijk_feat_list.extend(g.vs[[*dijk_ls, g.vcount()-1]]['lf'])
    if dijk_feat_list:
        dijk_feat = torch.cat(dijk_feat_list, axis=1)
        dijk_feat = guidepost_feature_to_shape(dijk_feat, (k1, k_dijk))
    else:
        dijk_feat = torch.zeros([k1, k_dijk])
    if cls:
        candidate_feat = torch.cat(g.vs[cls]['lf'] , axis=1)
        candidate_feat = guidepost_feature_to_shape(candidate_feat, (k1, k_candidate))
        item_feat = torch.stack(*[g.vs[cls]['lf']])
    else:
        candidate_feat = torch.zeros([k1, k_candidate])
        item_feat = torch.zeros([1, k1, 2])
    #TODO slbling_feature
    
    feature_rsp = dict(
        solution_feat=solution_feat,
        dijk_feat=dijk_feat,
        candidate_feat=candidate_feat,
        item_feat=item_feat,
    )
    return feature_rsp

def guidepost_init_context(df_stop, guidepost_config):
    k1 = guidepost_config['k1']
    g_sn = make_g_sn(df_stop, guidepost_config)
    df_stop_vertex = g_sn.get_vertex_dataframe() \
        .assign(location=lambda xdf: xdf.apply(axis=1, func=lambda row: (row.lng, row.lat))) \
        .query("nodetype=='s'") \
        .assign(lineid=lambda xdf: xdf.name.str.rsplit('_', 1).apply(take_0))
    kdtree = KDTree(df_stop_vertex[['lng', 'lat']].to_numpy())
    df_line_vertex = df_stop_vertex.drop("name", axis=1) \
        .groupby("lineid").location.agg(list) \
        .apply(lambda x: guidepost_feature_to_shape(x, (k1, 2))) \
        .to_frame() \
        .rename({"location": "lf"}, axis=1) \
        .reset_index()
    df_line_edge = pd.DataFrame(
            kdtree.query_ball_point(df_stop_vertex[['lng', 'lat']].to_numpy(), km_transfer/110), 
            columns=['neighbor']
        ) \
        .explode("neighbor") \
        .astype(dict(neighbor='int')) \
        .assign(current_lineid=df_stop_vertex.lineid) \
        .assign_by("neighbor", neighbor_lineid=df_stop_vertex.lineid) \
        .query("current_lineid!=neighbor_lineid") \
        .groupby(["current_lineid", 'neighbor_lineid']) \
        .size().rename("crosscount").reset_index()
    g = ig.Graph.DataFrame(df_line_edge, directed=True, vertices=df_line_vertex)
    g.add_vertex("placeholder", lf=torch.zeros([k1, 2]))
    lm = LocationManager(g, g_sn, df_stop_vertex)
    guidepost_ctx = dict(
        g=g, g_sn=g_sn, lm=lm, config=guidepost_config
    )
    return guidepost_ctx
class LocationManager(object):
    def __init__(self, g, g_sn, df_stop_vertex):
        self.g_sn = g_sn
        self.kdtree = KDTree(df_stop_vertex[['lng', 'lat']].to_numpy())
        
        _m = dict(g.get_vertex_dataframe().reset_index()[['vertex ID', 'name']].set_index("name")['vertex ID'])
        self.ls_for_kdtree = [_m[_] for _ in df_stop_vertex.lineid.tolist()]
    def get_ls(self, location, k=10):
        _, ri_list = self.kdtree.query(location, k)
        ri_list = ri_list.tolist()
        ls = [self.ls_for_kdtree[_] for _ in ri_list]
        output_ls = []
        for l in ls:
            if l not in output_ls:
                output_ls.append(l)
        return output_ls
    def get_gs(self, location, k=20):
        _, ri_list = self.kdtree.query(location, k)
        ri_list = ri_list.tolist()
        gs = functools.reduce(lambda x, y: x+y, self.g_sn.neighborhood(ri_list, mode='out'), [])
        ss = self.g_sn.vs[gs].select(nodetype='n').indices
        output_ss = []
        for s in ss:
            if s not in output_ss:
                output_ss.append(s)
        return output_ss
    def get_shortest_paths(self, origin, destination):
        oss = self.get_gs(origin)
        dss = self.get_gs(destination)
        paths = []
        for s in oss:
            for path in self.g_sn.get_shortest_paths(s, dss, mode='out', weights='weight', output='epath'):
                paths.append(path)
        return paths
    
def is_location_lf(lf):
    # return torch.count_nonzero(lf[0, ...]-lf[1, ...]).tolist()<=0
    return torch.count_nonzero(lf - torch.broadcast_to(lf[0, ...], lf.shape)).tolist()<=0

def lf_tensor_to_lineparts(self, lf):
    g_sn = self.g_sn
    lf = guidepost_feature_remove_zero(lf)
    k1 = lf.shape[0]
    result_lineids = []
    for i in range(k1):
        name = '_'.join([str(_) for _ in lf[i, ...].detach().numpy()])
        vids = g_sn.vs[g_sn.neighbors(name, mode='out')].select(nodetype='s').indices
        lineids = g_sn.vs[vids]['lineid']
        if not result_lineids:
            result_lineids.extend(lineids)
        else:
            result_lineids = [_ for _ in result_lineids if _ in lineids]
        if result_lineids.__len__() <= 1:
            break
    result = []
    for result_lineid in result_lineids:
        result.append((result_lineid, (0, 0)))
    return result
    