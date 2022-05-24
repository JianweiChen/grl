## Write only functions, not classes if necessary
# my: 2731289611338
# def ej():
#     import pathlib
#     exec(pathlib.Path("/Users/didi/Desktop/repos/grl/dev.py").open('r').read(), globals())

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

@monkey(ig.Graph, 'draw')
def _ig_graph_draw(g):
    nx.draw(g.to_networkx(), with_labels=True)

def connect_graph_in_circle_mode(g):
    """用于生成连通图
    """
    leaders = [_[0] for _ in g.components()]
    if leaders.__len__() == 1:
        return
    leader_edges = list(zip(leaders, [*leaders[1:], leaders[0]]))
    for leader_edge in leader_edges:
        g.add_edge(*leader_edge)

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

def to_pinyin(s):
    rs = pypinyin.pinyin(s, style=0)
    rs = [_[0] for _ in rs]
    r = '_'.join(rs)
    return r

@monkey(ig.Graph, 'plot3d')
def _ig_plot3d(g):
    layout_conf = dict()
    layout_conf['maxiter'] = 100_0000
    coords_array = np.array(g.layout_kamada_kawai(dim=2, **layout_conf).coords)
    # coords_array = np.array(g.layout_bipartite(**layout_conf).coords)
    df_vertex = g_s.get_vertex_dataframe()
    df_edge = g_s.get_edge_dataframe()
    df_vertex['x'] = coords_array[..., 0]
    df_vertex['y'] = coords_array[..., 1]
    df_vertex['z'] = 0.0
    df_edge = df_edge \
        .assign_by("source", src_x = df_vertex.x, src_y = df_vertex.y, src_z = df_vertex.z) \
        .assign_by("target", tgt_x = df_vertex.x, tgt_y = df_vertex.y, tgt_z = df_vertex.z) \
        .assign(null_x=np.nan, null_y=np.nan, null_z=np.nan)

    traces = []
    vertex_trace = go.Scatter3d(x=df_vertex.x.to_numpy(), y=df_vertex.y.to_numpy(), z=df_vertex.z.to_numpy(), 
                    name='vertex', text=df_vertex.name.tolist(),
                    mode='markers', hoverinfo='text', marker=dict(size=5., opacity=0.5))
    traces.append(vertex_trace)
    edge_trace = go.Scatter3d(
        x=df_edge[['src_x', 'tgt_x', 'null_x']].to_numpy().flatten(),
        y=df_edge[['src_y', 'tgt_y', 'null_y']].to_numpy().flatten(),
        z=df_edge[['src_z', 'tgt_z', 'null_z']].to_numpy().flatten(),
        name='edge', mode='lines', hoverinfo='none', marker=dict(size=5., opacity=0.5))
    traces.append(edge_trace)
    fig = go.Figure(data=[*traces], layout=go.Layout(hovermode='closest'))
    return fig

@monkey(pd.DataFrame, 'assign_gps')
def _assign_gps(df):
    df = df.assign_join_col(gps=('lng', 'lat'))
    return df

#### now, start to compile busgraph igraph

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

def fill_empty_gc_for_gate_sid(row):
    gc = row.gc
    gate = row.gate
    if eval(gc).__len__() == 0 and gate>0:
        gc = str([row.lng, row.lat])
    return gc

def load_desc_df(path=data_path_("desc.csv")):
    """
    >>> df_desc = load_desc_df()
    >>> df_desc.query("stopname.str.contains('知春路')")
    """
    df = pd.read_csv(path).set_index(['lid', 'seq']) \
        [['desc', 'stopid', 'lineid', 'lng', 'lat']] \
        .assign(stopname=lambda xdf: xdf.desc.str.split('_').apply(lambda x: x[1])) \
        .assign(linename=lambda xdf: xdf.desc.str.split('_').apply(lambda x: x[0])) \
        .assign(coordinate=lambda xdf: xdf.lng.astype('str')+','+xdf.lat.astype('str'))
    return df




class Vizmap(traitlets.HasTraits):
    """For taking latitude and longitude points in jupyter notebook
    >>> vm = Vizmap() 
    >>> vm.container    # show a ipyleaflet map
    >>> vm.gps          # get coordinate
    >>> vm.gpskey       # easy to copy
    """
    def __init__(self, coordinate=[116.35366,39.98274]):
        super().__init__()
        def on_click_btn(*args):
            self.dialog.v_model = True
        def on_m_interaction(**kwargv):
            if kwargv.get('type') == 'click':
                coordinate = kwargv.get("coordinates")
                coordinate = [
                    round(coordinate[0], 5),
                    round(coordinate[1], 5),
                ]
                self.marker.location = self.coordinate = coordinate
                # self.marker.location = self.coordinate = coordinate
                
        self.zoom = 13
        self.coordinate = coordinate[::-1]
        self.width = 800
        self.m = ipyleaflet.Map(
            basemap=ipyleaflet.basemap_to_tiles(ipyleaflet.basemaps.Gaode.Normal), 
            zoom=self.zoom,
            scroll_wheel_zoom=True,
            center=coordinate[::-1]
        )
        self.marker = ipyleaflet.Marker(location=coordinate[::-1], draggable=False)
        self.m.add_layer(self.marker)
        self.m.on_interaction(on_m_interaction)
        self.btn = ipyv.Btn(children=['pop'])
        self.dialog = \
            ipyv.Dialog(children=[
                ipyv.Card(children=[
                    self.m
                ], width=self.width),
            ], v_model=False, width=self.width)
        self.btn.on_event('click', on_click_btn)
        # ipywidgets.jslink((self.text_field, 'v_model'), (self, 'coordinate'))
        self.container = \
            ipyv.Container(children=[
                self.btn,
                self.dialog
            ])
    @property
    def gps(self):
        return self.coordinate[::-1]
    @property
    def gpskey(self):
        return ','.join([str(_) for _ in self.gps])

def get_prepare_df():
    df = pd.read_csv(data_path_("history/geo_stop_all_country.csv")) \
        .assign(subway=lambda xdf: (xdf.type==0).astype('int')) \
        [['lineid', 'sequenceno', 'lng', 'lat', 'stopname', 'linename', 'subway']] \
        .assign(lineid=lambda xdf: xdf.lineid.astype('str').apply(lambda x: x[:-4]).astype('int')) \
        .groupby(["lineid", 'sequenceno'], as_index=False).first()
    return df

def check_guidepost_context_input_df(df):
    """这个df是df_sid 要包含 lineid, stopname, linename, sequenceno, lng, lat, type(subway)
    在消重问题上也有一些要求
    TODO
    """
    return True

### line recommend
   
"""
最简单版本的特征构成 未来应该做成rnn可变长的
feature shape:
    k1 * (2 + k_history + k_lvp + k_candidate + 2)

context
"""

def get_default_guidepost_config():
    config = {
        "center": [116.34762, 39.97692],
        "r_km": 50,
        "transfer_km": 0.8,
        "k1": 32,
        "k_history": 32,
        "k_lvp": 512,
        "k_candidate": 256
    }
    return config

def get_location_ls(context, location):
    _, ri_list = context['location_kdtree'].query(location, 10)
    lineids = [context['m_locationid_to_lineid'].get(_) for _ in ri_list]
    vertex_ids = [context['m_lineid_to_vertexid'].get(_) for _ in lineids]
    new_vertex_ids = []
    for vertex_id in vertex_ids:
        if vertex_id in new_vertex_ids:
            continue
        new_vertex_ids.append(vertex_id)
    return new_vertex_ids

def extract_item_feature(ctx, lid):
    g = context['g']
    return g.vs[lid]['line_feature']
def make_extract_situation_feature_demo_req(ctx):
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

def build_guidepost_context(df, config):
    assert check_guidepost_context_input_df(df), "check_guidepost_context_input_df fail"
    assert 'center' in config

    k1 = config.get("k1", 32)

    center = config['center']
    r_km = config.get("r_km", 10)
    transfer_km = config.get("transfer_km", 0.8)
    
    
    if 'sequenceno' in df.columns and 'seq' not in df.columns:
        df = df.rename({'sequenceno': 'seq'}, axis=1)
    
    df = df \
        .assign(seq_mod = df.seq%k1) \
        .assign(seq_div = df.seq//k1)
    df = df.assign(lineid=df.lineid.astype('str')+'_'+df.seq_div.astype("str"))
    df = df.assign(seq=df.seq_mod)
    df = df.drop("seq_mod", axis=1).drop("seq_div", axis=1)

    all_location_arr = df[['lng', 'lat']].to_numpy()
    all_location_kdtree = KDTree(all_location_arr)
    
    ri_list = all_location_kdtree.query_ball_point(center, r_km/110)
    lineids = df.loc[ri_list].lineid.unique().tolist()
    df = df.set_index("lineid").loc[lineids].reset_index()
    df = df.sort_values(['lineid', 'seq']).reset_index(drop=True) \
        .rename_axis("locationid")


    df = df.assign(location=df.apply(axis=1, func=lambda row: (row.lng, row.lat)))
    location_arr = df[['lng', 'lat']].to_numpy()
    location_kdtree = KDTree(location_arr)

    df_line_vertex = df \
        .groupby("lineid").location.agg(list).apply(np.array).apply(lambda x: completion_arr(x, (k1, 2))) \
        .to_frame() \
        .rename({"location": "line_feature"}, axis=1) \
        .reset_index()
    sr_lineid = df.lineid
    
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
    g.add_vertex('placeholder_0', line_feature=np.zeros((k1, 2)))
    m_locationid_to_lineid = dict(df[['lineid']].reset_index().to_numpy().tolist())
    m_lineid_to_vertexid = dict(zip(g.vs['name'], g.vs.indices))
    m_desc = dict(
        df \
        .assign(key=lambda xdf: xdf.lineid+"_"+xdf.seq.astype("str")) \
            .assign(val=lambda xdf: xdf[['linename', 'stopname', 'lng', 'lat', 'subway']].apply(axis=1, func=dict)) \
            [['key', 'val']].to_numpy().tolist()
    )
    
    ctx = dict(
        g=g,
        location_kdtree=location_kdtree,
        m_locationid_to_lineid=m_locationid_to_lineid,
        m_lineid_to_vertexid=m_lineid_to_vertexid,
        m_desc=m_desc,
        config=config,
    )
    return ctx



def guidepost_query(ctx, req):
    #*****************临时记录**************
    candidate_ls = context['g'].neighbors(1, mode='out')
    item = random.choice(candidate_ls)
    origin = [116.34762, 39.97692]
    destination = [116.34762, 39.98692]
    ols = get_location_ls(context, origin)
    dls = get_location_ls(context, destination)
    req = dict(
        origin=origin,
        destination=destination,
        ols=ols,
        dls=dls,
        history=[1],
        candidate_ls=candidate_ls,
        item=item
    )
    while True:
        rsp = guidepost_query_one_step(ctx, req)
        if True:
            break
        # TODO: set a new req
    #*****************临时记录**************
    rsp = dict(
        history = history
    )
    return rsp

def guidepost_query_one_step(ctx, req):
    """一步搜索"""
    model = ctx['model']
    origin = req['origin']
    destination = req['destination']
    history = req['history']
    if not history:
        candidate_ls = [1,2,3] #TODO
    else:
        current = history[-1]
        candidate_ls = context['g'].neighbors(current, mode='out')
    feature_rsp = extract_feature(context, req)
    feature = feature_rsp['feature']
    item_feature = None #TODO
    preds = model(feature, item_feature)
    i = torch.argmax(preds)
    selected = candidate_ls[i]
    rsp = dict()
    return rsp

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
    """将数组变为指定形状
    与reshape有本质区别 采用的是裁剪和0补全的方法
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

def remove_zero_location(arr):
    """在构建线路类特征时 我们用0来补位 而plot时这些0占位需要去掉
    """
    return [_ for _ in arr if _[0]>0]

def draw_lines(context, *vertex_ids):
    to_draw_list = []
    for vertex_id in vertex_ids:
        xs, ys = zip(*remove_zero_location(context['g'].vs[vertex_id]['line_feature']))
        to_draw_list.append(xs)
        to_draw_list.append(ys)
    plt.plot(*to_draw_list)

def draw_all_neighbor_lines(context, vertex_id):
    draw_lines(context, *context['g'].neighbors(vertex_id, mode='out'))

def split_situation_feature(ctx, feat):
    """将situation_feature按照列分割为所设计的几类 history/lvp/candidate
    """
    config = ctx.get("config")
    k_history = config.get("k_history")
    k_lvp = config.get("k_lvp")
    k_candidate = config.get("k_candidate")
    
    ks = [k_history, k_lvp, k_candidate]
    feat_list = []
    n = 0
    for k in ks:
        part_feat = feat[..., n:n+k]
        n+=k
        feat_list.append(part_feat)
    return feat_list

def draw_guidepost_feat(feat):
    """将guidepost的线路类特征画出来
    >>> feature_history, feature_lvp, feature_candidate = split_situation_feature(context, feat)
    >>> draw_guidepost_feat(feat_candidate)
    """
    k1 = feat.shape[0]
    k2 = feat.shape[1]
    l_feature_list = torch.tensor(feat).reshape(k1, k2//2, 2).permute(1, 0, 2).tolist()
    draw_arg_list = []
    for l_feature in l_feature_list:
        if torch.tensor(l_feature).count_nonzero() == 0:
            continue
        xs, ys = zip(*l_feature)
        xs = [_ for _ in xs if _>0]
        ys = [_ for _ in ys if _>0]
        if xs and ys:
            draw_arg_list.extend([xs, ys])
    plt.plot(*draw_arg_list)
def feature_history_to_transit_routeplan(feature_history):
    """feature_history是指 dst-l3-l2-l1-org 这样顺序的 k_history * k1维度的二维数组
    不足的部分补零
    基于feature_history结合df_sid构成的静态数据结构 返回wind接受的transit_routeplan格式数据
    这一步有两个用途
    一是guidepost_model推断后 转变为可以返回的结果
    二是用于训练过程中计算weight 作为规则判别器的依赖数据
    """
    pass

class GuidepostModel(torch.nn.Module):
    """用于判断线路可选概率的模型
    """
    def __init__(self, context):
        super(GuidepostModel, self).__init__()
        config = context['config']
        k1 = config['k1']
        k_lvp = config['k_lvp']
        k_candidate = config['k_candidate']
        k_history = config['k_history']
        self.k1 = k1
        self.k2 = k_history + k_lvp + k_candidate
        self.flatten = torch.nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = torch.nn.Linear((self.k2+2)*self.k1, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 1)
        self.sigmoid = torch.nn.Sigmoid()

    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
