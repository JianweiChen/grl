
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
import numpy as np
import random
import itertools
import networkx as nx
import collections
import pypinyin
import plotly.graph_objects as go


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
import pandas as pd
import time

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

"""可以考虑将公交图分割成若干块
只有将这种管理上的连续化做好 事情才能更容易的做下去
需要cover住所有不确定的情况 step by step
两个点之间的各种距离 都可以使用四种特征来确定

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
schedule_type: detail/pattern 详情类型与模式类别
tripcount: 0
entrances: Akou#12314213

经纬度这个东西实际是均匀的
"""

def to_pinyin(s):
    rs = pypinyin.pinyin(s, style=0)
    rs = [_[0] for _ in rs]
    r = '_'.join(rs)
    return r

class IdManager(object):
    def __init__(self):
        self.mapping = dict()
    def get_id(self, category, input_id):
        if category not in self.mapping:
            self.mapping[category] = IdGenerator(category)
        output_id = self.mapping[category].get_id(input_id)
        return output_id
    def dump(self, path):
        dill.dump(self, pathlib.Path(path).open("wb"))

class IdGenerator(object):
    """为什么要将generator自增
    """
    def __init__(self, category):
        self.category = category
        self.mapping = dict()
    def get_id(self, input_id):
        if input_id not in self.mapping:
            self.mapping[input_id] = self.mapping.__len__()+1
        return self.mapping[input_id]

def join_desc_id(desc, id):
    assert isinstance(desc, str)
    assert isinstance(id, int) and id>0
    desc = desc.replace('#', '')
    return f"{desc}#{id}"

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

'''
from here we make three period of busgraph define
'''
def parse_line_group_info(line_group_info):
    line_group_info = line_group_info.replace(' ', '')
    parts = line_group_info.split('&')
    assert parts.__len__() >= 1
    line_group_name = parts[0]
    parts = parts[1:]
    line_group_attr = dict()
    for part in parts:
        part = part.replace(' ', '')
        if '=' not in part:
            line_group_attr[part] = True
        else:
            key, val = part.split('=', 1)
            line_group_attr[key] = val
    if 'stop_count' in line_group_attr:
        stop_count = line_group_attr.get("stop_count")
        stop_count = pd.to_numeric(stop_count)
        if pd.isnull(stop_count):
            stop_count = 10
        line_group_attr['stop_count'] = stop_count
    else:
        line_group_attr['stop_count'] = 10
    line_group_attr['name'] = line_group_name
    return line_group_attr

def generate_stop_info_list(line_group_infos, stop_name_pool, config):
    line_group_cross_pct = config.get("line_group_cross_pct", 0.1)
    stop_count_max_room = 0
    line_group_attr_list = []
    for line_group_info in line_group_infos:
        line_group_attr = parse_line_group_info(line_group_info)
        line_group_attr_list.append(line_group_attr)
        stop_count_max_room += line_group_attr.get("stop_count")
    stop_name_pool = list(set(stop_name_pool))
    assert len(stop_name_pool) >= stop_count_max_room
    stop_name_pool = stop_name_pool[:stop_count_max_room]
    line_group_count = line_group_attr_list.__len__()
    g_l = ig.Graph.Erdos_Renyi(line_group_count, line_group_cross_pct)
    g_l.vs['stop_count'] = [_['stop_count'] for _ in line_group_attr_list]
    g_l.vs['line_group'] = [_['name'] for _ in line_group_attr_list]
    g_l.vs['platform'] = [_.get("platform", False) for _ in line_group_attr_list]
    g_l.vs['loop'] = [_.get("loop", False) for _ in line_group_attr_list]
    g_l.vs['oneway'] = [_.get("oneway", False) for _ in line_group_attr_list]
    components = g_l.components()
    leaders = [_[0] for _ in components]
    if leaders.__len__() > 1:
        edges = list(zip(leaders, [*leaders[1:], leaders[0]]))
        g_l.add_edges(edges)
    
    stopkey2vid = dict()
    g_s = ig.Graph(directed=True)
    for v in g_l.vs:
        stop_count = v['stop_count']
        for seq in range(stop_count):
            key = (v.index, 0, seq)
            v_stop = g_s.add_vertex(key)
            
            stopkey2vid[key] = v_stop.index
            
            if seq>0:
                v_last = g_s.vs[stopkey2vid[(v.index, 0, seq-1)]]
                
                g_s.add_edge(v_last.index, v_stop.index)

    cliques = g_l.cliques(min=2)
    cliques = sorted(cliques, key=lambda x: x.__len__() , reverse=True)
    for clique in cliques:
        stop_counts = g_l.vs[clique]['stop_count']
        keys = []
        for i in range(clique.__len__()):
            stop_count = stop_counts[i]
            seq = random.randint(0, stop_count-1)
            
            keys.extend([
                (clique[i], 0, seq),
            ])
        for key1, key2 in itertools.product(keys, keys):
            if key1[0] != key2[0]:
                g_s.add_edge(stopkey2vid[key1], stopkey2vid[key2])
    
    return g_s

@monkey(pd.DataFrame, 'assign_busgraph_key')
def _assign_busgraph_key(df):
    df = df.assign_join_col(key=('lid', 'lng', 'lat'))
    return df
@monkey(pd.DataFrame, 'assign_busgraph_gc_count')
def _assign_busgraph_gc_count(df):
    df['gc_count'] = df['gc'].apply(lambda x: eval(x).__len__() // 2)
    return df

def make_desc_test_version():
    stop_path = data_path_("history/geo_stop_all_country.csv")
    df_stop = pd.read_csv(stop_path)
    df_lineid2lid = pd.read_csv(data_path_("lineid2lid.csv")).set_index("lineid")

    df_stop['desc'] = df_stop['linename']+'_'+df_stop['stopname']
    df_stop['stopid'] = df_stop.stopid.apply(lambda x: int(str(x)[:-4]))
    df_stop['lineid'] = df_stop.lineid.apply(lambda x: int(str(x)[:-4]))
    K = 100_000
    df_stop['lng'] = df_stop.lng.apply(lambda x: float(int(x*K)/K))
    df_stop['lat'] = df_stop.lat.apply(lambda x: float(int(x*K)/K))
    df_desc = df_stop[['stopid', 'lineid', 'desc', 'lng', 'lat']] \
        .groupby('stopid').first() \
        .reset_index() \
        .assign_by("lineid", lid=df_lineid2lid.lid) \
        [['lid', 'lng', 'lat', 'desc', 'lineid', 'stopid']]
    df_desc.to_csv(data_path_("desc.csv"), index=False)

def df_sid_join_gc_test_version(df_sid):
    df_desc = pd.read_csv(data_path_("desc.csv")).set_index("stopid")
    K = 100_000
    df_gate = pd.read_csv(data_path_("geo_stexit.csv")) \
        .assign(stopid=lambda xdf: xdf.stopid.apply(lambda x: int(str(x)[:-4]))) \
        .assign(lng=lambda xdf: xdf.lng.apply(lambda x: float(int(x*K)/K))) \
        .assign(lat=lambda xdf: xdf.lat.apply(lambda x: float(int(x*K)/K))) \
        .assign(gate_coordinate=lambda xdf: xdf.apply(axis=1, func=lambda row: (row.lng, row.lat))) \
        .groupby("stopid") \
        .agg(
            gate_coordinates=('gate_coordinate', 'unique')
        ) \
        .assign(gate_coordinates=lambda xdf: xdf.gate_coordinates \
                .apply(lambda x: str(np.array([list(_) for _ in x]).flatten().tolist())))
    df_gate = df_desc \
        .assign_busgraph_key() \
        .assign(gate_coordinates=df_gate.gate_coordinates) \
        .query("gate_coordinates==gate_coordinates") \
        .set_index("lineid") \
        .assign(lid=df_lineid2lid.lid) \
        .set_index("key")
    df_sid = df_sid \
        .reset_index() \
        .assign_join_col(key=('lid', 'lng', 'lat')) \
        .set_index('key') \
        .assign(gc=df_gate.gate_coordinates) \
        .fillna('[]') \
        .reset_index() \
        .set_index("sid") \
        [['lid', 'seq', 'onserve', 'lng', 'lat', 'gc']]
    return df_sid

def make_busgraph_df_sid_with_gate_info_test_version():
    df_sid = pd.read_csv(data_path_('sid.csv'))
    df_sid = df_sid_join_gc_test_version(df_sid)
    df_sid = df_sid.assign_by('lid', gate=df_lid.gate)
    df_sid.to_csv(data_path_('sid.csv'))

@monkey(pd.DataFrame, 'assign_gps')
def _assign_gps(df):
    df = df.assign_join_col(gps=('lng', 'lat'))
    return df

#### now, start to compile busgraph igraph

def fill_empty_gc_for_gate_sid(row):
    gc = row.gc
    gate = row.gate
    if eval(gc).__len__() == 0 and gate>0:
        gc = str([row.lng, row.lat])
    return gc

