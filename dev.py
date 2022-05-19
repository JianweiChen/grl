## Write only functions, not classes if necessary
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
import dill
import ipyvuetify as ipyv
import ipyleaflet
import traitlets
import redis


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
    df_desc = df_stop[['stopid', 'lineid', 'desc', 'lng', 'lat', 'sequenceno']] \
        .groupby('stopid').first() \
        .reset_index() \
        .assign_by("lineid", lid=df_lineid2lid.lid) \
        [['lid', 'lng', 'lat', 'desc', 'lineid', 'stopid', 'sequenceno']] \
        .rename(dict(sequenceno='seq'), axis=1)
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
    df = pd.read_csv(path).set_index(['lid', 'seq']) \
        [['desc', 'stopid', 'lineid', 'lng', 'lat']] \
        .assign(stopname=lambda xdf: xdf.desc.str.split('_').apply(lambda x: x[1])) \
        .assign(linename=lambda xdf: xdf.desc.str.split('_').apply(lambda x: x[0])) \
        .assign(coordinate=lambda xdf: xdf.lng.astype('str')+','+xdf.lat.astype('str'))
    return df

## use kdtree
from scipy.spatial import KDTree

def get_coordinate_by_city_name(df_city, city_name):
    df_city = df_city.set_index("city_name")
    if city_name not in df_city.index:
        return (0.0, 0.0)
    sr = df_city.loc[city_name]
    return (sr.center_lng, sr.center_lat)

def get_neighbor_within(kdtree, kdis, name, df_city):
    coordinate = get_coordinate_by_city_name(df_city, name)
    index_list = kdtree.query_ball_point(coordinate, r=kdis/110)
    df = df_city.loc[index_list] \
        .assign(query_lng=coordinate[0]) \
        .assign(query_lat=coordinate[1])
    gps_array = df[['center_lng', 'center_lat', 'query_lng', 'query_lat']].to_numpy()
    xy_array = np_coords(gps_array)
    theta = np.degrees(np.arctan2(xy_array[..., 1], xy_array[..., 0]))
    ro = np.linalg.norm(xy_array, axis=1)
    df['ro'] = ro
    
    theta += 360.0 * (theta<0).astype('float32')
    theta = 90.0 - theta
    clock = theta / 30
    clock += 12.0 * (clock<0).astype('float32')
    df['clock'] = clock
    df.pop('query_lng')
    df.pop('query_lat')
    df.pop("city_level")
    df.pop("car_prefix")
    df = df.set_index("city_id") 
    df = df.sort_values("clock") \
        .reset_index()
    return df

def load_df_sid(path):
    df = pd.read_csv(path).set_index("sid")
    df2 = df.query("gate>0").apply(axis=1, func=fill_empty_gc_for_gate_sid).rename("gc").to_frame()
    df.update(df2)
    return df

def make_gps_vertex_name_col(xdf):
    return xdf.gate.apply(lambda x: ['n', 'p'][x]) + '_' \
        +xdf.lng.astype(str)+'_' \
        +xdf.lat.astype(str)+'_' \
        +xdf.gate.astype(str)
def make_linestop_vertex_name_col(xdf):
    return 's_'+xdf.lid.astype(str)+'_'+xdf.seq.astype(str)+'_'+xdf.gate.astype(str)
def make_gc_gps_vertex_name_col(xdf):
    def make_gc_gps_vertex_name_col_by_row(row):
        return [f'n_{_[0]}_{_[1]}_1' for _ in np.array(eval(row.gc)).reshape((-1, 2)).tolist()]
    return xdf.apply(axis=1, func=make_gc_gps_vertex_name_col_by_row)

def get_edge_xx(df_x):
    df_x = df_x.reset_index(drop=True)
    kdtree = KDTree(df_x[['lng', 'lat']].to_numpy())

    neighbor_ids_column = kdtree.query_ball_point(df_x[['lng', 'lat']].to_numpy(), 0.8/110)

    df_edge_xx = df_x \
        .rename(dict(vertex_name='src'), axis=1) \
        [['src']] \
        .assign(tgt_id=neighbor_ids_column) \
        .explode('tgt_id') \
        .astype(dict(tgt_id='int')) \
        .set_index("tgt_id") \
        .assign(tgt=df_x.vertex_name) \
        .reset_index() \
        [['src', 'tgt']]
    return df_edge_xx
def compile_sid(df_sid):
    """对df_sid进行编译
    """
    df_l = df_sid.assign(vertex_name=make_linestop_vertex_name_col) \
        .reset_index()
    df_l_vertex = df_l[['vertex_name', 'lng', 'lat']] \
        .assign(nodetype='s')
    df_gps = df_l.assign(gps_vertex_name=make_gps_vertex_name_col)
    df_gps_with_gate = df_gps.query("gate>0").assign(gc=make_gc_gps_vertex_name_col)
    df_gps_without_gate = df_gps.query("gate==0")

    df_n1_vertex = df_gps_without_gate.groupby("gps_vertex_name").agg(lng=('lng', 'first'), lat=('lat', 'first')) \
        .rename_axis("vertex_name").reset_index() \
        [['vertex_name', 'lng', 'lat']]

    df_n2_vertex = df_gps_with_gate.gc.explode().to_frame() \
        .assign(
            lng=lambda xdf: xdf.gc.str.split("_").apply(lambda x: x[1]).astype(float),
            lat=lambda xdf: xdf.gc.str.split("_").apply(lambda x: x[2]).astype(float),) \
        .rename(dict(gc='vertex_name'), axis=1)

    df_n_vertex = pd.concat([df_n1_vertex, df_n2_vertex], axis=0).reset_index(drop=True) \
        .assign(nodetype='n') \
        .groupby("vertex_name", as_index=False).first()


    df_p_vertex = df_gps_with_gate \
        [["gps_vertex_name", 'lng', 'lat']] \
        .rename(dict(gps_vertex_name='vertex_name'), axis=1) \
        .assign(nodetype='p') \
        .groupby("vertex_name", as_index=False).first()


    df_vertex = pd.concat([
        df_l_vertex, df_n_vertex, df_p_vertex,
    ], axis=0).reset_index(drop=True)

    df_edge_ll = df_l[['vertex_name', 'lid']].rename(dict(vertex_name='src', lid='srclid'), axis=1)
    df_edge_ll2 = df_edge_ll.rename(dict(src='tgt', srclid='tgtlid'), axis=1) \
        .shift(-1) \
        .fillna(0) \
        .astype(dict(tgtlid='int')) \
        .reset_index(drop=True)
    df_edge_ll = df_edge_ll.assign(tgt=df_edge_ll2.tgt, tgtlid=df_edge_ll2.tgtlid) \
        .query("srclid==tgtlid") \
        [['src', 'tgt']] \
        .assign(edgetype='ll')

    df_edge_lp = df_gps_with_gate[['vertex_name', 'gps_vertex_name']] \
        .rename(dict(vertex_name='src', gps_vertex_name='tgt'), axis=1) \
        .assign(edgetype='lp')
    df_edge_pl = df_edge_lp.rename(dict(src='tgt', tgt='src'), axis=1).assign(edgetype='pl')

    df_edge_ln = df_gps_without_gate[['vertex_name', 'gps_vertex_name']] \
        .rename(dict(vertex_name='src', gps_vertex_name='tgt'), axis=1) \
        .assign(edgetype='ln')
    df_edge_nl = df_edge_ln.rename(dict(src='tgt', tgt='src'), axis=1).assign(edgetype='nl')

    df_edge_pn = df_gps_with_gate[['gps_vertex_name', 'gc']].rename(dict(gps_vertex_name='src', gc='tgt'), axis=1) \
        .assign(edgetype='pn') \
        .explode('tgt')
    df_edge_np = df_edge_pn.rename(dict(src='tgt', tgt='src'), axis=1).assign(edgetype='np')


    df_edge_nn = get_edge_xx(df_n_vertex).assign(edgetype='nn')
    df_edge_pp = get_edge_xx(df_p_vertex).assign(edgetype='pp')

    df_edge = pd.concat([
        df_edge_ll,
        df_edge_np,
        df_edge_pn,
        df_edge_lp,
        df_edge_pl,
        df_edge_ln,
        df_edge_nl,
        df_edge_nn,
        df_edge_pp,
    ], axis=0) \
    .reset_index(drop=True) \
    .groupby(['src', 'tgt'], as_index=False) \
    .agg(edgetype=('edgetype', 'first'))

    df_edge['edgetype'] = df_edge['edgetype'].str.replace("l", 's')

    return df_vertex, df_edge

def run_compile_busgraph():
    df_sid = load_df_sid(data_path_("sid.csv"))
    df_vertex, df_edge = compile_sid(df_sid)
    df_vertex.to_csv(data_path_("busgraph_vertex.csv"), index=None)
    df_edge.to_csv(data_path_("busgraph_edge.csv"), index=None)
def load_busgraph_vertex():
    df = pd.read_csv(data_path_("busgraph_vertex.csv"))
    return df
def load_busgraph_edge():
    df = pd.read_csv(data_path_("busgraph_edge.csv"))
    return df
def get_dfv_near(dfv_all, coordinate, a):
    d = a / 110
    dfv_all = dfv_all.reset_index(drop=True)
    kdtree = KDTree(dfv_all[['lng', 'lat']].to_numpy())
    vids = kdtree.query_ball_point(coordinate, d)
    dfv_selected = dfv_all.loc[vids].sort_values(['lng', 'lat']).reset_index(drop=True)
    return dfv_selected

def get_v_and_e_dataframe_near(dfv, dfe, coordinate, a):
    dfv_selected = get_dfv_near(dfv, coordinate, a)
    # dfv_nodetype = dfv_selected.set_index("vertex_name")['nodetype']
    # tc = time.time()
    # print(tc)
    # dfv_index = dfv_selected.set_index("vertex_name").index
    dfv_index = dfv_selected['vertex_name'].tolist()
    source_index_list = dfe.reset_index().set_index("src").loc[dfv_index]['index'].tolist()
    target_index_list = dfe.reset_index().set_index("tgt").loc[dfv_index]['index'].tolist()
    index_list = list(set(source_index_list+target_index_list))
    dfe_selected = dfe.loc[index_list]
    # print(time.time()-tc)
    # dfe_selected = dfe \
    #     .set_index("src").join(dfv_nodetype) \
    #     .rename(dict(nodetype='src_nodetype'), axis=1) \
    #     .rename_axis('src') \
    #     .reset_index() \
    #     .set_index('tgt').join(dfv_nodetype) \
    #     .rename(dict(nodetype='tgt_nodetype'), axis=1) \
    #     .rename_axis('tgt') \
    #     .reset_index() \
    #     .query("src_nodetype==src_nodetype or tgt_nodetype==tgt_nodetype") \
    #     [['src', 'tgt', 'edgetype']]
    dfv_selected = dfv.set_index("vertex_name") \
        .loc[list(set(dfe_selected.src.tolist() + dfe_selected.tgt.tolist()))] \
        .reset_index()
    return dfv_selected, dfe_selected
def add_weight_default_column(dfe):
    dfk = pd.DataFrame([
        ('ss', 0, 1, 55),
        ('nn', 0, 1, 4.5),
        ('np', 1.5/60, 1, 5.5),
        ('pn', 1.5/60, 1, 5.5),
        ('pp', 0, 1, 5.5),
        ('sn', 0, 0, 1),
        ('ns', 6/60, 0, 1),
        ('sp', 0, 0, 1),
        ('ps', 3/60, 0, 1),
    ], columns=['edgetype', 'k0', 'k1', 'k2']) \
    .set_index("edgetype")
    dfe = dfe \
        .assign_by('edgetype', _k0=dfk.k0, _k1=dfk.k1, _k2=dfk.k2) \
        .assign(weight0=lambda xdf: xdf._k0+xdf.mht*xdf._k1 / xdf._k2)
    dfe = dfe[['src', 'tgt', 'edgetype', 'mht', 'weight0']].reset_index(drop=True)
    return dfe
def get_g_near(dfv, dfe, coordinate, a):
    """
    >>> g = get_g_near(dfv, dfe, [123.08778,41.21388], 5)
    """
    dfv_selected, dfe_selected = get_v_and_e_dataframe_near(dfv, dfe, coordinate, a)
    v2lng = dfv_selected.set_index("vertex_name").lng
    v2lat = dfv_selected.set_index("vertex_name").lat

    arr = dfe_selected \
    .assign_by("src", srclng=v2lng, srclat=v2lat) \
    .assign_by("tgt", tgtlng=v2lng, tgtlat=v2lat) \
    [['srclng', 'srclat', 'tgtlng', 'tgtlat']] \
    .to_numpy()
    dfe_selected = dfe_selected.assign(mht=np_mht(arr))
    dfe_selected = add_weight_default_column(dfe_selected)
    g = ig.Graph.DataFrame(dfe_selected, directed=True, vertices=dfv_selected)
    return g

def get_busgraph_go_trace(arr, nodetype_column, name, color='blue'):
    #TODO
    assert arr.shape[1] in (2, 4)
    mode = 'markers' if arr.shape[1]==2 else 'line'
    nodetype2z = dict(
        n=0.0, p=1.0, s=2.0
    )
    nodetype2z = dict()
    trace = go.Scatter3d(
        x=arr[..., 0],
        y=arr[..., 1],
        z=nodetype_column.apply(lambda x: nodetype2z.get(x, 0.0)).tolist(),
        mode=mode,
        hoverinfo=None,
        marker=dict(
            size=5.,
            opacity=0.5,
            color=color
        )
    )
    return trace

def query_stopname(df_desc, pattern):
    """
    >>> query_stopname(df_desc, "上海")
    """
    return df_desc.query(f"stopname.str.contains('{pattern}')").sort_values("desc")
class Vizmap(traitlets.HasTraits):
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

def save_desc_to_redis():
    from tqdm import tqdm
    import redis
    client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
#     df_desc = load_desc_df(data_path_("desc.csv"))
    df = df_desc \
        .reset_index() \
        .assign(key=lambda xdf: xdf.lid.astype('str')+'_'+xdf.seq.astype('str')) \
        .sort_values(['lid','seq']) \
        [['key', 'desc']] \
        .reset_index(drop=True)
    i = 0
    B = 1024
    for i in tqdm(range(df.shape[0]//B+10)):
        _df = df.iloc[i*B: (i+1)*B]
        if _df.empty:
            break
        mapping = dict(_df.to_numpy().tolist())
        client.hset("desc", mapping=mapping)
    
def get_desc_from_bin(b):
    if not b:
        return 'x_x'
    else:
        return b.decode()

def get_g_vertex_gps_array_by(g, center):
    gps_arr = g.get_vertex_dataframe()[['lng', 'lat']].to_numpy()
    return np.concatenate([gps_arr, np.broadcast_to(center, gps_arr.shape)], axis=1)

def get_g_edge_gps_arr(g):
    df_coor = g.get_vertex_dataframe()[['lng', 'lat']]
    gps_arr = g.get_edge_dataframe() \
        .assign_by('source', src_lng=df_coor.lng, src_lat=df_coor.lat) \
        .assign_by("target", tgt_lng=df_coor.lng, tgt_lat=df_coor.lat) \
        [['src_lng', 'src_lat', 'tgt_lng', 'tgt_lat']] \
        .to_numpy()
    return gps_arr