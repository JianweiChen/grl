

import pathlib
import pandas as pd
import dill
import igraph as ig
import logging
from .np_util import *

def load_gcg_node():
    return pd.read_csv(data_path_("gcg_node.csv"))[['nodeid', 'nodename', 'lng', 'lat', 'nodetype']]

def load_gcg_edge():
    return pd.read_csv(data_path_("gcg_edge.csv"))[['src', 'tgt', 'edgetype']]

def load_gcg_city():
    return pd.read_csv(data_path_("gcg_city.csv"))

def make_vertex_map(v):
    nodename = v['nodename']
    nodetype = v['nodetype']
    parts = nodename.split("_")
    stopname, citynames, linename, subname, gpstype, stopid = None, None, None, None, None, None
    if nodetype == 'L':
        stopname, linename, citynames, stopid = parts
        stopid = int(stopid)
    else:
        stopname, subname, citynames, _, _ , gpstype= parts
        gpstype = int(gpstype)
    citynames = citynames.split("|")
    citycount = len(citynames)
    return dict(
        stopname=stopname, linename=linename, subname=subname, citynames=citynames,
        stopid=stopid, gpstype=gpstype, citycount=citycount
    )

class Gcg(object):
    """根据原始节点数据框与边数据框生成供查询的图数据
    阅读该代码要求阅读者对pandas比较熟练 否则会有阅读障碍
    功能：
    1. 最短路径搜索
    2. 节点查找
    3. 邻边查找
    4. 最近邻搜索
    """
    def __init__(self, df_node=pd.DataFrame(), df_edge=pd.DataFrame(), g=ig.Graph()):
        by_node_edge = not (df_node.empty or df_edge.empty)
        by_g = g.vcount() > 0
        assert (by_g and not by_node_edge) or (by_node_edge and not by_g), "by_node_edge is True or by_g is True!!!"
        if by_node_edge:
            # 消去df_edge中重复的部分 这是因为生成gcg_edge数据框时一个bug导致的 之后会修
            if True:
                df_edge = df_edge.groupby(['src', 'tgt'], as_index=False).first()
            # 检查df_edge中是否包括df_node没有的点（反过来没有必要检查）
            diffset = set(df_edge.src.tolist()) & set(df_edge.tgt.tolist()) - set(df_node.nodeid.tolist())
            assert not diffset, "check nodeid in df_edge which not exists in df_node"
            n2lng = df_node.set_index("nodeid").lng
            n2lat = df_node.set_index("nodeid").lat
            gps_arr = df_edge \
                .assign_by("src", src_lng=n2lng, src_lat=n2lat) \
                .assign_by('tgt', tgt_lng=n2lng, tgt_lat=n2lat) \
                [['src_lng', 'src_lat', 'tgt_lng', 'tgt_lat']].to_numpy()
            df_edge['mht'] = np_mht(gps_arr)
            
            df_edgetype_k = pd.DataFrame([
                ("Ll", 0.0, 1.0, 45.0),
                ("Ln", 0, 0.0, 1.0),
                ("Nl", 10/60.0, 0.0, 1.0),
                ("Lp", 0, 0.0, 1.0),
                ("Pl", 10/60.0, 0.0, 1.0),
                ("Pp", 0, 1.0, 5.5),
                ("Pn", 5/60.0, 1.0, 5.5),
                ("Np", 5/60.0, 1.0, 5.5),
                ("Nn", 0, 1.0, 4.5),
            ], columns=['edgetype', 'k0', 'k1', 'k2']).set_index("edgetype")
            df_k = df_edge \
                .assign_by("edgetype", **{kn: df_edgetype_k[kn] for kn in ['k0', 'k1', 'k2']}) \
                [['mht', 'k0', 'k1', 'k2']]
            df_edge['weight_default'] = df_k.k0 + df_k.k1 * df_k.mht / df_k.k2
            self.g = ig.Graph.DataFrame(df_edge, directed=True, vertices=df_node)
        else:
            self.g = g
        
        self.init_from_g(self.g)
    
    def init_from_g(self, g):
        self.g = g
        self.df_vertex = self.g.get_vertex_dataframe()
        self.df_edge = self.g.get_edge_dataframe()
        self.df_gps = self.df_vertex.groupby("nodetype").get_group("N").reset_index()
        from scipy.spatial import KDTree
        gps_array = self.df_gps[['lng', 'lat']].to_numpy()
        self.kdtree = KDTree(gps_array)
        self.set_vertex_attr()
        self.set_edge_attr()
        self.nvids = self.g.vs.select(nodetype='N').indices
    
    def get_random_vertex_id(self):
        vid = random.randint(0, self.df_vertex.shape[0])
        return vid        

    def set_vertex_attr(gcg):
        gcg.g.vs['_m'] = [make_vertex_map(v) for v in gcg.g.vs]
        for key in gcg.g.vs[0]['_m'].keys():
            gcg.g.vs[key] = [v['_m'][key] for v in gcg.g.vs]
        del gcg.g.vs['_m']
    def set_edge_attr(gcg):
        gcg.g.es['weight'] = gcg.g.es['weight_default']

    def epath_to_feat(gcg, epath):

        edgetypes = gcg.g.es.select(epath)['edgetype']
        weights = gcg.g.es.select(epath)['weight']
        mhts = gcg.g.es.select(epath)['mht']
        df = pd.DataFrame(dict(edgetype=edgetypes, mht=mhts, weight=weights, c=1))
        df = df.groupby("edgetype").sum()
        edgetype_list = [
            'Ll', 'Ln', 'Lp', 'Nl', 'Nn', 'Np', 'Pl', 'Pn', 'Pp'
        ]
        df = df.reindex(edgetype_list).fillna(0.0)
        sum_row = df.sum().tolist()
        df.loc['Ss'] = sum_row
        feat = df.to_numpy().T.flatten()
        return feat

    def generate_action_instance(self, vid, to):
        to = self.g.vs.select(to).select(nodetype='N').indices
        v = self.g.vs[vid]
        start_gps = [v['lng'], v['lat']]
        start_gps = np.broadcast_to(start_gps, (len(to), 2))
        tov = self.g.vs.select(to)
        end_gps = np.stack([tov['lng'], tov['lat']]).T
        gps = np.concatenate([start_gps, end_gps], axis=1)

        dis = np.stack([np_distance(gps), np_mht(gps)]).T
        gentype = np.zeros_like(dis)[..., :1]

        paths = self.g.get_shortest_paths(vid, to, weights='weight', output='epath')
        feat = np.stack([self.epath_to_feat(_) for _ in paths], axis=0)
        feat = np.concatenate([gps, dis, feat, gentype], axis=1)
        return feat

    def get_pmconv_dots(gcg, org, dst, **kwargv):
        """这是伪方法 只是拿到一条可用的路线 真正的pmconv待开发
        """
        ori_vid = gcg.df_gps.loc[gcg.kdtree.query(org)[1]][0]
        dst_vid = gcg.df_gps.loc[gcg.kdtree.query(dst)[1]][0]
        paths = gcg.g.get_shortest_paths(ori_vid, dst_vid, weights='weight')
        path = paths[0]
        vs = gcg.g.vs.select(path)
        return list(zip(vs['lng'], vs['lat'], vs['stopname'], vs['nodetype']))
        
class GcgDataLoader(object):
    def __init__(self, df_node=pd.DataFrame(), df_edge=pd.DataFrame(), df_city=pd.DataFrame()):
        self.df_node = df_node
        self.df_edge = df_edge
        self.df_city = df_city
        self.inited = False
    def init(self):
        logging.warning("gcg_data_loader init doing")
        if self.df_node.empty:
            self.df_node = load_gcg_node().set_index("nodeid")
        if self.df_edge.empty:
            self.df_edge = load_gcg_edge()[['src', 'tgt', 'edgetype']]
        if self.df_city.empty:
            self.df_city = load_gcg_city().set_index("city_id")
        self.inited = True
        logging.warning("gcg_data_loader init done")
    def init_from(self, gcg_data_loader):
        self.df_node = gcg_data_loader.df_node
        self.df_edge = gcg_data_loader.df_edge
        self.df_city = gcg_data_loader.df_city
        self.inited = True
    def display(self):
        display(self.df_node)
        display(self.df_edge)
        display(self.df_city)
    def _get_cityids_around_one_city(self, cityid, contain_center=1):
        df = self.df_city.query(f"city_id=={cityid}")
        if df.empty:
            neighbor_cityids = []
        else:
            neighbor_cityids = df.iloc[0].neighbor.split(',')
            neighbor_cityids = [int(_) for _ in neighbor_cityids]
        if contain_center:
            neighbor_cityids = [cityid, *neighbor_cityids]
        return neighbor_cityids
    
    def _get_nodeids_by_cityids(self, cityids):
        nodeids = self.df_city.loc[cityids].nodeids.str.split(',') \
            .explode() \
            .apply(int) \
            .rename('nodeid') \
            .to_frame() \
            .groupby("nodeid") \
            .size() \
            .index \
            .tolist()
        return nodeids
    
    def _get_df_node(self, cityids):
        nodeids = self._get_nodeids_by_cityids(cityids)
        df = self.df_node.loc[nodeids].reset_index()
        return df

    def _get_df_edge(self, cityids):
        df_node = self._get_df_node(cityids).set_index("nodeid")
        df = self.df_edge \
            .assign_by("src", src_nodename=df_node.nodename) \
            .assign_by("tgt", tgt_nodename=df_node.nodename) \
            .query("src_nodename==src_nodename") \
            .query("tgt_nodename==tgt_nodename") \
            [['src', 'tgt', 'edgetype']]
        return df

    def get_gcg(self, cityid, neighbor=0):
        if not self.inited:
            self.init()
        cityids = [cityid] if neighbor==0 else self._get_cityids_around_one_city(cityid)
        df_node = self._get_df_node(cityids)
        df_edge = self._get_df_edge(cityids)
        gcg = Gcg(df_node, df_edge)
        # g = ig.Graph.DataFrame(df_edge, directed=True, vertices=df_node)
        return gcg

    def show_city_node_count(self, cityid, neighbor=0):
        if neighbor:
            cityids = gcg_data_loader._get_cityids_around_one_city(cityid)
        else:
            cityids = [cityid]
        s = gcg_data_loader._get_df_node(cityids) \
        .assign(cityname=lambda xdf: xdf.nodename.str.split("_").apply(take_2)) \
        .groupby("cityname").size().sort_values(ascending=False).to_string()
        print(s)
    
    def save_gcg(self, cityid, neighbor=0):
        gcg = self.get_gcg(cityid, neighbor)
        dill.dump(gcg, pathlib.Path(data_path_(f"gcg_{cityid}_{neighbor}.dill")).open("wb"))

    
    def get_global_gcg(self):
        if not self.inited:
            self.init()
        gcg = Gcg(self.df_node.reset_index(), self.df_edge)
        return gcg
    
    def load_gcg_from_path(self, path):
        gcg = dill.load(pathlib.Path(data_path_(path)).open("rb"))
        gcg.init_from_g(gcg.g)
        return gcg
    def load_gcg(self, cityid, neighbor=0):
        gcg = dill.load(pathlib.Path(data_path_(f"gcg_{cityid}_{neighbor}.dill")).open("rb"))
        gcg.init_from_g(gcg.g)
        return gcg