from grl import *
import time
import pandas as pd

@monkey(pd.DataFrame, "assign_gps")
def _pandas_assign_gps(df):
    df = df \
        .assign(_lng=df.lng.apply(lambda x: round(x, 5))) \
        .assign(_lat=df.lat.apply(lambda x: round(x, 5))) \
        .assign_join_col(gps=('_lng', '_lat', 'gpstype')) \
        .drop('_lng', axis=1) \
        .drop('_lat', axis=1)
    return df

def make_df_edge_with_kdtree(df, nodetype):
    df = df.reset_index()
    idx2nodeid = df.rename_axis('idx').reset_index()[['idx', 'nodeid']].astype('int').set_index('idx').nodeid
    gps_array = df[['lng', 'lat']].to_numpy()
    from scipy.spatial import KDTree
    kdtree = KDTree(gps_array)
    neighbor_1km = kdtree.query_ball_point(gps_array, r=1/110)
    _, neighbor_topk_nearest = kdtree.query(gps_array, 2)
    idx2idx = pd.Series(neighbor_1km.tolist()) + pd.Series(neighbor_topk_nearest.tolist())
    idx2idx = idx2idx.explode()
    df_edge = idx2idx.rename('tgt_idx').rename_axis('src_idx').to_frame().query('src_idx!=tgt_idx').reset_index().astype('int64')

    df_edge = df_edge \
        .assign_by("src_idx", src=idx2nodeid).assign_by("tgt_idx", tgt=idx2nodeid) \
        [['src', 'tgt']]
    edgetype = nodetype + nodetype.lower()
    df_edge = df_edge.assign(edgetype=edgetype)
    return df_edge

def run():
    test = False
    with GrlReport("stop_and_sid") as report:
        df_city = pd.read_csv(data_path_("geo_city.csv")).set_index("city_id")
        df_stop = pd.read_csv(data_path_("geo_stop_all_country.csv")) \
        [['cityid', 'lineid', 'stopid', 'stopname', 'linename', 'plid', 'lng', 'lat', 'type', 's2']]

        if test:
            df_stop = df_stop.iloc[:10000]
        
        df_stexit = pd.read_csv(data_path_("geo_stexit.csv"))
        
        df_stop = df_stop.assign_by('cityid', cityname=df_city.city_name) \
            .query('cityname==cityname') \
            .assign(gpstype=df_stop.type) \
            .assign_by("gpstype", sub_stopname=pd.Series(data=['站台', '站牌'], index=[0, 1]))
        
        df_stop['stopname'] = df_stop.stopname.str.replace('_', 'x').replace('|', 'x')
        df_stop['linename'] = df_stop.linename.str.replace('_', 'x').replace('|', 'x')
        df_stop['cityname'] = df_stop.cityname.str.replace('_', 'x').replace('|', 'x')
        df_stexit['name'] = df_stexit.name.str.replace('_', 'x').replace('|', 'x')


        df_stop = df_stop \
            .assign_join_col(nodename=['stopname', 'linename', 'cityname', 'stopid']) \
            .assign_gps()
        df_sid = df_stop.set_index("stopid")[['stopname', 'nodename', 'cityname', 'lng', 'lat']]
        df_stop = df_stop.assign_by('s2', next_nodename=df_sid.nodename)
    
    with GrlReport("gps") as report:
        df_stexit = df_stexit \
            .assign_by('stopid', nodename=df_sid.nodename, stopname=df_sid.stopname, cityname=df_sid.cityname) \
            .query("stopname==stopname") \
            .assign(gpstype=2) \
            .rename(dict(name='sub_stopname'), axis=1)

        df_gps_raw = pd.concat([
                df_stop[['stopid', 'stopname', 'sub_stopname', 'cityname', 'lng', 'lat', 'gpstype', 'nodename']],
                df_stexit[['stopid', 'stopname', 'sub_stopname', 'cityname', 'lng', 'lat', 'gpstype', 'nodename']]
            ], axis=0) \
            .reset_index(drop=True) \
            .rename(dict(nodename='l_nodename'), axis=1) \
            .assign_gps()

        df_gps_cityinfo = df_gps_raw \
            .groupby("gps").cityname.unique() \
            .rename('citynames') \
            .to_frame() \
            .assign(citycount=lambda xdf: xdf.citynames.apply(len)) \
            .assign(cityname=lambda xdf: xdf.citynames.apply(lambda xs: '|'.join(xs))) \
            .drop("citynames", axis=1)

        df_gps = df_gps_raw \
            .groupby("gps") \
            .first() \
            .assign(citycount=df_gps_cityinfo.citycount, cityname=df_gps_cityinfo.cityname) \
            .reset_index() \
            .assign_join_col(gps_nodename=('stopname', 'sub_stopname', 'cityname', 'gps')) \
            .set_index("gps") \
            [['gps_nodename', 'gpstype', 'lng', 'lat', 'citycount']]

    with GrlReport("node") as report:
        df_node = pd.concat([
            df_stop[['nodename', 'lng', 'lat']].assign(nodetype='L'),
            df_gps \
            .assign_by('gpstype', nodetype=pd.Series(index=[0, 1, 2], data=['P', 'N', "N"])) \
            [['gps_nodename', 'lng', 'lat', 'nodetype']] \
                .rename(dict(gps_nodename='nodename'), axis=1)
        ]) \
        .reset_index(drop=True)
        df_node = df_node.assign(nodeid=df_node.index+1001).set_index('nodeid')

        nodename2nodeid = df_node.reset_index().set_index('nodename').nodeid

    with GrlReport("edge_part_one") as report:
        df_edge_l2l = df_stop \
        [['nodename', 'next_nodename']] \
        .rename(dict(nodename='src_nodename', next_nodename='tgt_nodename'), axis=1) \
        .query('tgt_nodename==tgt_nodename') 

        _df_edge_gps2l = df_gps_raw.assign_by("gps", gps_nodename=df_gps.gps_nodename) \
            [['gps_nodename', 'l_nodename']] \
            .rename(dict(gps_nodename='src_nodename', l_nodename='tgt_nodename'), axis=1)
        df_edge_e2l = _df_edge_gps2l \
            .assign(temp = lambda xdf: xdf.src_nodename.str.split("_").apply(lambda xs: xs[-1])).query("temp=='2'") \
            .drop("temp", axis=1)
        df_edge_p2l = _df_edge_gps2l \
            .assign(temp = lambda xdf: xdf.src_nodename.str.split("_").apply(lambda xs: xs[-1])).query("temp=='0'") \
            .drop("temp", axis=1)
        df_edge_gps2l = _df_edge_gps2l \
            .assign(temp = lambda xdf: xdf.src_nodename.str.split("_").apply(lambda xs: xs[-1])).query("temp!='2'") \
            .drop("temp", axis=1)
        df_edge_l2gps = df_edge_gps2l \
            .rename(dict(src_nodename='tgt_nodename', tgt_nodename='src_nodename'), axis=1)
        
        df_edge_n2p = df_edge_e2l.set_index("tgt_nodename") \
            .join(df_edge_p2l.set_index("tgt_nodename"), lsuffix='_n', rsuffix='_p') \
            .reset_index(drop=True) \
            .rename(dict(src_nodename_n='src_nodename', src_nodename_p='tgt_nodename'), axis=1)

        df_edge_p2n = df_edge_n2p.rename(dict(src_nodename='tgt_nodename', tgt_nodename='src_nodename'), axis=1) \
        [["src_nodename", 'tgt_nodename']]
        
        df_edge = pd.concat([
            df_edge_l2l, df_edge_gps2l, df_edge_l2gps, df_edge_n2p, df_edge_p2n
        ], axis=0)

        df_edge = df_edge \
            .assign_by('src_nodename', src=nodename2nodeid) \
            .assign_by('tgt_nodename', tgt=nodename2nodeid) \
            .assign_by('src', src_nodetype=df_node.nodetype) \
            .assign_by('tgt', tgt_nodetype=df_node.nodetype)

        df_edge = df_edge.assign(edgetype=df_edge.src_nodetype+df_edge.tgt_nodetype.str.lower())
        df_edge = df_edge[['src', 'tgt', 'edgetype']]
    with GrlReport("edge_part_two_neighbor_kdtree"):
        id2nodename = df_node.nodename
        df_edge_nn = make_df_edge_with_kdtree(df_node.query("nodetype=='N'"), 'N')
        df_edge_pp = make_df_edge_with_kdtree(df_node.query("nodetype=='P'"), 'P')
        df_edge_pp = df_edge_pp \
            .assign_by("src", src_stopname=id2nodename.str.split("_").apply(lambda xs: xs[0])) \
            .assign_by("tgt", tgt_stopname=id2nodename.str.split("_").apply(lambda xs: xs[0])) \
            .query("src_stopname==tgt_stopname") \
            .drop("src_stopname", axis=1) \
            .drop("tgt_stopname", axis=1)
        df_edge = pd.concat([
            df_edge, df_edge_nn, df_edge_pp,
        ], axis=0)
    with GrlReport("city_info"):
        df_city_nodeid = df_node \
            .nodename \
            .apply(lambda x: x.split("_")[2].split("|")) \
            .rename("cityname") \
            .to_frame() \
            .rename_axis("nodeid") \
            .reset_index() \
            .explode("cityname") \
            .groupby("cityname") \
            .agg(
                nodeids=('nodeid', 'unique'),
            ) \
            .assign(nodecount=lambda xdf: xdf.nodeids.apply(len))
        df_city = df_city \
            .assign_by("city_name", nodeids=df_city_nodeid.nodeids.apply(lambda xs: ','.join([str(_) for _ in xs])), nodecount=df_city_nodeid.nodecount)

    with GrlReport("save"):
        df_node.to_csv(data_path_("gcg_node.csv"))
        df_edge.to_csv(data_path_("gcg_edge.csv"))
        df_city.to_csv(data_path_("gcg_city.csv"))
run()