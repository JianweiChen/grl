

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
    # deprecated!! read get_dfvnp_near
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
    >>> dfv = load_busgraph_vertex()
    >>> dfe = load_busgraph_edge()
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
    dfv_selected = dfv_selected.sort_values(["nodetype", 'lng', 'lat'])
    g = ig.Graph.DataFrame(dfe_selected, directed=True, vertices=dfv_selected)
    set_busgraph_default_attr(g)
    return g

def set_busgraph_default_attr(g):
    g['kdtree'] = get_busgraph_kdtree(g)
    g.vs['gps'] = [[_['lng'], _['lat']] for _ in g.vs]

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

def get_desc_from_lidseq(lidseq_tps):
    assert isinstance(lidseq_tps, list)
    assert isinstance(lidseq_tps[0], tuple) or isinstance(lidseq_tps[0], list)
    assert lidseq_tps[0].__len__() == 2
    keys = [
        f"{_[0]}_{_[1]}" for _ in lidseq_tps
    ]
    vals = [get_desc_from_bin(_) for _ in client.hmget("desc", keys)]
    return dict(zip(keys, vals))
do_desc = get_desc_from_lidseq

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

def get_polar_coordinate(gps_arr, ka=10, kd=1, ta=1000):
    coordinate = np_coords(gps_arr)
    degree = np.degrees(np.arctan2(coordinate[..., 0], coordinate[..., 1]))
    degree += (degree<0).astype('float32') * 360
    amplitude = np.linalg.norm(coordinate, axis=1)
    
    degree *= kd
    amplitude *= ka
    
    degree = degree.astype('int')
    amplitude = amplitude.astype('int')
    return np.stack([degree, amplitude], axis=1)

def parse_orders(orders, precision=3):
    orders = orders.split(',')
    ods = []
    coords = []
    for order in orders:
        parts = order.split('_')
        coords.append([round(float(_), precision) for _ in parts[1:]])
    gps_arr = np.array(coords)
    mht = np_mht(gps_arr)
    dftv = pd.Series(gps_arr.reshape([-1, 2]).tolist()).apply(tuple) \
        .rename("name").to_frame() \
        .groupby("name") \
        .size() \
        .rename("vc") \
        .to_frame() \
        .reset_index() \
        .sort_values("vc", ascending=False)
    dfte = pd.DataFrame(dict(src=gps_arr[..., :2].tolist(), tgt=gps_arr[..., 2:].tolist(), mht=mht))
    dfte['src'] = dfte['src'].apply(tuple)
    dfte['tgt'] = dfte['tgt'].apply(tuple)
    dfte = dfte.groupby(['src', 'tgt'], as_index=False).agg(mht=('mht', 'first'), ec=('mht', 'count'))
    g = ig.Graph.DataFrame(dfte, directed=True, vertices=dftv)
    return g

def get_g_subway(dfv, dfe):
    dfv_selected = dfv \
        .assign(gate=dfv.vertex_name.str.split('_') \
        .apply(lambda x: x[-1]).astype('int')) \
        .query("gate>0") \
        .reset_index(drop=True)
    vnames = dfv_selected.vertex_name.tolist()
    dfe_selected = dfe \
        .set_index("src") \
        .loc[vnames] \
        .reset_index() \
        .set_index("tgt") \
        .loc[vnames] \
        .reset_index() \
        [['src', 'tgt', 'edgetype']]
    v2lng = dfv_selected.set_index("vertex_name").lng
    v2lat = dfv_selected.set_index("vertex_name").lat
    arr = dfe_selected \
        .assign_by("src", srclng=v2lng, srclat=v2lat) \
        .assign_by("tgt", tgtlng=v2lng, tgtlat=v2lat) \
        [['srclng', 'srclat', 'tgtlng', 'tgtlat']] \
        .to_numpy()
    dfe_selected = dfe_selected.assign(mht=np_mht(arr))
    dfe_selected = add_weight_default_column(dfe_selected)

    g = igraph.Graph.DataFrame(dfe_selected, directed=True, vertices=dfv_selected)
    return g

def load_g_subway(path=data_path_("g_subway.dill")):
    g = dill.load(pathlib.Path(path).open('rb'))
    g.vs['gps'] = [[_['lng'], _['lat']] for _ in g.vs]
    return g

def load_g_sample_subway(path=data_path_("g_sample_subway.dill")):
    return load_g_subway(path)

def get_busgraph_kdtree(g):
    kdtree = KDTree(g.get_vertex_dataframe().query("nodetype!='s'")[['lng', 'lat']].to_numpy())
    return kdtree

def query_ball_point_vids(g, coordinate, km):
    return g['kdtree'].query_ball_point(coordinate, km/110)


def get_candidates_by_gps(g, gps, km=0.8):
    vids = g['kdtree'].query_ball_point(gps, km/110)
    return g.vs[vids]

def get_candidates_by_vid(g, vid, km=0.8):
    vids = g.neighborhood(vid, g.vcount(), mindist=1, mode='out')
    if not vids:
        return g.vs[vids]
    vs = g.vs[vids]
    vids = functools.reduce(
        lambda x, y: x+y,
        g['kdtree'].query_ball_point(vs['gps'], km/110),
        []
    )
    vids = list(set(vids))
    return g.vs[vids]

def get_near_sid_g(df_sid, coordinate, km):
    df = df_sid.loc[kdtree.query_ball_point(coordinate, km/110)].sort_values(['lid', 'seq'])
    df['key'] = df.lid.astype('str') + "_" + df.seq.astype('str')
    df2 = df[['key', 'lid', 'seq']].sort_values(['lid', 'seq']).reset_index(drop=True)
    dfv_selected = df[['key', 'lid', 'seq', 'lng', 'lat']]
    dfe_selected = pd.concat([
        df2,
        df2.shift(-1).rename({_:_+"_shifted" for _ in df2.columns}, axis=1).reset_index(drop=True)
    ], axis=1) \
    .query("key_shifted==key_shifted") \
    .astype({'lid_shifted': 'int', 'seq_shifted': 'int'}) \
    .query("lid==lid_shifted") \
    [['key', 'key_shifted']] \
    .rename(dict(key='src', key_shifted='tgt'), axis=1)
    g = ig.Graph.DataFrame(dfe_selected, directed=True, vertices=dfv_selected)
    g.vs['gps'] = [(_['lng'], _['lat']) for _ in g.vs]
    g.vs['lidseq'] = [(_['lid'], _['seq']) for _ in g.vs]
    g['kdtree'] = KDTree(g.get_vertex_dataframe()[['lng', 'lat']].to_numpy())
    return g

def report_onestep(ctx, history, need_desc=False):
    """这个函数主要用于打印机器学习过程中的数值解
    """
    g = ctx['g']
    origin = ctx['origin']
    destination = ctx['destination']
    v = history[-1]
    stepcount = len(history)

    gps = v['gps']
    lidseq = v['lidseq']
    lid = v['lid']
    seq = v['seq']
    gps_arr = np.array([
        [*origin, *gps],
        [*gps, *destination]
    ])
    coords = np.round(np_coords(gps_arr), 4).tolist()
    
    to_report = dict(
        stepcount=stepcount,
        gps=gps,
        lidseq=lidseq,
        ori_coords = coords[0],
        dst_coords = coords[1]
    )
    if need_desc:
        to_report['desc'] = do_desc([lidseq]).get(f"{lid}_{seq}")
    
    for k, v in to_report.items():
        print(k, '=', v)
    print()


def query_guidepost(g, guidepost_req_jd):
    origin = guidepost_req_jd['origin']
    destination = guidepost_req_jd['destination']
    abparam = guidepost_req_jd['abparam']

    need_desc = abparam.get("need_desc", False)
    need_report = abparam.get("need_report", False)
    ctx = {
        'g': g,
        "origin": origin,
        "destination": destination,
        "guidepost_req": guidepost_req_jd,
        'history': []
    }
    history = ctx[
        'history'
    ]
    vs = get_candidates_by_gps(g, origin)
    for i in range(10):
        v = guidepost_recommend(ctx, vs)
        if not v: break
        history.append(v)
        
        vs = get_candidates_by_vid(g, v.index)
        if need_report:
            report_onestep(ctx, history, need_desc=need_desc)
    rsp = dict(
        indices= [v.index for v in history]
    )
    return rsp

def get_d_by_o(origin):
    destination = (origin[0], origin[1]+0.08)
    return destination

def recommend_next_vid(g, vs):
    """main function for guidepost!!
    """
    if len(vs) == 0:
        return None
    else:
        i = random.randint(0, len(vs)-1)
        v = vs[i]
        vid = v.index
        return v
    

def check_guidepost_input_df(df):
    #TODO
    return True

def get_guidepost_context(df, location, neighbor_km=0.8, km=10, line_seq_max_count=32):
    assert check_guidepost_input_df(df), "get_guidepost_context input df format error..."

    def make_line_location_feature(arr):
        arr = np.array(arr)
        extra_shape = (line_seq_max_count - arr.shape[0], arr.shape[1])
        assert extra_shape[0]>=0
        if extra_shape[0]>0:
            arr = np.concatenate([
                arr,
                np.broadcast_to(arr[-1, ...], extra_shape)
            ], axis=0)
        return arr
    df = df \
        .assign(seq_mod=lambda xdf: xdf.sequenceno % line_seq_max_count) \
        .assign(seq_div=lambda xdf: xdf.sequenceno // line_seq_max_count) \
        .assign(lineid=lambda xdf: xdf.lineid.astype('str')+'_'+xdf.seq_div.astype('str')) \
        .assign(seq=lambda xdf: xdf.seq_mod) \
        [['lineid', 'seq', 'lng', 'lat', 'stopname', 'linename']]
    #TODO get map of lineid-seq -> linename, stopname, lng, lat
    arr = df[['lng', 'lat']].to_numpy()
    kdtree = KDTree(arr)
    idxs = kdtree.query_ball_point(location, km/110)
    lineids = df.loc[idxs].lineid.unique()
    df = df.set_index("lineid").loc[lineids].reset_index()
    df_line_vertex = df \
        .assign(gps=lambda xdf: xdf.apply(axis=1, func=lambda row: [row.lng, row.lat])) \
        .groupby("lineid") \
        .gps.apply(list) \
        .apply(make_line_location_feature) \
        .rename("feature") \
        .to_frame() \
        .rename_axis("lineid") \
        .reset_index()
    arr = df[['lng', 'lat']].to_numpy()
    kdtree = KDTree(arr)
    df_line_edge = pd.Series(kdtree.query_ball_point(arr, neighbor_km/110)) \
        .explode() \
        .rename_axis("src").rename("tgt") \
        .to_frame().reset_index() \
        .astype(dict(src='int', tgt='int')) \
        .assign_by("src", src_lineid=df_line_vertex.lineid) \
        .assign_by("tgt", tgt_lineid=df_line_vertex.lineid) \
        .groupby(['src_lineid', 'tgt_lineid'], as_index=False) \
        .size().rename(dict(size="transfer_count"), axis=1) \
        .query("src_lineid!=tgt_lineid")
    tgt_lineids = list(set(df_line_edge.tgt_lineid.tolist()) & set(df_line_vertex.lineid.tolist()))
    df_line_edge = df_line_edge \
        .set_index("tgt_lineid") \
        .loc[tgt_lineids] \
        .reset_index() \
        [['src_lineid', 'tgt_lineid', 'transfer_count']]
    sr_linename = df.groupby("lineid").linename.first()
    df_line_vertex = df_line_vertex \
        .assign_by('lineid', linename=sr_linename)
    g = ig.Graph.DataFrame(df_line_edge, directed=True, vertices=df_line_vertex)
    
    # function: get_location_feature

    df_gps2lineids = df.groupby(['lng', 'lat']).lineid.unique() \
    .rename('lineids').to_frame() \
    .reset_index()
    stop_gps_arr = df_gps2lineids[['lng', 'lat']].to_numpy()
    stop_kdtree = KDTree(stop_gps_arr)
    m_idx2lineid = dict(df_gps2lineids['lineids'].reset_index().to_numpy().tolist())

    sr_lineid2vid = g \
        .get_vertex_dataframe() \
        ['name'] \
        .rename_axis("vid") \
        .rename("lineid") \
        .reset_index() \
        .set_index("lineid")['vid']
    m_lineid2vid = dict(sr_lineid2vid.reset_index().to_numpy().tolist())

    def get_location_feature(location, n=5):
        #TODO user context mode
        import functools
        rd_list, ri_list = stop_kdtree.query(location, n)
        rs = functools.reduce(lambda x, y: x+y, [
            list(zip(m_idx2lineid[ri].tolist(), [rd]*m_idx2lineid[ri].shape[0])) for ri, rd in zip(ri_list, rd_list)
        ], [])
        rs = [
            (m_lineid2vid[_[0]], _[1]) for _ in rs
        ]
        
        features = g.vs[[_[0] for _ in rs]]['feature']
        
        feature = np.concatenate(features, axis=1)
        k = 32
        feature = feature[..., :k]
        extra_shape = (feature.shape[0], k-feature.shape[1])
        if extra_shape[1]>0:
            extra_feature = np.broadcast_to(feature[..., -1:], extra_shape)
            feature = np.concatenate([
                feature, extra_feature
            ], axis=1)
        return feature
    # function: get_desc
    _df = df \
        .assign(key=lambda xdf: xdf.lineid+'_'+xdf.seq.astype('str')) \
        .assign(val=lambda xdf: xdf[['linename', 'stopname', 'lng', 'lat']].apply(axis=1, func=dict)) \
        [['key', 'val']]
    m_desc = dict(_df.to_numpy().tolist())
    def get_desc(lineid, seq=None):
        if not isinstance(lineid, str) or '_' not in lineid:
            lineid = str(lineid)+'_0'
        if not seq:
            key = lineid + "_1"
            return m_desc.get(key, dict()).get('linename')
        key = f"{lineid}_{seq}"
        return m_desc.get(key)
    # merge context
    context = {
        'g': g,
        'get_location_feature': get_location_feature,
        "m_desc": m_desc,
        "get_desc": get_desc,
    }
    return context

def get_getoff_seq(arr1, arr2):
    """
    TODO: judge arr1 and arr2 are oppo line
    """
    kdtree = KDTree(arr1)
    rd_list, ri_list = kdtree.query(arr2)
    i = np.argmin(rd_list)
    return ri_list[i]

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

