# tool for guidepost
# 在guidepost运作过程中无需使用的，放在tool.py里


def rand_situation_feat(context):
    config = context['config']
    k1 = config['k1']
    k_history = config['k_history']
    k_lvp = config['k_lvp']
    k_candidate = config['k_candidate']
    k2 = k_history + k_lvp + k_candidate
    feat = torch.rand(k1, k2)
    return feat

def random_location(context):
    r_km = context['config']['r_km']
    center = context['config']['center']
    a = random.random() * (r_km / 110)
    theta = random.random() * 2* np.pi * random.random()
    return [center[0]+a*np.cos(theta), center[1]+a*np.sin(theta)]

def split_situation_feature(ctx, feat):
    """Divide the situation_feature into several designed types history/lvp/candidate according to the column
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

def get_line_desc(context, line_vertex_ids):
    m_desc = context.get("m_desc")
    g = context['g']
    return [m_desc.get(f"{_}_1", dict()).get('linename') for _ in g.vs[line_vertex_ids]['name']]


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


def to_pinyin(s):
    rs = pypinyin.pinyin(s, style=0)
    rs = [_[0] for _ in rs]
    r = '_'.join(rs)
    return r


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
        coordinate = [120.95164, 31.28841] # 上海苏州边界
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
        self.btn = ipyv.Btn(children=['选择位置'])
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
    
### 开发中...

def guidepost_create_round_context(ctx):
    origin = guidepost_create_location(ctx)
    destination = guidepost_create_location(ctx)
    solutions = []
    round_ctx = dict(
        origin=origin,
        destination=destination,
        solutions=solutions,
    )
    return  round_ctx

def guidepost_create_query_req(ctx, round_ctx):
    query_req = dict(
        origin=round_ctx['origin'],
        destination=round_ctx['destination'],
        req_id=str(time.time()) # for join instance label
    )
    return query_req

def guidepost_run_round(ctx, round_ctx):
    config = ctx.get('config')
    solutions = round_ctx['solutions']
    max_query_count_per_round = config.get("max_query_count_per_round", 100)
    for i in range(max_query_count_per_round):
        query_req = guidepost_create_query_req(ctx, round_ctx)
        query_rsp = guidepost_query(ctx, query_req)
        solution = query_rsp['solution']
        solutions.append(solution)
    summary_list = [
        guidepost_summary_query_rsp(_) for _ in solutions
    ]

    return solutions #TODO remove it 

def guidepost_solution_is_complete(ctx, query_req, sls):
    #TODO
    if random.random() < 0.1 and sls.__len__() >= 3:
        return True
    return False

def guidepost_query(ctx, query_req):
    g = ctx['g']
    config = ctx['config']
    k_solution = config.get("k_solution")

    origin = query_req['origin']
    destination = query_req['destination']
    ols = get_location_ls(ctx, origin)
    dls = get_location_ls(ctx,destination)
    loop = k_solution - 1
    loop = 5
    sls = []
    cls = get_location_ls(ctx, origin)
    ols = get_location_ls(ctx, origin)
    dls = get_location_ls(ctx, destination)
    for i in range(loop):
        query_one_step_req = {
            k: v for k, v in query_req.items()
        }
        query_one_step_req['sls'] = sls
        query_one_step_req['cls'] = cls
        query_one_step_req['ols'] = ols
        query_one_step_req['dls'] = dls
        query_one_step_rsp = guidepost_query_one_step(ctx, query_one_step_req)
        sl = query_one_step_rsp['sl']
        sls.append(sl)
        cls = g.neighbors(sl, mode='out')
        if guidepost_solution_is_complete(ctx, query_req, sls):
            break
    query_rsp = dict(
        origin=origin,
        destination=destination,
        solution=sls
    )
    return query_rsp

def guidepost_query_one_step(ctx, query_one_step_req):
    cls = query_one_step_req.get("cls")
    instance_pool = ctx['instance_pool']
    sl = random.choice(cls) #TODO
    query_one_step_rsp = dict(
        sl=sl
    )
    return query_one_step_rsp

def get_location_ls(ctx, location):
    """A location is represented by several nearby lines. 
    The order of these lines should be insensitive to the model.
    """
    _, ri_list = ctx['location_kdtree'].query(location, 10)
    lineids = [ctx['m_locationid_to_lineid'].get(_) for _ in ri_list]
    vertex_ids = [ctx['m_lineid_to_vertexid'].get(_) for _ in lineids]
    new_vertex_ids = []
    for vertex_id in vertex_ids:
        if vertex_id in new_vertex_ids:
            continue
        new_vertex_ids.append(vertex_id)
    return new_vertex_ids


def guidepost_build_context(df_stop, config):
    if not config.get("has_default"):
        set_default_config(config)
    k1 = config['k1']

    center = config['center']
    km_area = config['km_area']
    km_transfer = config['km_transfer']

    df_stop = get_df_stop_specific_area(df_stop, guidepost_config) \
            .assign(location=lambda xdf: xdf.apply(axis=1, func=lambda row: (row.lng, row.lat)))

    location_arr = df_stop[['lng', 'lat']].to_numpy()
    location_kdtree = KDTree(location_arr)

    df_line_vertex = df_stop \
        .groupby("lineid").location.agg(list) \
        .apply(lambda x: guidepost_feature_to_shape(x, (k1, 2))) \
        .to_frame() \
        .rename({"location": "lf"}, axis=1) \
        .reset_index()
    sr_lineid = df_stop.lineid
    
    df_line_edge = pd.Series(location_kdtree.query_ball_point(location_arr, km_transfer/110)) \
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
    
    g.add_vertex('placeholder_0', lf=torch.zeros((k1, 2)))
    location_ls_parser = LocationLsParser(
        location_kdtree,
        dict(df_stop[['lineid']].reset_index().to_numpy().tolist()),
        dict(zip(g.vs['name'], g.vs.indices))
    )

    m_desc = dict(
        df_stop \
        .assign(key=lambda xdf: xdf.lineid+"_"+xdf.seq.astype("str")) \
            .assign(val=lambda xdf: xdf[['linename', 'stopname', 'lng', 'lat', 'subway']].apply(axis=1, func=dict)) \
            [['key', 'val']].to_numpy().tolist()
    )
    g_sn = make_g_sn(df_stop, config)

    # this two are used to join in `run_round`
    instance_pool = []
    reward_pool = []
    round_cache = dict()
    
    ctx = dict(
        g=g,
        g_sn=g_sn,
        location_ls_parser=location_ls_parser,
        m_desc=m_desc,
        config=config,
        instance_pool=instance_pool,
        reward_pool=reward_pool,
        round_cache=dict(),
    )
    return ctx
