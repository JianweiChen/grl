

def draw_subway_plot3d():
    gcg = GcgDataLoader().load_gcg_from_path(data_path_("gcg_subway.dill"))
    components = gcg.g.components()
    print("components len=", components.__len__())
    get_repr = lambda v: v.nodename.split("_")[0]+":"+v.nodetype
    g2 = gcg.g.subgraph(components[40])
    g2.plot3d(disable_text=True, text_func=get_repr, hover_func=get_repr, maxiter=100_000)

def make_picture():
    width = 5
    height = 1024
    channel = 5
    baseid = 1
    maxiter = 100_0

    stack = []
    v_base = gcg.g.vs[baseid]
    base_lng, base_lat = v_base['lng'], v_base['lat']
    data = []
    iter_number = 0


    m = dict()
    seen = set()

    stack.append([baseid])
    data.append([baseid])
    seen.add(baseid)

    while data.__len__() <= height+1 and data[-1].__len__() <= width+1:
        current_path = stack.pop(0)
        current_neighbors = gcg.g.neighbors(current_path[-1], mode='out')
        for current_neighbor in current_neighbors:
            if current_neighbor in seen:
                continue
            stack.append([*current_path, current_neighbor])
            seen.add(current_neighbor)
            v_neighbor = gcg.g.vs[current_neighbor]
            degree = float(gcg.g.degree(current_neighbor, mode='out'))
            lng = v_neighbor['lng']
            lat = v_neighbor['lat']
            lng_delta = lng - base_lng
            lat_delta = lat - base_lat
            n_onehot = float(v_neighbor['nodetype'] == 'N')
            p_onehot = float(v_neighbor['nodetype'] == 'P')
            
            
            pixel = [lng_delta, lat_delta, n_onehot, p_onehot, degree]
            assert len(pixel) == channel
            m[current_neighbor] = pixel[:channel]
            
            data.append(stack[-1])
        
        iter_number += 1
        if iter_number > maxiter:
            break
        
    rows = []
    for row in data:
        row = row[1:]
        if not row: continue
        if len(row)>width: continue
        row = [*row, *([row[-1]] * width)][:width]
        rows.append(row)
        
    d2 = [m[_] for _ in torch.tensor(rows).flatten().tolist()]
    tensor = torch.tensor(d2).reshape([-1, width, channel])