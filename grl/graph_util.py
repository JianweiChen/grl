
from .base_util import monkey
import networkx as nx
import igraph as ig
import numpy as np
import pandas as pd

@monkey(nx.Graph, "draw")
def _nx_draw(g):
    nx.draw(g, with_labels=True, font_color='white')


@monkey(ig.Graph, "draw")
def _ig_draw(g):
    nx.draw(g.to_networkx())
import plotly.graph_objects as go


### 不要过度开发这个功能 还是应该以文字工作为主
def get_graph_plotly_trace_triple(g, **conf):

    text_func = conf.get("text_func", None)
    hover_func = conf.get("hover_func", lambda x: str(dict(x)))
    disable_text = conf.get('disable_text', True)
    
    coords_array = np.array(g.layout_kamada_kawai(dim=3).coords)
    marks = ['x', 'y', 'z']
    m = {
        f"_{marks[i]}": coords_array[..., i]
        for i in range(3)
    }
    df_node = g \
        .get_vertex_dataframe() \
        .assign(**m)
    m_src = {
        f"_{marks[i]}_src": df_node[f'_{marks[i]}']
        for i in range(3)
    }
    m_tgt = {
        f"_{marks[i]}_tgt": df_node[f'_{marks[i]}']
        for i in range(3)
    }
    df_edge = g.get_edge_dataframe() \
        .assign_by("source", **m_src) \
        .assign_by("target", **m_tgt)
    
    m_node_and_text_xyz = {
        marks[i]: df_node[f'_{marks[i]}'].to_numpy()
        for i in range(3)
    }
    

    m_edge_xyz = dict()
    for i in range(3):
        key = marks[i]
        val = df_edge[[f'_{marks[i]}_src', f'_{marks[i]}_tgt']].to_numpy()
        val = np.insert(val, val.shape[1], np.nan, 1).flatten()
        m_edge_xyz[key] = val

    traces = []
    hover_array = df_node.apply(axis=1, func=hover_func).tolist()
    node_trace = go.Scatter3d(**m_node_and_text_xyz, text=hover_array, name='node',
                mode='markers', hoverinfo='text', marker=dict(size=5., opacity=0.5))
    traces.append(node_trace)
    edge_trace = go.Scatter3d(**m_edge_xyz, name='edge',
                mode='lines', line=dict(width=1.2), hoverinfo='none')
    traces.append(edge_trace)
    if text_func is not None and not disable_text:
        text_array = df_node.apply(axis=1, func=text_func).tolist()
        text_trace = go.Scatter3d(**m_node_and_text_xyz, text=text_array, name='text',
                    mode='text', hoverinfo='none', marker=dict(size=5., opacity=0.5))
        traces.append(text_trace)
    return tuple(traces)


@monkey(ig.Graph, 'plot3d')
def _ig_plot3d(g, **conf):
    traces = get_graph_plotly_trace_triple(g, **conf)
    fig = go.Figure(data=[*traces], layout=go.Layout(hovermode='closest'))
    display(fig)

@monkey(ig.Graph, "plotgcg")
def _ig_plotgcg(g, **conf):
    if 'text_func' not in conf:
        conf['text_func'] = lambda x: '|'.join([x['nodename'].split("_")[0], x['nodetype']])
    if 'hover_func' not in conf:
        conf['hover_func'] = lambda x: '|'.join([x['nodetype'], str(x['name']), x['nodename']])
    if 'disable_text' not in conf:
        conf['disable_text'] = True
    g.plot3d(**conf)

