
from .base_util import monkey
import networkx as nx

@monkey(nx.Graph, "draw")
def _nx_draw(g):
    nx.draw(g, with_labels=True, font_color='white')
