from grl import *
from grl.data_util import GcgDataLoader, Gcg
import dill
import igraph as ig
import networkx as nx
import ipyvuetify as v
import ipywidgets
import random
import treelib

gcg_data_loader = GcgDataLoader()



# gcg = gcg_data_loader.load_gcg(4,1)
gcg = gcg_data_loader.get_global_gcg()

pvid_list = gcg.df_vertex.query("nodetype=='P'").index.tolist()
subway_vids_list = gcg.g.neighborhood(pvid_list, mode='out', order=1, mindist=0)
subway_vid_list = pd.DataFrame(dict(x=[pd.Series(vids) for vids in subway_vids_list])).explode("x").x.tolist()
g_subway = gcg.g.subgraph(subway_vid_list)

gcg = Gcg(g=g_subway)

dill.dump(gcg, pathlib.Path(data_path_("gcg_subway.dill")).open("wb"))