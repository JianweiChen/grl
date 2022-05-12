import ipyvuetify as ipyv
import ipyleaflet
import pandas as pd

from pypinyin import pinyin
from ipywidgets import jslink

import dill
import igraph as ig
import networkx as nx
import ipyvuetify as ipyv
import ipywidgets
import random
import ipyleaflet
from grl.data_util import data_path_
from grl.data_util import GcgDataLoader, Gcg

def get_enname(hz):
    hz = hz.replace("市", '')
    return ''.join([_[0] for _ in pinyin(hz, style=0)])
def gid_to_gps(gid):
    gid = str(gid)
    lat_gid = gid[-10:-4]
    lng_gid = gid[:-10]
    lng = int(lng_gid) / 1000
    lat = int(lat_gid) / 1000
    return [lng, lat]

class GuidepostFe(object):
    """
    Usage: 
    >>> gpfe = GuidepostFe()
    >>> gpfe.display()
    """
    def __init__(self):
        self.zoom = 10
        self.focus_field_index = 0
        self.df_city = self.get_df_city()
        self.gcg_subway = self.get_gcg_subway() # 这一步会花1到2秒钟
        self.m = self.get_m()
        self.city_select = ipyv.Autocomplete(
            label="选择城市",
            items=list(self.df_city.index),
            v_model='beijing'
        )
        textfield_attr = dict(
            outlined=True, dense=True
        )
        self.origin_gps_field = ipyv.TextField(label='起点gps', v_model="", **textfield_attr)
        self.origin_frozen = ipyv.Checkbox(label="Lock", v_model=False)
        jslink((self.origin_gps_field, "disabled"), (self.origin_frozen, "v_model"))
        self.destination_gps_field = ipyv.TextField(label='终点gps', v_model="", **textfield_attr)
        self.destination_frozen = ipyv.Checkbox(label="Lock", v_model=False)
        jslink((self.destination_gps_field, "disabled"), (self.destination_frozen, "v_model"))

        self.text_field_list = [
            self.origin_gps_field,
            self.destination_gps_field,
        ]
        self.text_field_index = 0

        self.city_select.on_event("input", self.on_change_df_city)
        self.children =[
            self.get_card_title(),
            self.city_select,
            ipyv.Row(children=[
                ipyv.Icon(children=['vuetify']), 
                self.origin_frozen,
                ipyv.Divider(vertical=True),
                self.origin_gps_field,
                self.destination_frozen,
                ipyv.Divider(vertical=True),
                self.destination_gps_field,
            ]),
            self.m,
        ]
        self.dialog = ipyv.Dialog(children=[ipyv.Card(children=self.children, width=800)], v_model=False, width=800)
        self.origin_marker = ipyleaflet.Marker(location=[0, 0], draggable=False, icon=ipyleaflet.Icon(icon_url='https://img.icons8.com/flat-round/512/000000/home--v1.png', icon_size=[32, 32]))
        self.destination_marker = ipyleaflet.Marker(location=[0, 0], draggable=False)
        self.destination_marker.color = "#B4D455"
        self.m.add_layer(self.origin_marker)
        self.m.add_layer(self.destination_marker)

        self.move_center_to_city_name(self.city_select.v_model)
        display(self.dialog.children[0])
    def make_gps_string(self, gps):
        N = 100_000
        lng = str(int(gps[0]*N) / N)
        lat = str(int(gps[1]*N) / N)
        return f"{lng}, {lat}"


    def get_current_textfield_index(self):
        if self.destination_frozen.v_model and self.origin_frozen.v_model:
            return -1
        if self.destination_frozen.v_model:
            return 0
        if self.origin_frozen.v_model:
            return 1
        r = self.text_field_index
        self.text_field_index = 1-self.text_field_index
        return r
    def handle_m_interaction(self, **kwargs):
        # if kwargs.get("type") in ("mouseover", "mousemove", "mouseout"):
        #     return
        # print(kwargs.get('type'))
        if kwargs.get("type") != 'click':
            return
        coordinates = kwargs['coordinates']
        gps_string = self.make_gps_string(coordinates[::-1])
        if self.destination_frozen.v_model and self.origin_frozen.v_model:
            return
        elif self.destination_frozen.v_model:
            self.origin_gps_field.v_model = gps_string
            self.origin_marker.location = coordinates
            self.text_field_index = 1
        elif self.origin_frozen.v_model:

            self.destination_gps_field.v_model = gps_string
            self.destination_marker.location = coordinates
            self.text_field_index = 0
        else:
            if self.text_field_index == 0:
                self.origin_gps_field.v_model = gps_string
                self.destination_gps_field.v_model = ""
                self.origin_marker.location = coordinates
                self.destination_marker.location = [0.0, 0.0]
                self.text_field_index = 1
            else:
                self.destination_gps_field.v_model = gps_string
                self.destination_marker.location = coordinates
                self.text_field_index = 0
        self.render_solution()
    def render_solution(self):
        ori_gps = self.origin_marker.location[::-1]
        dst_gps = self.destination_marker.location[::-1]
        if not (ori_gps[0] > 0 and dst_gps[0] > 0):
            return
        self.m.layers = self.m.layers[:3]
        result = self.gcg_subway.get_pmconv_dots(ori_gps, dst_gps)
        for _ in result:
            popup = ipyleaflet.CircleMarker(
                location=[_[1], _[0]], 
                weight=1
                # child = ipywidgets.HTML(f"{_[3]}|{_[2]}"),
                # close_button=True,
                # auto_close=True,
                # close_on_escape_key=False
            )
            self.m.add_layer(popup)

    def on_change_df_city(self, widget, event, data):
        city_name = data
        self.move_center_to_city_name(city_name)
        self.origin_gps_field.v_model = self.destination_gps_field.v_model = ""
        self.origin_marker.location = self.destination_marker.location = [0.0, 0.0]
    def move_center_to_city_name(self, city_name):
        row = self.df_city.loc[city_name]
        gps = row.gps
        self.m.center = gps[::-1]

    def get_card_title(self):
        return ipyv.CardTitle(children=["Guidepost Dev System"])
    def get_gcg_subway(self):
        return GcgDataLoader().load_gcg_from_path(data_path_("gcg_subway.dill"))
    def get_m(self):
        m = ipyleaflet.Map(
            basemap=ipyleaflet.basemap_to_tiles(ipyleaflet.basemaps.Gaode.Normal), 
            zoom=self.zoom,
            scroll_wheel_zoom=True
        )
        m.on_interaction(self.handle_m_interaction)
        return m
    def display(self):
        self.dialog.v_model = True
    def get_df_city(self):
        df = pd.read_csv(data_path_("geo_city.csv")) \
            .query("subway_plid_count>0") \
            [['city_name', 'city_id', 'center_gid']]
        df['city_name'] = df['city_name'].apply(get_enname)
        df['gps'] = df['center_gid'].apply(gid_to_gps)
        df = df[['city_name', 'city_id', 'gps']].set_index("city_name")
        return df

