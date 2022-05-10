
import numpy as np

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