import time
from .adapter import Visualizer, meshcat
from trimesh import primitives, Trimesh
import numpy as np
import src.geom as geom

_zmq_url = 'tcp://127.0.0.1:6000'
VIZ = Visualizer(_zmq_url)


# ---------------------------------------------------
def as_path(*args):
    return '/'.join(map(str, args))


def _export_and_set(gm, handle, uid=None, mat=None):
    obj = gm.export(file_type='obj')
    mg = meshcat.geometry.ObjMeshGeometry(obj)
    if uid is not None:
        mg.uuid = uid
    VIZ[handle].set_object(mg, mat)


def viz_line(c1, c2, handle=None, thickness=0.05, opacity=None, mat=None, **kwargs):
    """
    Create a meshcat line from two points - dirty
    :param c1:
    :param c2:
    :param handle:
    :param viz:
    :param thickness:
    :return:
    """
    try:
        _p = np.array([0, 0, thickness])
        _y = np.array([0, thickness, 0])
        _x = np.array([thickness, 0, 0])
        handle = handle if handle else 'None'
        ls = Trimesh(vertices=[c1,      c2,
                               c1 + _p, c2 + _p,
                               c1 - _p, c2 - _p,
                               c1 + _y, c2 + _y,
                               c1 - _y, c2 - _y],
                     faces=[[2, 4, 6], [2, 4, 8],
                            [1, 3, 5], [1, 3, 7]])
        if opacity is not None:
             mat = meshcat.geometry.MeshPhongMaterial(transparent=True, opacity=opacity)
        else:
             mat = meshcat.geometry.MeshPhongMaterial(transparent=False)
        _export_and_set(ls.convex_hull, handle, mat=mat)
    except Exception as e:
        print(repr(e))


def viz_point(xyz, handle='points/', radius=0.2):
    px = primitives.Sphere(radius=radius, center=xyz)
    _export_and_set(px, handle)


def viz_point_bx(xyz, handle='points/', radius=0.2):
    px = primitives.Sphere(radius=radius, center=xyz).bounding_box
    _export_and_set(px, handle)


def viz_mesh(x, mat=None, handle=None, opacity=None, transparent=True, **kwargs):
    uid = str(x.id) if hasattr(x, 'id') else None
    handle = as_path(handle, x.id) if hasattr(x, 'id') else handle
    if mat is None and opacity is not None:
        mat = meshcat.geometry.MeshPhongMaterial(transparent=transparent, opacity=opacity)
    _export_and_set(x, handle, uid=uid, mat=mat)


def viz_bone(*args, handle=None,  **kwargs):
    if len(args) == 2:
        c1, c2 = args
    elif len(args) == 6:
        c1, c2 = args[:3], args[3:]
    else:
        return
    viz_line(c1, c2, handle=as_path(handle, 'line'), **kwargs)
    viz_point(c1, handle=as_path(handle, 'points/1'), **kwargs)
    viz_point(c2, handle=as_path(handle, 'points/2'), **kwargs)


# ----------------------------------------------------
def viz_recur(icad, handle=None, types=None, **kwargs):
    """
    recursively visualize objects and children
    :param icad:
    :param handle:
    :param types:
    :param kwargs:
    :return:
    """
    if types is None:
        types = [geom.MepCurve2d, geom.MeshSolid]

    if isinstance(icad, list):
        for child in icad:
            viz_recur(child, handle=handle, types=types, **kwargs)

    else:
        _hnd = icad.__class__.__name__
        if isinstance(icad, geom.MepCurve2d) and geom.MepCurve2d in types:
            viz_line(*icad.points, handle=as_path(handle, _hnd, icad.id))
        elif isinstance(icad, geom.MeshSolid) and geom.MeshSolid in types:
            viz_mesh(icad, handle=as_path(handle, _hnd), **kwargs)

        if icad.children is not None:
            viz_recur(icad.children, handle=handle, types=types, **kwargs)


def viz_if(icads, filter_fn, show_fn=viz_mesh, handle=None, **kwargs):
    for icad in icads:
        res = filter_fn(icad)
        if res:
            show_fn(icad, handle=handle, **kwargs)
        else:
            viz_point_bx([0, 0, 0], handle=as_path(handle, icad.id))


# ----------------------------------------------------
def visualize_indexed(solids, handle=None):
    handle = handle if handle else str(time.time()).split('.')[0]
    for x in solids:
        viz_mesh(x, handle)
    return handle


def visualize_points(points, handle=None, radius=0.2):
    handle = handle if handle else str(time.time()).split('.')[0]
    for x in points:
        viz_point(x, handle + '/' + str(x), radius)
    return handle


def visualize_lines(lines, handle=None, thickness=0.25, **kwargs):
    handle = handle if handle else str(time.time()).split('.')[0]
    for x, y in lines:
        viz_line(x, y, handle=handle, thickness=thickness)
    return handle







