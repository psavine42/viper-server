import time
from .adapter import Visualizer, meshcat
from trimesh import primitives, Trimesh
import numpy as np
import src.geom as geom

_zmq_url = 'tcp://127.0.0.1:6000'
VIZ = Visualizer(_zmq_url)
_default_material = meshcat.geometry.MeshPhongMaterial


# ---------------------------------------------------
def as_path(*args):
    return '/'.join(map(str, args))


def _export_and_set(gm, handle, uid=None, mat=None, **kwargs):
    obj = gm.export(file_type='obj')
    mg = meshcat.geometry.ObjMeshGeometry(obj)
    if uid is not None:
        mg.uuid = uid
    VIZ[handle].set_object(mg, mat)


def _with_material(mat=None, opacity=None, **kwargs):
    if mat is None and opacity is None:
        return None
    elif mat is not None:
        return mat
    elif opacity is not None:
        return _default_material(transparent=True, opacity=opacity)
    else:
        return _default_material(transparent=False)


def _default_handle(handle=None):
    return handle if handle else str(time.time()).split('.')[0]


def viz_line(c1, c2, handle=None, thickness=0.05, **kwargs):
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
        if type(c1).__module__ != 'numpy':
            c1 = np.asarray(c1)
        if type(c2).__module__ != 'numpy':
            c2 = np.asarray(c2)
        zr = np.eye(3) * thickness
        _x, _y, _p = zr[0], zr[1], zr[2]
        # _x = np.array([thickness, 0, 0])
        # _y = np.array([0, thickness, 0])
        # _p = np.array([0, 0, thickness])
        ls = Trimesh(vertices=[c1,      c2,
                               c1 + _p, c2 + _p,
                               c1 - _p, c2 - _p,
                               c1 + _y, c2 + _y,
                               c1 - _y, c2 - _y,
                               c1 + _x, c2 + _x,
                               c1 - _x, c2 - _x])
        _export_and_set(ls.convex_hull, _default_handle(handle),
                        mat=_with_material(**kwargs))
    except Exception as e:
        print(repr(e))


def viz_point(xyz, handle='points/', radius=0.2, **kwargs):
    _clear_if(handle, **kwargs)
    px = primitives.Sphere(radius=radius, center=xyz)
    _export_and_set(px, handle, mat=_with_material(**kwargs))


def viz_point_bx(xyz, handle='points/', radius=0.2):
    px = primitives.Sphere(radius=radius, center=xyz).bounding_box
    _export_and_set(px, handle)


def viz_mesh(x, handle=None, uid=None, **kwargs):
    uid = str(x.id) if hasattr(x, 'id') else uid
    handle = as_path(handle, uid)
    _export_and_set(x, handle, uid=uid, mat=_with_material(**kwargs))


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


def viz(arg, **kwargs):
    from lib.geo import Point, Line
    if isinstance(arg, Line):
        viz_line(np.array(list(arg.points[0].coordinates)),
                 np.array(list(arg.points[1].coordinates)), **kwargs)
    elif isinstance(arg, Point):
        viz_point(list(arg.coordinates), **kwargs)


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
def viz_by_index(items, handle='x', **kwargs):
    for i, k in enumerate(items):
        viz(k, handle=as_path(handle, i), **kwargs)


def visualize_indexed(solids, index, handle=None):
    """

    :param solids:
    :param index: ddict(set) ix from to
    :param handle:
    :return:
    """
    handle = _default_handle(handle)
    for i, tgts in index.items():
        for t in tgts:
            viz_line(solids[i].centroid, solids[t].centroid,
                     handle=as_path(handle, i, t))


def visualize_points(points, handle=None, radius=0.2):
    handle = _default_handle(handle)
    for x in points:
        viz_point(x, handle + '/' + str(x), radius)
    return handle


def visualize_lines(lines, handle=None, thickness=0.25, **kwargs):
    for x, y in lines:
        viz_line(x, y, handle=_default_handle(handle), thickness=thickness)
    return handle


# testing specific visualization ------------------------------
def _clear_if(handle, clear=False, **kwargs):
    if clear is True:
        VIZ[handle].delete()


def viz_ixs(vsys, ixs, handle='tsx', **kwargs):
    _clear_if(handle, **kwargs)
    for ix_ in ixs:
        viz_mesh(vsys.inputs[ix_], handle=handle + '/{}'.format(ix_), **kwargs)


def viz_conn_ixs(vs2, ixs, handle='conn', **kwargs):
    _clear_if(handle, **kwargs)
    for i in ixs:
        adj = vs2.inters[i]
        for j in adj.neigh_to_sphere.keys():
            if j in ixs:
                viz_line(vs2.inputs[i].centroid, vs2.inputs[j].centroid,
                         handle=handle + '/{}/{}'.format(i, j))


def viz_nodes(nodes, handle='nodes', **kwargs):
    _clear_if(handle, **kwargs)
    seen = set()
    counter = 0
    for i, n in enumerate(nodes):
        for neigh in n.neighbors(fwd=True, bkwd=True, edges=True):
            this = tuple(sorted(list(neigh.geom)))
            if this not in seen:
                seen.add(this)
                g1, g2 = neigh.geom
                pipe = str(neigh.get('is_pipe', None))
                viz_line(g1, g2, handle=as_path(handle, pipe, counter), **kwargs)
                counter += 1


def viz_edges(edges, handle='nodes', **kwargs):
    _clear_if(handle, **kwargs)
    seen = set()
    for i, e in enumerate(edges):
        if e.id not in seen:
            seen.add(e.id)
            g1, g2 = e.geom
            pipe = str(e.get('is_pipe', None))
            r = e.get('radius', 0.05) / 2
            viz_line(g1, g2, handle=as_path(handle, pipe, e.id), thickness=r, **kwargs)


def viz_iter(node, handle='nodes', **kwargs):
    _clear_if(handle, **kwargs)
    seen = set()
    for n in node.__iter__(fwd=True, bkwd=True):
        for neigh in n.neighbors(fwd=True, bkwd=True, edges=True):
            if neigh.id not in seen:
                seen.add(neigh.id)
                g1, g2 = neigh.geom
                pipe = str(neigh.get('is_pipe', None))
                r = neigh.get('radius') if neigh.get('radius') is not None else 0.01
                viz_line(g1, g2, handle=as_path(handle, pipe, neigh.id), thickness=r,  **kwargs)


def viz_edge_dir(edge, handle='edge', radius=0.02, t=0.005, **kwargs):
    crv = edge.curve
    crv = crv.extend(end=-(2*radius + t*3)).points_np()
    viz_line(*crv, handle=as_path(handle, edge.id, 'line'), thickness=t)
    viz_point(crv[1], handle=as_path(handle, edge.id, 'end'), radius=radius)


def viz_edge_geo(edge, handle, **kwargs):
    g1, g2 = edge.geom
    r = edge.get('radius') if edge.get('radius') is not None else 0.01
    viz_line(g1, g2, handle=as_path(handle, edge.id), thickness=r, **kwargs)


def viz_order(node, handle='nodes', **kwargs):
    _clear_if(handle, **kwargs)
    for n in node.__iter__(fwd=True):
        for neigh in n.successors(edges=True):
            # if neigh.id not in seen:
            # seen.add(neigh.id)
            viz_edge_dir(neigh, handle=handle)


def viz_heirarchical(root_node, handle, **kwargs):
    _clear_if(handle, **kwargs)
    # path = 0
    q = [ (root_node, [0]) ]
    while q:
        node, path_ix = q.pop(0)
        sucs = node.successors(both=True)
        if len(sucs) >= 1:
            suc_edge, suc_node = sucs[-1]
            pth = path_ix + ['data']
            viz_edge_geo(suc_edge, handle=as_path(handle, *pth), **kwargs)
            q.append((suc_node, path_ix))

        if len(sucs) == 2:
            suc_edge2, suc_node2 = sucs[0]
            first = path_ix[0:-1] + [0]
            last = path_ix[-1] + 1

            pth = first + [last, 'data']
            viz_edge_geo(suc_edge2, handle=as_path(handle, *pth), **kwargs)
            q.append((suc_node2, first + [last]))
        else:
            continue



def delete(k):
    VIZ[k].delete()

