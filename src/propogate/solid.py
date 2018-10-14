import numpy as np
from collections import defaultdict as ddict
from shapely.geometry import Point, LineString
from scipy import spatial
from .base import BasePropogator, EdgePropogator, RecProporgator
from src.structs import geom_merge, Node


def confidence():
    pass


class Confidence(BasePropogator):
    def __init__(self, name='confidence', **kwargs):
        super(Confidence, self).__init__(name, **kwargs)

    def on_first(self, node, _, **kwargs):
        node.write(self.var, 1)
        return node, np.asarray(node.successors(edges=True)[0].direction)

    def on_default(self, node, prev, **kwargs):
        sucs = node.successors(edges=True)
        if len(sucs) == 0:
            return node, prev
        suc = sucs[0]
        if suc.has_geom is False:
            new_dir = np.asarray([suc.source.geom, suc.target.geom])
            suc.write('mean_pt', np.mean(new_dir, axis=0))
            suc.write('len', spatial.distance.cdist([new_dir[0]], [new_dir[1]])[0][0])
            return node, prev
        else:
            """
            direction_adj = direction * confidence_in_direction
            confidence_in_direction = num * mean_deviation
            
            
            """
            new_dir = np.asarray(suc.direction)
            norm_diff = np.sum(np.abs(prev - new_dir)) / np.pi
            mean = np.mean(np.stack([prev, new_dir]), axis=0)
            suc.write(self.var, norm_diff)
            return node, new_dir


# -------------------------------------------
class EdgeFunc(EdgePropogator):
    def __init__(self, fn, **kwargs):
        super(EdgeFunc, self).__init__(name=fn.__name__, **kwargs)
        self._fn = fn

    def on_default(self, edge, _, **kwargs):
        self._fn(edge)
        return edge, _


class PropsEdge(EdgePropogator):
    def __init__(self, fn, name='propshow', **kwargs):
        super(PropsEdge, self).__init__(name, **kwargs)
        self._fn = fn

    def on_default(self, edge, prev, **kwargs):
        if edge.id not in self.seen:
            edge = self._fn(edge, **kwargs)
            self.seen.add(edge.id)
        return edge, prev


class PropsNode(BasePropogator):
    def __init__(self, fn, **kwargs):
        super(PropsNode, self).__init__(fn.__name__, **kwargs)
        self.seen = set()
        self._fn = fn

    def on_default(self, node, prev, **kwargs):
        if node.id not in self.seen:
            self.seen.add(node.id)
            node = self._fn(node, seen=self.seen, **kwargs)

        return node, prev

# -------------------------------------------
import src.geom


def fix_mesh_fn(edge, **kwargs):
    """
    Final repair step.
    """
    if edge.has_geom is True:
        obj = edge.obj.as_mesh()
        obj_clean = src.geom.remove_dangles(obj)
        edge._solid = obj_clean.convex_hull_ms

    return edge


def remove_empty_fn(edge, **kwargs):
    """
    ptnode1---fknode1===fknode2---ptnode2
    ptnode1--------midnode--------ptnode2
    """
    if edge.has_geom is False:  # and
        src_ = edge.source
        tgt_ = edge.target
        midpt = tuple(np.array([src_.geom, tgt_.geom]).mean(axis=0).tolist())
        new = Node(midpt)

        # sucs = tgt_.successors(edges=True)
        # sucs = tgt_.successors(edges=True)

        for out_edge in tgt_.successors(edges=True):
            out_edge.reverse()
            out_edge.reconnect(new)
            out_edge.reverse()
            edge = out_edge
        for in_edge in src_.predecessors(edges=True):
            in_edge.reconnect(new)
            edge = in_edge
        return edge
    return edge


def remove_empty_ndfn(node, **kwargs):
    """
    ptnode1---fknode1===fknode2---ptnode2
    ptnode1--------midnode--------ptnode2

    Handle everything by Giving EdgeSolid types:
        - Geometic         has a MeshSOlid
        - Temp             will be deleted
        - Non-Geometric    must stay but have not geometry - eg duct taps

    if it as 'tap' node aka midnode has many succers,
    need to find the most likely one to do connection to (nearest -face-face distance)
    """
    edges = node.successors(edges=True)
    # select the best face-to-face geometric
    for suc_edge in edges:
        if suc_edge.has_geom is False:
            tgt_ = suc_edge.target
            src_ = suc_edge.source
            midpt = tuple(np.array([src_.geom, tgt_.geom]).mean(axis=0).tolist())
            new = Node(midpt)

            return node
    return node


def print_func_fn(edge, properties=['conf'], **kwargs):
    if edge.has_geom is True:
        print([edge.get('conf') for p in properties])
    return edge


def print_geom_wfn(edge, **kwargs):
    if edge.has_geom is True:
        print(edge.source.geom, edge.target.geom)
    return edge


def edge_fix_fn(edge, **kwargs):
    if edge.has_geom is True:
        solid_mesh = edge.obj
        nodes = [edge.source, edge.target]
        sources = [edge.source.as_np, edge.target.as_np]
        # target = edge.target.as_np
        _, _, tri = solid_mesh.nearest.on_surface(sources)
        # px = solid_mesh.faces[tri]
        for ix, i in enumerate(tri):
            ixs, locs = np.where(solid_mesh.face_adjacency == i)
            # print(b2.face_normals[4])
            fn = solid_mesh.face_normals[solid_mesh.face_adjacency[ixs, np.abs(locs - 1)]]
            dif = fn - solid_mesh.face_normals[i]
            print(dif)
            nearest_norm = np.argmin(np.abs(dif).mean(axis=1))
            corr_face = np.unique(solid_mesh.faces[[nearest_norm, i]])
            mean = np.mean(solid_mesh.vertices[corr_face], axis=0)
            nodes[ix].update_geom(tuple(mean.tolist()))
        print(tri)



    return edge


# Edges -------------------------------------------
detail_mesh = PropsEdge(fix_mesh_fn)
remove_empty = PropsEdge(remove_empty_fn)
print_func = PropsEdge(print_func_fn)
print_geom = PropsEdge(print_geom_wfn)
# Nodes -------------------------------------------
remove_empty_nd = PropsNode(remove_empty_ndfn)
# ----------------------------------
reset_centroids = PropsEdge(edge_fix_fn)