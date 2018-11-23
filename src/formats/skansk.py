import numpy as np
import lib.geo as geo
import src.structs
import src.propogate as gp
import math, src.geom
import importlib
importlib.reload(gp)
from scipy.spatial import kdtree, distance
import src.propogate.base
importlib.reload(src.propogate.base)
from src.propogate.base import QueuePropogator
import itertools
from lib.geo import Point, Line

importlib.reload(src.structs)
from src.structs import Node, Edge
import src.structs.node_utils as gutil
from enum import Enum
from shapely.ops import nearest_points
from src.formats.revit import TeeCmd, ElbowCmd, CouplingCmd, Tee

_ngh_arg = dict(fwd=True, bkwd=True, edges=True)


def connect_by_pairs(cylinders, factor=0.3):
    """ returns points and loose connectivity """
    pts_ind = np.concatenate([x.line.numpy for x in cylinders])
    tree = kdtree.KDTree(pts_ind)
    pairs = tree.query_pairs(r=factor)
    return pairs, tree, pts_ind


def _other_ix(ix):
    """if even, check for odd + 1"""
    return ix + 1 if ix % 2 == 0 else ix - 1


def _process_index(node_dict, pts_index, ix, cylinder):
    if ix not in node_dict:
        node_dict[ix] = Node(gutil.tuplify(pts_index[ix]), pt_index=ix)
    other = _other_ix(ix)
    if other not in node_dict:
        node_dict[other] = Node(gutil.tuplify(pts_index[other]), pt_index=other)
    node_dict[ix].connect_to(node_dict[other], is_pipe=True, radius=cylinder.radius)
    return node_dict


def make_node_dict(pairs, pts_index, cylinders):
    """

    :param pairs:
    :param pts_index:
    :param cylinders:
    :return:
    """
    node_dict = dict()
    for ix11, ix21 in pairs:
        node_dict = _process_index(node_dict, pts_index, ix11, cylinders[ix11//2])
        node_dict = _process_index(node_dict, pts_index, ix21, cylinders[ix21//2])

        # determine which ends of pipes to connect.
        # any 2 pipes can only connect once
        ix12 = _other_ix(ix11)
        ix22 = _other_ix(ix21)
        if node_dict[ix12] not in node_dict[ix22].neighbors(True, True) \
           and node_dict[ix11] not in node_dict[ix22].neighbors(True, True) \
           and node_dict[ix11] not in node_dict[ix21].neighbors(True, True) \
           and node_dict[ix12] not in node_dict[ix21].neighbors(True, True):

            ir1 = np.array([ix11, ix12])
            ir2 = np.array([ix21, ix22])
            dst = distance.cdist(pts_index[ir1], pts_index[ir2])
            k1, k2 = np.unravel_index(np.argmin(dst, axis=None), dst.shape)
            node_dict[ir1[k1]].connect_to(node_dict[ir2[k2]], is_pipe=False)
    return node_dict


def connected_components(nodes, min_size=None):
    """
    Given a list of nodes finds the graphs that are within that list

    Returns
    --------
    list(Node) of graphs larger than min_size.
    if min_size is None, returns all subgraphs
    """
    notseen = {n.id for n in nodes}
    seen = set()
    cur = 0
    comps = []
    while len(seen) < len(nodes):
        cnt = 0
        for n in nodes[cur].__iter__(fwd=True, bkwd=True):
            seen.add(n.id)
            if n.id in notseen:
                cnt += 1
                notseen.remove(n.id)
        comps.append([cur, cnt])
        nexts = list(notseen.difference(seen))
        if len(nexts) > 1:
            cur = [i for i, x in enumerate(nodes) if x.id == nexts[0]][0]
        else:
            break

    if min_size is not None:
        comps = list(filter(lambda x: x[1] > min_size, comps))
    seen2 = set()
    comps_start = []
    final_list = []
    for k, v in comps:
        comps_start.append(len(final_list))
        for n in nodes[k].__iter__(fwd=True, bkwd=True):
            if n.id not in seen2:
                seen2.add(n.id)
                final_list.append(n)

    comps_starts = np.asarray(comps_start)
    root_nodes = [final_list[i] for i in comps_starts]
    return root_nodes


def direction_best(edge1, edge2):
    crv1 = edge1.curve if isinstance(edge1, Edge) else edge1
    crv2 = edge2.curve if isinstance(edge2, Edge) else edge2
    return np.min([crv1.direction - crv2.direction,
                   -1*crv1.direction, crv2.direction])


def similar_dir_abs(edge1, edge2, tol=1e-2):
    crv1 = edge1.curve if isinstance(edge1, Edge) else edge1
    crv2 = edge2.curve if isinstance(edge2, Edge) else edge2
    return np.allclose(crv1.direction, crv2.direction, atol=tol) or \
            np.allclose(-1*crv1.direction, crv2.direction, atol=tol)


def _is_pipe(edge):
    return edge.get('is_pipe', None) is True


def edge_iter(nd, fn=None):
    seen = set()
    for n in nd.__iter__(fwd=True, bkwd=True):
        edges = n.neighbors(**_ngh_arg)
        edges.sort(key=lambda x: x.curve.length)
        for e in edges:
            if e.id not in seen:
                seen.add(e.id)
                if fn is not None:
                    if fn(e):
                        yield e
                else:
                    yield e


def edges_iter(edge):
    seen = set()
    e1 = edge_iter(edge.target, _is_pipe)
    e2 = edge_iter(edge.source, _is_pipe)
    its = [e1, e2]
    while any(its):
        for it in its:
            v = next(it)
            if v.id not in seen:
                seen.add(v.id)
                yield v


def take_n(iter, n):
    return [x for _, x in zip(range(n), iter)]


def neighbor_pipes(edge, n):
    return [x for _, x in zip(range(n), edges_iter(edge))]


_REVIT_ANGLE = 89.0
_ELBOW_COUPLING_TOL = 5.0


# SKS specific --------------------------------------------------------------------
class DetectElbows(QueuePropogator):
    def __init__(self, **kwargs):
        super(DetectElbows, self).__init__(fwd=True, edges=True, **kwargs)
        self._res = set()

    def on_default(self, edge, **kwargs):
        """
            -edge cannot be a pipe
            -one edge is a pipe, other is not
            -the edge that is not a pipe can have only one neighbor (NPN)
            - NPN Must
            """
        if _is_pipe(edge) is True:
            return edge
        src = edge.source
        tgt = edge.target
        srcs = [x for x in src.neighbors(True, True, True) if x.id != edge.id]
        tgts = [x for x in tgt.neighbors(True, True, True) if x.id != edge.id]
        if len(srcs) != 1 or len(tgts) != 1:
            return edge
        if _is_pipe(srcs[0]) is False or _is_pipe(tgts[0]) is False:
            return edge
        if not similar_dir_abs(tgts[0], srcs[0]):
            edge.write('is_elbow', True)
            self._res.add(edge.id)

        return edge


class ResolveElbowEdge(QueuePropogator):
    def __init__(self):
        QueuePropogator.__init__(self, fwd=True, edges=True)

    def on_default(self, edge, **kwargs):
        """
        [src]
        ======+
              \ [edge]
              +
             || [tgt_edge]
              +
              \ [neigh1]
              +============ [pipe]

        [src]
        ======+
              \ [edge]
              +
             || [tgt_edge]
              +============ [pipe]
        """
        if edge.get('is_elbow', None) is not True:
            return edge
        src = edge.source
        tgt = edge.target
        srcs = [x for x in src.neighbors(fwd=False, edges=True, bkwd=True) ]
        tgts = [x for x in tgt.neighbors(edges=True) ]
        if len(srcs) != 1 or len(tgts) != 1:
            return edge
        src_edge, tgt_edge = srcs[0], tgts[0]
        angle = gutil.norm_angle(src_edge, tgt_edge)
        if angle > _REVIT_ANGLE:
            # elbow - try to abstract
            if tgt_edge.target.nsucs == 1:
                neigh1 = tgt_edge.target.neighbors(edges=True)[0]
                if not _is_pipe(neigh1):
                    pass
            pass
        return edge


def resolve_elbow_edge(edge):
    """
    [src]
    ======+
          \ [edge]
          +
         || [tgt_edge]
          +
          \ [neigh1]
          +============ [pipe]

    [src]
    ======+
          \ [edge]
          +
         || [tgt_edge]
          +============ [pipe]
    """
    if edge.get('is_elbow', None) is not True:
        return edge
    src = edge.source
    tgt = edge.target
    srcs = [x for x in src.neighbors(fwd=False, edges=True, bkwd=True)]
    tgts = [x for x in tgt.neighbors(edges=True)]
    if len(srcs) != 1 or len(tgts) != 1:
        return edge
    src_edge, tgt_edge = srcs[0], tgts[0]
    angle = gutil.norm_angle(src_edge, tgt_edge)
    if angle > _REVIT_ANGLE:
        # elbow - try to abstract
        if tgt_edge.target.nsucs == 1:
            neigh1 = tgt_edge.target.neighbors(edges=True)[0]
            if not _is_pipe(neigh1):
                pass
        pass


def _label_one_one(node, tol=_ELBOW_COUPLING_TOL):
    pred = node.predecessors(edges=True)[0]
    succ = node.successors(edges=True)[0]
    angle = gutil.norm_angle(pred, succ)
    if angle > tol:
        node.write('is_elbow', True)
    else:
        node.write('is_coupling', True)
    return node


def _label_one_two(node, tol=_ELBOW_COUPLING_TOL):
    pred = node.predecessors(edges=True)[0]
    suc1, suc2 = node.successors(edges=True)
    angle1 = gutil.norm_angle(pred, suc1)
    angle2 = gutil.norm_angle(pred, suc2)
    if angle1 < angle2:
        suc2.write('tap_edge', True)
    else:
        suc1.write('tap_edge', True)
    return node


class LabelConnector(QueuePropogator):
    def __init__(self):
        super(LabelConnector, self).__init__()

    def on_default(self, node, tol=_ELBOW_COUPLING_TOL, **kwargs):
        if node.npred == 1 and node.nsucs == 1:
            return _label_one_one(node, tol)
        elif node.npred == 1 and node.nsucs == 2:
            return _label_one_two(node, tol)
        return node


class DetectTriangles(QueuePropogator):
    """
    premature optimization is the root of all evil.
        - Donald Knuth
    """
    def __init__(self, build=False, **kwargs):
        super(DetectTriangles, self).__init__(fwd=True, bkwd=True, **kwargs)
        self._res = set()
        self._build = build

    def on_default(self, node, **kwargs):
        """
                ______________
               /              |
        --eP--n]=[0---P---1]=[2---P-- => NO


                 -(0)--P--(..)
        --eP--(n)   |            =>  YES
                \   |
                 +-(1)--P--(..)

        # todo - wrap this in an a grapgpropogator.
        # merge the triangle into a point(with label branch)
        # return the new node
        :param node:
        :param edge:
        :return:
        """
        # srcs = [x for x in node.neighbors(True, True, True) if _is_pipe(x) is False]

        for edge in node.neighbors(fwd=True, bkwd=True, edges=True):
            if _is_pipe(edge) is False:
                continue
            srcs = [x for x in node.neighbors(fwd=True, bkwd=True, edges=True)
                    if x.id != edge.id and _is_pipe(x) is False]
            if len(srcs) == 2:
                ng1, ng2 = [x.other_end(node) for x in srcs]
                e1 = gutil.edge_between(ng1, ng2)

                if e1 is not None and _is_pipe(e1) is False:
                    items = [node, ng1, ng2]
                    for itm in items:
                        if itm in self.q:
                            self.q.remove(itm)
                        if itm.id in self.seen:
                            self.seen.remove(itm.id)

                    node = resolve_triangle3(items)
                    self._res.add(node.id)
                    return node
        return node


# functions ------------------------------------
def remove_short(node, lim=1/24, **kwargs):
    """
           [suc_edge]   [edg]
    --> (node)->(suc_tgt)---->(nexts)

    ------>(node[mid])------->(nexts)
    """
    preds = node.predecessors(edges=True)
    data = preds[0].tmps if len(preds) > 0 else {}
    sucs = node.successors(both=True)
    for (suc_edge, suc_tgt) in sucs:
        if suc_edge.curve.length < lim:
            nexts = suc_tgt.successors(both=True)
            for (edg, nd) in nexts:
                edg.delete()
                node.connect_to(nd, **{**data, **edg.tmps})
            suc_edge.delete()
    return node


def add_ups(node, h=0.1, **kwargs):
    if node.get('has_head', None) is True:
        nd = np.array(list(node.geom))
        nd[-1] += h
        new = Node(gutil.tuplify(nd), is_head=True)
        node.connect_to(new, remove_head=True, tap_edge=True)
        node.write('has_head', False)
        node.write('head_tee', True)
        new.write('$create', '$head')
    return node


vert = lambda x, tol: 1 - np.abs(gutil.slope(x)) < tol


def is_vertical(edge, tol=0.005):
    if _is_pipe(edge) is True:
        slope = 1 - np.abs(gutil.slope(edge))
        if slope < tol:
            edge.write('is_vertical', True)
    return edge


def align_vertical(edge, tol=0.01):
    if _is_pipe(edge) is True:
        slope = 1 - np.abs(gutil.slope(edge))
        if slope < tol:
            src = edge.source.as_np
            tgt = edge.target.as_np
            tgt[0:2] = src[0:2]
            edge.target.geom = gutil.tuplify(tgt)
    return edge


def align_tap(edge, TOL=_REVIT_ANGLE):
    """

    :param edge:
    :param TOL:HARDCODED REVIT TOLERANCE. found empirically
     working: [89.22843573127959, 89.91857497862868, 89.22841701199027, 89.93205755597465, 89.2284428900399]
     not working: [88.89186491151713, 88.84667449752507, 88.89179632626258, 88.8919083924853, 88.89182755566546]
    :return:
    """
    if edge.get('tap_edge', None) is not True or edge.source.npred != 1:
        return edge
    egde_in = edge.source.predecessors(edges=True, ix=0)
    angle = math.degrees(egde_in.curve.line.angle_to(edge.curve.line))
    if angle < TOL:
        src = edge.source.as_np
        tgt = edge.target.as_np
        tgt[0:2] = src[0:2]
        edge.target.geom = gutil.tuplify(tgt)
        return edge
    return edge


def resolve_triangle3(tri_nodes):
    """
    this node will be moved to triangle center
    1). tri_nodes = get all triangle nodes
    2). get edges which are connected to triangle and are pipes
    3). COL = the two with closest direction
    3). the tri_edges
    4). pt = edge between COL.centroid
    5). move node to pt,
    6). connect tri_nodes to it

    :param tri_edges:
    :return:
    """
    outer_edge_ids = set()
    tri_edges, outer_edges, outer_dirs = [], [], []
    for n1, n2 in itertools.combinations(tri_nodes, 2):
        edge = gutil.edge_between(n1, n2)
        if edge not in tri_edges:
            tri_edges.append(edge)

    # 1). get edges which are connected to triangle and are pipes
    for n in tri_nodes:
        for edge in n.neighbors(fwd=True, bkwd=True, edges=True):
            if edge not in tri_edges and edge.id not in outer_edge_ids:
                outer_edges.append((edge, n))
                outer_edge_ids.add(edge.id)

    # 2). get edges which are connected to triangle and are pipes
    outer_edges_same_dir, nodes_on_main, best = [], [], 1e6
    for (e1, n1), (e2, n2) in itertools.combinations(outer_edges, 2):
        siml = np.abs(direction_best(e1.curve, e2.curve))   # + np.abs(n1.geom[-1] - n2.geom[-1])
        if siml < best:
            outer_edges_same_dir, nodes_on_main, best = [e1, e2], [n1, n2], siml

    # 3). the tri_edge between edges with same direction
    assert len(tri_nodes) == 3, 'expected 3, got {}'.format(len(tri_nodes))

    # 4). pt = edge between COL.centroid, Get new point location
    main1, main2 = nodes_on_main[0], nodes_on_main[1]
    new_pnt = geo.Point(main1.geom).midpoint_to(geo.Point(main2.geom))
    tap_node = [n for n in tri_nodes if n.id not in [main1.id, main2.id]][0]

    # get first node in order
    order = [x.get('order') for x in tri_nodes]
    node = tri_nodes[int(np.argmin(order))]

    for edge in tri_edges:
        edge.delete()

    if node == tap_node:
        node = tap_is_input(node, main1, main2, new_pnt)
    else:
        node_out2 = main2 if main1.id == node.id else main1
        node = tap_is_other(node, tap_node, node_out2, new_pnt)

    return node


def tap_is_input(node_in_tap, node_out1, node_out2, new_pnt):
    tap_edge, pred_node = node_in_tap.predecessors(both=True)[0]
    pt_tuple = gutil.tuplify(new_pnt.numpy)

    if tap_edge.curve.line.length < 1.:
        node_in_tap.geom = pt_tuple
        node_in = node_in_tap
        tap_edge.write('tap_edge', True)
    else:
        node_in = Node(pt_tuple, tee_index=0, tee_mid=True)
        node_in_tap.connect_to(node_in, **tap_edge.tmps, tap_edge=True)

    node_in.write('tap_input', False)
    node_in.write('is_tee', True)
    for n in [node_out1, node_out2]:
        o_edge, o_succ = n.successors(both=True)[0]
        node_in.connect_to(o_succ, **o_edge.tmps)
        o_edge.delete()
    return node_in


def tap_is_other(node_in, tap_node, node_out2, new_pnt):
    node_in.geom = gutil.tuplify(new_pnt.numpy)
    node_in.write('is_tee', True)
    node_in.write('tap_input', True)
    tap_edge, tap_succ = tap_node.successors(both=True)[0]

    if tap_edge.curve.line.length < 1.:
        node_in.connect_to(tap_succ, **tap_edge.tmps, tap_edge=True)
        tap_edge.delete()
    else:
        node_in.connect_to(tap_node, **tap_edge.tmps, tap_edge=True)

    o_edge, o_succ = node_out2.successors(both=True)[0]
    node_in.connect_to(o_succ, **o_edge.tmps)
    o_edge.delete()
    return node_in


# -- todo ------------------------------------------------------
def kdconnect(kdindexes):
    """"""
    return


def resolve_elbow_edges(node, lim=2/12, **kwargs):
    """
    [inn_edge]   [node]
    =============>+
                  \ [edge]
                  + [tgt]
                  ||
                  || [out_edge]
                  ||
                  +  [new_tgt]
    """
    if node.npred != 1 or node.nsucs != 1:
        return node

    edge, tgt = node.successors(both=True, ix=0)
    if edge.get('is_elbow', None) is True:
        inn_edge = node.predecessors(edges=True, ix=0)
        out_edge, out_node = edge.target.successors(both=True, ix=0)
        bl1 = inn_edge.curve.line
        bl2 = out_edge.curve.line
        inn_line = inn_edge.curve.extend(10, 10).line
        out_line = out_edge.curve.extend(10, 10).line
        pp1 = bl1.points[1].projected_on(out_line)
        pp2 = bl2.points[0].projected_on(inn_line)
        dst = pp1.distance_to(pp2)
        # print(node.id, dst)
        if dst < lim:
            # new_tgt = out_edge.target
            node.geom = gutil.tuplify(pp1.midpoint_to(pp2).numpy)
            node.connect_to(out_node, **out_edge.tmps)
            # edge.disconnect()
            tgt.deref()
            edge.delete()
            out_edge.delete()
            return node
    return node


# sks testing specific ------------------------------------------
class Annotator(QueuePropogator):
    """
        if node has propoerty k in mapping, then
        write value v to node with ket self.var
    Usages:
        rp.Annotator('$create', mapping={'dHead': 1, }),

    """
    def __init__(self, var, mapping={}, **kwargs):
        self.mapping = mapping
        self.var = var
        self._pos = 0
        super(Annotator, self).__init__(**kwargs)

    def on_default(self, node, **kwargs):
        for k, v in self.mapping.items():
            if node.get(k, None) is True:
                self._pos += 1
                node.write(self.var, v)
        return node


class LongestPath(QueuePropogator):
    """ Disconnect everything that is not on longets path """
    def __init__(self, **kwargs):
        QueuePropogator.__init__(self)

    def on_next(self, node):
        sucs = node.successors()
        if len(sucs) > 1:

            score, bestix = -1, -1
            for i, suc in enumerate(sucs):
                this_score = len(list(suc.__iter__(fwd=True)))
                if this_score > score:
                    score, bestix = this_score, i
                node.disconnect_from(suc)

            keep = sucs[bestix]
            node.connect_to(keep)
            return [keep]
        else:
            return sucs



class GatherTags(QueuePropogator):
    def __init__(self):
        super(GatherTags, self).__init__()
        self._res = set()

    def on_default(self, node, **kwargs):
        self._res.update(node.tmps.keys())
        return node


# BUILDING GRAPHS -----------------------------------------------
import src.formats.revit
importlib.reload(src.formats.revit)
from src.formats.revit import Command, Cmds, PipeCmd


_RevitFamily = {
    'is_tee': 'Tee - Generic',
    'is_elbow': 'Elbow - Generic',
    'is_coupling': 'Coupling - Generic'
}


def tee_nodes(node):
    return Tee.tee_nodes(node)


def tee_edges(node):
    return Tee.tee_edges(node)


is_tee = lambda node: node.get('is_tee', None) is True or node.get('head_tee', None) is True
is_elbow = lambda node: node.get('is_elbow', None) is True
is_coupling = lambda node: node.get('is_coupling', None) is True
is_head_tee = lambda node: node.get('head_tee', None) is True


class MakeInstructions(QueuePropogator):
    """
    walker version of 'SystemProcesser'

    instead of geom, returns a list of instructions.

    in order to ensure that geometry exists building has separate
    bookkeeping from iteration,

    """
    final_order = [Cmds.Noop]

    def __init__(self, **kwargs):
        QueuePropogator.__init__(self, **kwargs)
        self._built = set()

    def reset(self):
        self._res = []
        self._built = set()
        self.seen = set()

    def add(self, cmd, *data, **kwargs):
        if cmd == Cmds.Pipe and data[1].id in self._built:
            return
        else:
            self._res += Command.create(cmd, *data, **kwargs)
            for e in data:
                # ensure geometries are built
                if isinstance(e, Edge) and e.id not in self._built:
                    self._built.add(e.id)

    def on_default(self, node, shrink=None, **kwargs):
        """
        Generate instructions and add them to

        """
        preds = node.predecessors(edges=True)
        pred = preds[0] if node.npred == 1 else None

        if node.get('head_tee', None) is True:
            if node.nsucs == 2:
                suc_edge1, suc_edge2 = node.successors(edges=True)
                if suc_edge2.get('tap_edge', None) is True:
                    main_out, tee_edge = suc_edge1, suc_edge2
                elif suc_edge1.get('tap_edge', None) is True:
                    main_out, tee_edge = suc_edge2, suc_edge1
                else:
                    return node

                if pred is not None:
                    # create successors
                    self.add(Cmds.Pipe, pred, shrink=shrink)
                    self.add(Cmds.Pipe, main_out, shrink=shrink)
                    self.add(Cmds.Pipe, tee_edge, shrink=shrink)

                    # connect tee, delete top, create on face
                    self.add(Cmds.Tee, pred, main_out, tee_edge)
                    # self.add(Cmds.Delete, tee_edge)
                    self.add(Cmds.FamilyOnFace, tee_edge, 1, 0)
                    # self.add(Cmds.FamilyOnFace, node, 2, 0)
            return node

        elif node.get('is_coupling', None) is True:
            if pred is None or node.nsucs != 1:
                return node
            succ = node.successors(edges=True)[0]
            self.add(Cmds.Pipe, succ)
            self.add(Cmds.Coupling, pred, succ)
            return node

        elif node.get('is_tee', None) is True:
            suc1, suc2 = node.successors(edges=True)
            self.add(Cmds.Pipe, suc1, shrink=shrink)
            self.add(Cmds.Pipe, suc2, shrink=shrink)
            edgein, edgemain, edge_tap = tee_edges(node)
            if edgein is None:
                return node
            self.add(Cmds.Tee, edgein, edgemain, edge_tap)
            return node

        elif node.get('is_elbow', None) is True:
            tgt = node.successors(edges=True)[0]
            self.add(Cmds.Pipe, tgt, shrink=shrink)
            self.add(Cmds.Elbow, pred, tgt)
            return node

        # edge specific
        for (edge, suc_node) in node.successors(both=True):

            if edge.id in self._built:
                continue

            elif edge.get('is_pipe', None) is True:
                self.add(Cmds.Pipe, edge, shrink=shrink)

            elif edge.get('is_elbow', None) is True:
                src = edge.source.predecessors(edges=True)[0]
                tgt = edge.target.successors(edges=True)[0]
                self.add(Cmds.Pipe, edge, shrink=shrink)
                self.add(Cmds.Pipe, tgt, shrink=shrink)
                self.add(Cmds.Elbow, src, edge)
                self.add(Cmds.Elbow, edge, tgt)

        return node

    def _on_complete_ovr(self, command_list):
        res = []
        cmd_vls = [x.value for x in self.final_order]

        rest_ix = cmd_vls.index(0)
        arrs = [[] for i in range(len(self.final_order))]

        for cmd in command_list:
            arr_ix = cmd_vls.index(cmd[0]) if cmd[0] in cmd_vls else rest_ix
            arrs[arr_ix].append(cmd)

        for arr in arrs:
            res += arr
        return res

    def on_complete(self, node):
        return self._on_complete_ovr(self._res)


class MakeInstructionsNodeBased(MakeInstructions):
    """
    Create Fittings then connect Pipes between them

    """
    final_order = [Cmds.Noop, Cmds.Pipe, Cmds.FamilyOnFace, Cmds.MoveEnd, Cmds.Coupling]

    def __init__(self, **kwargs):
        super(MakeInstructions, self).__init__(**kwargs)
        self._built = set()

    def on_default(self, node, shrink=None, **kwargs):
        """
        Generate instructions
        """
        for e, nd in node.successors(both=True):
            if nd.nsucs == 0:
                self.add(Cmds.Pipe, PipeCmd.ConnectorPoint, e, shrink=shrink)
            elif nd.get('is_coupling', None) is True:
                nxt_egde, nxt_node = nd.successors(both=True, ix=0)
                self.add(Cmds.Pipe, PipeCmd.ConnectorPoint, e, shrink=shrink)
                self.add(Cmds.Pipe, PipeCmd.PointConnector, nxt_egde, shrink=shrink)
            else:
                self.add(Cmds.Pipe, PipeCmd.Connectors, e)

        if node.get('head_tee', None) is True:
            n1, n2, edge_tee = tee_nodes(node)
            if n1 is None:
                return node
            self.add(Cmds.FamilyOnPoint, n1, node, n2, family='Tee - Generic')
            self.add(Cmds.FamilyOnFace, edge_tee, 1, 0)

        elif node.get('is_coupling', None) is True:
            p, t = node.predecessors(ix=0, edges=True), node.successors(ix=0, edges=True)
            self.add(Cmds.Coupling, p, t, family='Coupling - Generic')

        elif node.get('is_tee', None) is True:
            n1, n2, _ = tee_nodes(node)
            self.add(Cmds.FamilyOnPoint, n1, node, n2, family='Tee - Generic')

        elif node.get('is_elbow', None) is True:
            p, t = node.predecessors(ix=0), node.successors(ix=0)
            self.add(Cmds.FamilyOnPoint, p, node, t, family='Elbow - Generic')
        return node


class MakeInstructionsV3(MakeInstructions):
    final_order = [Cmds.FamilyOnPoint, Cmds.Pipe, Cmds.Noop]

    def __init__(self, **kwargs):
        super(MakeInstructions, self).__init__(**kwargs)
        self._built = set()
        self._backups = {}

    def on_default(self, node, shrink=None, **kwargs):
        """
        Generate instructions
        """
        for succ_edg, succ_nd in node.successors(both=True):
            eid = succ_edg.id
            if is_tee(node) and is_tee(succ_nd):
                self.add(Cmds.Pipe, PipeCmd.Connectors, succ_edg)

            elif is_tee(node) and not is_tee(succ_nd):
                self.add(Cmds.Pipe, PipeCmd.ConnectorPoint, succ_edg, shrink=shrink)
                self.add(Cmds.Connect, succ_edg, succ_nd)

            elif not is_tee(node) and is_tee(succ_nd):
                self.add(Cmds.Pipe, PipeCmd.PointConnector, succ_edg, shrink=shrink)
                self.add(Cmds.Connect, succ_edg, node)
            else:
                self.add(Cmds.Pipe, PipeCmd.Points, succ_edg, shrink=shrink)

            # ends of pipe
            if succ_nd.nsucs == 0:
                self.add(Cmds.Pipe, PipeCmd.ConnectorPoint, succ_edg, shrink=shrink)

        if is_head_tee(node):
            n1, n2, edge_tee = tee_nodes(node)
            if n1 is None:
                return node
            self.add(Cmds.FamilyOnPoint, n1, node, n2, family='Tee - Generic')
            self.add(Cmds.FamilyOnFace, edge_tee, 1, 0)

        elif is_coupling(node):
            p, t = node.predecessors(ix=0, edges=True), node.successors(ix=0, edges=True)
            self.add(Cmds.Coupling, p, t)

        elif is_tee(node):
            n1, n2, _ = tee_nodes(node)
            self.add(Cmds.FamilyOnPoint, n1, node, n2, family='Tee - Generic')

        elif is_elbow(node):
            pe, pn = node.predecessors(ix=0, both=True)
            te, tn = node.successors(ix=0, both=True)
            line1 = geo.Line(geo.Point(node.as_np), geo.Point(pn.as_np))
            line2 = geo.Line(geo.Point(node.as_np), geo.Point(tn.as_np))
            if np.dot(line1.unit_vector, line2.unit_vector) > 0:
                self.add(Cmds.Elbow, pe, te)
            else:
                self.add(Cmds.FamilyOnPoint, pn, node, tn, family='Elbow - Generic')

        return node

    def on_complete2(self):
        main = self._on_complete_ovr(self._res)
        return {'main': main, 'backup': self._backups}


def make_actions(node, shrink=None, **kwargs):
    """
            Generate instructions
            """
    # create = Command.create
    create = Command.action
    res = []
    for succ_edg, succ_nd in node.successors(both=True):
        succ_edg.write('$create', 'pipe')
        if is_tee(node) and is_tee(succ_nd):
            res += create(PipeCmd.Connectors, succ_edg)

        elif is_tee(node) and not is_tee(succ_nd):
            res += create(PipeCmd.ConnectorPoint, succ_edg, shrink=shrink)
            res += create(Cmds.Connect, succ_edg, succ_nd)

        elif not is_tee(node) and is_tee(succ_nd):
            res += create(PipeCmd.PointConnector, succ_edg, shrink=shrink)
            res += create(Cmds.Connect, succ_edg, node)
        else:
            res += create(PipeCmd.Points, succ_edg, shrink=shrink)

        # ends of pipe
        if succ_nd.nsucs == 0:
            res += create(PipeCmd.ConnectorPoint, succ_edg, shrink=shrink)

    if is_head_tee(node):
        node.write('$create', 'tee')
        n1, n2, edge_tee = tee_nodes(node)
        if n1 is None:
            return res
        res += create(Cmds.FamilyOnPoint, n1, node, n2, family='Tee - Generic')
        res += create(Cmds.FamilyOnFace, edge_tee, 1, 0)

    elif is_coupling(node):
        node.write('$create', 'coupling')
        p, t = node.predecessors(ix=0, edges=True), node.successors(ix=0, edges=True)
        res += create(Cmds.Coupling, p, t)

    elif is_tee(node):
        n1, n2, _ = tee_nodes(node)
        node.write('$create', 'tee')
        res += create(TeeCmd.FamilyOnPoint, n1, node, n2, family='Tee - Generic')

    elif is_elbow(node):
        node.write('$create', 'elbow')
        res += create(ElbowCmd.Connectors, node)

    return res

# ------------------------------------------------------------
def resolve_heads(root_node, spr_points, tol=1.):
    kdprop = gp.KDTreeIndex()
    _ = kdprop(root_node)
    data = np.array(kdprop._data)

    for six, spr in enumerate(spr_points):
        cdist = distance.cdist([spr], data)[0]
        if np.min(cdist) < tol:
            best_ix = np.argmin(cdist)
            nd = kdprop[best_ix]
            nde = gutil.node_with_id(root_node, nd.id)
            connect_heads2(nde)
    return kdprop


def connect_heads2(node):
    neighs = node.neighbors(fwd=True, bkwd=True, edges=True)
    if node.get('has_head') is True:
        return  # node, False

    elif len(neighs) == 1:
        node.write('has_head', True)
        return  # node, True

    elif len(neighs) == 2:
        e1, e2 = neighs
        if _is_pipe(e1) is True and _is_pipe(e2) is False:
            edge, data = e2, e1.tmps
        elif _is_pipe(e1) is False and _is_pipe(e2) is True:
            edge, data = e1, e2.tmps
        else:
            return

        p1, p2 = edge.geom
        center = Point(p1).midpoint_to(Point(p2))
        new_main = edge.source

        new_main.geom = gutil.tuplify(center.numpy)
        new_main.remove_edge(edge)
        new_main.write('has_head', True)

        rem_node = edge.target
        for s in rem_node.successors():
            new_main.connect_to(s, **data)
        rem_node.deref()
        return

