import numpy as np
import lib.geo as geo
from src.structs import Node, Edge
import src.propogate as gp

import importlib
importlib.reload(gp)
from scipy.spatial import kdtree, distance
import src.propogate.base
importlib.reload(src.propogate.base)
from src.propogate.base import QueuePropogator, FunctionQ
import itertools
from collections import Counter
from lib.geo import Point, Line
_ngh_arg = dict(fwd=True, bkwd=True, edges=True)


def connect_by_pairs(cylinders, factor=0.3):
    pts_ind = np.concatenate([x.line.numpy for x in cylinders])
    tree = kdtree.KDTree(pts_ind)

    pairs = tree.query_pairs(r=factor)

    return pairs, tree, pts_ind


def tuplify(np_pt):
    return tuple(np_pt.tolist())


def _other_ix(ix):
    if ix % 2 == 0:  # if even, check for odd + 1
        return ix + 1
    else:
        return ix - 1


def process_index(node_dict, pts_index, ix, cylinder):
    if ix not in node_dict:
        node_dict[ix] = Node(tuplify(pts_index[ix]), pt_index=ix)

    other = _other_ix(ix)

    if other not in node_dict:
        node_dict[other] = Node(tuplify(pts_index[other]), pt_index=other)
    node_dict[ix].connect_to(node_dict[other], is_pipe=True, radius=cylinder.radius)

    return node_dict


def make_node_dict(pairs, pts_index, cylinders):
    node_dict = dict()
    for ix11, ix21 in pairs:
        node_dict = process_index(node_dict, pts_index, ix11, cylinders[ix11//2])
        node_dict = process_index(node_dict, pts_index, ix21, cylinders[ix21//2])

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


def connected_components(nodes):
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

    final_comps = list(filter(lambda x: x[1] > 2, comps))
    seen2 = set()
    comps_start = []
    final_list = []
    for k, v in final_comps:
        comps_start.append(len(final_list))
        for n in nodes[k].__iter__(fwd=True, bkwd=True):
            if n.id not in seen2:
                seen2.add(n.id)
                final_list.append(n)
    return final_list, comps_start


def direction_best(edge1, edge2):
    crv1 = edge1.curve if isinstance(edge1, Edge) else edge1
    crv2 = edge2.curve if isinstance(edge2, Edge) else edge2
    return np.min([crv1.direction - crv2.direction,
                   -1*crv1.direction, crv2.direction] )


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


def edge_between(n1, n2):
    e1 = n1.edge_to(n2)
    if e1 is not None:
        return e1
    return n2.edge_to(n1)


def node_with_id(nd, eid):
    for n in nd.__iter__(fwd=True, bkwd=True):
        if n.id == eid:
            return n


# ------------------------------------------------
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
        if len(srcs) != 1 and len(tgts) != 1:
            return edge
        if _is_pipe(srcs[0]) is False or _is_pipe(tgts[0]) is False:
            return edge
        if not similar_dir_abs(tgts[0], srcs[0]):
            edge.write('is_elbow', True)
            self._res.add(edge.id)
        return edge


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
        for edge in node.neighbors(fwd=True, bkwd=True, edges=True):
            if _is_pipe(edge) is False:
                return node
            srcs = [x for x in node.neighbors(fwd=True, bkwd=True, edges=True)
                    if x.id != edge.id and _is_pipe(x) is False]
            if len(srcs) == 2:
                ng1, ng2 = [x.other_end(node) for x in srcs]
                e1 = edge_between(ng1, ng2)
                if e1 is not None and _is_pipe(e1) is False:
                    res = srcs + [e1]
                    if self._build is True:
                        node = resolve_triangle(res)
                        self._res.add(tuple(sorted([x.id for x in res])))
                    else:
                        for x in res:
                            x.write('is_triangle', [x.id for x in res])
                        self._res.add(tuple(sorted([x.id for x in res])))
        return node


def resolve_triangle(tri_edges, node=None):
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
    tri_ids = {x.id for x in tri_edges}
    tri_nodes, outer_edges = [], []
    outer_edge_ids = set()

    # 1). tri_nodes = get all triangle nodes
    # 2). get edges which are connected to triangle and are pipes
    for n in tri_edges:
        if n.source not in tri_nodes:
            tri_nodes.append(n.source)
            for en in n.source.neighbors(fwd=True, bkwd=True, edges=True):
                if en.id not in tri_ids and en.id not in outer_edge_ids:
                    outer_edges.append(en)
                    outer_edge_ids.add(en.id)
        if n.target not in tri_nodes:
            tri_nodes.append(n.target)
            for en in n.target.neighbors(fwd=True, bkwd=True, edges=True):
                if en.id not in tri_ids and en.id not in outer_edge_ids:
                    outer_edges.append(en)
                    outer_edge_ids.add(en.id)

    outer_edges_same_dir, best = [], 1e6
    # 3). COL = the two with closest direction
    for e1, e2 in itertools.combinations(outer_edges, 2):
        siml = np.abs(direction_best(e1.curve, e2.curve))
        if siml < best:
            outer_edges_same_dir, best = [e1, e2], siml

    # 3). the tri_edge between edges with same direction
    this_edge = None
    nds = itertools.chain(*[[x.source, x.target] for x in outer_edges_same_dir])
    for n1, n2 in itertools.combinations(nds, 2):
        this_edge = edge_between(n1, n2)
        if this_edge is not None and this_edge.id in tri_ids:
            break

    # 4). pt = edge between COL.centroid
    tn1, tn2 = this_edge.source, this_edge.target
    # node which is not on the main line
    other_node = [n for n in tri_nodes if n.id not in [tn1.id, tn2.id]]
    tap_point = geo.Point(other_node[0].geom)
    line = geo.Line(geo.Point(tn1.geom), geo.Point(tn2.geom))

    # 5) Get new point location
    new_pnt = tap_point.projected_on(line)
    if node is None:
        # source node - must be closest to origin than any other
        order = [x.get('order') for x in tri_nodes]
        node = tri_nodes[int(np.argmin(order))]

    # update the node
    node.update_geom(tuplify(new_pnt.numpy))
    node.write('is_tee', True)

    # line from other_node to new point - check if it should be a new drop
    tap_edge = [x for x in outer_edges if x not in outer_edges_same_dir][0]

    # disconnect everything
    for edge in tri_edges:
        node.remove_edge(edge)
        et, es = edge.target, edge.source
        if et is not None:
            et.remove_edge(edge)
        if es is not None:
            es.remove_edge(edge)

    # 6). connect tri_nodes to it
    for o_edge in outer_edges:
        line_arg = dict(is_pipe=True, radius=o_edge.get('radius'))
        osrc, otgt = o_edge.source, o_edge.target
        if osrc == node or otgt == node:
            continue

        osrc.remove_edge(o_edge)
        otgt.remove_edge(o_edge)
        if new_pnt.distance_to(Point(osrc.geom)) < new_pnt.distance_to(Point(otgt.geom)):
            tgt_cls, tgt_far = osrc, otgt
        else:
            tgt_cls, tgt_far = otgt, osrc

        if o_edge.id == tap_edge.id:
            line_to_tap = Line(new_pnt, Point(tgt_cls.geom))
            tap_dir = tap_edge.source.as_np
            dir_similar = similar_dir_abs(line_to_tap, tap_edge.curve, 1e-3)
            xy_close = np.allclose(tap_dir[0:2], new_pnt.numpy[0:2], 1e-2)

            # tgt_cls.connect_to(tgt_far, **line_arg)
            # node.connect_to(tgt_cls, **line_arg)

            if xy_close and dir_similar:
                # line pointing to projection point, just extend
                # node.connect_to(tgt_far, **line_arg)
                tgt_cls.connect_to(tgt_far, **line_arg)
                node.connect_to(tgt_cls, **line_arg)
            elif xy_close and not dir_similar:
                # direction is not similar. use closest node to
                tgt_cls.connect_to(tgt_far, **line_arg)
                node.connect_to(tgt_cls, **line_arg)

            elif not xy_close and dir_similar:
                # horizantal tap - extend is fine
                node.connect_to(tgt_far, **line_arg)

            else:
                # neither is close. Just make new line
                tgt_cls.connect_to(tgt_far, **line_arg)
                node.connect_to(tgt_far, **line_arg)

        else:
            node.connect_to(tgt_far, **line_arg)

    return node


def kdconnect(kdindexes):
    """"""
    return


def resolve_elbow(edge):
    # solve it in xy, then project z
    in_edge = edge.source.predecessors(edges=True)[0]
    out_edge = edge.target.successors(edges=True)[0]
    extend1 = in_edge.curve.extend(2.)
    extend2 = out_edge.curve.extend(2.)
    print(extend1.points, extend2.points)
    import shapely.ops as ops
    p1, p2 = ops.nearest_points(extend1, extend2)
    # z will
    print(list(p1.coords), list(p2.coords))
    return


class KDTreeIndex(QueuePropogator):
    def __init__(self, **kwargs):
        super(KDTreeIndex, self).__init__(**kwargs)
        self._data = []
        self._index = []
        self._root = None

    def get_node(self, nid):
        return node_with_id(self._root, nid)

    def __getitem__(self, index):
        node_id = self._index[index]
        return self.get_node(node_id)

    def nearest_point(self, other):
        """"""
        pass

    # waling interface ----------------
    def on_first(self, node, **kwargs):
        self._root = node
        if not isinstance(node, list):
            return [node]
        return node

    def on_default(self, node, **kwargs):
        self._data.append(list(node.geom))
        self._index.append(node.id)
        return node

    def on_complete(self, node):
        self._res = kdtree.KDTree(np.array(self._data))
        return self._res


class RemoveSubLoops(QueuePropogator):
    def __init__(self, **kwargs):
        super(RemoveSubLoops, self).__init__(fwd=True, **kwargs)

    def on_default(self, node, depth=5, **kwargs):
        # prev_data.append(node.id)
        seen_local = {node.id}
        depth_cnt = 0

        halt = {x.id for x in node.neighbors(**kwargs)}

        q = node.neighbors(**kwargs) + [None]
        print(node.id, len(q))

        while q and depth_cnt < depth:
            this_nd = q.pop(0)
            # print(this_nd)
            if this_nd is None:
                q.append(None)
                depth_cnt += 1
                print(depth_cnt)
                continue

            elif this_nd.id in halt and depth_cnt > 0:
                edge = edge_between(node, this_nd)
                if edge is None:
                    print('err', this_nd)

                elif edge.get('is_triangle', None) is None \
                        and _is_pipe(edge) is False:

                    node.remove_edge(edge)
                    this_nd.remove_edge(edge)
                    print('REMOVED', edge.id)

                else:
                    print('triangle')

            elif this_nd.id in seen_local:
                continue

            seen_local.add(this_nd.id)
            q += this_nd.neighbors()
        return node


class SpatialRoot(QueuePropogator):
    def __init__(self):
        super(SpatialRoot, self).__init__(bkwd=True)
        self.dist = 0
        self.best = None

    def on_default(self, node, **kwargs):
        if len(node.neighbors(fwd=True, bkwd=True)) == 1:
            mag = np.sum(np.array(node.geom) ** 2) ** 0.5
            if mag > self.dist:
                self.dist = mag
                self.best = node.id
        return node


class NearestTo(SpatialRoot):
    def __init__(self, loc, exclude=None):
        super(NearestTo, self).__init__()
        self._loc = np.array(loc)
        self._exclude = exclude

    def on_default(self, node, **kwargs):
        if node.id in self._exclude:
            return node
        mag = np.sum( (np.array(node.geom) - self._loc) ** 2) ** 0.5
        if mag > self.dist:
            self.dist = mag
            self.best = node.id
        return node


class EdgeDirectorQ(QueuePropogator):
    """ Creates direction for each edge"""
    def __init__(self,  **kwargs):
        super(EdgeDirectorQ, self).__init__(**kwargs)

    def on_default(self, node, **kwargs):
        for pred in node.predecessors():
            if pred.id not in self.seen:
                edge = pred.edge_to(node)
                edge.reverse()
        return node


def rotator(orienting_edge):
    rnt = np.array([orienting_edge.source.geom,
                    orienting_edge.target.geom])

    rnt[:, -1] = 0 # no z

    norm = Line(Point(rnt[1]), Point(rnt[0]))
    base = Line(Point(0., 0., 0.), Point(1., 0, 0))
    m = geo.Movement(norm, base)
    return m


class MovementQ(QueuePropogator):
    """ apply movement to each Node Point """
    def __init__(self, m):
        super(MovementQ, self).__init__()
        self.M = m

    def on_default(self, node, **kwargs):
        pt = Point(node.geom)
        pt2 = self.M.on_point(pt)
        node.geom = tuplify(pt2.numpy)
        return node


class Rotator(MovementQ):
    def __init__(self, angle):
        l1 = Line(Point(0, 0, 0), Point(1, 0, 0))
        l2 = Line(Point(0, 0, 0), Point(np.cos(angle), np.sin(angle), 0))
        M = geo.Movement(l2, l1)
        MovementQ.__init__(self, M)


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


# -------------------------------
def resolve_heads(root_node, spr_points, tol=1.):
    kdprop = KDTreeIndex()
    _ = kdprop(root_node)
    data = np.array(kdprop._data)

    for six, spr in enumerate(spr_points):
        cdist = distance.cdist([spr], data)[0]
        if np.min(cdist) < tol:
            best_ix = np.argmin(cdist)
            nd = kdprop[best_ix]
            nde = node_with_id(root_node, nd.id)
            connect_heads2(nde)


def connect_heads2(node):
    neighs = node.neighbors(fwd=True, bkwd=True, edges=True)
    if node.get('has_head') is True:
        return # node, False

    elif len(neighs) == 1:
        node.write('has_head', True)
        # node.write('end_head', True)
        return # node, True
    elif len(neighs) == 2:
        for edge in neighs:
            if _is_pipe(edge) is False:
                edge.write('has_head', True)
                # rebuild ?
                p1, p2 = edge.geom
                center = Point(p1).midpoint_to(Point(p2))
                new_main = edge.source

                new_main.geom = tuplify(center.numpy)
                new_main.remove_edge(edge)
                new_main.write('has_head', True)

                rem_node = edge.target
                for s in rem_node.successors():
                    new_main.connect_to(s)
                rem_node.deref()
                return
                # remove
                # return node, True
    # return node, False




# DEPR -------------------------------
def connect_heads(node_dict,  sprinks):
    sdists = {i: 1e7 for i in range(sprinks.shape[0])}
    sbests = {i: None for i in range(sprinks.shape[0])}
    seen = set()
    for k, n in node_dict.items():
        for neigh in n.neighbors(fwd=True, bkwd=True, edges=True):
            g1, g2 = neigh.geom
            this = tuple([g1, g2])
            if this not in seen:
                seen.add(this)
                cp = geo.Point(g1).midpoint_to(geo.Point(g2)).numpy[:2]
                ds = np.sqrt(np.sum((sprinks - cp) ** 2, axis=-1))
                minix = np.argmin(ds)
                minds = ds[minix]
                if minds < sdists[minix]:
                    sdists[minix] = minds
                    sbests[minix] = neigh
    return sdists, sbests



