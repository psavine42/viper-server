
import networkx as nx
import numpy as np

from src.geom import MepCurve2d, rebuild_ls, rebuild, split_ls, to_Nd, to_mls
from collections import defaultdict as ddict

import src.walking as walking
from src.geomType import GeomType

from src import visualize

from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge, nearest_points, shared_paths

from src.rules.graph import Node, Edge
import src.propogate.geom_prop as gp
import src.propogate.propagators as ppg
import pprint

class GeomFilter(object):
    def __init__(self):
        pass

    def __call__(self, mls):
        pass


class System(object):
    def __init__(self, segments=None, symbols=[], **kwargs):
        self._data = dict()
        self.result = []
        self._root = None
        self._keep_layers = []
        self._edge_ord, self._node_ord = 0, 0
        self.G = nx.DiGraph() # dg with cycles allowed bro
        # self._lim = [1e10, 1e10, -1e10, -1e10]
        self._symbols = {s.points[0]: s for s in symbols}
        self._endpoints = []
        if segments:
            self.build(segments, **kwargs)

    # Util---------------------------------------------
    def gplot(self, **kwargs):
        visualize.gplot(self.G, **kwargs)

    def reverse_edge(self, src, tgt):
        data = self.G.get_edge_data(src, tgt, None)
        self.G.add_edge(tgt, src, id=data.get('id', None))
        self.G.remove_edge(src, tgt)
        return tgt, src

    def edit_node(self, old, new, sucs=None, pred=None, data=None):
        data = self.G.nodes[old] if data is None else data
        sucs = list(self.G.successors(old)) if sucs is None else sucs
        pred = list(self.G.predecessors(old)) if pred is None else pred
        self.G.add_node(new, **data)
        for x in sucs:
            edata = self.G.get_edge_data(old, x, {})
            self.G.add_edge(new, x, **edata)
        for x in pred:
            edata = self.G.get_edge_data(x, old, {})
            self.G.add_edge(x, new, **edata)
        self.G.remove_node(old)
        return new

    def propogate_node(self, start, op):
        pass

    # walking internal---------------------------------
    def _remove_colinear_fn(self, el, pred, sucs, seen):
        if self.G.nodes[el].get('symbol', None) is not None:
            return
        npred, nsucs = len(pred), len(sucs)
        if nsucs == 1 and npred == 1:
            p1, p2 = pred[0], sucs[0]
        elif nsucs == 2 and npred == 0:
            p1, p2 = sucs[1], sucs[0]
        elif nsucs == 0 and npred == 2:
            p1, p2 = pred[1], pred[0]
        else:
            return
        pnt = Point(el)
        if pnt.within(LineString([p1, p2]).buffer(.01)) is True:
            self.G.remove_node(el)
            self.G.add_edge(p1, p2)

    def _comp_direction_fn(self, e1, pred, sucs, seen):

        for src in pred:
            if src not in seen:
                self.reverse_edge(src, e1)

    def _write_attrs(self, el, pred, sucs, seen):
        """ counter keeps track of build order """
        nx.set_node_attributes(self.G, {el: {'order': self._node_ord}})
        self._node_ord += 1

        for p in sucs:
            crv = MepCurve2d(el, p)
            nx.set_edge_attributes(
                self.G,
                {(el, p): {'direction': np.array(crv.direction),
                           'length': crv.length,
                           'order': self._edge_ord,
                           'id': crv.id}})
            self._edge_ord += 1

    def _set_order(self, el, pred, sucs, seen):
        nx.set_node_attributes(self.G, {el: {'order': self._node_ord}})
        self._node_ord += 1
        for p in sucs:
            nx.set_edge_attributes(self.G, {(el, p): {'order': self._edge_ord}})
            self._edge_ord += 1

    def _remove_duplicates(self, el, pred, sucs, seen):
        dups = set(pred).intersection(sucs)
        for dup in dups:
            if len(sucs) > len(pred):
                self.G.remove_edge(dup, el)
            else:
                self.G.remove_edge(el, dup)

    # Interface-----------------------------------------
    def walk(self, source, fn):
        walking.do_walk(self.G, source, fn)

    def compute_direction(self, source):
        seen = set()
        q = [(self._root, x) for x in self.G.successors(self._root)]
        while q:
            el = q.pop(-1)
            if el not in seen:
                seen.add(el)
                e1, e2 = el
                sucs = list(self.G.successors(e2))
                nxts = list(set(self.G.predecessors(e2)).difference(e1))
                new = [(e2, x) for x in sucs]
                for src in nxts:
                    if (src, e2) not in seen:
                        new_edge = self.reverse_edge(src, e2)
                        new.append(new_edge)
                q.extend(new)

    def remove_colinear(self, source):
        count = len(self.G)
        done = False
        while not done:
            # print('iter', count)
            self.walk(source, self._remove_colinear_fn)
            self.walk(source, self._remove_duplicates)
            if len(self.G) == count:
                done = True
            else:
                count = len(self.G)

    def bake_attributes(self, source, full=True):
        """ bake whatever is needed for pattern recognition """
        self._edge_ord = 0
        self._node_ord = 0
        self._endpoints = []
        fn = self._write_attrs if full is True else self._set_order
        walking.walk_dfs(self.G, source, fn)

    # Maintenance---------------------------------------
    def add(self, *segments):
        for seg in segments:
            self._data[seg.id] = seg
            p1, p2 = seg.points
            self.G.add_edge(p1, p2, id=seg.id, gtype=seg.geomType)

    def remove(self, *segments):
        for x in segments:
            seg_id = x.id if isinstance(x, MepCurve2d) else x
            if seg_id in self._data:
                seg = self._data.pop(seg_id)
                p1, p2 = seg.points
                self.G.remove_edge(p1, p2)

    @property
    def root(self):
        return self._root

    def stat(self):
        print('Stat:', len(self.G), len(self.G.edges), len(self._data))

    # Setup---------------------------------------------
    def get_intersects(self, lines, pt=None, norm=0.1):
        """

        :param lines: list of Linestring
        :param pt:
        :param norm:
        :return:
        """

        if pt is None:
            pt = Point(0, 0, 0)
        elif isinstance(pt, tuple):
            pt = Point(*list(pt))
        root, dist, n = None, 1e10, norm
        ind1 = ddict(set)

        def add_subseg(crv, l, px):
            ind1[crv.id].update(px)
            ind1[l.id].update(px)

        while lines:
            crv = lines.pop(0)
            if pt.distance(crv) < dist:
                root, dist = crv, pt.distance(crv)
            for l in lines:
                px1 = l.intersection(crv)
                if not px1.is_empty:
                    add_subseg(crv, l, list(px1.coords))
                else:
                    crv2 = crv.extend_norm(n, n)
                    px2 = l.extend_norm(n, n).intersection(crv2)
                    if not px2.is_empty:  # and isinstance(px2, geometry.LineString):
                        add_subseg(crv, l, list(px2.coords))

        p1, p2 = [Point(x) for x in root.points]
        self._root = tuple(list(p1.coords if pt.distance(p1) < pt.distance(p2) else p2.coords)[0])
        self._keep_layers.append(root.layer)
        return ind1

    def add_to_graph(self, line_dict, ind1):
        for id, crv in line_dict.items():
            if crv.type != GeomType.SYMBOL:
                if crv.layer not in self._keep_layers:
                    continue

            pts = list(crv.points)
            ns = list(ind1.get(id, []))
            if None in ns:
                continue

            tups = list(set(ns + pts))
            tups.sort()
            for i in range(1, len(tups)):
                self.add(MepCurve2d(tups[i - 1], tups[i]))

    def build(self, segments=[], root=None, ext=0.2):
        """
        # todo deal with Symbols
        :param segments:
        :param symbols:
        :param ext:
        :return:
        """
        segments.sort()
        line_dict = {l.id: l for l in segments}
        ind1 = self.get_intersects(segments, pt=root, norm=ext)
        self.add_to_graph(line_dict, ind1)

    def _write_symbols(self):
        for pt, sym in self._symbols.items():
            if pt in self.G:
                print('found:   ', sym)
                nx.set_node_attributes(self.G, {pt: sym.to_dict})
            else:
                self.G.add_node(pt, id=sym.id, gtype=sym.geomType)
                print('missing:  ', sym)

    def bake(self, root=None):
        if root is None:
            root = self._root
        self._write_symbols()
        self.remove_colinear(root)      # clean up
        self.compute_direction(root)    # bake
        self.bake_attributes(root)
        self.stat()
        return self


class SystemV2(System):
    def __init__(self, **kwargs):
        self.mls = []
        super(SystemV2, self).__init__(**kwargs)

    def gplot(self, **kwargs):
        visualize.ord_plot(self.G, **kwargs)

    def annotate(self, pnt, data):
        nx.set_node_attributes(self.G, {pnt: data})

    def _nearest(self, pt):
        self.best_dist = 1e10
        self.best_cord = None

        def near_fn(el, pred, sucs, seen):
            dist = pt.distance(Point(el))
            if dist < self.best_dist:
                self.best_dist = dist
                self.best_cord = el

        self.walk(self._root, near_fn)
        return self.best_cord, self.best_dist

    def _write_symbols(self, tol=5.0):
        for pt, sym in self._symbols.items():
            npg, dist = self._nearest(sym)
            if dist < tol:
                self.annotate(npg, sym.to_dict)


class SystemV3(System):
    def __init__(self, **kwargs):
        self._node_dict = dict()
        super(SystemV3, self).__init__(**kwargs)

    def filter_keep(self, lines, pt):
        if pt is None:
            pt = Point(0, 0, 0)
        elif isinstance(pt, tuple):
            pt = Point(*list(pt))

        root, dist = None, 1e10
        for crv in lines:
            if pt.distance(crv) < dist:
                root, dist = crv, pt.distance(crv)
        self._keep_layers.append(root.layer)
        fls = []
        for l in lines:
            if l.layer == root.layer or l.type == GeomType.SYMBOL:
                fls.append(l)
        return fls

    def get_intersects(self, lines, pt=None, norm=0.1):
        lines = self.filter_keep(lines, pt)
        mls_ob = linemerge(lines)
        root, _ = nearest_points(mls_ob, Point(pt))
        root = to_Nd(root, 3)

        merged = list(mls_ob)
        print('building', len(merged))
        dists = ddict(dict)
        node_dict = dict()
        acomps = []
        for i in range(len(merged)):
            ndict = dict()
            cords = list(merged[i].coords)
            for k in range(len(cords)):
                pnt = cords[k]
                if pnt not in ndict:
                    ndict[pnt] = Node(pnt)
                if k > 0:
                    ndict[cords[k - 1]].connect_to(ndict[pnt])
            acomps.append(ndict)
            for j in range(i+1, len(merged)):
                dist = merged[i].distance(merged[j])
                dists[i][j] = dist
                dists[j][i] = dist

        q = [0]
        while q and len(dists) != 1 and len(acomps) != 1:
            i = q.pop()
            mk, mv = min(dists[i].items(), key=lambda x: x[1])

            this_comp = acomps[i]
            othr_comp = acomps[mk]

            # compute nearest point
            pi, po = [to_Nd(x) for x in nearest_points(merged[i], merged[mk])]
            in_this = pi in this_comp
            in_othr = po in othr_comp

            if in_this and in_othr:
                this_comp[pi].connect_to(othr_comp[po])

            elif in_this and not in_othr:
                if pi != po:
                    new_node = Node(po)
                    pa = gp.PointAdder(new_node)
                    pa(list(othr_comp.values())[0])
                    this_comp[pi].connect_to(new_node)
                else:
                    pa = gp.PointAdder(this_comp[pi])
                    pa(list(othr_comp.values())[0])

            elif not in_this and in_othr:
                if pi != po:
                    new_node = Node(pi)
                    pa = gp.PointAdder(new_node)
                    pa((list(this_comp.values())[0]))
                    othr_comp[po].connect_to(new_node)
                else:
                    pa = gp.PointAdder(othr_comp[po])
                    pa(list(this_comp.values())[0])

            else:
                print('ERR')

            for k, nd in othr_comp.items():   # move nodes to current comp
                this_comp[k] = nd

            merged[i] = merged[i].union(merged[mk]) # update geometry
            merged[mk] = None                       # remove previous
            acomps[mk] = None                       # remove previous

            for k, v in dists[mk].items():
                if k in dists[i]:
                    min_dist = min([dists[i][k], v])
                    dists[i][k] = min_dist
                dists[k].pop(mk)
            dists.pop(mk)

            if len(dists) == 1:
                break

            nextk, bestm = None, 1e10
            for k, v in dists.items():
                mk, mv = min(v.items(), key=lambda x: x[1])
                if mv < bestm:
                    nextk = k
                    bestm = mv
                if mv == 0:
                    break
            q.append(nextk)

        print('{} components, {} nodes'.format(len(acomps), sum([len(x) for x in acomps if x])))
        for i in range(len(acomps)):
            if acomps[i] is None:
                continue
            for k, v in acomps[i].items():
                if v.id not in node_dict:
                    if v.geom == root:
                        self._root = v
                    # node_dict[v.id] = v
                else:
                    print('ovelap', v)

        print('{} components, {} nodes'.format(len(acomps), len(node_dict)))
        assert self._root is not None
        gp.EdgeRouter()(self._root)
        return node_dict

    def add_to_graph(self, line_dict, indexes):
        # intersect_ids_to_pts, points_to_ids, id_to_id, pts_to_pts = indexes

        seen = set()
        for k, vs in indexes.items():

            seen.add(k)
            if k not in self._node_dict:
                self._node_dict[k] = Node(k)

            for v in vs:
                if v not in self._node_dict:
                    self._node_dict[v] = Node(v)
                # if self._node_dict[v] not in self._node_dict[k].neighbors():
                self._node_dict[k].connect_to(self._node_dict[v])

        print(self._root)
        print(len(self._node_dict))
        p1 = self._root
        self._root = self._node_dict [p1]

    def build(self, segments=[], root=None, ext=0.2):
        line_dict = {l.id: l for l in segments}
        indexes = self.get_intersects(segments, pt=root, norm=ext)

    def gplot(self, **kwargs):
        G = nodes_to_nx(self._root, **kwargs)
        visualize.gplot(G)

    def aplot(self, **kwargs):
        G = sys_to_nx(self)
        visualize.gplot(G, **kwargs)

    def bake(self, root=None):
        baker = ppg.Chain(
            ppg.EdgeDirector(),
            ppg.GraphTrim(),
            ppg.DirectionWriter(),
            ppg.DistanceFromSource(),
            ppg.BuildOrder(),
            ppg.DistanceFromEnd(),
        )

        baker(self._root)
        return self


def nx_to_nodes(system):
    G, root = system.G, system.root
    seen, q = set(), [root]
    tmp = {}
    while q:
        el = q.pop(0)
        if el not in seen:
            seen.add(el)
            pred = list(G.predecessors(el))
            sucs = list(G.successors(el))

            data = G.nodes[el]
            if 'symbol_id' in data:
                chld = data.pop('children', [])
            nd = Node(el, **data)
            for x in pred:
                if x in tmp:
                    tmp[x].connect_to(nd, **G[x][el])
            for x in sucs:
                if x in tmp:
                    nd.connect_to(tmp[x], **G[el][x])
            tmp[nd.geom] = nd
            q.extend(pred + sucs)

    root_node = tmp[root]
    return root_node


def sys_to_nx(system):
    import networkx as nx
    G = nx.DiGraph()
    for _, node in system._node_dict.items():
        for n in node.successors():
            G.add_edge(node.geom, n.geom)
        # for n in node.predecessors():
        #    G.add_edge(n.geom, node.geom)
    return G


def nodes_to_nx(root, fwd=True, bkwd=False):
    import networkx as nx
    G = nx.DiGraph()
    for node in root.__iter__(fwd=fwd, bkwd=bkwd):
        G.add_node(node.geom, **{**node.data, **node.tmps})

    for node in root.__iter__(fwd=fwd, bkwd=bkwd):
        for n in node.successors():
            G.add_edge(node.geom, n.geom)
        # for n in node.predecessors():
        #    G.add_edge(n.geom, node.geom)

    return G

