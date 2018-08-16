
import lib.figures as F
from src.geom import MepCurve2d
from collections import defaultdict as ddict
import networkx as nx
from shapely import geometry
import numpy as np
import src.walking as walking
from src.geomType import GeomType
import src.visualize
from shapely.ops import nearest_points
from shapely.ops import linemerge

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
        src.visualize.gplot(self.G, **kwargs)

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
        pnt = geometry.Point(el)
        if pnt.within(geometry.LineString([p1, p2]).buffer(.01)) is True:
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
        #if len(sucs) == 0:
        #    self._endpoints.append(el)
        #else:
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
            print('iter', count)
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
            pt = geometry.Point(0, 0, 0)
        elif isinstance(pt, tuple):
            pt = geometry.Point(*list(pt))
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

        p1, p2 = [geometry.Point(x) for x in root.points]
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
        print('num segs: ', len(segments))
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

        self.stat()
        self._write_symbols()
        self.remove_colinear(root) # clean up
        self.compute_direction(root) # bake
        self.bake_attributes(root)

        return self


class SystemV2(System):
    def __init__(self, **kwargs):
        self.mls = []
        super(SystemV2, self).__init__(**kwargs)

    def gplot(self, **kwargs):
        src.visualize.ord_plot(self.G, **kwargs)

    def annotate(self, pnt, data):
        nx.set_node_attributes(self.G, {pnt: data})

    def _nearest(self, pt):
        self.best_dist = 1e10
        self.best_cord = None

        def near_fn(el, pred, sucs, seen):
            dist = pt.distance(geometry.Point(el))
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
