
import numpy as np
from collections import defaultdict as ddict
from shapely.geometry import Point, LineString

from .base import BasePropogator, EdgePropogator, RecProporgator
from src.structs import geom_merge, Node


class EdgeRouter(BasePropogator):
    """
    Graph Cleaning operation - if there are overlapping lines,

    """

    def __init__(self, name=None, **kwargs):
        super(EdgeRouter, self).__init__(name=name, **kwargs)
        self.seen = set()

    def next_fn(self, node):
        return [x for x in node.neighbors() if x.id not in self.seen]

    def on_default(self, node, _, fuzz=0.0, bkwd=False, **kwargs):

        if node.id not in self.seen:
            self.seen.add(node.id)
            edges = node.neighbors(bkwd=bkwd, edges=True)
            to_del = []

            for i in range(len(edges)):
                ei = edges[i]
                gi = LineString(ei.geom)
                for j in range(i+1, len(edges)):
                    ej = edges[j]
                    gj = LineString(ej.geom)
                    inters = gj.intersection(gi)

                    if ej == ei or ej.inverse_of(ei):
                        to_del.append(ej)
                    elif isinstance(inters, LineString):
                        # rerouting op here
                        if gj.length > gi.length:
                            keep, rem = ei, ej
                        elif gj.length < gi.length:
                            keep, rem = ej, ei
                        else:
                            keep, rem = ej, ei

                        to_del.append(rem)
                        next_end = rem.other_end(node)
                        other_end = keep.other_end(node)
                        other_end.connect_to(next_end)
                        self.seen.difference_update([other_end.id, next_end.id])

            if len(to_del) > 0:
                self.seen.remove(node.id)
                for edge in to_del:
                    node.remove_edge(edge)

        return node, _


class PropMerge(BasePropogator):
    def __init__(self, prop, **kwargs):
        super(PropMerge, self).__init__(name=prop, **kwargs)
        self.seen = set()
        self._syms = ddict(list)

    def on_default(self, node, _, **kwargs):
        if node.id not in self.seen:
            self.seen.add(node.id)
            prop = node.get(self.var, None)
            if prop and node not in self._syms[prop]:
                self._syms[prop].append(node)
        return node, _

    def __call__(self, node, **kwargs):
        super(PropMerge, self).__call__(node, **kwargs)
        for k, nodes in self._syms.items():
            if len(nodes) > 1:
                # print(k, nodes)
                geom = np.mean([np.array(n.geom) for n in nodes])
                ndx = Node(geom)
                for n in nodes:
                    n.connect_to(ndx)


class PointAdder(EdgePropogator):
    """
    point_node Node with geom such that intersects an edge

    that edge will be split by the point_to add

    """
    def __init__(self, point_node, **kwargs):
        super(PointAdder, self).__init__(name=point_node, **kwargs)
        self.point = point_node
        self.geom = Point(point_node.geom)
        self.seen = set()

    def on_default(self, edge, _, **kwargs):
        self.seen.add(edge.id)
        return edge, _

    def next_fn(self, edge):
        return [x for x in edge.target.neighbors(edges=True)  \
                + edge.source.neighbors(edges=True) if x.id not in self.seen]

    def is_terminal(self, edge, _, **kwargs):
        ix = self.geom.intersects(LineString(edge.geom))
        return ix

    def on_terminal(self, edge, _, **kwargs):
        e2 = edge.split(self.point)
        return e2, _


class PPrinter():
    pass

# Anti Patterns ---------------------------------------------------------------


class Cluster(RecProporgator):
    def on_default(self, node, dat, tol=1., **kwargs):
        s = []
        q = [node]
        while q:
            el = q.pop(0)
            s.append(el)
            for n in el.successors():
                if n.as_point.distance(el.as_point) < tol:
                    q.append(n)
        if len(s) == 3:
            node = geom_merge(*s)
        return node, dat


class DirCluster(RecProporgator):
    def on_default(self, node, _, dtol=1., bkwd=True, **kwargs):
        s = []
        q = [node]
        while q:
            el = q.pop(0)
            s.append(el)
            for n in el.neighbors(bkwd=bkwd):
                if n.as_point.distance(el.as_point) < dtol:
                    q.append(n)
        if len(s) == 3:
            node = geom_merge(*s)
        return node, _