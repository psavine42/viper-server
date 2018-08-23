import lib.geo
from src import geom
from .base import BasePropogator, EdgePropogator
from src.rules.graph import Cell, Node
import numpy as np
from shapely.geometry import Point, LineString


def direction(p1, p2):
    return lib.geo.normalized(np.array(p2) - np.array(p1))


class EdgeRouter(BasePropogator):
    def __init__(self, name=None, **kwargs):
        super(EdgeRouter, self).__init__(name=name, **kwargs)
        self.seen = set()

    def next_fn(self, node):
        return [x for x in node.neighbors() if x.id not in self.seen]

    def on_default(self, node, mls, **kwargs):

        if node.id not in self.seen:
            self.seen.add(node.id)
            edges = node.neighbors(edges=True)
            to_del = []

            for i in range(len(edges)):
                ei = edges[i]
                gi = LineString(ei.geom)
                for j in range(i+1, len(edges)):
                    ej = edges[j]
                    gj = LineString(ej.geom)
                    inters = gj.intersection(gi)
                    # print(gj, gi, inters)

                    if ej == ei or ej.inverse(ei):
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
                        # print(other_end, next_end)
                    elif inters == Point(node.geom):
                        pass

            if len(to_del) > 0:
                self.seen.remove(node.id)
                for edge in to_del:
                    node.remove_edge(edge)

        return node, mls


class PointAdder(EdgePropogator):
    def __init__(self, point_node, **kwargs):
        super(PointAdder, self).__init__(name=point_node, **kwargs)
        self.point = point_node
        self.geom = Point(point_node.geom)
        self.seen = set()

    def on_default(self, edge, _, **kwargs):
        # print(edge)
        self.seen.add(edge.id)
        return edge, _

    def next_fn(self, edge):
        return [x for x in edge.target.neighbors(edges=True) \
               + edge.source.neighbors(edges=True) if x.id not in self.seen]

    def is_terminal(self, edge, _, **kwargs):
        # self.seen.add(edge.id)
        ix = self.geom.intersects(LineString(edge.geom))
        # print(LineString(edge.geom), ix)
        return ix

    def on_terminal(self, edge, _, **kwargs):
        e2 = edge.split(self.point)
        return e2, _


class PPrinter():
    pass
