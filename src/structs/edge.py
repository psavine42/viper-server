import numpy as np

from src.geom import MepCurve2d
from src.structs.base import GraphData


class Edge(GraphData):
    __slots__ = ['_src', '_tgt']

    def __init__(self, source, target, **kwargs):
        super(Edge, self).__init__(**kwargs)
        self._src = source
        self._tgt = target
        target.add_in_edge(self)
        source.add_out_edge(self)

    @property
    def geom(self):
        return self._src.geom, self._tgt.geom

    @property
    def direction(self):
        return MepCurve2d(*self.geom).direction

    @property
    def curve(self):
        return MepCurve2d(*self.geom)

    @property
    def source(self):
        return self._src

    @property
    def target(self):
        return self._tgt

    def similar_direction(self, other):
        return np.allclose(other.direction, self.direction)

    def split(self, new_geom, **kwargs):
        """
        x ---> y

        x ---> new ---> y

        turn edge into node, and connect
        :param geom:
        :param kwargs:
        :return:
        """
        #if isinstance(new_geom, Node):
        newnode = new_geom
        #else:
        #    newnode = Node(new_geom, **kwargs)
        prev_target = self._tgt
        prev_target.remove_edge(self)

        newnode.add_in_edge(self)
        self._tgt = newnode
        edge = newnode.connect_to(prev_target)
        return edge

    def reconnect(self, node):
        prev_tgt = self._tgt
        prev_tgt.remove_edge(self)
        node.add_in_edge(self)
        self._tgt = node

    def other_end(self, node):
        if node == self._src:
            return self._tgt
        elif node == self._tgt:
            return self._src
        else:
            return None

    def propogate(self):
        pass

    def reverse(self):
        target = self._tgt
        source = self._src
        target.remove_edge(self)
        source.remove_edge(self)
        self._tgt = source
        self._src = target
        target.add_out_edge(self)
        source.add_in_edge(self)
        return self

    def __eq__(self, other):
        return self._src.__eq__(other._src) \
               and self._tgt.__eq__(other._tgt)

    def inverse_of(self, other):
        return self._src.__eq__(other._tgt) \
               and self._tgt.__eq__(other._src)

    def __len__(self):
        return self.curve.length

    def __del__(self):
        self._tgt.remove_edge(self)
        self._src.remove_edge(self)
        self._tgt, self._src = None, None

    def __str__(self):
        st = '<Edge>:{}, {}'.format(self.geom, self.tmps)
        return st