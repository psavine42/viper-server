import numpy as np

from src.geom import MepCurve2d
from src.structs.base import GraphData


class Edge(GraphData):
    __slots__ = ['_src', '_tgt']

    def __init__(self, source, target, **kwargs):
        GraphData.__init__(self, **kwargs)
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
        newnode = new_geom
        prev_target = self._tgt
        prev_target.remove_edge(self)

        newnode.add_in_edge(self)
        self._tgt = newnode
        edge = newnode.connect_to(prev_target, **kwargs)
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
        """ src -> tgt
        Return
        src <- tgt
        """
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
        if not isinstance(other, self.__class__):
            return False
        return self._src.__eq__(other._src) \
               and self._tgt.__eq__(other._tgt)

    def inverse_of(self, other):
        return self._src.__eq__(other._tgt) \
               and self._tgt.__eq__(other._src)

    def __len__(self):
        return self.curve.length

    def __del__(self):
        if self._tgt is not None:
            self._tgt.remove_edge(self)
        if self._src is not None:
            self._src.remove_edge(self)
        self._tgt, self._src = None, None

    def delete(self):
        if self._tgt is not None:
            self._tgt.remove_edge(self)
        if self._src is not None:
            self._src.remove_edge(self)
        self._tgt, self._src = None, None

    def __str__(self):
        st = '<Edge>:{}, {}'.format(self.geom, self.tmps)
        return st

    def __iter__(self, fwd=True, bkwd=False, edges=False):
        if bkwd is True:
            if edges is True:
                for e in self._src.neighbors(edges=True, bkwd=True, fwd=False):
                    yield e
            else:
                yield self._src
        if fwd is True:
            if edges is True:
                for e in self._tgt.neighbors(edges=True, bkwd=False, fwd=True):
                    yield e
            else:
                yield self._tgt





# class SuperEdge(EdgeSolid):
#     """
#     Edge that can take additional input output
#
#     Attrs:
#     ----------
#         - internal:
#         Internal Graph to be accessed by __iter__
#
#
#     """
#     def __init__(self, source, target,  **kwargs):
#         EdgeSolid.__init__(self, source, target, **kwargs)
#         self._inner_succ = []
#         self._inner_pred = []
#
#     def add_predecessor(self, node):
#         self._inner_pred.append(node)
#
#     def add_successor(self, node):
#         self._inner_succ.append(node)
#
#     def __inner_iter(self, inners):
#         for n in inners:
#             yield n
#
#     def __iter__(self, fwd=True, bkwd=False, edges=False):
#         if edges is False:
#             if bkwd is True:
#                 yield self._src
#                 for n in self._inner_pred:
#                     yield n
#             if fwd is True:
#                 yield self._tgt
#                 for n in self._inner_succ:
#                     yield n
#         elif edges is True:
#             if bkwd is True:
#                 for n in self._inner_pred:
#                     for e in n.neighbors(edges=True, bkwd=True, fwd=False):
#                         yield e
#             if fwd is True:
#                 for n in self._inner_succ:
#                     for e in n.neighbors(edges=True, bkwd=False, fwd=True):
#                         yield e




