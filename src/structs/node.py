
from copy import deepcopy
from shapely.geometry import Point

from .edge import Edge
from .base import GraphData


class Node(GraphData):
    __slots__ = ['_pred', '_sucs', '_geom', '_cond', '_done']
    def __init__(self, geom, **kwargs):
        super(Node, self).__init__(**kwargs)
        self._geom = geom
        self._pred = []
        self._sucs = []
        self._cond = {}
        self._done = False

    # Access -------------------------------------------
    @property
    def nsucs(self):
        return len(self._sucs)

    @property
    def npred(self):
        return len(self._pred)

    @property
    def geom(self):
        return self._geom

    @property
    def as_point(self):
        return Point(self.geom)

    def predecessors(self, edges=False):
        if edges is True:
            return self._pred
        return [x.source for x in self._pred]

    def successors(self, edges=False):
        if edges is True:
            return self._sucs
        return [x.target for x in self._sucs]

    def neighbors(self, edges=False, fwd=True, bkwd=True):
        res = []
        if fwd is True:
            res = self.successors(edges=edges)
        if bkwd is True:
            return res + self.predecessors(edges=edges)
        return res

    # mutation -------------------------------------------
    def update_geom(self, val):
        self._geom = val

    def remove_edge(self, edge):
        if edge in self._sucs:
            self._sucs.remove(edge)
        if edge in self._pred:
            self._pred.remove(edge)

    def add_out_edge(self, edge):
        self._sucs.append(edge)

    def add_in_edge(self, edge):
        self._pred.append(edge)

    def connect_to(source, target, **edge_data):
        if source.id != target.id and target not in source.neighbors():
            edge = Edge(source, target, **edge_data)
            return edge

    def edge_to(self, other):
        for x in self._sucs:
            if x.target == other:
                return x

    def get(self, v, d=None):
        if hasattr(self, v):
            return getattr(self, v, d)
        else:
            return super(Node, self).get(v, d)

    def deref(self):
        for edge in self.successors(edges=True):
            other = edge.other_end(self)
            other.remove_edge(edge)
            self.remove_edge(edge)
            edge._tgt = None
            edge._src = None

    @property
    def complete(self):
        return self.get('solved', False)

    def propogate(self):
        return

    # python -------------------------------------------
    def __eq__(self, other):
        return self.id == other.id

    def __contains__(self, item):
        return self.__getitem__(item) is not None

    def __getitem__(self, item, **kwargs):
        item_fn = self._make_item_fn(item)
        for node in self.__iter__(**kwargs):
            if item.__eq__(item_fn(node)):
                return node
        return None

    def __iter__(self, fwd=True, bkwd=False,  seen=None):
        """defaults to DFS on successors """
        if seen is None:
            seen = set()
        seen.add(self.id)
        yield self
        for n in self.neighbors(fwd=fwd, bkwd=bkwd):
            if n.id not in seen:
                for x in n.__iter__(seen=seen, fwd=fwd, bkwd=bkwd):
                    # if edges is False:
                    yield x
                    # else:
                    #    yield n.edge_to(x)
        del seen

    def __call__(self, fn, acc, **kwargs):
        acc = fn(self, acc, **kwargs)
        for suc in self.successors():
            acc = suc.__call__(fn, acc, **kwargs)
        return acc

    def __str__(self):
        st = '<{}>:{} : {}'.format(self.__class__.__name__, self._geom, str(self.tmps))
        return st

    def __len__(self, local=False, seen=None):
        if local is True:
            return len(self.neighbors(edges=True))
        return len(list(self.__iter__()))

    # Util -----------------------------------------------
    def _on_mutate(self):
        self._tmp = deepcopy(self._data)

    def _make_item_fn(self, item):
        if isinstance(item, type(self.geom)):
            item_fn = lambda x: x.geom
        elif isinstance(item, Node):
            item_fn = lambda x: x
        else:
            item_fn = lambda x: x.data
        return item_fn


class GeomNode(Node):
    def __init__(self, geom, **kwargs):
        super(GeomNode, self).__init__(geom, **kwargs)

    def __eq__(self, other):
        return self.geom.__eq__(other.geom)








