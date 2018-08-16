import networkx as nx
import src.geom
from uuid import uuid4
from collections import defaultdict as ddict
from copy import deepcopy

###############################################################
#
###############################################################


class GraphData(object):
    def __init__(self, **kwargs):
        self._data = kwargs
        self._id = uuid4()
        self._tmp = deepcopy(self._data)

    @property
    def id(self):
        return self._id

    @property
    def data(self):
        return self._data

    @property
    def tmps(self):
        return self._tmp

    def get(self, item, d=None):
        return self._tmp.get(item, d)

    def update(self, k, v):
        self._data[k] = v

    def write(self, k, v):
        self._tmp[k] = v


class IEdge(object):
    def __init__(self, source, target, **kwargs):
        self._src = source
        self._tgt = target
        target.add_in_edge(self)
        source.add_out_edge(self)

    @property
    def source(self):
        return self._src

    @property
    def target(self):
        return self._tgt


class Edge(GraphData):
    def __init__(self, source, target, **kwargs):
        super(Edge, self).__init__(**kwargs)
        self._src = source
        self._tgt = target
        self._data = kwargs
        target.add_in_edge(self)
        source.add_out_edge(self)

    @property
    def geom(self):
        return self._src.geom, self._tgt.geom

    @property
    def source(self):
        return self._src

    @property
    def target(self):
        return self._tgt

    def split(self, geom, **kwargs):
        """
        x ---> y

        x ---> new ---> y

        turn edge into node, and connect
        :param geom:
        :param kwargs:
        :return:
        """
        newnode = Node(geom, **kwargs)
        prev_target = self._tgt
        prev_target.remove_edge(self)
        newnode.add_in_edge(self)
        self._tgt = newnode
        newnode.connect_to(prev_target)

    def propogate(self):
        pass

    def __del__(self):
        self._tgt.remove_edge(self)
        self._src.remove_edge(self)
        self._tgt, self._src = None, None


class Node(GraphData):
    def __init__(self, geom, **kwargs):
        super(Node, self).__init__(**kwargs)
        self._geom = geom
        self._pred = []
        self._sucs = []
        self._cond = {}

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
    def update_geom(self, fn):
        self._geom = fn(self._geom)

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
        if source.id != target.id:
            edge = Edge(source, target, **edge_data)
            # 3source._on_mutate()
            # target._on_mutate()
            return edge

    def get(self, v, d=None):
        if hasattr(self, v):
            return getattr(self, v, d)
        else:
            return super(Node, self).get(v, d)

    def propogate(self):
        pass

    # python -------------------------------------------
    def __eq__(self, other):
        return self.id == other.id

    def __contains__(self, item):
        return self.__getitem__(item) is not None

    def __getitem__(self, item):
        item_fn = self._make_item_fn(item)
        for node in self.__iter__():
            if item.__eq__(item_fn(node)):
                return node
        return None

    def __iter__(self, fwd=True, bkwd=False, seen=None):
        """defaults to DFS on successors """
        if seen is None:
            seen = set()
        seen.add(self.id)
        yield self
        for n in self.neighbors(fwd=fwd, bkwd=bkwd):
            if n.id not in seen:
                for x in n.__iter__(seen=seen, fwd=fwd, bkwd=bkwd):
                    yield x
        del seen

    def __call__(self, fn, acc, **kwargs):
        acc = fn(self, acc, **kwargs)
        for suc in self.successors():
            acc = suc.__call__(fn, acc, **kwargs)
        return acc

    def __str__(self):
        st = '<NODE>:' + str(self._geom)
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


class XGraph(object):
    def __init__(self, ):
        return


