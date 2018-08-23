import src.geom
from uuid import uuid4
from collections import defaultdict as ddict
from copy import deepcopy
import random


# ---------------------------------------------------------------
class Cell(object):
    """
    In this context, a cell is a hyperedge, which
    are built up

    """

    def __init__(self, var, val=None, propg=None, graph_objs=None):
        self._id = random.randint(0, 1e6)
        self._var = var
        self._value = val
        self._graph_objs = graph_objs if graph_objs else []
        self._activate = []

    @property
    def var(self):
        return self._var

    @property
    def value(self):
        return self._value

    def write(self, node, value):
        self._var = value
        for propagator in self._activate:
            propagator(node, data=value)

    def attach(self, graph_obj):
        self._graph_objs.append(graph_obj)


# ---------------------------------------------------------------
class GraphData(object):
    def __init__(self, **kwargs):
        self._data = kwargs
        self._id = random.randint(0, 1e6)
        self._tmp = deepcopy(self._data)
        self._cell_refs = {}

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
        if k in self._cell_refs:
            self._cell_refs[k].write(self, v)
        self._tmp[k] = v

    def add_cell(self, cell):
        self._tmp[cell.var] = cell.value
        self._tmp[cell.var] = cell


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


    # def gdirection(self):


    @property
    def source(self):
        return self._src

    @property
    def target(self):
        return self._tgt

    def split(self, new_geom, **kwargs):
        """
        x ---> y

        x ---> new ---> y

        turn edge into node, and connect
        :param geom:
        :param kwargs:
        :return:
        """
        if isinstance(new_geom, Node):
            newnode = new_geom
        else:
            newnode = Node(new_geom, **kwargs)
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

    def inverse(self, other):
        return self._src.__eq__(other._tgt) \
               and self._tgt.__eq__(other._src)

    def __del__(self):
        self._tgt.remove_edge(self)
        self._src.remove_edge(self)
        self._tgt, self._src = None, None

    def __str__(self):
        st = '<Edge>:{}, {}'.format(self.geom, self.tmps)
        return st


class Node(GraphData):
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

    def merge(self, other):
        pass

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

            # del edge
        # self.d


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
        st = '<{}>:{} :'.format(self.__class__.__name__, self._geom, str(self.tmps))
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


def delete_between(node, source, target):
    prd = source.edge_to(node)
    suc = target.edge_to(node)
    source.connect_to(target)

    node.remove_edge(suc)
    node.remove_edge(prd)
    source.remove_edge(prd)
    target.remove_edge(suc)



