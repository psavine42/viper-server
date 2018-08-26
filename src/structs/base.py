
import random
from copy import deepcopy


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
        graph_obj.add_cell(self)
        self._graph_objs.append(graph_obj)


# ---------------------------------------------------------------
class GraphData(object):
    __slots__ = ['_id', '_data', '_tmp', '_cell_refs']
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
        self._cell_refs[cell.var] = cell


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




