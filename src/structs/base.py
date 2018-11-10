import random
from copy import deepcopy
from .arithmetic import *
from .cell import Cell


def uid_fn():
    return random.randint(0, 1e7)


# -------------------------------------------------
class GraphData(object):
    __slots__ = ['_id', '_data', '_tmp', '_cells']

    def __init__(self, **kwargs):
        self._data = kwargs
        self._id = uid_fn()
        self._tmp = deepcopy(self._data)
        self._cells = {}

    @property
    def id(self):
        return self._id

    @property
    def data(self):
        return self._data

    @property
    def tmps(self):
        return self._tmp

    @property
    def cells(self):
        return self._cells

    def get_cell(self, var):
        if var in self._cells:
            return self._cells[var]
        else:
            cell = Cell(var=var)
            self.add_cell(cell)
            return cell

    def get(self, item, d=None):
        if item in self._cells:
            if self._cells[item].value:
                return self._cells[item].value
        return self._tmp.get(item, d)

    def update(self, k, v):
        self._data[k] = v

    def write(self, k, v, **kwargs):
        if k in self._cells.keys():
            self._cells[k].add_contents(v)
        else:
            self._tmp[k] = v

    def add_cell(self, cell):
        self._cells[cell.var] = cell




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




