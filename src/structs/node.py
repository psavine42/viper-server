
from copy import deepcopy
from shapely.geometry import Point
import numpy as np
from .edge import Edge
from .base import GraphData




class Node(GraphData):
    __slots__ = ['_pred', '_sucs', '_geom', '_cond', '_done']

    def __init__(self, geom, **kwargs):
        GraphData.__init__(self, **kwargs)
        self._geom = geom
        self._pred = []
        self._sucs = []
        self._cond = {}
        self._done = False
        self._init_cells()

    def _init_cells(self):
        pass

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

    @geom.setter
    def geom(self, value):
        self._geom = value

    @property
    def as_np(self):
        return np.asarray(list(self.geom))

    @property
    def as_point(self):
        return Point(self.geom)

    def predecessors(self, edges=False, both=False):
        if edges is True:
            return self._pred
        elif both is True:
            return [(x, x.source) for x in self._pred]
        else:
            return [x.source for x in self._pred]

    def successors(self, edges=False, both=False):
        if edges is True:
            return self._sucs
        elif both is True:
            return [(x, x.target) for x in self._sucs]
        else:
            return [x.target for x in self._sucs]

    def neighbors(self, edges=False, fwd=True, bkwd=True, both=False):
        res = []
        if fwd is True:
            res = self.successors(edges=edges, both=both)
        if bkwd is True:
            return res + self.predecessors(edges=edges, both=both)
        return res

    # mutation -------------------------------------------
    def update_geom(self, val):
        """ corresponds to moving the skeleton node """
        self._geom = val

    def remove_edge(self, edge):
        """ remove edge """
        if edge in self._sucs:
            # tgt = edge.target
            self._sucs.remove(edge)
        if edge in self._pred:
            self._pred.remove(edge)

    def disconnect_from(self, node):
        if isinstance(node, self.__class__):
            edge = None
            for e in self.successors(edges=True):
                if e.target.id == node.id:
                    edge = e
                    break
            for e in self.predecessors(edges=True):
                if e.source.id == node.id:
                    edge = e
                    break
            if edge is not None:
                edge.delete()
        elif isinstance(node, Edge):
            self.remove_edge(node)

    def add_out_edge(self, edge):
        self._sucs.append(edge)

    def add_in_edge(self, edge):
        self._pred.append(edge)

    def connect_to(source, target, edge_cls=Edge, **edge_data):
        if source.id != target.id and target not in source.neighbors():
            edge = edge_cls(source, target, **edge_data)
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

    def _remove(self, edge):
        other = edge.other_end(self)
        other.remove_edge(edge)
        self.remove_edge(edge)
        edge._tgt = None
        edge._src = None

    def deref(self, fwd=True, bkwd=True):
        """ remove all references to this node
            todo - both directions
        """
        if fwd is True:
            for edge in self.successors(edges=True):
                self._remove(edge)
        if bkwd is True:
            for edge in self.predecessors(edges=True):
                self._remove(edge)

    @property
    def complete(self):
        return self.get('solved', False)

    def propogate(self):
        return

    # python -------------------------------------------
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False

    def __contains__(self, item):
        return self.__getitem__(item) is not None

    def __getitem__(self, item, **kwargs):
        """ DEPRECATED """
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
                    yield x
        del seen

    def __call__(self, fn, acc, **kwargs):
        """ DEPRECATED """
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
        keys = ['npred', 'nsucs', 'type']
        for k in keys:
            # self.write()
            pass

    def fill_cells(self):
        keys = ['npred', 'nsucs']
        for k in keys:

            val = self.get(k, None)
            print(k, val)
            #if val:
                #v.set_contents(val)

    def _make_item_fn(self, item):
        if isinstance(item, type(self.geom)):
            item_fn = lambda x: x.geom
        elif isinstance(item, Node):
            item_fn = lambda x: x
        else:
            item_fn = lambda x: x.data
        return item_fn


class Geom(object):
    def __init__(self, obj=None, **kwargs):
        super(Geom, self).__init__()
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, v):
        self._obj = v

    @property
    def has_geom(self):
        return self._obj is not None


class GeomEdge(Edge, Geom):
    def __init__(self, source, target, obj=None, **kwargs):
        Edge.__init__(self, source, target, **kwargs)
        Geom.__init__(self, obj)


class GeomNode(Node, Geom):
    def __init__(self, geom, obj=None, **kwargs):
        Node.__init__(self, geom,  **kwargs)
        Geom.__init__(self, obj)


