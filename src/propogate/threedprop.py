from .base import QueuePropogator
import lib.geo
import importlib
importlib.reload(lib.geo)
from lib.geo import Line, Point, Movement
from ..structs.node_utils import tuplify, node_with_id
import numpy as np
from scipy.spatial import kdtree


class SpatialRoot(QueuePropogator):
    """
    Find a node that has only one connected neighbor
    and is at a geometric extent of the graph
    """
    def __init__(self):
        super(SpatialRoot, self).__init__(bkwd=True)
        self.dist = 0
        self.best = None

    def on_first(self, node, **kwargs):
        if not isinstance(node, list):
            return [node]
        return node

    def on_default(self, node, **kwargs):
        if len(node.neighbors(fwd=True, bkwd=True)) == 1:
            mag = np.sum(np.array(node.geom) ** 2) ** 0.5
            if mag > self.dist:
                self.dist = mag
                self.best = node.id
        return node

    def on_complete(self, node):
        return node_with_id(node, self.best)


class NearestTo(SpatialRoot):
    def __init__(self, loc, exclude=None):
        super(NearestTo, self).__init__()
        self._loc = np.array(loc)
        self._exclude = exclude

    def on_default(self, node, **kwargs):
        if node.id in self._exclude:
            return node
        mag = np.sum((np.array(node.geom) - self._loc) ** 2) ** 0.5
        if mag > self.dist:
            self.dist = mag
            self.best = node.id
        return node


class KDTreeIndex(QueuePropogator):
    def __init__(self, **kwargs):
        super(KDTreeIndex, self).__init__(**kwargs)
        self._data = []
        self._index = []
        self._root = None

    def get_node(self, nid):
        return node_with_id(self._root, nid)

    def __getitem__(self, index):
        node_id = self._index[index]
        return self.get_node(node_id)

    def get_index_id(self, index):
        return self._index[index]

    @property
    def data(self):
        return self._res.data

    def query_ball_tree(self, other, r):
        """"""
        return self._res.query_ball_tree(other._res, r)

    # walking interface ----------------
    def on_first(self, node, **kwargs):
        self._root = node
        if not isinstance(node, list):
            return [node]
        return node

    def on_default(self, node, **kwargs):
        self._data.append(list(node.geom))
        self._index.append(node.id)
        return node

    def on_complete(self, node):
        self._res = kdtree.KDTree(np.array(self._data))
        return self._res


class MovementQ(QueuePropogator):
    """ apply movement to each Node Point """
    def __init__(self, m):
        super(MovementQ, self).__init__()
        self.M = m

    def on_default(self, node, **kwargs):
        pt = Point(node.geom)
        pt2 = self.M.on_point(pt)
        node.geom = tuplify(pt2.numpy)
        return node


class Rotator(MovementQ):
    """ rotate around 0 with angle """
    def __init__(self, angle):
        l1 = Line(Point(0, 0, 0), Point(1, 0, 0))
        l2 = Line(Point(0, 0, 0), Point(np.cos(angle), np.sin(angle), 0))
        M = Movement(l2, l1)
        MovementQ.__init__(self, M)


class Scale(QueuePropogator):
    def __init__(self, scale):
        QueuePropogator.__init__(self)
        self.scale = scale

    def on_default(self, node, **kwargs):
        pt = self.scale * Point(node.geom).numpy
        node.geom = tuplify(pt)
        return node


class SlopeChange(QueuePropogator):
    def __init__(self, scale):
        QueuePropogator.__init__(self)
        self.scale = scale




def rotate_graph(root_node, angle):
    return Rotator(angle)(root_node)






