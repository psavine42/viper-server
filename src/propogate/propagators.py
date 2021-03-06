from src import geom
from .base import BasePropogator, EdgePropogator, QueuePropogator

import numpy as np


# ---------------------------------------------------------------
class DistanceFromSource(BasePropogator):
    def __init__(self, name='distance_to_source', **kwargs):
        super(DistanceFromSource, self).__init__(name, **kwargs)

    def on_first(self, node, prev_data, **kwargs):
        node.write(self._var, 0)
        return node, 0

    def on_default(self, node, prev_data, **kwargs):
        new_data = prev_data + 1
        current = node.get(self._var, None)
        if current is None:
            node.write(self._var, new_data)
            return node, new_data
        elif current < new_data:
            return node, new_data
        else:
            node.write(self._var, new_data)
            return node, new_data


class BuildOrder(QueuePropogator):
    """
    Writes the build order to each node and edge.
    The build order is used to index revit Builds
    """
    def __init__(self, name='order', **kwargs):
        super(BuildOrder, self).__init__(**kwargs)
        self.var = name
        self.node_cnt = 0
        self.edge_cnt = 0
        self.edge_seen = set()

    def on_default(self, node,  **kwargs):
        node.write(self.var, self.node_cnt)
        self.node_cnt += 1
        for edge in node.successors(edges=True):
            if edge.id not in self.edge_seen:
                self.edge_seen.add(edge.id)
                edge.write(self.var, self.edge_cnt)
                self.edge_cnt += 1
        return node


class ElevationChange(BasePropogator):
    def __init__(self, name='elevation', **kwargs):
        super(ElevationChange, self).__init__(name, **kwargs)

    def on_first(self, node_and_edge, data, **kwargs):
        node, edge = node_and_edge
        new_geom = geom.add_coord(node.geom, z=data)
        new_node = edge.split(new_geom)
        return new_node, data

    def on_default(self, node, elevation_delta, **kwargs):
        new_geom = geom.add_coord(node.geom, z=elevation_delta)
        node.update_geom(new_geom)
        return node, elevation_delta


class Where(QueuePropogator):
    """ Searches the graph for first node which condition(node) == True.
        if the search is complete without the condition
        being met, return None
    """
    def __init__(self, condition, **kwargs):
        super(Where, self).__init__(**kwargs)
        self.fn = condition

    def is_terminal(self, node, **kwargs):
        return self.fn(node, **kwargs)

    def on_complete(self, node):
        return None


class DistanceFromEnd(BasePropogator):
    def __init__(self, name='dist_to_end', **kwargs):
        super(DistanceFromEnd, self).__init__(name=name, **kwargs)

    def is_terminal(self, node, prev_data, **kwargs):
        return len(node.successors()) == 0

    def on_terminal(self, node, _, **kwargs):
        DistanceFromSource(self.var, reverse=True)(node)
        return node, _


class LoopDetector(BasePropogator):
    def __init__(self, name='on_loop', **kwargs):
        super(LoopDetector, self).__init__(name=name, **kwargs)

    def is_terminal(self, node, prev_data, **kwargs):
        return node.id in prev_data

    def on_terminal(self, node, prev_data, **kwargs):
        step = prev_data.index(node.id)
        path = prev_data[step:]
        for n in node.__iter__(bkwd=True, fwd=False):
            if n.id in path:
                n.write(self.var, True)
        return node, prev_data

    def on_default(self, node, prev_data, **kwargs):
        prev_data.append(node.id)
        return node, prev_data


class DirectionWriter(EdgePropogator):
    def __init__(self, name='direction_change',  **kwargs):
        super(DirectionWriter, self).__init__(name=name, **kwargs)

    def on_default(self, edge, prev_direction, **kwargs):
        """ Branch
               ^        <--+-->
               |           ^
            -->+-->        |

            two lines are colinear, and one is orthagonal
        """
        new_dir = edge.direction
        res = False if prev_direction is None else np.allclose(new_dir, prev_direction)

        if res is True:
            edge.write(self.var, False)
        else:
            edge.write(self.var, True)

        # if edge is not None:
        #    angle = np.angle(new_dir - prev_direction)
        #    edge.write('angle', angle)
        return edge, new_dir


class FuncPropogator(BasePropogator):
    def __init__(self, fn, **kwargs):
        super(FuncPropogator, self).__init__(name=fn.__name__, **kwargs)
        self._fn = fn

    def on_default(self, node_or_edge, prev_data, **kwargs):
        res = self._fn(node_or_edge, prev_data, **kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            node, new_data = res
            return node, new_data
        return node_or_edge, prev_data

    def __call__(self, node, **kwargs):
        super(FuncPropogator, self).__call__(node, **kwargs)
        self.seen = set()
        return node


# ---------------------------------------------------------------
class EdgeDirector(QueuePropogator):
    """ Creates direction for each edge
        Input is a node attached to a undrirected graph, which
        should become directed starting from that node
    """
    def __init__(self,  **kwargs):
        super(EdgeDirector, self).__init__(**kwargs)

    def on_default(self, node, **kwargs):
        for pred in node.predecessors():
            if pred.id not in self.seen:
                edge = pred.edge_to(node)
                edge.reverse()
        return node


class GraphTrim(QueuePropogator):
    """
    If two edges have same direction, merge them
    """
    def __init__(self,  **kwargs):
        super(GraphTrim, self).__init__(name=None, **kwargs)

    def on_default(self, node, tol=0.001, **kwargs):
        sucs = node.successors(edges=True)
        pred = node.predecessors(edges=True)
        if len(sucs) == 1 and len(pred) == 1:
            suc, prd = sucs[0], pred[0]
            crv1 = geom.MepCurve2d(*prd.geom)
            crv2 = geom.MepCurve2d(*suc.geom)
            if np.allclose(crv1.direction, crv2.direction, atol=tol):
                cur_src = prd.source
                new_tgt = suc.target
                cur_src.connect_to(new_tgt, **{**prd.tmps, **suc.tmps})
                node.remove_edge(suc)
                node.remove_edge(prd)
                cur_src.remove_edge(prd)
                new_tgt.remove_edge(suc)
                return cur_src
        return node


class Chain(object):
    def __init__(self, *props):
        self._props = props

    def __call__(self, root):
        for prop in self._props:
            root = prop(root)
        return root







def ApplyProps(node, fn):
    for n in node.__iter__():
        pred = n.predecessors(edges=True)
        sucs = n.successors(edges=True)
        for p in pred:
            for s in sucs:
                Cell()


