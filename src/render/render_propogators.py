from src.propogate import BasePropogator, FuncPropogator
from src.propogate.base import RecProporgator
from src.geom import set_coord
from src.structs import Node
from src.geom import MepCurve2d
import numpy as np


class RedrawPropogator(BasePropogator):
    def __init__(self, var, fn=None, geom=None, **kwargs):
        self.geom = geom
        self._fn = fn
        super(RedrawPropogator, self).__init__(name=var, **kwargs)
        self.seen = set()

    def on_default(self, node, _, **kwargs):
        if node.id not in self.seen:
            self.seen.add(node.id)
            if node.get(self.var, None) is True:
                return self._fn(node, self.geom), _
        return node, _


class Annotator(RecProporgator):
    """
        if node has propoerty k in mapping, then
        write value v to node with ket self.var
    Usages:
        rp.Annotator('$create', mapping={'dHead': 1, }),

    """
    def __init__(self, var, mapping={}, **kwargs):
        self.mapping = mapping
        super(Annotator, self).__init__(name=var, **kwargs)

    def on_default(self, node, _, **kwargs):
        for k, v in self.mapping.items():
            if node.get(k, None) is True:
                node.write(self.var, v)
            return node, _


class AddSlope(RecProporgator):
    def on_default(self, node, prev_data, slope=0.0, **kwargs):
        leveled = list(prev_data)
        leveled[-1] = node.geom[-1]
        crv = MepCurve2d(node.geom, leveled)
        this_z = list(prev_data)[-1] + crv.length * slope
        node = set_z(node, this_z)
        return node, this_z


class Translate(RecProporgator):
    def on_default(self, node, prev_data, **kwargs):
        new = np.array(node.geom) + prev_data
        node.update_geom(tuple(new.tolist()))
        return node, prev_data


def set_z(node, z):
    geom = list(node.geom)
    geom[-1] = z
    node.update_geom(tuple(geom))
    return node


def update_z(node, z):
    geom = list(node.geom)
    geom[-1] += z
    node.update_geom(tuple(geom))
    return node


def vertical(node,  z):
    base_g = node.geom
    geom = list(base_g)
    geom[-1] = 0.
    node.update_geom(tuple(geom))
    edge = node.successors(edges=True)[0]
    new_node = Node(base_g)
    edge.split(new_node)

    geom[-1] = z
    rise_node = Node(tuple(geom))
    new_node.connect_to(rise_node)
    return new_node


def riser_fn(node,  z):
    edge = node.successors(edges=True)[0]
    new_node = Node(node.geom)
    edge.split(new_node)
    new_node = FuncPropogator(update_z)(new_node, data=z)
    return new_node


def drop_fn(node, z):
    """
    o----x


    o
    :param node:
    :param var:
    :param z:
    :return:
    """
    edge = node.predecessors(edges=True)[0]
    new_node = Node(node.geom)
    node.update_geom(set_coord(node.geom, z=z))
    edge.split(new_node)
    node.connect_to(new_node)
    return new_node


def vbranch(node, z):
    for edge in node.successors(edges=True):
        if edge.get('direction_change') is True:
            new_node = Node(node.geom)
            edge.split(new_node)
            FuncPropogator(update_z)(new_node, data=z)
            return node



