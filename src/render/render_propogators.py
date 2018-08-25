from src.propogate.propagators import BasePropogator, FuncPropogator
from src.geom import add_coord, set_coord
from src.rules.graph import Node



class RedrawPropogator(BasePropogator):
    def __init__(self, var, fn=None, geom=None, **kwargs):
        self.geom = geom
        self._fn = fn
        super(RedrawPropogator, self).__init__(name=var, **kwargs)
        self.seen = set()

    def on_default(self, node, _, **kwargs):
        if node.id not in self.seen:
            self.seen.add(node.id)
            return self._fn(node, self.var, self.geom), _
        return node, _


def set_z(node, z):
    geom = list(node.geom)
    geom[-1] = z
    node.update_geom(tuple(geom))
    return node


def riser_fn(node, var, z):
    if node.get(var, False) is True:
        edge = node.successors(edges=True)[0]
        # print(edge)
        geom = add_coord(node.geom, z=z)
        new_node = Node(geom)
        edge.split(new_node)
        new_node = FuncPropogator(set_z)(new_node, data=z)
        # print(new_node)
        return new_node
    return node


def drop_fn(node, var, z):
    """
    o----x


    o
    :param node:
    :param var:
    :param z:
    :return:
    """
    if node.get(var, False) is True:
        edge = node.predecessors(edges=True)[0]
        # node
        new_node = Node(node.geom)
        node.update_geom(set_coord(node.geom, z=z))
        edge.split(new_node)
        node.connect_to(new_node)
        return new_node
    return node


def vbranch(node, var, z):
    # todo ----------------------------------
    if node.get(var, False) is True:
        for edge in node.successors(edges=True):
            if edge.get('direction_change') is True:

                geom = add_coord(node.geom, z=z)
                new_node = Node(geom)
                edge.split(new_node)
                new_node = FuncPropogator(set_z)(new_node, data=z)
                return new_node
    return node


