from itertools import chain
import math


def tuplify(np_pt):
    return tuple(np_pt.tolist())


def edge_between(n1, n2):
    e1 = n1.edge_to(n2)
    if e1 is not None:
        return e1
    return n2.edge_to(n1)


def norm_angle(pred_edge, suc_edge, degrees=True):
    a = pred_edge.curve.line.angle_to(suc_edge.curve.line)
    if degrees is True:
        return math.degrees(a)
    return a


def edge_with_id(root_node, eid):
    roots = [root_node] if not isinstance(root_node, list) else root_node
    for root in roots:
        for n in root.__iter__(True, True):
            for e in n.neighbors(True, True, True):
                if e.id == eid:
                    return e


def node_with_id(root_node, eid):
    roots = [root_node] if not isinstance(root_node, list) else root_node
    for root in roots:
        for n in root.__iter__(fwd=True, bkwd=True):
            if n.id == eid:
                return n


def common_node(*edges):
    u = list(set.intersection(*[{e.source.id, e.target.id} for e in edges]))
    if len(u) == 1:
        return edges[0].source if edges[0].id == u[0] else edges[0].target


def slope(edge):
    line = edge.curve.line.numpy
    return (line[1, -1] - line[0, -1])/edge.curve.line.length

