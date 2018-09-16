from .node import Node
from shapely.geometry import MultiPoint
from src.geom import to_Nd


def delete_between(node, source, target):
    prd = source.edge_to(node)
    suc = target.edge_to(node)
    source.connect_to(target)

    node.remove_edge(suc)
    node.remove_edge(prd)
    source.remove_edge(prd)
    target.remove_edge(suc)


def node_at(root, coords):
    for n in root.__iter__():
        if n.geom == coords:
            return n


def geom_merge(*nodes):
    res = []
    ids = set()
    for n in nodes:
        ids.add(n.id)
        res.append(n.as_point)
    mlp_center = MultiPoint(res).centroid
    node = Node(to_Nd(mlp_center))

    for n in nodes:
        for ins in n.predecessors(edges=True):
            if ins.other_end(n).id not in ids:
                ins.other_end(n).connect_to(node)
                n.remove_edge(ins)
                ins.other_end(n).remove_edge(ins)
                del ins
        for ins in n.successors(edges=True):
            if ins.other_end(n).id not in ids:
                node.connect_to(ins.other_end(n))
                n.remove_edge(ins)
                ins.other_end(n).remove_edge(ins)
                del ins
    for n in nodes:
        del n
    return node


# def apply_ppg(node):


