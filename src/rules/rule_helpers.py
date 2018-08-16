import numpy as np


def is_main(node, acc, target=None, seen=None):
    """ is node reachable by a loop in the graph ? """
    if seen is None:
        seen = set()
    seen.add(node.id)

    res = []
    for n in node.successors():
        if n.id == target:
            return n
        if n.id not in seen:
            itm = is_main(n, acc, target, seen=seen)
            if any(itm):
                res.append(n)

    if any(res):
        return res
    return [False]


def translate(node, x=0, y=0, z=0):
    arr = np.array([x, y, z])

    def update_z(xyz):
        xyz = np.array(list(xyz))
        xyz += arr
        return tuple(xyz.tolist())

    for n in iter(node):
        n.update_geom(update_z)
    return node





