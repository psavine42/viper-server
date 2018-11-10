
def tuplify(np_pt):
    return tuple(np_pt.tolist())


def edge_between(n1, n2):
    e1 = n1.edge_to(n2)
    if e1 is not None:
        return e1
    return n2.edge_to(n1)


def node_with_id(nd, eid):
    for n in nd.__iter__(fwd=True, bkwd=True):
        if n.id == eid:
            return n


