import networkx as nx


__ROUND = 4


def round_tup(pts, r=__ROUND):
    return tuple([round(p, r) for p in pts])


def props_to_nx(root):

    G = nx.DiGraph()
    q = [root]
    while q:
        n = q.pop(0)
        str_nd = str(n)
        G.add_node(str_nd)
        for p in n.pre_conditions():
            G.add_edge(str(p), str_nd)
            q.append(p)
    return G
