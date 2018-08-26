import networkx as nx
from src.structs import Node

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


def nx_to_nodes(system):
    G, root = system.G, system.root
    seen, q = set(), [root]
    tmp = {}
    while q:
        el = q.pop(0)
        if el not in seen:
            seen.add(el)
            pred = list(G.predecessors(el))
            sucs = list(G.successors(el))

            data = G.nodes[el]
            if 'symbol_id' in data:
                chld = data.pop('children', [])
            nd = Node(el, **data)
            for x in pred:
                if x in tmp:
                    tmp[x].connect_to(nd, **G[x][el])
            for x in sucs:
                if x in tmp:
                    nd.connect_to(tmp[x], **G[el][x])
            tmp[nd.geom] = nd
            q.extend(pred + sucs)

    root_node = tmp[root]
    return root_node


def sys_to_nx(system):
    import networkx as nx
    G = nx.DiGraph()
    for _, node in system._node_dict.items():
        for n in node.successors():
            G.add_edge(node.geom, n.geom)

    return G


def nodes_to_nx(root, fwd=True, bkwd=False):
    import networkx as nx
    G = nx.DiGraph()
    for node in root.__iter__(fwd=fwd, bkwd=bkwd):
        G.add_node(node.geom, **{**node.data, **node.tmps})

    for node in root.__iter__(fwd=fwd, bkwd=bkwd):
        for n in node.successors():
            e = node.edge_to(n)
            G.add_edge(node.geom, n.geom, **e.tmps)

    return G


