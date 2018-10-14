import networkx as nx
from src.structs import Node
from rtree import index

__ROUND = 4


def make_index():
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.storage = index.RT_Memory
    p.overwrite = True
    idx3 = index.Index(properties=p)
    return idx3


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


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


def cells_to_nx(res):
    """ visualize propagator network with nx """
    q = []
    for r in res:
        q += r.neighbors
    G = nx.DiGraph()
    seen = set()
    while q:
        el = q.pop(0)
        if el.id not in seen:
            seen.add(el.id)
            G.add_node(el.id, type='prop', fn=str(el._fn.__name__), cnt=el._cnt)
            out = el.output
            G.add_node(out.id, type='cell', content=str(out.contents), var=out._var)
            G.add_edge(el.id, out.id, weight=el._cnt)
            q.extend(out.neighbors)
            for n in el.inputs:
                if n.id not in seen:
                    G.add_node(n.id, type='cell', content=str(n.contents), var=n._var)
                    q.extend(n.neighbors)
                    G.add_edge(n.id, el.id, weight=el._cnt)
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


def sys_to_nx(system, G=None, **kwargs):
    G = G if G else nx.DiGraph()
    for node in system:
        for n in node.neighbors():
            G.add_edge(node.geom, n.geom)
    return G


def nodes_to_nx(root, fwd=True, bkwd=False, G=None):

    G = G if G else nx.DiGraph()
    for node in root.__iter__(fwd=fwd, bkwd=bkwd):
        G.add_node(node.geom, **{**node.data, **node.tmps})

    for node in root.__iter__(fwd=fwd, bkwd=bkwd):
        for n in node.successors():
            e = node.edge_to(n)
            G.add_edge(node.geom, n.geom, **e.tmps)

    return G


def bunch_to_nx(nodes_list, **kwargs):
    G = nx.DiGraph()
    for n in nodes_list:
        G = sys_to_nx(n, G=G, **kwargs)

    return G
