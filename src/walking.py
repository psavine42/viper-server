

def _extend_both(pred, sucs):
    return pred + sucs


def _extend_pred(pred, sucs):
    return pred


def _extend_sucs(pred, sucs):
    return sucs


def _pred_fn(G, el, e2):
    return list(G.predecessors(el))


def _sucs_fn(G, el, e2):
    return list(G.successors(e2))


def _prepare(root):
    seen, terminal = set(), False
    q = root if isinstance(root, list) else [root]
    return seen, terminal, q


def walk_G(G, root, fn):
    seen, terminal, q = _prepare(root)
    while q and terminal is not True:
        el = q.pop(0)
        if el not in seen:
            seen.add(el)
            pred = list(G.predecessors(el))
            sucs = list(G.successors(el))
            fn(G, el, pred, sucs, seen)
            q.extend(set(pred + sucs).difference(seen))


def do_walk(G, root, fn, p_index=0, ext_fn=_extend_both):
    seen, terminal, q = _prepare(root)
    while q and terminal is not True:
        el = q.pop(p_index)
        if el not in seen:
            seen.add(el)
            pred = list(G.predecessors(el))
            sucs = list(G.successors(el))
            terminal = fn(el, pred, sucs, seen)
            new = ext_fn(pred, sucs)
            q.extend(new)


def search(G, root, fn, p_index=0, ext_fn=_extend_both):
    seen, terminal, q = _prepare(root)
    while q and terminal is not True:
        el = q.pop(p_index)
        if el not in seen:
            seen.add(el)
            pred = list(G.predecessors(el))
            sucs = list(G.successors(el))
            terminal, sucs, pred = fn(el, pred, sucs, seen)
            new = ext_fn(pred, sucs)
            q.extend(set(new).difference(seen))


def walk_edges(G, root, fn, p_index=0, ext_fn=_extend_both,
               predf=_pred_fn, sucsf=_sucs_fn):
    seen, terminal = set(), False
    rsucs = G.successors(root)
    q = [(root, x) for x in rsucs]
    while q and terminal is not True:
        el = q.pop(p_index)
        if el not in seen:
            seen.add(el)
            e1, e2 = el
            pred = predf(G, e1, e2)
            sucs = sucsf(G, e1, e2)
            terminal = fn(e1, e2, pred, sucs, seen)
            new = ext_fn(pred, sucs)

            q.extend([(e2, x) if x in sucs else (x, e1)
                      for x in new])


def walk_dfs(G, root, fn):
    do_walk(G, root, fn, p_index=-1, ext_fn=_extend_both)


def walk_dfs_forward(G, root, fn):
    do_walk(G, root, fn, p_index=-1, ext_fn=_extend_sucs)


def walk_edges_dfs(G, root, fn):
    walk_edges(G, root, fn, p_index=-1, ext_fn=_extend_both)


def walk_backward(G, root, fn):
    do_walk(G, root, fn, p_index=0, ext_fn=_extend_pred)
