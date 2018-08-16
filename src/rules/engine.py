from src.rules.graph import Node, Edge
from collections import defaultdict as ddict
from collections import Counter
from .property import Property
from . import opers

class KB(object):
    def __init__(self):
        self._facts = {}

    def add_rule(self, prop):
        self._facts[prop.id] = prop

    def root(self):
        return



class RuleEngine(object):
    def __init__(self, root=None, term_rule=None):
        self.root = root
        self._term_rule = term_rule
        self._first_op = term_rule
        self._rules = dict() # todo ordered dict
        self._freq = None
        self._post_conds = ddict(set)
        self._pre_conds = ddict(set)
        if term_rule is not None:
            self._preprocess(term_rule)

    def _preprocess(self, terminal):
        q = [terminal]
        cnt = Counter()
        while q:
            el = q.pop(0)

            self._rules[el.id] = el
            pre = el.pre_conditions()
            for x in pre:
                self._post_conds[x.id].add(el.id)
                self._pre_conds[el.id].add(x.id)
            q.extend(pre)
            cnt[el.id] += 1
        self._freq = cnt
        # todo compute order of ideal applications
        # this is some F of likelyhood of transitions

    def post_conditions(self, rule):
        if isinstance(rule, Property):
            for p in rule.post_conditions():
                yield p
        else:
            for p in self._post_conds[rule.id]:
                yield self._rules[p]

    def plot(self, root, labels):
        import random
        import src.visualize
        meta = {}
        for n in labels:
            meta[n] = {'size': random.randint(0, 250), 'color': random.random()}

        print(meta)
        for n in root.__iter__():
            print(str(n), n.nsucs, n.npred, n.tmps, )
            for mk in meta.keys():
                if n.tmps.get(mk, None) is True:
                    n.write('type', mk)

        def label_fn(n, d):
            sym = d.get('type', '')
            return '{} {}'.format(sym, n)

        nxg = nodes_to_nx(root)
        src.visualize._plot(nxg, label_fn, meta=meta)


    def yield_queue(self, root):
        """
        computing direction:
            if node has 'count':
                put in todoq
                if distance to source decreases:
                    relabel
                    continue
            elif not has count:
                pcount = pred_node.count
                this.count =  pcount + 1

        idea is automaton

        if rule.not_stuck:
            rule.go_next(node)
        else:
            yield (rule, node)

        :param root:
        :return:
        """
        op_q = [self._rules[x] for x, v in self._freq.most_common()]
        seen = set()
        cnt = 0
        while op_q:
            op = op_q.pop(0)
            q = [(op, root)]

            while q:
                rule, node = q.pop(0)

                for result, n_node in rule.walk2(node):
                    cnt += 1
                    if result is True:
                        # print(rule, str(node), node.tmps)
                        for post in self.post_conditions(rule):
                            q.append((post, n_node))
        print('Num iters :{}'.format(cnt))
        return root


    def propogate(self, root_node):
        """
        Write
        :return:
        """
        # node_q = [(root_node, None, []) ]
        node_q = [root_node]

        while node_q:

            node, op_q, mutex = node_q.pop(0)
            op_q = list(self._rules.values())
            mutex = []

            while node.complete is False and op_q:

                this_op = op_q.pop(0)
                res = this_op(node)

                if res is True:
                    op_q.remove(this_op.mutex)
                    op_q.remove(this_op.pre_conditions())

                    # op_q.extend(this_op.post_pos())
                    # mutex.extend(this_op.mutex)

                elif res is False:
                    # rule removed
                    op_q.remove(this_op.post_pos())
                    op_q.remove(this_op.pre_conditions())

                    # op_q.extend(this_op.post_pos())
                    # if false, then the posts of True are removed

                elif res is None:
                    # rule could not be applied
                    # a precondition could not be met
                    # op_q = []
                    break
                else:
                    raise Exception

            node.propogate()
            node_q.extend(node.successors())

            if node.complete is False:
                node_q.append(node)

        return root_node


def props_to_nx(root):
    import networkx as nx
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


def nx_to_nodes(G, root):
    seen, q = set(), [root]
    tmp = {}
    while q:
        el = q.pop(0)
        if el not in seen:
            seen.add(el)
            pred = list(G.predecessors(el))
            sucs = list(G.successors(el))

            nd = Node(el, **G.nodes[el])
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


def nodes_to_nx(root):
    import networkx as nx
    G = nx.DiGraph()
    for node in iter(root):
        G.add_node(node.geom, **{**node.data, **node.tmps})

    for node in iter(root):
        for n in node.successors():
            G.add_edge(node.geom, n.geom)
        for n in node.predecessors():
            G.add_edge(n.geom, node.geom)

    return G


