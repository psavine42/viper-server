from src.rules.graph import Node, Edge
from collections import defaultdict as ddict
from collections import Counter
from .property import Property
from . import opers
from copy import deepcopy
from viper import nodes_to_nx
import random


class KB(object):
    def __init__(self, root):
        self._root = root
        self._dict = {}
        self._init_rules = None
        self._build_agenda()

    def __str__(self):
        st = 'KB with {} rules, initial rules {}'.format(len(self._dict), len(self.agenda))
        return st

    def show(self):
        for k, v in self._dict.items():
            print(v, v.post_pos, v.post_neg)

    def get_vars(self):
        q = [self._root]
        vars = set()
        seen = ()
        while q:
            el = q.pop(0)
            if el.id not in seen:
                if el.var is not None:
                    vars.add(el.var)
                for x in el.pre_conditions:
                    q.append(x)
        return vars

    def compute_additional_mutexes(self):
        pass

    def plot(self):
        vars_ = self.get_vars()

    def _build_agenda(self):
        q = [self._root]
        vars = dict()
        seen = ()
        while q:
            el = q.pop(0)
            if el.id not in seen:
                self._dict[el.id] = el
                vars[el.id] = len(el.pre_conditions)
                for x in el.pre_conditions:
                    q.append(x)
        return [x[0] for x in sorted(vars.items(), key=lambda x: x[1]) if x[1] == 0]

    def __getitem__(self, item):
        return self._dict[item]

    @property
    def agenda(self):
        if self._init_rules is None:
            self._init_rules = self._build_agenda()
        return self._init_rules

    @property
    def root(self):
        return self._root


class Recorder(object):
    def __init__(self, debug=False, mx=1e6, nlog=10):
        self._debug = debug
        self._history = []
        self._max = mx
        self._log_every = nlog
        self._cnt = 0

    @property
    def step(self):
        return self._cnt

    def inc(self):
        self._cnt += 1

    def report(self, *args):
        if (self._debug is True) and (self._cnt % self._log_every == 0):
            print('ITER', self._cnt, *args)

    def stop(self):
        if self._max < self._cnt:
            return True
        return False

    def on_op(self, op, node, res):
        self._history.append((op.id, node.id, res))


class RuleEngine(object):
    def __init__(self, root=None, term_rule=None, **kwargs):
        self.root = root
        self._term_rule = term_rule
        self._first_op = term_rule
        self._rules = dict() # todo ordered dict
        self._freq = None

        self._post_conds = ddict(set)
        self._pre_conds = ddict(set)

        self.logger = Recorder(**kwargs)
        if term_rule is not None:
            self._preprocess(term_rule)

    def _preprocess(self, terminal):
        q = [terminal]
        cnt = Counter()
        while q:
            el = q.pop(0)

            self._rules[el.id] = el
            pre = el.pre_conditions
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


    def annotate_type(self, root, labels):
        meta = {}
        for n in labels:
            meta[n] = {'size': random.randint(0, 250), 'color': random.random()}
        for n in root.__iter__():
            for mk in meta.keys():
                if n.tmps.get(mk, None) is True:
                    n.write('type', mk)
        return meta

    def plot(self, root, labels):
        import src.visualize
        meta = self.annotate_type(root, labels)
        def label_fn(n, d):
            sym = d.get('type', '')
            return '{}'.format(sym)
        nxg = nodes_to_nx(root)
        src.visualize._plot(nxg, label_fn,  meta=meta)

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

    def report(self, *args):
        self.logger.report(*args)

    def log(self, *args):
        self.logger.on_op(*args)
        self.logger.report(*args)
        self.logger.inc()

    def alg2(self, root_node, kb):
        """
        Write
        :return:
        """
        agenda = kb.agenda
        nodes_q = [root_node]
        opers_q = [deepcopy(agenda)]

        while nodes_q:

            node = nodes_q.pop(0)
            op_q = opers_q.pop(0)

            self.report(node, len(op_q), len(nodes_q))
            if node.complete is True:
                continue

            while node.complete is False and op_q:

                this_op = kb[op_q.pop(0)]
                res = this_op.apply(node)
                self.log(this_op, node, res)

                if res is True:

                    remove = this_op.mutexes + this_op.pre_conditions + this_op.post_neg
                    for op in remove:
                        if op.id in op_q:
                            op_q.remove(op.id)

                    for op in this_op.post_pos:
                        if op.id not in op_q:
                            op_q.insert(0, op.id)

                elif res is False:

                    remove = this_op.post_pos + this_op.pre_conditions
                    for op in remove:
                        if op.id in op_q:
                            op_q.remove(op.id)

                elif res is None:
                    op_q.append(this_op.id)
                    break
                else:
                    raise Exception

            node.propogate()    # placeholder

            for suc in node.successors():
                if not suc.complete  and suc not in nodes_q:
                    nodes_q.append(suc)
                    opers_q.append(deepcopy(agenda))

            if node.complete is False and op_q:
                nodes_q.append(node)
                opers_q.append(op_q)

            self.report(node, len(op_q), len(nodes_q))
            self.report(op_q)
            self.report('-----------')

            if self.logger.stop() is True:
                break
        print(*nodes_q)
        print('solved in {} iterations'.format(self.logger.step))
        return root_node









