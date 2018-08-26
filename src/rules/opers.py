from uuid import uuid4
import random


# ------------------------------------------------
# utils
def random_id():
    return random.randint(0, 1e6)


def mutex(*args):
    for i in range(len(args)):
        for j in range(len(args)):
            if i != j:
                if args[j] not in args[i].mutexes:
                    args[i].mutexes.append(args[j])


# ------------------------------------------------
class Condition(object):
    def __init__(self, *args, trigger=False, var=None, node=True, edge=True, **kwargs):
        self._prop = None
        self._var = var
        self._fn = None
        self._trigger = trigger
        self._id = random_id()
        self._on_node = node
        self._on_edge = edge
        self._frequency = 0
        self._post_pos = []
        self._post_neg = []
        self._pre_conds = []
        self.mutexes = []
        self._hist = {}

    def _get_conditions(self, xs):
        res = []
        if hasattr(xs, 'IConditional'):
            res.append(xs)
        elif type(xs) in [tuple, list]:
            for x in xs:
                res += self._get_conditions(x)
        return res

    def add_post_pos(self, cond):
        if self.iscond(cond):
            cond._pre_conds.append(self)
            self._post_pos.append(cond)

    def add_pre_pos(self, cond):
        if self.iscond(cond):
            cond._post_pos.append(self)
            self._pre_conds.append(cond)

    def add_post_neg(self, cond):
        if self.iscond(cond):
            cond._pre_conds.append(self)
            self._post_neg.append(cond)

    def add_pre_neg(self, cond):
        if self.iscond(cond):
            cond._post_neg.append(self)
            self._pre_conds.append(NOT(cond))

    def walk1(self, node):
        for n in iter(node):
            yield self.__call__(n)

    def walk2(self, node):
        for n in iter(node):
            res = self.__call__(n)
            yield (res, n)

    # flow ------------------------------------------------
    @property
    def var(self):
        return self._var

    @property
    def pre_conditions(self):
        return self._pre_conds

    @property
    def post_pos(self):
        return self._post_pos

    @property
    def post_neg(self):
        return self._post_neg

    @staticmethod
    def iscond(obj):
        return getattr(obj, 'IConditional', None) is True

    @property
    def IConditional(self):
        return True

    @property
    def _cls(self):
        return '<{}>'.format(self.__class__.__name__)

    @property
    def id(self):
        return '<{}>:{}'.format(self.__class__.__name__, str(self._id))

    def __iter__(self, node):
        res = self.__call__(node)
        if isinstance(list, res):
            for r in res:
                yield r
        else:
            yield res

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return self._cls + '' if not self._var else self._var

    # flow ------------------------------------------------
    def check_conditions(self, node):
        return True

    def activate(self, node, res):
        if res is True:
            for cond in self.post_pos:
                cond.apply(node)
            for cond in self.post_neg:
                cond.on_eval(node, False)

        elif res is False:
            pass

    def on_eval(self, node, res):
        if self._var is not None:
            node.write(self._var, res)

    def __call__(self, node):
        raise NotImplementedError

    def apply(self, node):
        self.check_conditions(node)
        res = self.__call__(node)
        self.on_eval(node, res)
        if self._trigger is True:
            self.activate(node, res)
        return res


# ------------------------------------------------
class Mutex(Condition):
    def __init__(self,  *args, **kwargs):
        super(Mutex, self).__init__(**kwargs)
        for arg in args:
            self.add_pre_pos(arg)
        mutex(args)

    def __call__(self, node):
        n_true = 0
        for cond in self.pre_conditions:
            res = cond(node)
            if res is True:
                n_true += 1
        return n_true == 1


class IF(Condition):
    def __init__(self, prop, fn, val, **kwargs):
        super(IF, self).__init__(**kwargs)
        self._name = prop
        self._fn = fn
        self._val = val
        self.add_pre_pos(fn)

    def __str__(self):
        st = super(IF, self).__str__()
        return st + str(self._name) + str(self._fn) + str(self._val)

    def __call__(self, node):
        prop = node.get(self._name, None)
        if prop is None:
            return False
        return self._fn(prop, self._val)


class NOT(Condition):
    def __init__(self, cond, **kwargs):
        super(NOT, self).__init__(**kwargs)
        self._op = cond
        self.add_pre_pos(cond)

    def __call__(self, node):
        pre = self._op(node)
        if pre is None:
            return None
        return not pre


class HAS(Condition):
    def __init__(self, prop, **kwargs):
        super(HAS, self).__init__(**kwargs)
        self._prop = prop

    def __str__(self):
        return '<{}>: {} "{}"'.format(self._cls, self.id, self._prop)

    def __call__(self, node):
        prop = node.get(self._prop, None)
        return prop is not None


class OR(Condition):
    """
        cond | node
        -----+------
        many | one
    """
    def __init__(self, *args, **kwargs):
        super(OR, self).__init__(**kwargs)
        self.conds = args
        for cond in args:
            self.add_pre_pos(cond)

    def __call__(self, node):
        return any([c(node) for c in self.conds])


class AND(Condition):
    """
        cond | node
        -----+------
        many | one
    """
    def __init__(self, *conds, **kwargs):
        super(AND, self).__init__(*conds, **kwargs)
        for cond in conds:
            self.add_pre_pos(cond)

    def __call__(self, node):
        for c in self.pre_conditions:
            res = c(node)
            if res is not True:
                return res
        return True


class ANY(Condition):
    """
        cond | node
        -----+------
         one | many
    """
    def __init__(self, cond, **kwargs):
        super(ANY, self).__init__(**kwargs)
        self.cond = cond

    def __call__(self, nodes):
        return any([self.cond(c) is True for c in nodes])


class INSUCS(Condition):
    def __init__(self, op, fn, **kwargs):
        super(INSUCS, self).__init__(**kwargs)
        self.cond = op
        self._fn = fn

    def __call__(self, node):
        alls = []
        for x in node.__iter__():
            res = self.cond(x)
            alls.append(res)
        return self._fn(alls)


