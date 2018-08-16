from .property import Property
from uuid import uuid4


class Condition:
    def __init__(self, *args, **kwargs):
        self._prop = None
        self.cond = args
        self._fn = []
        self._id = uuid4()
        self._frequency = 0
        self._post_pos = []
        self._post_neg = []
        self._pre_pos = []
        self._pre_neg = []

    def __call__(self, node):
        raise BaseException('not implemented in base class')

    def __str__(self):
        return self.__class__.__name__

    def _get_conditions(self, xs):
        res = []
        if hasattr(xs, 'IConditional'):
            res.append(xs)
        elif type(xs) in [tuple, list]:
            for x in xs:
                res += self._get_conditions(x)
        return res

    def pre_conditions(self):
        res = []
        for k, v in self.__dict__.items():
            res += self._get_conditions(v)
        return res

    def post_conditions(self):
        return []

    def __iter__(self, node):
        res = self.__call__(node)
        if isinstance(list, res):
            for r in res:
                yield r
        else:
            yield res

    def walk1(self, node):
        for n in iter(node):
            yield self.__call__(n)

    def walk2(self, node):
        for n in iter(node):
            yield (self.__call__(n), n)

    @property
    def IConditional(self):
        return True

    @property
    def id(self):
        return '<{}>:{}'.format(self.__class__.__name__, str(self._id))

    def __eq__(self, other):
        return self.id == other.id

    @classmethod
    def as_prop(cls, name, *args, **kwargs):
        return Property(name, cls(*args), **kwargs)


##############################################
def add_mutex(*args):
    for i in range(len(args)):
        args[i]._neg.append(args[0:i-1] + args[i:len(args)-1])


class IF(Condition):
    def __init__(self, prop, fn, val):
        super(IF, self).__init__()
        self._prop = prop
        self._fn = fn if getattr(fn, 'IConditional', None) is None else fn
        # self._o
        self._val = val

    def __str__(self):
        st = super(IF, self).__str__()
        return st + str(self.cond)

    def __call__(self, node):
        prop = node.get(self._prop, None)
        if prop is None:
            return False
        return self._fn(prop, self._val)


class NOT(Condition):
    def __init__(self, cond):
        super(NOT, self).__init__()
        self.cond = cond

    def __call__(self, node):
        pre = self.cond(node)
        return not pre


class HAS(Condition):
    def __init__(self, prop, fn=None):
        super(HAS, self).__init__()
        self._prop = prop
        self._op = fn

    def __call__(self, node):
        prop = node.get(self._prop, None)
        return prop is not None


class OR(Condition):
    """
        cond | node
        -----+------
        many | one
    """
    def __init__(self, *args):
        super(OR, self).__init__()
        self.conds = args

    def __call__(self, node):

        # print(node, res, self.conds)
        return any([c(node) for c in self.conds])


class AND(Condition):
    """
        cond | node
        -----+------
        many | one
    """
    def __init__(self, *conds):
        super(AND, self).__init__(*conds)

    def __call__(self, node):

        return all([c(node)  for c in self.cond])


class ANY(Condition):
    """
        cond | node
        -----+------
         one | many
    """
    def __init__(self, cond):
        super(ANY, self).__init__()
        self.cond = cond

    def __call__(self, nodes):
        return any([self.cond(c) is True for c in nodes])


class INSUCS(Condition):
    def __init__(self, op, fn):
        super(INSUCS, self).__init__()
        self.cond = op
        self.fn = fn

    def __call__(self, node):
        alls = []
        for x in node.__iter__():
            res = self.cond(x)
            alls.append(res)
        return self.fn(alls)


