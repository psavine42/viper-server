from .graph import Node
from uuid import uuid4


class Walker(object):
    def __init__(self, name, **kwargs):
        self.name= name

    def is_terminal(self):
        return

    def sample(self, sucs):
        return

    def _apply_op(self, node):
        return node

    def __call__(self, node):
        res = self._apply_op(node)
        return


class Property(object):
    def __init__(self, name, *conds, fn=None, node=True, edge=True):
        self._name = name
        self._id = uuid4()
        self._on_node = node
        self._on_edge = edge
        self._write_fn = fn
        self._range = -1
        self._sucs = []
        self._pred = []
        self._conds = []
        for cond in conds:
            if isinstance(cond, Property):
                self.add_pred(cond)
            else:
                self._conds.append(cond)
    @property
    def IConditional(self):
        return True

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return '<Prop>:<' + self.name + ">:" + str(self._id)

    def pre_conditions(self):
        return self._pred + self._conds

    def post_conditions(self):
        return self._sucs

    def add_pred(self, other):
        other._sucs.append(self)
        self._pred.append(other)

    def add_sucs(self, other):
        self._sucs.append(other)
        other._pred.append(self)

    def is_complete(self):
        """
        complete is when self is done,
        and
        :return:
        """

        pass

    def check_conditions(self, node):
        """

        ISEND
            if NSUCS(node) == 0 then True

        :param node: (Node)
        :return:
        """
        # if property is written, return that
        prop = node.get(self._name, None)
        if prop is not None:
            return prop

        # check the node in cache, and whether

        # otherwise calculate it on that node
        for cond in self.pre_conditions():
            if cond(node) is False:
                return False
        return True

    def _write_to(self, node, res):
        node.write(self._name, res)

    def apply(self, node):
        res = self.check_conditions(node)
        if res is True:
            if self._write_fn is not None:
                data = self._write_fn(node)
                node.write(self._name, data)
                return res
        node.write(self._name, res)
        return res

    def __call__(self, node):
        self.apply(node)

    def __str__(self):
        st = '<Prop>:<{}>'.format(self._name)
        return st

    def walk1(self, node):
        for n in iter(node):
            yield self.__call__(n)

    def walk2(self, node):
        for n in node.__iter__():
            res = self.apply(n)
            yield (res, n)