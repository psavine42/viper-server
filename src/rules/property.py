from .graph import Node
from uuid import uuid4
from .opers import Condition

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


class Property(Condition):
    def __init__(self, name, cond):
        super(Property, self).__init__()
        self._name = name
        self._base = cond
        self._range = -1

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return '<Prop>:<' + self.name + ">:" + str(self._id)

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

    def on_eval(self, node, res):
        if res is not None:
            node.write(self._name, res)

    def apply(self, node):
        self.check_conditions(node)
        res = self._base.apply(node)
        self.on_eval(node, res)
        return res

    def __str__(self):
        st = '<Prop>:<{}>'.format(self._name)
        return st

