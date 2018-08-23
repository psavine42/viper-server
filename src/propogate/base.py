from uuid import uuid4
from src.rules.graph import Node, Edge

class BasePropogator(object):
    def __init__(self, name, nodes=True, edges=False, domain=None, reverse=False, **kwargs):
        self._id = uuid4()
        self._var = name
        self._domain = domain
        self._reverse = reverse
        self._use_edges = edges
        self._use_nodes = nodes
        self._in_cells = []
        self._out_cells = []

    @property
    def var(self):
        return self._var

    def on_first(self, node, prev_data, **kwargs):
        return self.on_default(node, prev_data, **kwargs)

    def on_terminal(self, node, prev_data, **kwargs):
        return self.on_default(node, prev_data, **kwargs)

    def on_default(self, node, prev_data, **kwargs):
        return node, prev_data

    def is_terminal(self, node, prev_data, **kwargs):
        return False

    def next_fn(self, node):
        if self._reverse is False:
            return node.successors()
        else:
            return node.predecessors()

    def __call__(self, node, data=None, first=True, **kwargs):

        if self.is_terminal(node, data, **kwargs) is True:
            self.on_terminal(node, data, **kwargs)
            return None
        elif first is True:
            node, new_data = self.on_first(node, data, **kwargs)
        else:
            node, new_data = self.on_default(node, data, **kwargs)

        for next_node in self.next_fn(node):
            self.__call__(next_node, new_data, first=False, **kwargs)

    def __str__(self):
        st = '<{}>'.format(self.__class__.__name__)
        return st


class EdgePropogator(BasePropogator):
    def __init__(self, name,  **kwargs):
        super(EdgePropogator, self).__init__(name,  **kwargs)
        self.seen = set()

    def on_default(self, edge, _, **kwargs):
        self.seen.add(edge.id)
        return edge, _

    def next_fn(self, edge):
        if self._reverse is False:
            return edge.target.successors(edges=True)
        else:
            return edge.source.predecessors(edges=True)

    def __call__(self, node, data=None, first=True, **kwargs):
        if isinstance(node, Node):
            for edge in node.neighbors(edges=True):
                super(EdgePropogator, self).__call__(edge, data, first=True, **kwargs)
        else:
            super(EdgePropogator, self).__call__(node, data, first=True, **kwargs)


class Propogator(BasePropogator):
    def __init__(self, name, on_default=None, terminal=None, on_terminal=None, on_first=None, **kwargs):
        super(Propogator, self).__init__(name, **kwargs)
        self._on_default = on_default
        self._on_first = on_first
        self._on_terminal = on_terminal
        self._is_terminal = terminal

    def on_first(self, node, prev_data, **kwargs):
        if self._on_first is not None:
            return self._on_first(node, prev_data, **kwargs)
        return super(Propogator, self).on_first(node, prev_data, **kwargs)

    def on_terminal(self, node, prev_data, **kwargs):
        if self._on_terminal is not None:
            return self._on_terminal(node, prev_data, **kwargs)
        return super(Propogator, self).on_terminal(node, prev_data, **kwargs)

    def on_default(self, node, prev_data, **kwargs):
        if self._on_default is not None:
            return self._on_default(node, prev_data, **kwargs)
        return super(Propogator, self).on_default(node, prev_data, **kwargs)

    def is_terminal(self, node, prev_data, **kwargs):
        if self._is_terminal is not None:
            return self._is_terminal(node, prev_data, **kwargs)
        return False

    def next_fn(self, node):
        if self._reverse is False:
            return node.successors()
        else:
            return node.predecessors()



