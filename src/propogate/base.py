from uuid import uuid4
from src.structs.node import Node


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
            self.__call__(next_node, data=new_data, first=False, **kwargs)

    def __str__(self):
        st = '<{}>'.format(self.__class__.__name__)
        return st


class UndirectedProp(BasePropogator):
    def __init__(self, name=None,  **kwargs):
        name = name if name else self.__class__.__name__
        super(UndirectedProp, self).__init__(name,  **kwargs)
        self.seen = set()

    def next_fn(self, node):
        return node.neighbors(fwd=True, bkwd=True)


class RecProporgator(BasePropogator):
    def __init__(self, name=None,  **kwargs):
        name = name if name else self.__class__.__name__
        super(RecProporgator, self).__init__(name,  **kwargs)
        self.seen = set()

    def __call__(self, node, data=None, first=True, **kwargs):

        if self.is_terminal(node, data, **kwargs) is True:
            self.on_terminal(node, data, **kwargs)
            return None
        elif first is True:
            node, new_data = self.on_first(node, data, **kwargs)
        else:
            node, new_data = self.on_default(node, data, **kwargs)

        for next_node in self.next_fn(node):
            if next_node.id not in self.seen:
                self.seen.add(next_node.id)
                self.__call__(next_node, data=new_data, first=False, **kwargs)


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


def unique_edge_neighs(node_or_edge, fwd=True, bkwd=True):
    if isinstance(node_or_edge, Node):
        return node_or_edge.neighbors(fwd=fwd, bkwd=bkwd, edges=True)
    else:
        alls = node_or_edge.source.neighbors(fwd=fwd, bkwd=bkwd, edges=True) \
               + node_or_edge.target.neighbors(fwd=fwd, bkwd=bkwd, edges=True)
        seen = {node_or_edge.id}
        res = []
        for e in alls:
            if e.id not in seen:
                seen.add(e.id)
                res.append(e)
    return res


class UndirectedEdgeProp(BasePropogator):
    def __init__(self, name=None, **kwargs):
        name = name if name else self.__class__.__name__
        super(UndirectedEdgeProp, self).__init__(name, **kwargs)
        self.seen = set()

    def next_fn(self, edge):
        return unique_edge_neighs(edge)


class QueuePropogator(object):
    """
    dfs: if True will take from start of queue, resulting in dfs
    otherwise will use end of queue for bfs
    """
    def __init__(self,
                 fwd=True,
                 dfs=True,
                 bkwd=False,
                 edges=False,
                 **kwargs):
        self._pop_end = 0 if dfs is True else -1
        self.fwd = fwd
        self.bkwd = bkwd
        self.edges = edges
        self.seen = set()
        self.q = []
        self._res = []

    @property
    def result(self):
        return self._res

    def reset(self):
        self.seen = set()
        self._res = []

    def on_first(self, node, **kwargs):
        if not isinstance(node, list):
            return [node]
        return node

    def on_terminal(self, node, **kwargs):
        return self.on_default(node, **kwargs)

    def on_default(self, node, **kwargs):
        """
        operation to apply to each node or edge.
        should return same type as input
        :param node:
        :param kwargs:
        :return: node
        """
        return node

    def is_terminal(self, node, **kwargs):
        return False

    def on_record(self, node):
        self.seen.add(node.id)

    def on_complete(self, node):
        return node

    def on_next(self, node_or_edge):
        """
        specify elements to queue next
        :param node_or_edge:
        :return:
        """
        res = []
        if self.fwd is True:
            if self.edges is True:
                res += node_or_edge.target.successors(edges=True)
            else:
                res += node_or_edge.successors()
        if self.bkwd is True:
            if self.edges is True:
                res += node_or_edge.source.predecessors(edges=True)
            else:
                res += node_or_edge.predecessors()

        return res

    def __call__(self, node, num_iter=None, debug=False, **kwargs):
        """

        :param node:
        :param num_iter: if not None, only run on num_iter elements
        :param debug:
        :param kwargs:
        :return:
        """
        self.q = self.on_first(node, **kwargs)
        cntr = 0
        while self.q:

            el = self.q.pop(self._pop_end)
            if el.id in self.seen:
                continue

            self.on_record(el)
            if self.is_terminal(el, **kwargs) is True:
                self.on_terminal(el, **kwargs)
                return el
            else:
                el = self.on_default(el,  **kwargs)

            for next_el in self.on_next(el):
                # if next_el not in q:
                self.q.append(next_el)

            cntr += 1
            if num_iter is not None and cntr > num_iter:
                break
        print(self.__class__.__name__, 'seen nodes : ', cntr)
        return self.on_complete(node)

    def __repr__(self):
        st = self.__class__.__name__
        st += ': edge {}, fwd {} , bkwd {}'.format(self.edges, self.fwd, self.bkwd)
        return st


class FunctionQ(QueuePropogator):
    def __init__(self, fn):
        super(FunctionQ, self).__init__()
        self.fn = fn

    def on_default(self, node, **kwargs):
        return self.fn(node)


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



