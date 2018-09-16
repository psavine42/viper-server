from numpy import ndarray
from .arithmetic import Constant, NP, Var
from random import randint

def alert_propagators(to_do):
    for n in to_do:
        n()


def wrap_value(value):
    if type(value) in [int, float, bool]:
        return Constant(value)
    elif isinstance(value, str):
        return Var(value)
    elif isinstance(value, ndarray):
        return NP(value)
    else:
        return value


class Support(object):
    __slots__ = ['_support', '_dirty', '_roots', '_depth']
    def __init__(self, support=None):
        self._support = [support] if support else []
        self._dirty = True
        self._roots = []
        self._depth = []

    def add_support(self, other):
        if other not in self._support:
            self._support.append(other)
            self._dirty = True

    def support_roots(self):
        """
        if modified, recalculate support
        :return:
        """
        if self._dirty is True:
            self._depth = []
            self._roots = []

            q = [ (0, x) for x in self._support ]
            while q:
                n, s = q.pop()
                if isinstance(s, Cell):
                    q.extend([(n+1, x) for x in s.support])
                elif s not in self._roots:
                    self._roots.append(s)
                    self._depth.append(n + 1)

            self._dirty = False
            return self._roots
        else:
            return self._roots

    @property
    def support(self):
        return self._support

    def remove_support(self, sup):
        if sup in self._support:
            self._support.remove(sup)
            self._dirty = True

    def __del__(self):
         while self._roots:
             self._roots.pop()
         while self._support:
             self._support.pop()

    def __str__(self):
        st = ''
        return st



class Cell(object):
    """


    """
    __slots__ = ['_var', '_content', '_neighbors', '_profile', '_support']
    def __init__(self, *args, var=None, profile=False, support=None):
        self._var = var
        self._neighbors = []  # list of propagators
        self._profile = profile
        self._content = Cell._check_args(*args)
        self._support = Support(support)

    # provenance -----------------------------------
    def add_support(self, sup):
        self._support.add_support(sup)

    def remove_support(self, sup):
        self._support.remove_support(sup)

    @property
    def support_roots(self):
        return self._support.support_roots()

    @property
    def support(self):
        return self._support.support

    # aotp cell api -----------------------------------
    @staticmethod
    def _merge(own_content, other_content):
        # todo decide where to put merge
        return own_content.merge(other_content)

    @staticmethod
    def _contradicts(content, answer):
        # todo implement world views
        return False

    @property
    def id(self):
        return id(self)

    @property
    def contents(self):
        """
        returns the wrapped cell contents
        if cell is empty, returns 'None'
        """
        if not self.is_empty:
            return self._content

    @property
    def value(self):
        """
        retrieve unwrapped value in cell contents
        if the cell is empty, returns 'None'
        """
        if self.contents:
            return self.contents.value

    def add_contents(self, content):
        """
        adds contents to cell.
        if this changes cell contents, alerts propagators
        :param content:
        """
        content = wrap_value(content)
        self._status('before', content)

        if content is None or content.is_empty:
            return

        elif self.is_empty is True:
            self._content = content
            alert_propagators(self._neighbors)

        elif self._content != content:

            answer = self._merge(self._content, content)
            if answer.__eq__(self._content):
                return
            elif self._contradicts(self._content, answer):
                print('Ack! inconsistency')
            else:
                self._content = answer
                self._status('on_change')
                alert_propagators(self._neighbors)

        self._status('after', content)

    def set_contents(self, value):
        """ user changes cell value """
        self._content = wrap_value(value)
        alert_propagators(self._neighbors)

    def new_neighbor(self, propagator):
        """ """
        if propagator not in self._neighbors:
            self._neighbors.append(propagator)
            alert_propagators(self._neighbors)

    def remove_neighbor(self, propagator):
        if propagator in self._neighbors:
            self._neighbors.remove(propagator)
            alert_propagators(self._neighbors)

    # --------------------------------------------------
    @staticmethod
    def _check_args(*args):
        if not args:
            return None
        else:
            return wrap_value(args[0])

    @property
    def var(self):
        return self._var

    @property
    def is_empty(self):
        if self._content is None:
            return True
        else:
            return self._content.is_empty

    def predecessors(self):
        for n in self._neighbors:
            for pred_cell in n.inputs:
                yield pred_cell

    @property
    def neighbors(self):
        return self._neighbors

    # python --------------------------------------------------
    def __del__(self):
        self._support.__del__()
        self._support = None
        self._var = None
        self._content = None
        self._neighbors = []  # list of propagators

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return id(self) == id(other)
        return False

    def __str__(self, neigh=False):
        st = ''
        if self._var:
            st += self._var
        if not self.is_empty:
            st += str(self._content)
        else:
            st += '<empty>'

        #if self.support is not None:
        #    st += ' , support: ' + str(self.support)

        if neigh is True:
            st += '\n'
            st += ', '.join([str(x) for x in self.neighbors])
            st += '\n'
        return '<Cell>:{}'.format(st)

    def _status(self, step, c=None):
        if self._profile is True:
            st = '{}, Var: {}, Empty? {}, Content: {} '.format(
                step, self._var, self.is_empty, self._content)
            t2 = ''
            if c is not None:
                t2 = 'content eq? {}, answer: {} '.format(
                    self._content == c, self._merge(self._content, c))
            print(st + t2)

