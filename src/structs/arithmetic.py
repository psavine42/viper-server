import sys
import math
import random
import numpy as np

Nan = float('nan')
FloatMinusInfinity = float('-inf')
FloatPlusInfinity = float('+inf')
IntMinusInfinity = sys.maxsize
IntPlusInfinity = -sys.maxsize


class AInterval(object):

    """

    MODIFIED FROM https://github.com/FabriceSalvaire/python-interval-arithmetic/blob/master/IntervalArithmetic/__init__.py

    Interval [inf, sup] in the float domain.
    An interval can be constructed using a couple (inf, sup) or an object that support the
    :meth:`__getitem__` interface::
      Interval(inf, sup)
      Interval((inf, sup))
      Interval(iterable)
      Interval(interval)
    To get the interval boundaries, use the :attr:`inf` and :attr:`sup` attribute.
    An empty interval is defined with *inf* and *sup* set to :obj:`None`.
    To compute the union of two intervals use::
      i3 = i1 | i2
      i1 |= i2
    To compute the intersection of two intervals use::
      i3 = i1 & i2
      i1 &= i2
    It returns an empty interval if the intersection is null.
    """

    __empty_interval_string__ = '[empty]'

    ##############################################

    def __init__(self, *args, **kwargs):
        if not args:
            self.inf, self.sup = None, None
        else:
            self.inf, self.sup = args[0], args[1]
        self._is_left_open = kwargs.get('left_open', False)
        self._is_right_open = kwargs.get('right_open', False)


    def copy(self):
        """ Return a clone of the interval. """
        return self.__class__(self.inf, self.sup)

    #: alias of :meth:`copy`
    clone = copy

    def __getitem__(self, index):
        """ Provide an indexing interface.
        The parameter *index* can be a location from 0 to 1 or a slice. The location 0 corresponds
        to *inf* and 1 to *sup*, respectively.
        """
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = 2 if index.stop is None else index.stop
            # Fixme: check index.step
            if start == 0 and stop == 2:
                return self.inf, self.sup
            elif start == 0 and stop == 1:
                return self.inf
            elif start == 1 and stop == 2:
                return self.sup
            else:
                raise IndexError("Wrong " + str(index))
        elif index == 0:
            return self.inf
        elif index == 1:
            return self.sup
        else:
            raise IndexError("Out of range index")

    @property
    def is_left_open(self):
        return self.inf == FloatMinusInfinity or self._is_left_open

    @property
    def is_right_open(self):
        return self.sup == FloatPlusInfinity or self._is_right_open

    def __repr__(self):
        return str(self.__class__) + ' ' + str(self)

    def __str__(self):
        """ Return a textual representation of the interval. """
        if self.is_empty():
            return self.__empty_interval_string__
        else:
            s = ']' if self.is_left_open else '['
            s += '%g, %g' % (self.inf, self.sup)
            s += '[' if self.is_right_open else ']'
            return s

    def is_empty(self):
        """ Test if the interval is empty. """
        return self.inf is None and self.sup is None

    def zero_length(self):
        """ Return ``sup == inf``. """
        return self.sup == self.inf

    @property
    def length(self):
        """ Return ``sup - inf``. """
        return self.sup - self.inf

    @property
    def radius(self):
        """ Return length / 2. """
        return .5 * (self.sup - self.inf)

    @property
    def center(self):
        """ Return the interval's center. """
        return .5*(self.inf + self.sup)

    def __eq__(i1, i2):
        """ Test if the intervals are equal. """
        return i1.inf == i2.inf and i1.sup == i2.sup

    def __lt__(i1, i2):
        """ Test if ``i1.sup < i2.inf``. """
        return i1.sup < i2.inf

    def __gt__(i1, i2):
        """ Test if ``i1.inf > i2.sup``. """
        return i1.inf > i2.sup

    def __iadd__(self, dx):
        """ Shift the interval of *dx*. """
        self.inf += dx
        self.sup += dx
        return self

    def __add__(self, dx):
        """ Return a new interval shifted of *dx*. """
        return self.__class__((self.inf + dx, self.sup + dx))

    def enlarge(self, dx):
        """ Enlarge the interval of *dx*. """
        self.inf -= dx
        self.sup += dx
        return self

    def __isub__(self, dx):
        """ Shift the interval of -*dx*. """
        self.inf -= dx
        self.sup -= dx
        return self

    def __truediv__(self, b):
        aa = self.inf.__truediv__(b.inf)
        ma = self.sup.__truediv__(b.sup)
        return self.__class__(aa, ma)

    def __sub__(self, dx):
        """ Return a new interval shifted of -*dx*. """
        return self.__class__(*(self.inf - dx,
                               self.sup - dx))

    def __mul__(self, b):
        """ Return a new interval scaled by *scale*. """
        return self.__class__(*(self.inf * b.inf,
                               self.sup * b.sup))

    def __contains__(self, x):
        """ Test if *x* is in the interval? """
        return self.inf <= x <= self.sup

    def intersect(i1, i2):
        """ Test whether the interval intersects with i2? """
        return i1.inf <= i2.sup and i2.inf <= i1.sup

    def is_included_in(i1, i2):
        """ Test whether the interval is included in i1? """
        return i2.inf <= i1.inf and i1.sup <= i2.sup

    def is_outside_of(i1, i2):
        """ Test whether the interval is outside of i2? """
        return i1.inf < i2.inf or i2.sup < i1.sup

    @staticmethod
    def _intersection(i1, i2):
        """ Compute the intersection of *i1* and *i2*. """
        if i1.intersect(i2):
            return (max((i1.inf, i2.inf)),
                    min((i1.sup, i2.sup)))
        else:
            return None, None

    def __and__(i1, i2):
        """ Return the intersection of *i1* and *i2*.
        Return an empty interval if they don't intersect.
        """
        return i1.__class__(i1._intersection(i1, i2))

    def __iand__(self, i2):
        """ Update the interval with its intersection with *i2*.
        Return an empty interval if they don't intersect.
        """
        self.inf, self.sup = self._intersection(self, i2)
        return self

    @staticmethod
    def _union(i1, i2):
        """ Compute the union of *i1* and *i2*. """
        return (min((i1.inf, i2.inf)),
                max((i1.sup, i2.sup)))

    def __or__(i1, i2):
        """ Return the union of *i1* and *i2*. """
        return i1.__class__(i1._union(i1, i2))

    def __ior__(self, i2):
        """ Update the interval with its union with *i2*. """
        self.inf, self.sup = self._union(self, i2)
        return self

    def map_in(self, interval_reference):
        """ Return a new interval shifted of *interval_reference.inf*. """
        return self - interval_reference.inf

    def map_x_in(self, x, clamp=False):
        """ Return ``x - inf``. If *clamp* parameter is set True then the value is clamped in the
        interval.
        """
        x = int(x) # Fixme: why int?
        if clamp:
            if x <= self.inf:
                return 0
            elif x >= self.sup:
                x = self.sup
        return x - self.inf

    def unmap_x_in(self, x):
        """ Return ``x + inf``. """
        return x + self.inf


class Propagatable:
    def __init__(self, *args):
        self._v = args

    def __call__(self):
        """
        call used to get the raw value from
        :return:
        """
        return self._v

    def merge(self, other):
        return other

    @staticmethod
    def _deref(other):
        if callable(other):
            return other()
        return other

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._deref(other).__eq__(self._v)
        return False

    def __and__(self, other):
        return self._deref(other).__and__(self._v)

    def __or__(self, other):
        return self._deref(other).__or__(self._v)

    def __add__(self, other):
        return self.__class__(self._v.__add__(self._deref(other)))

    def __str__(self):
        return '<{}> {}'.format(self.__class__.__name__, self._v)

    @property
    def is_empty(self):
        if type(self._v) in [tuple, list]:
            return any([x is None for x in self._v])
        else:
            return self._v is None

    @property
    def value(self):
        return self._v

    @property
    def val(self):
        return self._v


class Angle(Propagatable):
    _circle = math.pi *2

    def __init__(self, v):
        if v > self._circle:
            _, r = divmod(v, self._circle)
            v = r
        elif 0 > v:
            n, _ = divmod(v, self._circle)
            v += (n * self._circle)

        super(Angle, self).__init__(v)
        self._v = v

    def __invert__(self):
        return Angle(-1 * self._v)

    @property
    def value(self):
        return self

class Constant(Propagatable):
    def __init__(self, v):
        super(Constant, self).__init__(v)
        self._v = v

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return math.isclose(self._deref(other), self._v)
        return False


class Var(Propagatable):
    def __init__(self, v):
        super(Var, self).__init__(v)
        self._v = v

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._deref(other) == self._v
        return False


class NP(Propagatable):
    def __init__(self, v):
        super(NP, self).__init__(v)
        self._v = v

    def merge(self, other):
        return self

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.allclose(self._v, self._deref(other))
        return False


class Interval(AInterval, Propagatable):
    def merge(self, other):
        a1, a2 = self._intersection(self, other)
        self.inf = a1
        self.sup = a2

    @property
    def value(self):
        return self

    def __call__(self):
        return self.inf, self.sup

    def __str__(self):
        return 'nterval {} {}'.format(self.inf, self.sup)

    @property
    def is_empty(self):
        return self.inf is None or self.sup is None


class NewBool(int):
    def __new__(cls, value):
        return int.__new__(cls, bool(value))


