from .cell import Cell
from .propagator import Propagator, CondPropagator
import operator
import numpy as np
import itertools
import math
from .arithmetic import Angle

def make_cell():
    return Cell()


def constant(value):
    return fn_propagator_constructor(lambda *x: value)


def lift_to_cell_contents(fn):
    """
    The procedure lift-to-cell-contents ensures
    that if any cell contents are still nothing
    the result is nothing ,
    """
    def inner(*args):
        if any([True for x in args if x is None]):
            return None
        else:
            return fn(*args)
    return inner


def listify(*args):
    lst = []
    for a in args:
        if type(a) in [list, tuple]:
            lst += [x for x in a]
        else:
            lst.append(a)
    return lst


def fn_propagator_constructor(fn):
    """
    wrap a function with a propagator

    example:

        adder = fn_propagator_constructor(operator.add)

        n1 = Cell()
        n2 = Cell()
        res = Cell()

        adder(n1, n2, res)

        n1.add_contents(5)
        n2.add_contents(2)

        res.value
        >> 7

    :param fn: function to wrap
    :return: a function with cells as inputs
    """
    def inner(*cells):
        cells = listify(cells)
        return Propagator(cells[:-1], cells[-1], fn)
    return inner


def linear_classifier(class_cells, label_cells, result_cell):
    assert len(class_cells) == len(label_cells)
    for cell, label in zip(class_cells, label_cells):
        Propagator([cell, label], result_cell, take_last)


def prop_fn_to(*args, **kwargs):
    """

    :param args: first is function, rest list of cells
    :param kwargs: profile, var
    :return:
    """
    fn = args[0]
    cells = args[1:]
    out_cell = Cell(**kwargs)

    Propagator(cells, out_cell, fn)
    return out_cell


def conditional(predicate, if_true, fn, out=None, false=None, **kwargs):
    """

    wrapper for creating conditional propagator

    :param predicate: Cell
    :param if_true: Cell
    :param fn:      function (not propagator)
    :param false:   Cell
    :param out:     Cell
    :return:
    """
    if false is None:
        false_cell = Cell(False, **kwargs)
    elif isinstance(false, Cell):
        false_cell = false
    else:
        false_cell = Cell(false,  **kwargs)
    if out is None:
        output = Cell( **kwargs)
    else:
        output = out
    CondPropagator(predicate, output, if_true, false_cell, fn, **kwargs)
    return output, false_cell


# base operators ---------------------------------
adder = fn_propagator_constructor(operator.add)
subtracter = fn_propagator_constructor(operator.sub)
divider = fn_propagator_constructor(operator.truediv)
multiplier = fn_propagator_constructor(operator.mul)
equaler = fn_propagator_constructor(operator.eq)
ander = fn_propagator_constructor(operator.and_)
orrer = fn_propagator_constructor(operator.or_)
noter = fn_propagator_constructor(operator.not_)


#  ---------------------------------
def _angle(x, y):
    """
    angle between zero-based vectors x, y

    :param x: np.array
    :param y: np.array
    :return: float
    """
    return Angle(math.acos(min(1, abs(np.dot(x, y)))))


def _rotate(x, angle):
    """

    :param x:
    :param y:
    :return:
    """
    from lib.meshcat.src.meshcat import transformations
    _d = np.array([0, 0, 1])
    _t = transformations.rotation_matrix(angle.val, _d)
    return np.dot(_t[:3, :3] , x)



angler = fn_propagator_constructor(_angle)
rotater = fn_propagator_constructor(_rotate)
inverser = fn_propagator_constructor(lambda angl: ~angl)


def p_angle(x, y, angle):
    """
    angle relationship between two directions

    :param x: Cell with zero-norm np.array direction
    :param y: Cell with zero-norm np.array direction
    :param z: Cell for angle between x, y
    """
    angler(x, y, angle)     # angle x, y -> ang
    rotater(x, angle, y)    # rotate vec x by angle z -> y

    _inv = Cell()           # cell for negative angle
    inverser(angle, _inv)   # negative angle
    angler(y, x, _inv)      # angle y, x -> -1 * angle
    rotater(y, _inv, x)     # rotate vec y by -1*angle -> x


def pnotnone(x):
    """ DEP """
    if not x:
        return False
    else:
        return True

def pall(*args):
    return all(list(args))

def pproduct(x, y, total):
    multiplier(x, y, total)
    divider(total, y, x)
    divider(total, x, y)


def pdivide(x, y, result):
    # lol its not commutative
    divider(x, y, result)
    multiplier(result, y, x)



def p1equals(x, y, z):
    equaler(x, y, z)
    equaler(y, x, z)


def p2equals(x, y, z):
    equaler(x, y, z)
    equaler(y, x, z)
    equaler(z, x, y)


def pand(x, y, z):
    " conjunctive and "
    ander(x, y, z)
    ander(y, x, z)
    ander(z, x, y)


def take_last(*contents):
    """ this is a bit tricky ...
        # need better support
    """
    in_cell = contents[0]
    if True == in_cell or in_cell == True:
        return contents[-1]


def discrete_cells(values, **kwargs):
    """ generate cells with values """
    return [Cell(n, var='value:{}'.format(n), **kwargs) for n in values]


def p_for_classes(in_cell, values):
    """

    :param in_cell:
    :param values:
    :return:
    """
    classifiers = []
    for val_cell in values:
        out_cell = Cell(var='output of: eq {}'.format(val_cell.contents()))
        equaler(in_cell, val_cell, out_cell)
        classifiers.append(out_cell)
    return classifiers


def set_trace(cells, c=False, p=False):
    q = []
    for r in cells:
        q += list(r.neighbors)
    seen = set()
    while q:
        el = q.pop(0)   # this is a propagator
        if el.id not in seen:
            seen.add(el.id)
            out = el.output
            el._profile = p
            out._profile = c
            q.extend(out.neighbors)
            for n in el.inputs:
                if n.id not in seen:
                    n._profile = c
                    q.extend(n.neighbors)



