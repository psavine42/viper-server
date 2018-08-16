from lib.geo import *
import lib.geo
import uuid
import math
import numpy as np
from shapely import ops
from lib.sweep import SweepLine
from shapely.geometry import LineString, Point
from copy import deepcopy

class FamilySymbol(Point):
    def __init__(self, *args, **kwargs):
        self._direction = 0
        self._length = 0
        if isinstance(args[0], list):
            coord, r = self.to_xg(*args)
        else:
            if len(args) == 4:          # [x, y, z, r]
                coord = args[0:3]
                self._length = args[3] # Radius
            elif len(args) == 3:
                coord = args
            else:
                coord = args
        super(FamilySymbol, self).__init__(*coord)
        self._opts = kwargs
        self._type = kwargs.get('type', None)
        self._layer = kwargs.get('layer', None)
        self._uuid = kwargs.get('uid', uuid.uuid4())
        self._d = None

    # ------------------------------------------------
    @property
    def to_dict(self):
        return self._opts

    @property
    def direction(self):
        return self._direction

    @property
    def points(self):
        p1 = list(self.coords)
        return p1

    @property
    def length(self):
        return self._length

    @property
    def layer(self):
        return self._layer

    @property
    def id(self):
        return self._uuid

    @property
    def geomType(self):
        return self._type

    def __gt__(self, other):
        self.points.__gt__(other.points)

    def extend_norm(self, *args):
        l = 1 if self.length == 0 else self.length
        return self.buffer(l*args[0])

    def extend(self, *args):
        return self

    # ------------------------------------------------
    def to_xg(self, *points):
        r_cm, A = pointset_mass_distribution(points)
        val, vec = np.linalg.eigh(A)
        self._direction = np.asarray(vec[:, 2]).reshape(3)
        self._length = math.sqrt(val[2] / 2)
        r = r_cm
        r2 = r_cm + self._length * self._direction
        return r, r2

    def __str__(self):
        st = '{} {}'.format(self._type, self.points)
        return st


class MepCurve2d(LineString):
    def __init__(self, xy1, xy2, uid=None, layer=None, **kwargs):
        super(MepCurve2d, self).__init__([xy1, xy2])
        self._uuid = uuid.uuid4() if uid is None else uid
        self._layer = layer
        self._opts = kwargs

    def _init_pnt(self, pnt):
        if len(pnt) == 3:
            _tmp = list(pnt)
            h1 = list(_tmp).pop(-1)
            pnt = tuple(_tmp)
        else:
            h1 = 0
        return pnt, h1

    # ------------------------------------------------
    @property
    def geomType(self):
        return None

    @property
    def points(self):
        p1, p2 = list(self.coords)
        return p1, p2

    @property
    def layer(self):
        return self._layer

    @property
    def id(self):
        return self._uuid

    @property
    def direction(self):
        p1, p2 = list(self.coords)
        return lib.geo.normalized(np.array(p2) - np.array(p1))

    def __gt__(self, other):
        self.points.__gt__(other.points)

    # ------------------------------------------------
    def __reversed__(self):
        p1, p2 = self.points
        return MepCurve2d(p2, p1, uid=self.id, layer=self.layer)

    def points_np(self):
        p1, p2 = list(self.coords)
        return np.array(p1), np.array(p2)

    def to_points_(self):
        (x1, y1), (x2, y2)= list(self.coords)
        a = Point(x1, y1)
        b = Point(x2, y2)
        return a, b

    def extend(self, start=0., end=0.):
        p1, p2 = self.points_np()
        ep1 = p1 - self.direction * start
        ep2 = p2 + self.direction * end
        return MepCurve2d(ep1.tolist(), ep2.tolist(), uid=self.id, layer=self.layer)

    def extend_norm(self, start=0., end=0.):
        s = self.length * start
        e = self.length * end
        # e = max([24, self.length * end])
        return self.extend(s, e)

    def buffer_norm(self, norm=1.):
        return self.buffer(self.length * norm)

    def split(self, pt):
        for new in ops.split(self, pt):
            p1, p2 = list(new.coords)
            yield MepCurve2d(p1, p2, layer=self.layer)

    def to_dict(self, **kwargs):
        p1, p2 = self.points
        dct = {'start': p1, 'end':p2, 'system':''}
        dct.update(**kwargs)
        return dct


def pointset_mass_distribution(points):
    cm = np.zeros((1, 3))
    for p in points:
        cm += np.array(p)
    cm /= len(points)
    A = np.asmatrix(np.zeros((3, 3)))
    for p in points:
        r = np.asmatrix(np.array(p) - cm)
        A += r.transpose()*r
    return np.asarray(cm).reshape(3), A


def to_point(xyz):
    if isinstance(xyz, Point):
        return xyz
    else:
        return Point(xyz)


def add_coord(coord, x=0, y=0, z=0):
    crd = deepcopy(coord)
    if isinstance(crd, tuple):
        crd = list(crd)
    crd[0] += x
    crd[1] += y
    crd[2] += z
    return tuple(crd)
