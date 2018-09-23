from lib.geo import *
import lib.geo
import uuid
import math
import itertools
import numpy as np
from trimesh import Trimesh
from src.geomType import GeomType
from shapely import ops
from shapely.geometry import MultiPolygon, LineString, Point, MultiLineString, Polygon
from copy import deepcopy
from itertools import chain

class CadInterface(object):
    _ROUND = 6

    def __init__(self,
                 geomType=None,
                 uid=None,
                 layer=-1,
                 layerName=None,
                 children=[],
                 **kwargs):
        self._vtype = GeomType(geomType if geomType else 0)
        self._layer_id = layer
        self._layer_name = layerName
        self._uuid = uuid.uuid4() if uid is None else uid
        self._children = children

    @property
    def id(self):
        return self._uuid

    @property
    def layer(self):
        if self._layer_id == -1 and self.children:
            for c in self.children:
                cl = c.layer
                if cl and cl not in [0, -1]:
                    return cl
        return self._layer_id

    @layer.setter
    def layer(self, value):
        self._layer_id = value

    @property
    def contains_solids(self):
        return False

    @property
    def geomType(self):
        return self._vtype

    @property
    def base_args(self):
        return {'uid': self.id,
                'layer': self._layer_id,
                'geomType': self.geomType,
                'layerName': self._layer_name,
                'children': list([x.base_args for x in self.children])}

    @property
    def children(self):
        return self._children

    @property
    def points(self):
        raise NotImplemented('points not implemented in base')

    @classmethod
    def factory(cls, points=[], **kwargs):
        geom_type = kwargs.get('geomType', 0)
        geom_type = GeomType(geom_type if geom_type else 0)

        if geom_type in [GeomType['ARC'], GeomType['CIRCLE']]:
            pt = [round(p, cls._ROUND) for p in points]
            yield FamilySymbol(*pt, **kwargs)

        elif geom_type == GeomType['SOLID']:
            yield MEPSolidLine(**kwargs)

        elif geom_type == GeomType['FACE']:
            yield MEPFace(**kwargs)

        elif geom_type == GeomType['LINE']:
            yield MepCurve2d.from_viper(**kwargs)

        elif geom_type == GeomType['POLYLINE']:
            xyzs = [points[i:i + 3] for i in range(0, len(points), 3)]
            for i in range(1, len(xyzs)):
                kgs = deepcopy(kwargs)
                kgs['points'] = xyzs[i - 1] + xyzs[i]
                kgs['children'] = None
                yield MepCurve2d.from_viper(**kgs)

        elif geom_type == GeomType['MESH']:
            xyzs = [tuple(points[i:i + 3]) for i in range(0, len(points), 3)]
            xyz_set = list(set(xyzs))
            faces = []
            for i in range(0, len(xyzs), 3):
                faces.append([xyz_set.index(xyzs[i+x]) for x in range(3)])
            xyz_set = list(map(list, xyz_set))
            msh = MeshSolid(vertices=xyz_set, faces=faces, **kwargs)
            try:
                valid = msh.centroid
                yield msh
            except:
                yield None

        elif geom_type == GeomType['SYMBOL']:
            if kwargs.get('use_syms', False) is True:
                yield GometryInstance(**kwargs)
            else:
                for x in kwargs.get('children', []):
                    for r in cls.factory(**x):
                        yield r

    def children_of_type(self, geomtype):
        if self.geomType == geomtype:
            return self
        if self.children:
            itms = [x.children_of_type(geomtype) for x in self.children]
            return list(chain(filter(None, itms)))
        return None

    def __eq__(self, other) -> bool:
        return self.points.__eq__(other.points)

    def __str__(self):
        return self.__repr__()

    def __repr__(self) -> str:
        st = 'class: {}, type: {}, layer: {}'.format(
            self.__class__.__name__, self._vtype, self.layer)
        return st


class MEPFace(CadInterface, Polygon):
    contains_solids = False

    def __init__(self, children=[], **kwargs):
        CadInterface.__init__(self, **kwargs)
        self._lines = [MepCurve2d.from_viper(**x) for x in children
                       if x.get('geomType', None) is not None]
        Polygon.__init__(self, [x.points[0] for x in self._lines])

    @property
    def points(self):
        """ returns edge loop """
        return list(self.exterior.coords)

    @property
    def children(self):
        return self._lines

    def direction(self):
        """ face normal """
        return

    def __repr__(self):
        st = 'class: {}, type: {}, nlines: {}, {}'.format(
            self.__class__.__name__, self._vtype, len(self._lines), self._layer_id)
        return st


class MeshSolid(Trimesh, CadInterface):
    contains_solids = True

    def __init__(self, faces=None, vertices=None, **kwargs):
        """
        Arbitrary thing assumed to be a line-based geometry
        """
        CadInterface.__init__(self, **kwargs)
        Trimesh.__init__(self, vertices=vertices, faces=faces)

    @property
    def points(self):
        return self.vertices

    def __repr__(self):
        st = 'class: {}, type: {}, npoints: {}'.format(
            self.__class__.__name__, self._vtype, '')
        return st


class MEPSolidLine(CadInterface):
    contains_solids = True

    def __init__(self, children=[], **kwargs):
        """
        Arbitrary thing assumed to be a line-based geometry
        """
        CadInterface.__init__(self, **kwargs)
        self._faces = [MEPFace(**x) for x in children
                       if x.get('geomType', None) is not None
                       and GeomType(x.get('geomType', 0)) == GeomType['FACE']]
        # MultiPolygon.__init__(self, self._faces)

    def __getitem__(self, item):
        return self._faces[item]

    @property
    def centroid(self):
        return pointset_mass_distribution(self.points)[0]

    @property
    def children(self):
        return self._faces

    @property
    def points(self):
        """ returns computed line-based represntation of the object """
        return list(itertools.chain(*[x.points for x in self._faces]))

    def __repr__(self):
        st = 'class: {}, type: {}, nfaces: {}'.format(
            self.__class__.__name__, self._vtype, len(self._faces))
        return st


class FamilySymbol(CadInterface, Point):
    def __init__(self, *args, **kwargs):
        self._direction = 0
        self._length = 0
        if isinstance(args[0], list):
            #
            coord, r = self.to_xg(*args)
        else:
            if len(args) == 4:          # [x, y, z, r]
                coord = args[0:3]
                self._length = args[3]  # Radius
            elif len(args) == 3:
                coord = args
            else:
                coord = args
        Point.__init__(self, *coord)
        CadInterface.__init__(self, **kwargs)
        self._opts = kwargs
        self._d = None
        if kwargs.get('use_syms', None) is True:
            self._children = list([self.factory(**x) for x in self._children])

    # ------------------------------------------------
    @property
    def to_dict(self):
        return {**self._opts, **{'symbol_id': self._uuid}}

    @property
    def centroid(self):
        _own = list(itertools.chain(*[x.points for x in self._children]))
        c, _ = pointset_mass_distribution(_own)
        # print(self.points, c, self.layer)
        return c

    @property
    def direction(self):
        return self._direction

    @property
    def points(self):
        return list(self.coords)

    def points_np(self):
        return np.array(self.points)

    @property
    def length(self):
        return self._length

    def __eq__(self, other):
        self.points.__eq__(other.points)

    def __gt__(self, other):
        self.points.__gt__(other.points)

    def extend_norm(self, *args):
        l = 1 if self.length == 0 else self.length
        return self.buffer(l*args[0])

    def extend(self, *args):
        return self

    @property
    def contains_solids(self):
        return any([x.contains_solids is True for x in self.children])

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
        st = '{} {}'.format(self._vtype, self.points)
        return st


class GometryInstance(CadInterface, Point):
    def __init__(self, children=[], **kwargs):
        CadInterface.__init__(self, **kwargs)
        self._children = list(filter(None,
                                     chain(*[list(self.factory(**x))
                                             for x in children])))
        Point.__init__(self, self.centroid)
        _layer = self.layer
        self._layer_id = _layer
        for x in self._children:
            x.layer = _layer

    @property
    def points(self):
        return list(itertools.chain(*[x.points for x in self._children]))

    @property
    def centroid(self):
        return pointset_mass_distribution(self.points)[0]

    @property
    def contains_solids(self):
        return any([x.contains_solids is True for x in self.children])


class MepCurve2d(LineString, CadInterface):
    contains_solids = False

    def __init__(self, xy1, xy2, **kwargs):
        LineString.__init__(self, [xy1, xy2])
        CadInterface.__init__(self, **kwargs)
        self._opts = kwargs

    # ------------------------------------------------
    @property
    def points(self):
        p1, p2 = list(self.coords)
        return p1, p2

    @classmethod
    def from_viper(cls, points=[], rnd=6, twod=False, **kwargs):
        if len(points) not in [4, 6]:
            return None
        x1, y1, z1, x2, y2, z2 = [round(p, rnd) for p in points]
        if not (x1 == x2 and y1 == y2):
            if twod is True:
                z1, z2 = 0, 0
            p1, p2 = sorted([(x1, y1, z1), (x2, y2, z2)])
            return cls(p1, p2, **kwargs)

    @property
    def direction(self):
        p1, p2 = list(self.coords)
        return lib.geo.normalized(np.array(p2) - np.array(p1))

    def __gt__(self, other):
        self.points.__gt__(other.points)

    def __le__(self, other):
        self.points.__le__(other.points)

    def __eq__(self, other):
        return np.allclose(sorted(self.points), sorted(other.points))

    def __reversed__(self):
        p1, p2 = self.points
        return MepCurve2d(p2, p1, uid=self.id, layer=self.layer)

    # ------------------------------------------------
    def same_direction(self, other):
        return np.allclose(np.abs(self.direction), np.abs(other.direction))

    def points_np(self):
        p1, p2 = list(self.coords)
        return np.array(p1), np.array(p2)

    def to_points_(self):
        (x1, y1), (x2, y2) = list(self.coords)
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
        return self.extend(s, e)

    def buffer_norm(self, norm=1.):
        return self.buffer(self.length * norm)

    def split(self, pt):
        for new in ops.split(self, pt):
            p1, p2 = list(new.coords)
            yield MepCurve2d(p1, p2, layer=self.layer)

    def to_dict(self, **kwargs):
        p1, p2 = self.points
        dct = {'start': p1, 'end': p2, 'system': ''}
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


def angle_to(c1, c2):
    cc1 = c1.direction if isinstance(c1, MepCurve2d) else c1
    cc2 = c2.direction if isinstance(c2, MepCurve2d) else c2
    return math.acos(min(1, abs(np.dot(cc1, cc2))))


def add_coord(coord, x=0, y=0, z=0):
    crd = deepcopy(coord)
    if isinstance(crd, tuple):
        crd = list(crd)
    crd[0] += x
    crd[1] += y
    crd[2] += z
    return tuple(crd)


def set_coord(coord, x=None, y=None, z=None):
    crd = deepcopy(coord)
    if isinstance(crd, tuple):
        crd = list(crd)
    crd[0] = x if x else crd[0]
    crd[1] = y if y else crd[1]
    crd[2] = z if z else crd[2]
    return tuple(crd)


def split_ls(line_string, point):
    coords = line_string.coords
    j = None
    for i in range(len(coords) - 1):
        if LineString(coords[i:i + 2]).intersects(point):
           j = i
           break
    assert j is not None
    # Make sure to always include the point in the first group
    if Point(coords[j + 1:j + 2]).equals(point):
        return coords[:j + 2], coords[j + 1:]
    else:
        return coords[:j + 1], coords[j:]


def to3dp(point):
    return Point(to_Nd(point))


def to_Nd(point, n=3):
    if isinstance(point, Point):
        x = list(list(point.coords)[0])
    elif isinstance(point, tuple):
        x = list(point)
    else:
        x = point
    this_dim = len(x)
    if this_dim != n:
        if this_dim + 1 == n:
            x += [0]
        elif this_dim -1 == n:
            x = x[:2]
    elif this_dim == 3:
        if math.isnan(x[-1] ):
            x[-1] = 0
    return tuple(x)


def to_mls(line_str):
    if isinstance(line_str, LineString):
        line_str = list(line_str.coords)
    elif isinstance(line_str, MultiLineString):
        return line_str
    ds = []
    for i in range(1, len(line_str)):
        if line_str[i-1] != line_str[i]:
            ds.append((line_str[i - 1], line_str[i]))
    return MultiLineString(ds)


def add_line_at_end(mls, new_l, ix):
    if isinstance(mls, LineString):
        mls = to_mls(mls)
    o = []
    if ix == 0:
        o += list(new_l.coords)
        for x in mls.geoms:
            o += list(x.coords)
    else:

        for x in mls.geoms:
            o += list(x.coords)
        o += list(new_l.coords)
    print(o)
    return MultiLineString(list(zip(o[::2], o[1::2])))


def rebuild_mls(mls, point_to_add, **kwargs):
    proj1 = mls.project(point_to_add, normalized=True)
    pt_list = []

    found = False

    for geom in mls.geoms:
        for pt in geom.coords:
            nd = len(pt)
            if found is False:
                isf = mls.project(Point(pt), normalized=True)
                if isf > proj1:
                    pt_list.append(to_Nd(point_to_add, nd))
                    found = True
            pt_list.append(pt)

    return to_mls(pt_list)


def direction(p1, p2):
    return lib.geo.normalized(np.array(p2) - np.array(p1))


def rebuild(linestr, point_to_add, **kwargs):
    if isinstance(linestr, LineString):
        linestr = to_mls(list(linestr.coords))
    return rebuild_mls(linestr, point_to_add, **kwargs)





