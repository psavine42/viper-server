from lib.geo import *
import lib.geo
import uuid
import math
import itertools
import numpy as np
from trimesh import Trimesh
from src.geomType import GeomType
from shapely import ops
import  shapely.geometry as sg

from shapely.geometry import MultiPolygon, LineString, Point, MultiLineString, Polygon
from copy import deepcopy
from itertools import chain
from trimesh import bounds
# from pyobb.obb import OBB
from scipy import spatial
from .geombase import *
import importlib
importlib.reload(lib.geo)


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
        self._meta = kwargs

    @property
    def id(self):
        return self._uuid

    @property
    def layer(self):
        if self._layer_id in [0, -1] and self.children:
            for c in self.children:
                cl = c.layer
                if cl and cl not in [0, -1]:
                    return cl
        return self._layer_id

    @layer.setter
    def layer(self, value):
        self._layer_id = value

    @property
    def geomType(self):
        return self._vtype

    @property
    def base_args(self):
        """
        retrieve the arguments as they went in during creation
        :return: (dict)
        """
        return {'uid': self.id,
                'layer': self._layer_id,
                'geomType': self.geomType,
                'meta': self._meta,
                'layerName': self._layer_name,
                'children': list([x.base_args for x in self.children])}

    @classmethod
    def handle_class(cls, geom_type):
        return {GeomType['MESH']: MeshSolid,
                GeomType['ARC']: FamilySymbol,
                GeomType['CIRCLE']: FamilySymbol,
                GeomType['SOLID']: MEPSolidLine,
                GeomType['FACE']: MEPFace,
                GeomType['LINE']: MepCurve2d.from_viper,
                GeomType['POLYLINE']: MepCurve2d.from_viper,
                GeomType['SYMBOL']: GometryInstance,
                }.get(geom_type, None)

    @property
    def meta(self):
        return self._meta

    def get_item(self, k, d=None):
        return self._meta.get(k, d)

    # -------------------------------------------------------------
    # the main external interface { children, points, centroid }
    # managed by CadInterface. specific geometric ops mangaged by libs
    @property
    def children(self):
        """
        returns ICadInterface objects
        """
        return self._children

    @property
    def contains_solids(self):
        return False

    @property
    def points(self):
        raise NotImplemented('points not implemented in base')

    @property
    def centroid(self):
        raise NotImplemented('points not implemented in base')

    # -------------------------------------------------------------
    @classmethod
    def factory(cls, points=[], **kwargs):
        """
        Factory method to recursively / sequentially build geometry
        takes an dictionary with kwargs, and returns a generator

        :param points:
        :param kwargs:
        :return:
        """
        geom_type = kwargs.get('geomType', 0)
        geom_type = GeomType(geom_type if geom_type else 0)
        GeomClass = cls.handle_class(geom_type)

        if geom_type in [GeomType['ARC'], GeomType['CIRCLE']]:
            pt = [round(p, cls._ROUND) for p in points]
            yield GeomClass(*pt, **kwargs)

        elif geom_type == GeomType['SOLID']:
            yield GeomClass(**kwargs)

        elif geom_type == GeomType['FACE']:
            yield MEPFace(**kwargs)

        elif geom_type == GeomType['LINE']:
            yield GeomClass(points=points, **kwargs)

        elif geom_type == GeomType['POLYLINE']:
            xyzs = [points[i:i + 3] for i in range(0, len(points), 3)]
            for i in range(1, len(xyzs)):
                kgs = deepcopy(kwargs)
                kgs['points'] = xyzs[i - 1] + xyzs[i]
                kgs['children'] = None
                yield GeomClass(**kgs)

        elif geom_type == GeomType['MESH']:
            # group into points - [ x, y, z ]
            xyzs = [tuple(points[i:i + 3]) for i in range(0, len(points), 3)]
            xyz_set = list(set(xyzs))

            # group into faces - [ v1, v2, v3 ]
            faces = []
            for i in range(0, len(xyzs), 3):
                faces.append([xyz_set.index(xyzs[i+x]) for x in range(3)])

            xyz_set = list(map(list, xyz_set))
            msh = GeomClass(vertices=xyz_set, faces=faces, **kwargs)
            # print(msh)
            try:
                # if the centroid is not valid (computed in trimesh)
                # an error gets thrown, so none should be yielded
                # print(msh.points)
                valid = msh.centroid[0]
                # print(valid.shape)
                yield msh
            except:
                yield None

        elif geom_type == GeomType['SYMBOL']:
            if kwargs.get('use_syms', False) is True:
                yield GeomClass(**kwargs)
            else:
                for x in kwargs.get('children', []):
                    for r in cls.factory(**x):
                        yield r

    def children_of_type(self, geomtype):
        """

        :param geomtype:
        :return:
        """
        if isinstance(geomtype, str):
            geomtype = GeomType[geomtype]

        if self.geomType == geomtype:
            yield self
        elif self.children:
            for x in self.children:
                for c in x.children_of_type(geomtype):
                    yield c

    def contains_type(self, geomtype):
        return len(list(self.children_of_type(geomtype))) < 0

    def to_meshcat(self):
        pass

    def __eq__(self, other) -> bool:
        return self.points.__eq__(other.points)

    def __str__(self):
        return self.__repr__()

    def __repr__(self) -> str:
        st = 'class: {}, type: {}, layer: {}, n_kids: {}'.format(
            self.__class__.__name__, self._vtype, self.layer,
            len(self.children) if self.children else 0)
        return st


class RevitInterface(CadInterface):
    """
    SuperClass for using Revit Parameters as attributes
    """
    def __init__(self, *args, **kwargs):
        CadInterface.__init__(self, **kwargs)

    @property
    def category(self):
        return self.meta['category']

    @classmethod
    def handle_class(cls, geom_type):
        return {GeomType['MESH']: MeshSolidRV,
                GeomType['ARC']: FamilySymbol,
                GeomType['CIRCLE']: FamilySymbol,
                GeomType['SOLID']: MEPSolidLine,
                GeomType['FACE']: MEPFace,
                GeomType['LINE']: MepCurve2d.from_viper,
                GeomType['POLYLINE']: MepCurve2d.from_viper,
                GeomType['SYMBOL']: GometryInstance,
                }.get(geom_type, None)


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


class MeshSolid(CadInterface, Trimesh):
    contains_solids = True
    """
    Trimesh Handles almost everything
    
    """
    def __init__(self, faces=None, vertices=None, **kwargs):
        """
        Arbitrary thing assumed to be a line-based geometry
        """
        CadInterface.__init__(self, **kwargs)
        Trimesh.__init__(self, vertices=vertices, faces=faces)
        self._reps = {}
        self._facet_centroids = None
        self._current_rep = 'mesh'
        self._convex_comps = None
        # axis is two facet_box indices
        self._axis = []

    def reset(self):
        self._reps = {}
        self._facet_centroids = None
        self._current_rep = 'mesh'
        self._convex_comps = None
        # axis is two facet_box indices
        self._axis = []

    @classmethod
    def from_trimesh(cls, trimesh, **kwargs):
        return MeshSolid(faces=trimesh.faces, vertices=trimesh.vertices, **kwargs)

    def to_trimesh(self):
        res = self.as_mesh()
        kwargs = self.base_args
        tm = trimesh.Trimesh(faces=res.faces, vertices=res.vertices)
        return  tm

    def as_box(self):
        """ create bounding box and return a switcheroo """
        if self._current_rep == 'box':
            return self

        elif 'box' in self._reps:
            bx = self._reps['box']
            bx._reps[self._current_rep] = self
            bx._current_rep = 'box'
            return bx
        else:
            try:
                return self._as_box2()
            except:
                return None

    def _as_box2(self):
        b = Trimesh(vertices=self.vertices).convex_hull.bounding_box_oriented
        bx = self.__class__(vertices=b.vertices, faces=b.faces, **self.base_args)
        bx._reps[self._current_rep] = self
        bx._current_rep = 'box'
        return bx

    def as_mesh(self):
        if self._current_rep == 'mesh':
            return self
        else:
            res = self._reps['mesh']
            res._current_rep = 'mesh'
            res._reps = self._reps
            return res

    def _as(self, k):
        if k in self._reps:
            return self._reps[k]
        return None

    @property
    def as_lines(self):
        res = self._as('lines')
        if res:
            return res
        else:
            xg = []
            for e in self.points[self.edges_unique]:
                p1, p2 = e.tolist()
                xg.append(sg.LineString([p1, p2]))
            fnl = sg.MultiLineString(xg)
            self._reps['lines'] = fnl
            return fnl

    @property
    def convex_components(self):
        if self._convex_comps is None:
            self._convex_comps = self.convex_decomposition(engine='vhacd')
        return self._convex_comps

    @property
    def num_convex(self):
        cvx = self.convex_components
        if isinstance(cvx, Trimesh):
            return 1
        elif isinstance(cvx, list):
            return len(cvx)

    @property
    def facets_centroids(self):
        """
        centroids of facets

        Returns
        --------------
            np.array(len(self.facets), 3)
        """
        if self._facet_centroids is not None:
            return self._facet_centroids
        vert_face = self.vertices[self.faces[self.facets]]
        fct_cnts = vert_face.reshape(self.facets.shape[0], -1, 3).mean(axis=1)
        self._facet_centroids = fct_cnts
        return self._facet_centroids

    @property
    def bounding_cylinder_oriented(self):
        return None

    @property
    def convex_hull_ms(self):
        cvx = self.convex_hull
        bx = self.__class__(faces=cvx.faces, vertices=cvx.vertices, uid=self.id)
        return bx

    @property
    def as_obb(self):
        if 'obb' in self._reps:
            return self._reps['obb']
        else:
            obb = OBB(vertices=self.vertices)
            self._reps['obb'] = obb
            return obb

    @property
    def confidence(self):
        return 0

    @property
    def points(self):
        return self.vertices

    def intersect_line(self, origins, directions):
        return self.ray.intersects_location(origins, directions)

    @property
    def centroid(self):
        return np.mean(self.points, axis=0)

    def to_meshcat(self):
        from src.ui.adapter import meshcat
        obj = self.export(file_type='obj')
        mesh_geom = meshcat.geometry.ObjMeshGeometry(obj)
        mesh_geom.uuid = str(self.id)
        return mesh_geom


class ConnectionResult(object):
    def __init__(self):
        self._cntype = None

    def resolve(self, m1, m2):
        """
        point: face-to-face
            each vertex has corresponding corner on other face & areas of tris with tol

        tap: partial to face
        """
        r1 = h_points_close_to_face(m1, m2)
        if r1 is not None:
            pass


class OBB(Trimesh):
    """
    Utility class for doing stuff with OBBs
    """
    def __init__(self, vertices=None, mesh=None, debug=False, **kwargs):
        if mesh is None and vertices is not None:
            mesh = Trimesh(vertices=vertices).convex_hull.bounding_box_oriented

        Trimesh.__init__(self, vertices=mesh.vertices, faces=mesh.faces)
        vert_face = self.vertices[self.faces[self.facets]]

        # facet centroids np.shape([6, 3])
        self._facet_centers = vert_face.reshape(self.facets.shape[0], -1, 3).mean(axis=1)

        fct_norm = self.facets_normal
        mn1, mn2 = np.where(np.isclose(spatial.distance.cdist(fct_norm, -fct_norm), 0.))

        # pairs of axis points of obb : [ 3, 2]
        self._pairs = np.unique([sorted(v) for v in zip(mn1, mn2)], axis=0)
        lines = []
        for n1, n2 in zip(self.axes_vertices[:, 0, :], self.axes_vertices[:, 1, :]):
            lines.append(lib.geo.Line(lib.geo.Point(n1), lib.geo.Point(n2)))
        self._axis_skel = AxisSkeleton(lines)

    @property
    def facets_centers(self):
        return self._facet_centers

    @property
    def axes(self):
        return self._pairs

    @property
    def axes_vertices(self):
        return self._facet_centers[self._pairs]

    @property
    def axes_lengths(self):
        return self.skeleton.axis_lengths

    @property
    def skeleton(self):
        return self._axis_skel

    def skeleton_distance(self, other):
        if isinstance(other, OBB):
            return self.skeleton.distance_to(other.skeleton)

    def as_spheres(self, r=None):
        """
        generate spheres along longest axis
        :param r: radius of spheres
         if None, will use the second longest axis of the obb
        :return:
        np.array of shape([m, 4])
        """
        middle_axis_ix = self.skeleton.middle_axis
        if r is None:
            r = self.axes_lengths[middle_axis_ix]

        longst_axis_ix = self._axis_skel.longest_axis
        num_sphere = max([1, int(self.axes_lengths[longst_axis_ix] // r)])
        if num_sphere == 1:
            return np.expand_dims(np.concatenate([self.centroid, [r]], axis=0), axis=0)
        else:
            longest_axis = self.facets_centers[self.axes[longst_axis_ix]]
            offset = (longest_axis[0, :] - longest_axis[1, :]) / num_sphere
            sphere_points = longest_axis[1, :] + np.asarray([i * offset for i in range(num_sphere + 1)])
            rads = np.tile(r, (sphere_points.shape[0], 1))
            return np.concatenate([sphere_points, rads], axis=1)


class AxisSkeleton(object):
    def __init__(self, lines):
        self._lines = lines

    @property
    def longest_axis(self):
        return np.argmax(self.axis_lengths)

    @property
    def shortest_axis(self):
        return np.argmin(self.axis_lengths)

    @property
    def middle_axis(self):
        return {0, 1, 2}.difference([self.shortest_axis, self.longest_axis]).pop()

    @property
    def axis_lengths(self):
        return np.asarray([x.length for x in self.lines])

    @property
    def lines(self):
        return self._lines

    @property
    def points(self):
        return list(itertools.chain(*[l.points for l in self.lines]))

    def closest_points(self, obj):
        """ closest points from self.points to other (lines)

        Returns
        ----------
        projected  : float,   closest point on other
        best_dist  : float,   float, distance
        ix_this    : int,     index of closest point for each point
        """
        if isinstance(obj, list):
            return [self.closest_points(x) for x in obj]

        best_dist, ix_this, ix_other = self.nearest(obj)
        if len(ix_this) == 0:
            return ([], [], [])
        closest_point = self.points[ix_this[0]]

        if isinstance(obj, AxisSkeleton):
            projected = closest_point.projected_on(obj.lines[ix_other[0]])
            return projected, best_dist, ix_this

        elif type(obj) in [lib.geo.Point, lib.geo.Line, lib.geo.Plane]:
            projected = closest_point.projected_on(obj)
            return projected, best_dist, ix_this

    def distance_to(self, obj):
        best_dist, ix_this, ix_other = self.nearest(obj)
        return best_dist

    def nearest(self, obj):
        """
        indices of nearest object
        :param obj:

        Returns
        ----------
        projected  : float,   closest point (projected to other)
        ix_this    : float,   float, distance
        ix_other    : int,     index of closest triangle for each point
        """
        best, ixt, ixo = 1e9, [], []

        def check_dist(line, other, i, j, b, ixt, ixo):
            d = line.distance_to(other)
            if d < b:
                 b, ixt, ixo = d, [i], [j]
            elif d == b:
                ixt.append(i)
                ixo.append(j)
            return b, ixt, ixo

        # todo this sux, but its 3 x 3 so ok for now
        if isinstance(obj, AxisSkeleton):
            for j, o in enumerate(obj.lines):
                for i, line in enumerate(self.points):
                    best, ixt, ixo = check_dist(line, o, i, j, best, ixt, ixo)

        elif type(obj) in [lib.geo.Point, lib.geo.Line, lib.geo.Plane]:
            for i, line in enumerate(self.points):
                best, ixt, ixo = check_dist(line, obj, i, 0, best, ixt, ixo)

        elif isinstance(obj, list):
            return [self.nearest(x) for x in obj]
        return best, ixt, ixo



class MeshSolidRV(RevitInterface, MeshSolid):
    def __init__(self, faces=None, vertices=None, **kwargs):
        """
        SuperClass for Revitinterface option
        """
        RevitInterface.__init__(self, **kwargs)
        Trimesh.__init__(self, vertices=vertices, faces=faces)
        self._reps = {}
        self._facet_centroids = None
        self._convex_comps = None
        self._current_rep = 'mesh'
        self._axis = []


class RepresentationManager(object):
    def __init__(self):
        self._reps = {}

    def __getitem__(self, item):
        if item in self._reps:
            return self._reps[item]




# -------------------------------------------------
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
    def numpy(self):
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
        for x in self.children:
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


class MepCurve2d(CadInterface, LineString):
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

        if not (x1 == x2 and y1 == y2 and z1 == z2):
            if twod is True:
                z1, z2 = 0, 0
            p1, p2 = sorted([(x1, y1, z1), (x2, y2, z2)])
            return MepCurve2d(p1, p2, **kwargs)

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

    @property
    def numpy(self):
        p1, p2 = list(self.coords)
        return np.array([p1, p2])

    @property
    def line(self):
        pts = lib.geo.Line(*map(lib.geo.Point, self.points))
        return pts

    @property
    def length(self):
        return self.line.length

    def points_np(self):
        p1, p2 = list(self.coords)
        return np.array(p1), np.array(p2)

    def to_points_(self):
        (x1, y1), (x2, y2) = list(self.coords)
        a = Point(x1, y1)
        b = Point(x2, y2)
        return a, b

    def extend(self, start=0., end=0.):
        """
        returns Curve with same id (for external indexing)
        :param start: start extension
        :param end: end extension
        :return:
        """
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



def connector_approx(mesh, conn_ix):
    adj_ixs = np.where(mesh.face_adjacency == conn_ix)[0]
    angles = np.where(np.isclose(mesh.face_adjacency_angles[adj_ixs], 0.0, atol=1e-04))
    adj_with_same = adj_ixs[angles].squeeze()
    # print('adj_with_same', adj_with_same)
    adj_pair = mesh.face_adjacency[adj_with_same]
    vertices = mesh.faces[adj_pair]
    unique_v = np.unique(np.asarray(vertices))

    center = mesh.vertices[unique_v].mean(axis=0)
    return center


def connector_face(ch1, ch2):
    """
    Create connection node on faces
    :param mesh1:
    :param mesh2:
    :return:
    """
    loc1, ray_ix1, tri_ix1 = ch1.ray.intersects_location([ch1.centroid], [ch1.centroid - ch2.centroid])
    loc2, ray_ix2, tri_ix2 = ch2.ray.intersects_location([ch2.centroid], [ch2.centroid - ch1.centroid])

    cent1 = connector_approx(ch1, tri_ix1)
    cent2 = connector_approx(ch2, tri_ix2)
    return cent1, cent2


def box_from_bounds(mn, mx):
    """
    in a right-hand system :
                  3-------7
                 /|      /|
                2-------6 |
                | 1-----|-5
                |/      |/
                0-------4
    OBB returns points thusly:
        0 (self.max[0], self.max[1], self.min[2])),
        1 (self.min[0], self.max[1], self.min[2])),
        2 (self.min[0], self.max[1], self.max[2])),
        3 (self.max),
        4 (self.min),
        5 self.max[0], self.min[1], self.min[2]),
        6 self.max[0], self.min[1], self.max[2]),
        7 self.min[0], self.min[1], self.max[2])
    """
    vs = [mn,
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mn[1], mx[2]], mx]

    fcs = np.asarray([
        [0, 1, 2], [1, 2, 3],
        [0, 1, 4], [1, 4, 5],
        [0, 2, 4], [2, 4, 6],

        [2, 3, 6], [2, 6, 7],
        [4, 5, 6], [5, 6, 7],
        [3, 5, 7], [1, 3, 5],
    ])
    return Trimesh(vertices=vs, faces=fcs, process=True).bounding_box



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





