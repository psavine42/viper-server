import unittest
import json, time
import pprint
import itertools
import numpy as np
from shapely.geometry import LinearRing, LineString, Point, MultiLineString, Polygon
from src.geom import MEPFace, MepCurve2d, MEPSolidLine, MeshSolid, GeomType, CadInterface
from src.viper import SolidSystem
from collections import Counter, defaultdict
from src.misc import utils, visualize
import src.geom as geom
import random
from scipy import spatial
from trimesh import Scene

from src.ui import visual
from src.ui.visual import visualize_indexed

# AOTP
from src.structs import aotp
from src.structs.cell import Cell
from src.structs.arithmetic import Interval


zmq_url = 'tcp://127.0.0.1:6000'


def to_structure(segs, use_syms=False):
    solids = []
    for x in segs:
        for j in CadInterface.factory(use_syms=use_syms, **x):
            if j:
                solids.append(j)
    return solids


class Test3DGeom(unittest.TestCase):
    """
    Testing basic serialization, deserialization of Meshy things
    """
    def setUp(self):
        with open('./data/sample/geom2.json', 'r') as f:
            self.segs = json.load(f)
            f.close()
        with open('./data/sample/mesh_sample.json', 'r') as f:
            self.small = json.load(f)
            f.close()

    def _valid_meshes(self, xs):
        xs = [x for x in xs if isinstance(x, MeshSolid) and x.centroid[0] < 1000.]
        return xs

    def test_parse(self):
        solids = to_structure(self.segs)

        solids[0].show()

    def test_layer(self):
        solids = to_structure(self.small, use_syms=True)
        solid = solids[0]
        # print(solid)
        assert len(solid.children) == 3
        for x in solid.children:
            print(x)
        assert solid.layer == 1040148

    def test_layer_s(self):
        xs = to_structure(self.segs, use_syms=False)
        xs = self._valid_meshes(xs)
        visualize_indexed(xs)

    def test_export_mesh(self):
        xs = to_structure(self.segs, use_syms=False)
        xs = self._valid_meshes(xs)
        # for x in xs:
        print(len(xs))
        ds = xs[0].export(file_type='obj')
        print(ds)

    def test_parse2(self):
        xs = to_structure(self.segs, use_syms=True)
        xs = [x for x in xs if x.children_of_type(geomtype=GeomType['MESH'])]
        scene = Scene()
        for gm in xs:
            scene.add_geometry(gm)
        scene.show()

    def test_axes_(self):
        xs = to_structure(self.small, use_syms=False)
        xs = self._valid_meshes(xs)
        solid = xs[0]
        print(solid.obb_centroid)
        print(solid.obb_axes)
        print(solid.bounds)
        print(solid.extents)


class TestTrain(unittest.TestCase):

    def setUp(self):
        import importlib
        importlib.reload(geom)
        st = '/home/psavine/source/viper'
        with open(st + '/data/sample/test5.json', 'r') as f:
            self.train = json.load(f)
            f.close()

    def preprocess(self, categories=None):
        return self.preprocess_(self.train)

    @staticmethod
    def preprocess_(data, categories=None):
        """
        Create Meshes
        Pickle Meshes
        """
        res = []
        for x in data:
            if len(x['points']) < 3:
                continue
            if x['geomType'] == 3:
                x['base_curve'] = x['points'][:6]
                x['points'] = x['points'][6:]

            elif x['geomType'] == 9:
                x['base_curve'] = x['points'][:3]
                x['points'] = x['points'][3:]
            else:
                continue
            dst = {**x['attrs'], **x['attrd'], **x['attri']}
            x['attrs'] = dst
            x.pop('attrd')
            x.pop('attri')
            x['ogeomType'] = x['geomType']
            x['geomType'] = 8
            x['uid'] = int(x['attrs']['ElementID'])
            x['category'] = x['attrs']['Category']
            if categories is None or x['category'] in categories:
                rs = list(filter(None, geom.RevitInterface.factory(**x)))
                res.extend(rs)
        return res

    def test_preproc(self):
        res = self.preprocess()
        for r in res:
            assert r.id == int(r.meta['attrs']['ElementID'])


class TestHvacSys(unittest.TestCase):
    def setUp(self):
        st = '/home/psavine/source/viper'
        with open(st + '/data/sample/test2.json', 'r') as f:
            xs = json.load(f)
            self.segs = json.loads(xs['data'])[0]['children']
            f.close()

    def _valid_meshes(self, xs):
        res = []
        for x in xs:
            for j in x.children_of_type(GeomType['MESH']):
                res.append(j)
        return res

    def test_load(self):
        segs = list(filter(lambda x: x['geomType'] == 6, self.segs))
        print(len(segs))
        dct = segs[0]
        fc = MEPSolidLine(**dct)
        print(fc.children[0])
        print(fc.children[0].layer)
        print(fc, fc.layer)
        print(*fc.children[0].exterior.coords)

    def to_structure(self, use_syms=False):
        return to_structure(self.segs, use_syms=use_syms)

    def test_parse(self):
        solids = self.to_structure(use_syms=False)
        c = Counter([type(x) for x in solids])
        print(c)
        xs = [x for x in solids if isinstance(x, MeshSolid)]
        xs = [x for x in xs if x.centroid[0] < 1000]
        print('num solids ', len(xs))
        sys = SolidSystem(xs)
        res = sys.res
        return sys

    def test_build2(self):
        base = self.to_structure(use_syms=True)
        print(Counter([type(x) for x in base]))

        xs = self._valid_meshes(base)
        xs = [x for x in xs if x.centroid[0] < 1000]
        assert len(xs) > 500
        print(Counter([type(x) for x in xs]))
        sys = SolidSystem()
        sys._build2(xs)
        res = sys.res
        return sys

    def test_intersect(self):
        base = self.to_structure(use_syms=True)
        print(Counter([type(x) for x in base]))
        start = time.time()
        xs = self._valid_meshes(base)
        xs = [x for x in xs if x.centroid[0] < 1000]
        assert len(xs) > 500
        end = time.time()
        print('preprocessing : ', end - start)

        # get most common 'layer'
        _index_layers = [s.layer for s in xs]
        lyr_cnt = Counter(_index_layers)
        test_layer = lyr_cnt.most_common(1)[0][0]
        print('test_layer', test_layer)
        xs = [x for x in xs if x.layer == test_layer]

        _point_tree = []
        _centers = []
        _index_starts = [0]
        _pt_to_solid = dict()
        for i, s in enumerate(xs):
            for j in range(len(s.points)):
                _pt_to_solid[_index_starts[-1] + j] = i
            _index_starts.append(_index_starts[-1] + len(s.points))
            _point_tree.append(s.points)
            _centers.append(s.centroid)

        _centers = spatial.KDTree(_centers)
        _point_tree = np.concatenate(_point_tree)
        print('all_pts', _point_tree.shape)
        start = time.time()

        res = defaultdict(set)
        for i, solid in enumerate(xs):

            dist_mat, ixs = _centers.query([solid.centroid], k=4)

            for ix in ixs[0][1:].tolist():
                if i in res[ix]:
                    continue
                other = spatial.distance.cdist(solid.points, xs[ix].points)
                ixr, mins = np.where(other < 0.1)
                if len(ixr) > 1:
                    res[i].add(ix)
                    res[ix].add(i)

        end = time.time()
        pprint.pprint(res)
        print('naive solution : ', end - start)
        visualize_indexed(xs)

    def test_batch_pts(self):
        # todo is this faster than loop ??
        import operator
        res = [np.random.random((random.randint(4, 10), 3)) for i in range(4)]
        lns = [y.shape[0] for y in res]
        lns = list(itertools.accumulate(lns, operator.add))
        print(lns)
        stacked = np.concatenate(res)
        target = np.random.random((7, 3))
        other = spatial.distance.cdist(stacked, target)
        ixr, mins = np.where(other < 0.31)
        ixs = set(ixr.tolist())
        for ix in ixs:
            np.where(np.logical_and(ix >= 6, ix <= 10))
        print(ixr)

    def test_build3(self):
        import src.viper
        base = self.to_structure(use_syms=True)
        xs = self._valid_meshes(base)
        xs = [x for x in xs if x.centroid[0] < 1000]
        assert len(xs) > 500

        sys = src.viper.SolidSystem()
        res = sys._build3(xs)
        pp = Counter([len(v) for k, v in res.items()])
        print(pp)
        return sys, xs

    def test_buildnl(self):
        import src.viper
        import importlib
        importlib.reload(src.viper)
        base = self.to_structure(use_syms=True)
        xs = self._valid_meshes(base)
        xs = [x for x in xs if x.centroid[0] < 1000]
        assert len(xs) > 500

        sys = src.viper.SolidSystem()
        res = sys._build_no_layer(xs)
        pp = Counter([len(v) for k, v in res.items()])
        print(pp)
        return sys, res

    def test_prop0(self):
        start = time.time()
        sys, solids = self.test_build3()
        print('setup in : {}'.format(time.time() - start))

        with2 = [k for k, v in sys.res.items() if len(v) == 2][0]

        # solid and two neighbors
        item = solids[with2]
        n1, n2 = [solids[i] for i in sys.res[with2]]
        lines = []

        seen_c = set()
        for k, v in sys.res.items():
            if len(v) > 0:
                c1 = tuple(solids[k].centroid.tolist())
                seen_c.add(c1)
                for vl in v:
                    c2 = tuple(solids[vl].centroid.tolist())
                    lines.append(MepCurve2d(c1, c2))
                    seen_c.add(c2)

        h = visual.visualize_points(list(seen_c))
        visual.visualize_lines(lines, handle=h)

    def test_prop(self):
        from src.structs import arithmetic
        import math
        start = time.time()
        sys, solids = self.test_build3()
        print('setup in : {}'.format(time.time() - start))

        with2 = [k for k, v in sys.res.items() if len(v) == 2][0]

        item = solids[with2]
        neigh_ix = list(sys.res[with2])
        bbx = item.bounding_box_oriented
        extents = np.copy(bbx.primitive.extents)
        transform = np.copy(bbx.primitive.transform)

        # extents = data['extents']
        print(bbx, extents, transform)
        print(bbx.face_normals)

        # compute the axes from face normals
        # def cells_for_mesh(msh):
        norm3 = item.obb_axes
        right = Cell(arithmetic.Constant(math.pi / 2))


        centroid_cell = Cell()

        # estimator OBB
        obb_ax0 = [Cell(norm3[0][i]) for i in range(3)]
        obb_ax1 = Cell(norm3[1])
        obb_ax2 = Cell(norm3[2])


        aotp.p_angle()

        # propogate these to intervals
        interval = ''

        # estimate of how good OBB estimator is

        # estimator using Neigh Centroids
        def estimate_centroid(msh1, msh2):
            cnt1 = msh1.centroid
            cnt2 = msh2.centroid
            vec1 = cnt1 - cnt2

        neigh_ax0 = [Cell() for i in range(3)]

        to_interval_cone = Cell()
        # Interval(54.9, 56.1)
        for c1, c2 in zip(neigh_ax0, obb_ax0):
            Interval()

        # intersect the estimate Intervals

        # estimator using Neigh OBB

        #

    def test_vecto1(self):
        c1, c2, c3 = Cell(), Cell(), Cell()
        vec = Cell(profile=True)
        aotp.vectorizer(c1, c2, c3, vec)

        c1.add_contents(1)
        assert vec.contents is None
        c2.add_contents(2)
        c3.add_contents(3)
        print(vec.contents)
        assert np.array_equal(vec.contents.value, np.array([1, 2, 3]))

    def test_vecto2(self):
        c1, c2, c3 = Cell(), Cell(), Cell()
        vec = Cell()
        aotp.to_vec_np(c1, c2, c3, vec)

        c1.add_contents(1)
        assert vec.contents is None
        c2.add_contents(2)
        print(vec.contents, c2.contents)
        c3.add_contents(3)
        print(vec.contents)
        assert np.array_equal(vec.contents.value, np.array([1, 2, 3]))

        vec.add_contents(np.array([2, 3, 1]))
        print(c1.value)
        assert c1.value == 2


    def test_centroid_vecs(self):
        """
        a box

        :return:
        """
        np.allclose()
        center_est1 = np.array([10, 10, 0])
        center_est2 = np.array([12, 11, 0])

        center_othr1 = np.array([20, 10, 0])
        center_othr2 = np.array([20, 11, 0])

        xyz = center_est1 - center_othr1
        xyz2 = [8, 8, 0]

        cntr_cells = aotp.make_cells(3)
        axis_cells = aotp.make_cells(3)
        cntr_x_obb = Cell()
        cntr_y_obb = Cell()
        cntr_z_obb = Cell()

        cntr_x_obb.add_contents(Interval(center_est1[0], center_est2[0]))
        cntr_y_obb.add_contents(Interval(center_est1[1], center_est2[1]))
        cntr_z_obb.add_contents(Interval(center_est1[2], center_est2[2]))



    def test_parsesym(self):
        solids = self.to_structure(use_syms=True)
        c = Counter([type(x) for x in solids])
        c = Counter([type(x) for x in solids])
        print(c)
        xs = [x for x in solids if x.contains_solids]
        # [print(x) for x in xs[0].children]
        print('num solids ', len(xs))
        sys = SolidSystem(xs)
        res = sys.res
        return sys

    def visualize(self, sys):
        nodes = list(itertools.chain(*[xs.values() for xs in sys.res]))
        print('visualizeing {} nodes'.format(len(nodes)))
        G = utils.bunch_to_nx(nodes)
        visualize.gplot(G)

    def visualize_pts(self):
        solids = self.to_structure()
        vec = []
        for s in solids:
            vec += list(s.points)
        vec = np.asarray(list(filter(lambda x: 500 < x[0] < 1000, vec)))
        colors = [visualize.hcolor() for i in range(len(vec))]
        print(vec.shape)
        visualize._3darr(vec, colors=colors)

    def test_rules(self):
        pass

    def test_mlstuff(self):
        x1 = LineString([(0, 0), (1, 1)])
        x2 = LineString([(1, 1), (2, 2)])
        x3 = LineString([(2, 2), (0, 0)])
        # mls = MultiLineString([x1, x2, x3])
        # lnr = LinearRing(mls)
        plg = Polygon([(0, 0, 0), (1, 1, 1), (2, 2, 2)])
        # print(mls)
        print(plg)

    def test_npm(self):
        xs = np.array([[2, 2, 2], [1, 1, 1]])
        x2 = np.array([1, 1, 1])

        x3 = xs - x2
        print(x3)