import unittest
import json
import pprint
import itertools
import numpy as np
from shapely.geometry import LinearRing, LineString, Point, MultiLineString, Polygon
from src.geom import MEPFace, MEPSolidLine, MeshSolid, GeomType, CadInterface
from src.viper import SolidSystem
from collections import Counter
from src.misc import utils, visualize
import random
from trimesh import Scene


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

    def test_parse(self):
        solids = to_structure(self.segs)

        solids[0].show()

    def test_parse2(self):
        xs = to_structure(self.segs)
        print(len(xs))

        c = Counter([type(x) for x in xs])
        print(c)
        solids = [x for x in xs if isinstance(x, MeshSolid)]
        print(len(solids))
        scene = Scene()
        for gm in solids:
            # print(gm.centroid)
            scene.add_geometry(gm)

        scene.show()

class TestHvacSys(unittest.TestCase):
    def setUp(self):
        with open('./data/sample/test2.json', 'r') as f:
            xs = json.load(f)
            self.segs = json.loads(xs['data'])[0]['children']
            f.close()
        # with open('./data/sample/geom2.json', 'w') as f:
        #     json.dump(self.segs, f)

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
        solids = self.to_structure()
        c = Counter([type(x) for x in solids])
        print(c)
        xs = [x for x in solids if isinstance(x, MeshSolid)]
        # [print(x) for x in xs[0].children]
        xs = [x for x in xs if x.centroid[0] < 1000]
        print('num solids ', len(xs))
        sys = SolidSystem(xs)
        res = sys.res
        return sys

    def test_child_lyt(self):
        base = self.to_structure(use_syms=True)
        c = Counter([type(x) for x in base])

        xs = [x for x in base if x.children_of_type(geomtype=GeomType['MESH'])]

        xs = [x for x in xs if x.centroid[0] < 1000]
        assert len(xs) > 500
        sys = SolidSystem(xs)
        res = sys.res
        return sys

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

    def visualize(self):
        sys = self.test_parse()
        nodes = list(itertools.chain(*[xs.values() for xs in sys.res]))
        print(len(nodes))
        G = utils.bunch_to_nx(nodes)
        visualize.gplot(G)

    def visualize_pts(self):
        solids = self.to_structure()
        vec = []

        for s in solids:
            # print(list(s.points))

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