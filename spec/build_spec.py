import unittest
import importlib
from src import viper
import sys
import src.geom
from shapely.geometry import Point, LineString
from src.geom import MepCurve2d
from lib.figures import draw_save
import src.process
import src.misc.visualize
import src.render
import src.factory
import time
import json
from pprint import pprint
import src.rules.engine as engine
import src.rules.rule_helpers
from spec.seg_data import *

_segments = SEGMENTS

_ROOT = (2, 1, 0)

_fp_labels = {
    (2, 1): 'source',
    (2, 5): 'elbow',
    (4, 5): 'split',
    (4, 4): 'end',
    (4, 10): 'split',
    (2, 10): 'end',
    (6, 10): 'branch',
    (6, 12): 'end',
    (8, 10): 'branch',
    (8, 8): 'end',
    (10, 10): 'elbow',
    (10, 5): 'end',
}

_rv_render = [
    (2.0, 5.0, 4.0, 5.0) ,
    (4.0, 5.0, 4.0, 10.0) ,
    (4.0, 5.0, 4.0, 4.0) ,
    (4.0, 10.0, 6.0, 10.0) ,
    (4.0, 10.0, 2.0, 10.0) ,
    (6.0, 10.0, 6.0, 12.0) ,
    (6.0, 10.0, 8.0, 10.0) ,
    (8.0, 10.0, 8.0, 8.0) ,
    (8.0, 10.0, 10.0, 10.0) ,
    (10.0, 10.0, 10.0, 5.0) ,
]

_rv_idx = [

]

_segments2 = SEGMENTS2


_inters = [[0, 1], [1, 2], [3, 4]  ]


_object = {'color', 'layer', 'eid'}

_global = {
    'drop_head_z',
    'vert_head_z',
    'drop_head_offset',
    'vert_head_offset',
    'base_z',

    'branch_offset',    # if 0, remove
    'slope',
    'system_type',

}


class TestBuild2(unittest.TestCase):
    def setUp(self):
        self.lines = [MepCurve2d(x[0][0:2], x[1][0:2])
                      for x in _segments]

    def test_extend(self):
        _targets_ext = [[[-1, 0], [3, 0]],
                        [[3, 0], [-1, 0]],
                        [[-0.7, -0.7], [1.7, 1.7]],
                       ]
        knowns = [[[0, 0, 0], [2, 0, 0]],
                  [[2, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [1, 1, 0]],
                  ]
        ops = [1, 1, 1.5]
        for op, t, x in zip(ops, _targets_ext, knowns):
            crv = MepCurve2d(x[0][0:2], x[1][0:2])
            crv2 = crv.extend(op/2, op/2)
            self.assertAlmostEqual(crv2.length, (crv.length + op))

    def test_build2(self):
        sys = viper.System()
        sys.build(_segments)
        # sys.plot()
        root = (2, 1)
        sys.compute_direction(root)
        assert (4, 9.5) in sys.G
        sys.remove_colinear(root)
        assert (4, 9.5) not in sys.G
        draw_save(sys.G)

    def test_build(self):
        root = (2, 1)
        sys = viper.System(_segments)
        sys.bake(root)
        heur = HeuristicsFP()
        heur.apply(sys)
        for k, v in _fp_labels.items():
            lbl = sys.G.nodes[k].get('label', None)
            assert v == lbl

    def test_dist(self):
        p = Point(-245, 40)
        a = MepCurve2d((-242.2608, 50.1927), (-241.6667, 48.9786))
        print(a.distance(p))
        assert a.distance(p) != 0.

    def test_withins(self):
        p = Point(214, 60.6452)
        p1 = LineString([(214, 17.6452), (214, 217.6452)])
        #  p2 = Point(-921932, 17.6452)
        assert p.within(p1)

    def test_sane(self):
        a = MepCurve2d((4.0, 10.6), (4.0, 10.0))
        b = MepCurve2d((4.0, 9.5), (4.0, 4.0))
        assert a.intersects(b) is False
        assert b.intersects(a) is False

    def test_z(self):
        a = MepCurve2d((4.0, 10.6, 5), (4.0, 10.0, 5))
        print(a.points)

    def test_reduce(self):
        _sg = [(10, 5, 10, 1), (10, 2, 10, 5),
               (10, 1, 10, 8), (8, 8, 10, 8)]
        segs = []
        for x in _sg:
            p1, p2 = sorted([(x[0], x[1]), (x[2], x[3])])
            segs.append(MepCurve2d(p1, p2))
        sys = viper.System(segs, root=(8, 8))
        sys.bake()
        sys.gplot()


class TestData(unittest.TestCase):

    def setUp(self):
        with open('./data/sample_req.json', 'r') as f:
            xs = json.load(f)
            self.segs = xs.copy()
            f.close()

    def test_read(self):
        start = time.time()
        print(len(self.segs))
        system = src.factory.SystemFactory.from_request(self.segs, root=(-246, 45, 0))
        system = system.bake()
        end = time.time()
        print(end - start)

        system = HeuristicsFP()(system)
        # system.gplot()
        return system

    def test_rvt_fmt(self):
        system = self.test_read()

    def test_rvprep(self):
        tmp_arg = {'shrink': 0, 'base_z': 0}
        root = (2, 1)
        system = src.factory.SystemFactory.from_segs(_segments, root=root, lr='a')
        system = system.bake()
        system = HeuristicsFP()(system)

        # src.visualize.ord_plot(system.G)
        # system.gplot()
        geom, inds = src.render.RenderSystem(**tmp_arg)(system)
        pprint([(i, j) for i, j in enumerate(geom)])
        pprint(inds)

    def test_webinter(self):
        proc = src.process.SystemProcessor()
        out = proc.process(self.segs, [])
        pprint(out['geom'][0:20])


class TestData2(unittest.TestCase):
    def setUp(self):
        with open('./data/data1.json', 'r') as f:
            xs = json.load(f)
            self.segs = json.loads(xs['data'])[0]['children']
            f.close()

    def make_system(self):
        return src.factory.SystemFactory.from_serialized_geom(
            self.segs, root=(-246, 45, 0))

    def test_nested_sys(self):
        system = self.make_system()
        system = system.bake()

        system = HeuristicsFP()(system)
        system.gplot()

    def test_syms(self):
        with open('./data/data2.json', 'r') as f:
            xs = json.load(f)
            f.close()
        cls = src.factory.SystemFactory
        print(xs)
        _, fs = cls.handle_symbol_abstract(xs)
        fs = fs[0]
        print(fs)
        # print(fs.points)

    def test_mls(self):
        _F = src.factory.SystemFactory
        mls = _F.to_multi_line_string(self.segs)
        print(mls)

    def test_sys2(self):
        _F = src.factory.SystemFactory
        _S = viper.SystemV2
        start = time.time()
        # MultiLineString.
        system = _F.from_serialized_geom(self.segs, sys=viper.SystemV2, root=(-246, 45, 0))
        system = system.bake()

        # for (p, d) in system.G.nodes(data=True):
        #     print(d)
        system = HeuristicsFP()(system)
        system.stat()
        end = time.time()
        assert isinstance(list(system._data.values()).pop(0), MepCurve2d) is True
        print(end - start)
        return system
        # src.visualize.plot(system.mls)

    def test_oo(self):
        s = self.test_sys2()
        root = engine.nx_to_nodes(s.G, s.root)
        print(root)
        print(root.successors(), root.predecessors())
        print(len(root))

        assert len(root) == len(s.G), 'unequal number of nodes'
        n = root.successors()[0]
        pth = nx.shortest_path(s.G,  n.successors()[0].geom, n.geom)
        print(pth)
        acc = set()

        rs = src.rules.rule_helpers.is_main(n, acc, n.id)
        # print(rs)
        # assert rs is True
        assert len(rs) == len(pth), '{} {} '.format(len(rs), len(pth))
        rs2 = src.rules.rule_helpers.is_main(n, acc, root.id)
        assert rs2 is False

    def test_split(self):
        dga = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        root = engine.nx_to_nodes(dga, 1)
        assert len(root) == len(dga)
        e1 = root.successors(edges=True)[0]
        e1.split(27)
        assert len(root) == len(dga) + 1
        assert 27 in root
        assert 42 not in root
        node4 = root[4]
        assert node4 is not None
        assert node4.geom == 4

    def test_branch_translate(self):
        # dga = nx.DiGraph([tuple([tuple(x[0]), tuple(x[1])]) for x in _segments2])
        _root = (2, 1, 0)
        system = src.factory.SystemFactory.from_segs(_segments, root=_root, lr='a')
        system = system.bake()
        root = engine.nx_to_nodes(system.G, system.root)
        # system.gplot()
        print(root)
        node = root[(8., 8., 0)]

        assert node is not None
        src.rules.rule_helpers.translate(node, z=2)
        assert (8., 8., 2.) in root
        assert node.geom == (8., 8., 2.)
        for n in iter(node):
            assert list(n.geom)[-1] == 2.




def rl():
    importlib.reload(sys.modules[__name__])
    importlib.reload(src.geom)
    importlib.reload(viper)
    importlib.reload(src.render)


def setup():
    rl()
    T = TestData2()
    T.setUp()
    return T.make_system()

def go():
    rl()
    unittest.main()