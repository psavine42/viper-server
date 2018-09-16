from __future__ import absolute_import, division, print_function

import time, math
import numpy as np
from lib.meshcat.src import meshcat
from src.ui import adapter
from spec.seg_data import *
from src.ui.adapter import geometry, ViewAdapter, ViewEdge
from unittest import TestCase
from lib.meshcat.src.meshcat import transformations as T
from src import MepCurve2d, SystemFactory, System, RuleEngine, KB, RenderNodeSystem
from src.rules import heursitics

zmq_url = 'tcp://127.0.0.1:6000'
_base = MepCurve2d((0., 0., 0.), (0., 1., 0.))


class TestGeom(TestCase):
    def test_angle(self):

        #xform1 = T.translation_matrix(np.array(list(origin)))
        pass


class TestCat(TestCase):

    def setUp(self):
        self.vis = ViewAdapter(zmq_url=zmq_url)

    def test_curve(self, crv, name):
        print(name)
        gx = geometry.Cylinder(crv.length, 0.2)
        xformx = ViewEdge.transform(crv)
        # print(xformx)

        self.vis[name].set_object(gx)
        self.vis[name].set_transform(xformx)
        # self.vis[name].set

    def test_vis(self):
        curve_on_x = MepCurve2d((15., 0., 0.), (5., 0., 0.))
        curve_on_y = MepCurve2d((0., 2., 0.), (0., 8., 0.))

        curve_on_z = MepCurve2d((10., 0., 1.), (10., 0.,5.))

        offset_on_x = MepCurve2d((5., 5., 0.), (15., 5., 0.))
        self.test_curve(curve_on_x, 'xx')
        self.test_curve(curve_on_y, 'yy')
        self.test_curve(curve_on_z, 'zz')

        self.test_curve(offset_on_x, 'xof')


    def test_mat(self):
        _base = MepCurve2d((0., 0., 0.), (0., 1., 0.))
        curve = MepCurve2d((10., 0., 0.), (20., 0., 0.))
        dr = math.acos(min(1, abs(np.dot(curve.direction, _base.direction))))
        print(dr)
        print(curve.direction, _base.direction)

    def test_adapter_nodes(self):
        data = load_segs(fl='1535158393.0-revit-signal')
        system = SystemFactory.from_serialized_geom(
            data, sys=System, root=(-246, 45, 0))
        system = system.bake()
        rules = heursitics.EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root, mx=2500, debug=False, nlog=20)
        Kb = KB(rules.root)
        root = Eng.alg2(system.root, Kb)
        print('nodes ', len(root))
        renderer = RenderNodeSystem()
        root = renderer.render(root)
        self.vis.build(root)

    def tearDown(self):
        self.vis.delete()

class TestPCat(TestCat):
    def tearDown(self):
        pass