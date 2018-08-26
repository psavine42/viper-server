from __future__ import absolute_import, division, print_function

import time
import numpy as np

import meshcat
import meshcat.geometry as g
from unittest import TestCase

verts = np.random.random((3, 1000)).astype(np.float32)

class TestCat(TestCase):
    def test_vis(self):
        zmq_url = 'tcp://127.0.0.1:6000'
        vis = meshcat.Visualizer(zmq_url=zmq_url).open()
        vis.set_object(g.Points(
            g.PointsGeometry(verts, color=verts),
            g.PointsMaterial()
        ))
        vis.close()