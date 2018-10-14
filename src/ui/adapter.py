import numpy as np
from src.geom import MepCurve2d, angle_to
from lib.meshcat.src import meshcat
from lib.meshcat.src.meshcat import geometry, Visualizer
from lib.meshcat.src.meshcat import transformations as T


_base = MepCurve2d((0., 0., 0.), (0., 1., 0.))

class Visualized:
    def __init__(self, graph_obj, viewer):
        self._obj = graph_obj
        self._viewer = viewer

    @property
    def path(self):
        return str(self._obj.id)


class ViewNode(Visualized):
    def __init__(self, *args):
        super(ViewNode, self).__init__(*args)

    def build(self):
        origin = self._obj.geom
        xform = T.translation_matrix(np.array(list(origin)))
        return geometry.Sphere(0.4), xform


class ViewEdge(Visualized):
    def __init__(self, *args):
        super(ViewEdge, self).__init__(*args)

    @staticmethod
    def transform(curve):
        origin, p2 = curve.points_np()
        mag = origin + (p2 - origin) / 2
        xform1 = T.translation_matrix(mag)
        angle = angle_to(_base, curve)

        xf2 = xform1 - np.eye(4)
        if curve.direction[-1] != 0.:
            xf2 += T.rotation_matrix(angle, np.array([1., 0., 0.]))
        else:
            xf2 += T.rotation_matrix(angle, np.array([0., 0., 1.]))

        return xf2

    def build(self):

        # print(origin)
        curve = self._obj.curve
        xform = self.transform(curve)
        return geometry.Cylinder(curve.length, radius=0.1), xform


class ViewAdapter(Visualizer):
    def __init__(self, root=None, **kwargs):
        super(ViewAdapter, self).__init__(**kwargs)
        self._mapping = {}
        if root is not None:
            self.build(root)

    def add(self, view_obj):
        geom, xforms = view_obj.build()
        self[view_obj.path].set_object(geom)
        self[view_obj.path].set_transform(xforms)
        self._mapping[view_obj.path] = view_obj


    def build(self, root):
        for n in root.__iter__():
            vn = ViewNode(n, self)
            self.add(vn)
            for edge in n.successors(edges=True):
                ve = ViewEdge(edge, self)
                self.add(ve)

    def on_select(self):
        pass


