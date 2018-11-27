import networkx as nx
import numpy as np

import dxfgrabber
from dxfgrabber.dxfentities import Body, Insert
import src.formats.sat as sab
import src.propogate as gp
from lib.geo import Point, Line
from src.misc.utils import nxgraph_to_nodes, nodes_to_nx
from src.formats import skansk as sks


# '/home/psavine/source/viper/data/out/commands/graphtest_full.pkl'
# BUILDING GRAPHS -----------------------------------------------
class ViperDxf(object):
    def __init__(self,
                 in_path=None,
                 out_path=None,
                 block_names=['3DM_UPR']
                 ):
        # paths
        self._dxf_path = in_path
        self._out_path = out_path

        # these come from dxf, but start with no_op transforms
        self._CAD_TRANSLATE = np.array([0., 0., 0])
        self._CAD_SCALE = 1

        # these still need to be populated dynamically...
        self.ANGLE = 346.9658914
        self.REVIT_SCALE = 1 / 0.4799

        # for connections - set manually
        self.FILTER_TOL = 0.3
        self.CONNECT_TOL = 0.4
        self._kd_tol = 0.75
        self._block_names = block_names

        # data cache
        self.root_nodes = []
        self._fam_points = []

    @property
    def CAD_ANGLE(self):
        return np.radians(360 - self.ANGLE)

    def as_linecyl(self, this):
        line = (this.line.numpy * self._CAD_SCALE) + self._CAD_TRANSLATE
        radius = this.radius * self._CAD_SCALE
        this.line = Line(Point(line[0]), Point(line[1]))
        this.radius = radius
        return this

    @staticmethod
    def as_line_xf(this, trs=np.array([0., 0., 0.]), scale=1.):
        pts = this.children_of_type('point')
        if len(pts) == 2:
            xf = this.children_of_type('transform')
            if len(xf) == 1:
                try:
                    p1, p2 = pts
                    p11 = (xf[0].apply(p1.geom) * scale) + trs
                    p12 = (xf[0].apply(p2.geom) * scale) + trs
                    return Line(Point(p11), Point(p12))
                except Exception as e:
                    return None
        return None

    def setup_point_based(self, inserts):
        self._fam_points = np.array(
            [self._CAD_TRANSLATE + (np.asarray(x.insert) * self._CAD_SCALE)
             for x in inserts if x.name in self._block_names]
        )
        self._fam_points[:, -1] = 0.

    @staticmethod
    def _is_valid(x):
        return x is not None and x.valid is True

    def _length_filter(self, x):
        return self._is_valid(x) and x.line.length > x.radius and x.line.length > self.FILTER_TOL

    def build_roots(self):
        reader = sab.SABReader()
        dxf = dxfgrabber.readfile(self._dxf_path)

        # read header info
        self._CAD_SCALE = 1. / dxf.header.get('$DIMALTF', 1.)

        # convert binary acis data to something logical
        acs_ents = [reader.read_single(x.acis) for x in dxf.entities if isinstance(x, Body)]

        # get the translation to a logical 0
        self._CAD_TRANSLATE = -1 * self.as_line_xf(acs_ents[0], scale=self._CAD_SCALE).points[0].numpy

        # read families
        self.setup_point_based(filter(lambda x: isinstance(x, Insert), dxf.entities))

        pipes_long = list(
            filter(self._length_filter, map(self.as_linecyl, filter(self._is_valid, map(sab.SACCylinder, acs_ents))))
        )

        pairs, tree, pts_ind = sks.connect_by_pairs(pipes_long, factor=self.CONNECT_TOL)
        node_dict = sks.make_node_dict(pairs, pts_ind, pipes_long)
        root_nodes = sks.connected_components(list(node_dict.values()), min_size=2)
        final_roots = sks.kdconnect(root_nodes, tol=self._kd_tol)
        return final_roots

    def xprocess(self, node):
        root = gp.SpatialRoot()(node)   # Find Logical Root
        root = gp.EdgeDirector()(root)
        root = gp.BuildOrder()(root)

        tree = sks.resolve_heads(root, self._fam_points)

        setup = gp.Chain(
            gp.Rotator(self.CAD_ANGLE),
            gp.EdgeDirector(),
            gp.BuildOrder(),
            sks.DetectTriangles(build=True),
            gp.EdgeDirector()
            )
        root = setup(root)
        root = gp.FunctionQ(sks.add_ups)(root, h=0.1)
        root = gp.GraphTrim()(root, tol=0.01)

        root = gp.EdgeDirector()(root)
        root = gp.FunctionQ(sks.remove_short)(root, lim=0.25)

        sks.DetectElbows()(root.neighbors(edges=True))
        root = gp.EdgeDirector()(root)
        root = gp.Scale(self.REVIT_SCALE)(root)

        root = sks.LabelConnector()(root)
        root = gp.FunctionQ(sks.resolve_elbow_edges)(root)
        gp.FunctionQ(sks.align_tap, edges=True)(root.neighbors(edges=True))
        root = sks.MakeInstructionsV4()(root, use_radius=True)
        return root

    def load(self):
        with open(self._out_path, 'rb') as F:
            nx_graph = nx.read_gpickle(F)
            self.root_nodes = nxgraph_to_nodes(nx_graph)

    def build(self):
        G = nx.DiGraph()
        root_nodes = self.build_roots()
        for i in range(len(root_nodes)):
            root_i = self.xprocess(root_nodes[i])
            root_i.write(root=True)
            G = nodes_to_nx(root_i, G=G)
            self.root_nodes.append(root_i)

        # save to an nx graph
        nx.write_gpickle(G, self._out_path)



