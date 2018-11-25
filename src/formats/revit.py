
"""

Revit utilities
"""
import json
import os
import math
from enum import Enum
import lib.geo as geo
import numpy as np
import src.geom
import src.geombase as geombase
import transforms3d.taitbryan as tb
import src.structs.node_utils as gutil
from src.structs import Node, Edge
import importlib
import src.formats.revit_base
import src.formats.revit_base as rvb
importlib.reload(rvb)

_TOLERANCE = 128    # 1/128 inch


def round_inches(x, tol=_TOLERANCE):
    return round(x * tol) / tol


def round_slope(x, tol=_TOLERANCE):
    return round(12 * x * tol) / tol


class Cmds(Enum):
    Noop = 0
    Pipe = 1
    Connect = 2
    Delete = 3
    Elbow = 4
    Tee = 5
    FamilyOnFace = 6
    Tap = 7
    Coupling = 8
    Transition = 9
    CapEnd = 10
    MoveEnd = 11
    TEST = 12
    Query = 13
    FamilyOnPoint = 12
    PipeBetween = 15


class CmdType(Enum):
    Create = 0
    Delete = 1
    Update = 2
    BuildFull = 3
    Connect = 4


class FamilyCmd(Enum):
    Connectors = 0
    LocationGeometry = 1
    FromAdjusted = 2


class PipeCmd(Enum):
    Connectors = 0
    Points = 1
    ConnectorPoint = 2
    PointConnector = 3
    MoveConnect = 4


class CouplingCmd(Enum):
    Connectors = 0
    LocationGeometry = 1
    ConnectorPoint = 2
    MoveConnect = 4


class ElbowCmd(Enum):
    Connectors = 0
    LocationGeometry = 1
    FromAdjusted = 2


class TeeCmd(Enum):
    Connectors = 0
    LocationGeometry = 1
    FromAdjusted = 2


_built = 'built'
_conn1 = 'conn1'
_conn2 = 'conn2'


_cmd_types = {'tee': TeeCmd,
              'coupling': CouplingCmd,
              'pipe': PipeCmd,
              'elbow': ElbowCmd}


def is_built(graph_obj):
    return rvb.is_complete(graph_obj)


def ensure_built(graph_obj, strategy=None):
    """
    if a graph objects needs to exist in revit env during some command,
    while preparing that command, call this on the graph object to generate
    commands for this object
    :param graph_obj:
    :return:
    """
    if is_built(graph_obj):
        return []
    klass_name = graph_obj.get('$create', None)
    cmd = _cmd_types.get(klass_name, None)
    klass = Command.get_cls_map().get(cmd, None)
    if klass is None:
        print('missing creator class', klass_name)
        return []
    tried = graph_obj.get('$attempts', [])
    todo_ky = [o for o in klass.strategy if o not in tried]
    fn = klass.get_func_map()[todo_ky]
    if fn is None:
        print('missing fn')
        return []
    return fn(graph_obj)


class CommandFactory(object):
    # https://thebuildingcoder.typepad.com/blog/2010/06/place-family-instance.html
    def __init__(self):
        self._tried = {}
        self._strategies = set()

    @classmethod
    def get_func_map(cls):
        return {}

    @classmethod
    def create(cls, cmd, *data, **kwargs):
        fn = cls.get_func_map().get(cmd.value, None)
        if fn is not None:
            return fn(*data, **kwargs)


class CurvePlacement(CommandFactory):
    """
    All functions that take an instance of 'Edge'
    """
    strategy = [0, 1, 4]

    # Helpers --------------------------------
    @classmethod
    def radius_from_edge(cls, edge, use_radius=True):
        if use_radius is True and edge.get('radius') is not None:
            return edge.get('radius') * 2
        return 1 / 12

    # Main functions --------------------------------
    @classmethod
    def from_connectors(cls, edge, use_radius=True, **kwargs):
        r = cls.radius_from_edge(edge, use_radius)
        cdef = [Cmds.Pipe.value, CmdType.BuildFull.value, edge.id]
        return [cdef + [edge.id, edge.source.id, edge.target.id, r]]

    @classmethod
    def from_points(cls, edge, **kwargs):
        cdef = [Cmds.Pipe.value, edge.id]
        return [cdef + cls.from_2points(edge, **kwargs)]

    @classmethod
    def from_2points(cls, edge, shrink=True, use_radius=True, **kwargs):
        start, end = edge.source, edge.target
        crv = src.geom.MepCurve2d(start.geom, end.geom)
        r = cls.radius_from_edge(edge, use_radius)
        if shrink is not None:
            s1 = -r * 0.2 if edge.source.npred != 0 else 0.
            s2 = -r * 0.2 if edge.target.nsucs != 0 else 0.
            p1, p2 = crv.extend(s1, s2).points
        else:
            p1, p2 = crv.points
        return list(p1) + list(p2) + [r]

    @classmethod
    def from_connector_point(cls, edge, use_radius=True, **kwargs):
        start, end = edge.source, edge.target
        r = cls.radius_from_edge(edge, use_radius)
        cdef = [Cmds.Pipe.value, CmdType.Create.value, edge.id]
        return [cdef + [edge.id, start.id] + list(end.geom) + [r]]

    # composite commands ---------------------------
    @classmethod
    def move_end_to_point(cls, edge, end_index, xyz, **kwargs):
        cdef = [Cmds.MoveEnd.value, CmdType.Update.value, edge.id]
        return [cdef + [edge.id, end_index] + xyz]

    @classmethod
    def move_end_to_connector(cls, edge, end=1, **kwargs):
        """ existing edge """
        tgt = edge.source if end == 0 else edge.target
        cdef = [Cmds.MoveEnd.value, CmdType.Connect.value, edge.id]
        return [cdef + [edge.id, tgt.id]]

    @classmethod
    def move_ends_to_connector(cls, edge, **kwargs):
        """ existing edge """
        cmd1 = cls.move_end_to_connector(edge, end=0)
        cmd2 = cls.move_end_to_connector(edge, end=1)
        return cmd1 + cmd2

    @classmethod
    def from_point_connector(cls, edge, **kwargs):
        cmd1 = cls.from_points(edge, **kwargs)      # create from points
        cmd2 = cls.move_end_to_connector(edge, end=1)   # connect target to its node
        return cmd1 + cmd2

    @classmethod
    def get_func_map(cls):
        return {0: cls.from_connectors,
                1: cls.from_points,
                2: cls.from_connector_point,
                3: cls.from_point_connector,
                4: cls.move_end_to_connector
                }


class Command(CommandFactory):
    @classmethod
    def cmd_tee_simple(cls, main_in, main_out, tee_out, index=[1, 0, 0], **kwargs):
        """
           /|\
            |
        --->+----->
        Returns:
        ----------
        command index, node
        [ CmdId, node.id, main_edge_in1.id, 1, main_edge_out2.id, 0 tee_edge.id, 0 ]
        """
        node = [main_in.source, main_in.target][index[0]]
        return [node.id, main_in.id, index[0], main_out.id, index[1], tee_out.id, index[2]]

    @classmethod
    def cmd_tee_branch_out(cls, tee_in, main_out1, main_out2, **kwargs):
        """
            |
           \|/
        <---+----->
        :param tee_in:
        :param main_out1:
        :param main_out2:
        :return:
        """
        return cls.cmd_tee_simple(main_out1, main_out2, tee_in, index=[0, 0, 1])

    @classmethod
    def delete(cls, node, **kwargs):
        return [[Cmds.Delete.value, node.id]]

    @classmethod
    def cmd_family_on_face(cls, item, index, family_index, **kwargs):
        """
        [ CmdId, edge_or_tee, index_of_connector ]
        """
        return [[Cmds.FamilyOnFace.value, item.id, index, family_index]]

    @classmethod
    def cmd_connect(cls, src, tgt, **kwargs):
        return [[Cmds.Connect.value, src.id, tgt.id]]

    @classmethod
    def connectn(cls, *graph_objs):
        return list(map(lambda x: x.id, graph_objs))

    # Inteface ------------------------------------
    @classmethod
    def cmd_pipe(cls, cmd, *data, **kwargs):
        return CurvePlacement.create(cmd, *data, **kwargs)

    @classmethod
    def cmd_coupling(cls, src, tgt, **kwargs):
        return [[Cmds.Coupling.value, src.id, 1, tgt.id, 0]]

    @classmethod
    def cmd_elbow(cls, src, tgt, **kwargs):
        return [[Cmds.Elbow.value, src.id, 1, tgt.id, 0]]

    # Family Placement

    @classmethod
    def _directed_combos(cls, fam_mat, line1, line2):
        a11 = geo.angle_between(fam_mat[1], line1.direction)
        a12 = geo.angle_between(fam_mat[1], line2.direction)
        a21 = geo.angle_between(fam_mat[2], line1.direction)
        a22 = geo.angle_between(fam_mat[2], line2.direction)
        return a11, a12, a21, a22

    @classmethod
    def _directs(cls, va, tgt_cross, line1, line2, angle):
        eul = tb.axangle2euler(tgt_cross, angle)
        mat = tb.euler2mat(*eul)
        va2 = np.dot(mat, va.T).T

        a11, a12, a21, a22 = cls._directed_combos(va2, line1, line2)

        t1 = np.array([a11 / line1.length, a22 / line2.length]).sum()
        t2 = np.array([a12 / line1.length, a21 / line2.length]).sum()
        return min([t1, t2])

    @classmethod
    def family_directions(cls, family_name):
        mp = {"Elbow - Generic": [np.array([-1, 0, 0]), np.array([0, 1, 0])],
              "Tee - Generic": [np.array([-1, 0, 0]), np.array([0, -1, 0])],
              "Coupling - Generic": [np.array([-1, 0, 0]), np.array([1, 0, 0])]
            }
        return mp.get(family_name, (None, None))

    @classmethod
    def _create_geometry_rot2(cls, line1, line2, family=None, use_angle=False, **kw):
        """

        :param line1:
        :param line2:
        :param family:
        :param use_angle:
        :return:
        """
        own_angle = line1.angle_to(line2)
        famx, famy = cls.family_directions(family)
        famz = np.array([0, 0, 1])
        if use_angle is True:
            famy = np.array([np.sin(own_angle), np.cos(own_angle), 0])

        all_axes = np.stack([famz, famx, famy]).T

        # cross product of lines to get their plane normal
        tgt_cross = np.cross(line1.unit_vector, line2.unit_vector)

        # axis - angle to align normals
        M = geo.rotation_matrix(famz, tgt_cross)
        zrot, yrot, xrot = tb.mat2euler(M)
        axa = tb.euler2axangle(zrot, yrot, xrot)

        # create euler matrix for first transformation
        m1, m2, m3 = tb.euler2mat(zrot, 0, 0), tb.euler2mat(0, yrot, 0), tb.euler2mat(0, 0, xrot)
        A = geombase.euler_composition(m1, m2, m3)

        # create euler matrix for first transformation
        family_transformed1 = np.dot(A, all_axes).T

        # now the normal planes are aligned, but need to get 2nd rotation
        # which occurs around target plane's normal
        a11, a12, a21, a22 = cls._directed_combos(family_transformed1, line1, line2)
        tests = [a11, -a11, a12, -a12, a21, -a21, a22, -a22]

        final_angle, best = 0, 1e6
        while tests:
            t = tests.pop()
            res = cls._directs(family_transformed1, tgt_cross, line1, line2, t)
            if res < best:
                best, final_angle = res, t

        if np.dot(line1.unit_vector, line2.unit_vector) > 0:
            own_angle = math.pi - own_angle

        tsf1 = axa[0].tolist() + [axa[1]]
        tsf2 = tgt_cross.tolist() + [final_angle, own_angle]
        return line1.numpy[0].tolist() + tsf1 + tsf2

    @classmethod
    def _create_geometry_1rot(cls, line1, family_axis):
        """

        :param line1: axis of
        :param family_axis: unit vector of direction to align to
        :return: list(7)  of [ placement_x, placement_y, placement_z, vec1, vec2, vec3, rotation_angle ]
        """
        M = geo.rotation_matrix(family_axis, line1.unit_vector)
        zrot, yrot, xrot = tb.mat2euler(M)
        axis, angle = tb.euler2axangle(zrot, yrot, xrot)
        place_point = line1.numpy[0].tolist()
        return place_point + axis.tolist() + [angle]

    @classmethod
    def place_family_point_angle(cls, node_in, node, node_up, family=None, **kwargs):
        """
        families default to placement at 0,0,0 with base connector at (-1,0,0)
        need to rotate the instance about an axis

        FamilyName, node.id, Location(3), direction(3), angle(1)

        """
        line1 = geo.Line(geo.Point(node.as_np), geo.Point(node_in.as_np))
        line2 = geo.Line(geo.Point(node.as_np), geo.Point(node_up.as_np))
        xforms = cls._create_geometry_rot2(line1, line2, family=family, **kwargs)
        t_edge = gutil.edge_between(node_in, node)
        radius = CurvePlacement.radius_from_edge(t_edge) * 0.5
        return [[Cmds.FamilyOnPoint.value, family,  node.id, 2] + xforms + [radius]]

    @classmethod
    def family_point_angle(cls, node_in, node, node_up, family=None, **kwargs):
        """
        families default to placement at 0,0,0 with base connector at (-1,0,0)
        need to rotate the instance about an axis

        FamilyName, node.id, Location(3), direction(3), angle(1)

        """
        line1 = geo.Line(geo.Point(node.as_np), geo.Point(node_in.as_np))
        line2 = geo.Line(geo.Point(node.as_np), geo.Point(node_up.as_np))
        xforms = cls._create_geometry_rot2(line1, line2, family=family, **kwargs)
        t_edge = gutil.edge_between(node_in, node)
        radius = CurvePlacement.radius_from_edge(t_edge) * 0.5
        return xforms + [radius]

    @classmethod
    def get_func_map(cls):
        return {1: cls.cmd_pipe,
                2: cls.cmd_connect,
                3: cls.delete,
                4: cls.cmd_elbow,
                5: cls.cmd_tee_simple,
                6: cls.cmd_family_on_face,
                7: cls.cmd_tee_branch_out,
                8: cls.cmd_coupling,
                9: None,
                10: None,
                12: cls.place_family_point_angle,
                11: CurvePlacement.move_end_to_point,
                14: cls.place_family_point_angle,
            }


# build Strategies ----------------------------------------------
class Pipe(rvb.ICommandManager):
    def __init__(self, graph_obj, strategy=None, **kwargs):
        super(Pipe, self).__init__(graph_obj, **kwargs)
        self._strategies = [self.ConnectorPoint, self.FromPoints]
        self._init_strategy(strategy, **kwargs)

    # Templates ---------------------------------------------------
    class PipeBase(rvb.IStartegy):
        base_val = Cmds.Pipe.value
        sub_cmd = -1

        @property
        def cmd_base(self):
            return [self.base_val, self.obj.id, self.sub_cmd]

    # Create Cmds : Connectors, Points, Point Connector ------------
    class FromPoints(PipeBase):
        sub_cmd = PipeCmd.Points.value

        def action(self, edge, **kwargs):
            geom = CurvePlacement.from_2points(edge, shrink=True)
            yield [self.cmd_base + geom]

        def success(self, edge, action):
            edge.write(built=True)

    class ConnectorPoint(PipeBase):
        sub_cmd = PipeCmd.ConnectorPoint.value

        def action(self, edge, **kwargs):
            start, end = edge.source, edge.target
            geom = CurvePlacement.from_2points(edge, shrink=True)
            yield [self.cmd_base + [start.id] + geom[3:]]

        def success(self, edge, action):
            edge.write(built=True, conn1=True)

        def fail(self, edge, action, msg):
            """
            - rebuild the pipe from points.
            - connect the start of the edge to source connector
            """
            yield Pipe.FromPoints(self.parent)
            yield Pipe.ConnectStartToNode(self.parent)

    class Connectors(PipeBase):
        sub_cmd = PipeCmd.Connectors.value

        def action(self, edge, **kwargs):
            yield [self.cmd_base + [edge.source.id, edge.target.id]]

        def success(self, edge, action):
            edge.write(built=True, conn1=True, conn2=True)

        def fail(self, edge, action, msg):
            yield Pipe.FromPoints(self.parent)

    # Updates Connect start or end ------------------------------
    class ConnectBase(rvb.IStartegy):
        """
        revit connectors are indexed to 1
        connect [ edge, connector at end -
        """
        base_val = Cmds.MoveEnd.value

        @property
        def cmd_base(self):
            return [self.base_val, self.obj.id]

    class ConnectEndToNode(ConnectBase):
        def action(self, edge, **kwargs):
            yield [self.cmd_base + [edge.id, 1, edge.target.id]]

        def success(self, edge, action):
            edge.write(conn2=True)

    class ConnectStartToNode(ConnectBase):
        def action(self, edge, **kwargs):
            yield [self.cmd_base + [edge.id, 0, edge.source.id]]

        def success(self, edge, action):
            edge.write(conn1=True)


# Template strategies -------------------------------------------------
class _GeomPlacementBase(rvb.IStartegy):
    """
    for Placing a geometry without reference to connectors
    """
    def success(self, node, action):
        """
            -record node was built
            -yield instructions to create connections to neighbor edges
        """
        node.write(built=True)
        for edge in node.successors(edges=True):
            yield Pipe.on(edge, Pipe.ConnectorPoint)
        for edge in node.predecessors(edges=True):
            yield Pipe.on(edge, Pipe.ConnectEndToNode)

    def fail(self, node, action, msg):
        """ if building a node fails, still generate the
            builders for successor edges
        """
        node.write(built=False)
        for edge in node.successors(edges=True):
            yield Pipe.on(edge, Pipe.FromPoints)


class _FromConnectorsBase(rvb.IStartegy):
    """
    Template for Creating from connectors
    Assumes that successor edges have been built and not connected
    """
    def _record(self, node, true_or_false):
        """
        """
        node.write(built=true_or_false)
        for edge in node.successors(edges=True):
            edge.write(conn1=true_or_false)
        for edge in node.predecessors(edges=True):
            edge.write(conn2=true_or_false)

    def success(self, node, action):
        """
        if successful,
            - node is created
            - predecessor edge conn2 is connected
            - successor edge conn1 is connected
        """
        self._record(node, True)

    def fail(self, node, action, msg):
        """
        if the node cannot be built from connectors,
        successor edges are already there so not much more to do
        """
        self._record(node, False)


# Concrete Builders ----------------------------------------------
class Elbow(rvb.ICommandManager):
    """
    Generally it works to create elbows from two connectors

    -most fail cases are from acute angles

    """
    def __init__(self, parent, strategy=None, **kwargs):
        super(Elbow, self).__init__(parent, **kwargs)
        self._strategies = [self.FromConnectors, self.FromGeometryPlacement]
        self._init_strategy(strategy, **kwargs)

    @staticmethod
    def elbow_neigh(node, **kwargs):
        pred = node.predecessors(ix=0, **kwargs)
        succ = node.successors(ix=0, **kwargs)
        return pred, succ

    class ElbowBase(rvb.IStartegy):
        fam = 'Elbow - Generic'
        base_val = Cmds.Elbow.value

    class FromConnectors(ElbowBase):
        def action(self, node, **kwargs):
            """
            Main strategy
                1. place target Edge,
                2. connect pred and succ with elbow
            """
            yield Pipe.on(node.successors(ix=0, edges=True), Pipe.FromPoints)
            yield Elbow.ConnectPipeTo(self.parent)

        def success(self, node, action):
            node.write(built=True)

    class ConnectPipeTo(ElbowBase, _FromConnectorsBase):
        def action(self, node, **kw):
            pred, succ = Elbow.elbow_neigh(node, edges=True)
            yield [self.cmd_base + [pred.id, 1, succ.id, 0]]

        def success(self, node, action):
            _FromConnectorsBase.success(self, node, action)

        def fail(self, node, action, msg):
            yield Elbow.FromGeometryPlacement(self.parent)

    class FromGeometryPlacement(ElbowBase):
        base_val = Cmds.FamilyOnPoint.value

        def action(self, node, **kwargs):
            n1, n2 = Elbow.elbow_neigh(node)
            xform = Command.family_point_angle(n1, node, n2, family=self.fam, **kwargs)
            yield [self.cmd_base + [self.fam] + xform]

        def success(self, node, action):
            pred, succ = Elbow.elbow_neigh(node, edges=True)
            yield Pipe.on(pred, Pipe.ConnectEndToNode)
            yield Pipe.on(succ, Pipe.ConnectStartToNode)


class Coupling(rvb.ICommandManager):
    def __init__(self, parent, strategy=None, **kwargs):
        super(Coupling, self).__init__(parent, **kwargs)
        self._strategies = [ self.FromConnectors ]
        self._init_strategy(strategy, **kwargs)

    class CouplingBase(rvb.IStartegy):
        fam = 'Coupling - Generic'
        base_val = Cmds.Coupling.value

    class FromConnectors(CouplingBase):
        def action(self, node, **kwargs):
            """
            default strategy:
                1. place target Edge,
                2. connect pred and succ with a coupling
            """
            yield Pipe.on(node.successors(ix=0, edges=True), Pipe.FromPoints)
            yield Coupling.ConnectPipeTo(self.parent)

        def success(self, node, *args):
            node.write(built=True)

        def fail(self, node, action, msg):
            yield Coupling.FromGeometryPlacement(self.parent)

    class ConnectPipeTo(CouplingBase, _FromConnectorsBase):
        def action(self, node, **kw):
            pred, succ = Elbow.elbow_neigh(node, edges=True)
            yield [self.cmd_base + [pred.id, 1, succ.id, 0]]

        def success(self, node, action):
            _FromConnectorsBase.success(self, node, action)

    class FromGeometryPlacement(CouplingBase):
        """
        todo - this must be face based -
        """
        base_val = Cmds.FamilyOnPoint.value

        def action(self, node, **kwargs):
            in_edge, _ = Elbow.elbow_neigh(node, edges=True)
            line = in_edge.curve.line
            xform = Command._create_geometry_1rot(line, np.array([-1., 0, 0]))
            yield [self.cmd_base + [self.fam] + xform]

        def success(self, node, **kwargs):
            pred, succ = Elbow.elbow_neigh(node, edges=True)
            yield Pipe.on(pred, Pipe.ConnectEndToNode)
            yield Pipe.on(succ, Pipe.ConnectStartToNode)


class ITee(rvb.ICommandManager):
    def __init__(self, parent, strategy=None, **kwargs):
        rvb.ICommandManager.__init__(self, parent,  **kwargs)
        self._strategies = [self.FromGeometryPlacement]
        self._init_strategy(strategy, **kwargs)

    @staticmethod
    def tee_nodes(node):
        # assert node.get('is_tee') is True
        preds = node.predecessors(edges=True)
        succs = node.successors(edges=True)
        pred = preds[0] if node.npred == 1 else None

        if node.nsucs == 2 and node.npred == 1:
            suc_edge1, suc_edge2 = succs

            if suc_edge2.get('tap_edge', None) is True:
                return pred.source, suc_edge2.target, suc_edge2

            elif suc_edge1.get('tap_edge', None) is True:
                return pred.source, suc_edge1.target, suc_edge1

            elif pred.get('tap_edge', None) is True:
                return suc_edge1.source, pred.target, pred

        elif node.nsucs == 2 and node.npred == 0:
            suc_edge1, suc_edge2 = succs
            if suc_edge2.get('tap_edge', None) is True:
                return suc_edge1.target, suc_edge2.target, suc_edge2

            elif suc_edge1.get('tap_edge', None) is True:
                return suc_edge2.target, suc_edge1.target, suc_edge1

        elif node.nsucs == 1 and node.npred == 1:
            suc_edge = succs[0]
            if suc_edge.get('tap_edge', None) is True:
                return pred.source, suc_edge.target, None

        return None, None, None

    @staticmethod
    def tee_edges(node):
        preds = node.predecessors(edges=True)
        succs = node.successors(edges=True)
        pred = preds[0] if node.npred == 1 else None
        if node.nsucs == 2 and node.npred == 1:
            suc_edge1, suc_edge2 = succs

            if suc_edge2.get('tap_edge', None) is True:
                return pred, suc_edge1, suc_edge2

            elif suc_edge1.get('tap_edge', None) is True:
                return pred, suc_edge2, suc_edge1

            elif pred.get('tap_edge', None) is True:
                return suc_edge1, suc_edge2, pred

        elif node.nsucs == 2 and node.npred == 0:
            suc_edge1, suc_edge2 = succs
            if suc_edge2.get('tap_edge', None) is True:
                return suc_edge1, suc_edge1, suc_edge2

            elif suc_edge1.get('tap_edge', None) is True:
                return suc_edge2, suc_edge2, suc_edge1
        return None, None, None

    class TeeBase(_GeomPlacementBase):
        fam = 'Tee - Generic'
        base_val = Cmds.Tee.value

        @property
        def cmd_base(self):
            return [self.base_val, self.obj.id, self.fam]

    class FromGeometryPlacement(TeeBase):
        base_val = Cmds.FamilyOnPoint.value

        def action(self, node, **kwargs):
            n1, n2, _ = ITee.tee_nodes(node)
            yield [self.cmd_base + Command.family_point_angle(n1, node, n2, family=self.fam, **kwargs)]

    class FromConnectors(rvb.IStartegy):
        def action(self, node, **kwargs):
            n1, n2, tee_n = ITee.tee_nodes(node)
            _, out1, out2 = ITee.tee_edges(node)

            # create the target edges.
            yield Pipe.on(out1, Pipe.FromPoints)
            yield Pipe.on(out2, Pipe.FromPoints)
            yield [self.cmd_base[0:2] + Command.connectn(n1, n2, tee_n)]


class Skip(rvb.ICommandManager):
    def __init__(self, parent, strategy=None, **kwargs):
        rvb.ICommandManager.__init__(self, parent,  **kwargs)
        self._strategies = [self.SkipOp]
        self._init_strategy(strategy, **kwargs)

    class SkipOp(_GeomPlacementBase):
        def action(self, node, **kwargs):
            yield [Cmds.Noop.value, -1, -1 ]

        def success(self, node, action):
            p = node.predecessors(ix=0, edges=True)
            p.write(conn2=True, skipped=True)
            node.write(built=True, skipped=True)

        def fail(self, node, action, msg):
            node.write(built=False, skipped=True)


class IFamily(rvb.ICommandManager):
    """ """
    def __init__(self,
                 parent,
                 strategy=None,
                 fam='Coupling Generic',
                 axis=np.array([1., 0., 0.])):
        super(IFamily, self).__init__(parent)
        self._strategies = [self.FromFace ]
        self.fam = fam
        self.axis = axis
        self._init_strategy(strategy)

    class FamBase(_GeomPlacementBase):
        base_val = Cmds.FamilyOnFace.value

    class FromPrevGeometryPlacement(FamBase):
        def action(self, node, **kwargs):
            edge = node.predecessors(ix=0, edges=True)
            geom = Command._create_geometry_1rot(edge.curve.line, self.parent.axis)
            yield [self.cmd_base + geom]

        def success(self, node, action):
            yield [Cmds.Connect, node.id, node.id, node.predecessors(ix=0, edges=True).id]

        def fail(self, node, action, msg):
            if node.nsucs == 0:
                yield IFamily.FromFace(node)
            else:
                yield IFamily.FromSuccGeometryPlacement(node)

    class FromSuccGeometryPlacement(FamBase):
        def action(self, node, **kwargs):
            axis = -1 * self.parent.axis
            edge = node.successors(ix=0, edges=True)
            geom = Command._create_geometry_1rot(edge.curve.line, axis)
            yield [self.cmd_base + geom]

        def fail(self, node, action, msg):
            yield IFamily.FromFace(node)

    class FromFace(FamBase):
        def action(self, node, **kwargs):
            edge = node.predecessors(ix=0, edges=True)
            yield [self.cmd_base + [edge.id, 1, 0]]

        def success(self, node, action):
            node.write(built=True)
            node.predecessors(ix=0, edges=True).write(conn2=True)

        def fail(self, node, action, msg):
            node.write(built=False)


_cmd_cls = {'tee': ITee,
            'family': IFamily,
            'coupling': Coupling,
            '$head':IFamily,
            'pipe': Pipe,
            'elbow': Elbow,
            }


def make_actions_for(node):
    command_creator = _cmd_cls.get(node.get('$create', None), None)
    if command_creator is not None:
        return command_creator.action(node)
    return Skip.action(node)



