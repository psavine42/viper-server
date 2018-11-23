
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
    if graph_obj.get('$built', None) is True:
        return True
    return False


def is_complete(graph_obj):
    if isinstance(graph_obj, Node) and graph_obj.get('built', None) is True:
        return True
    elif isinstance(graph_obj, Edge) \
            and graph_obj.get('built', None) is True \
            and graph_obj.get('conn1', None) is True \
            and graph_obj.get('conn2', None) is True:
        return True
    return False


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
    def from_points(cls, edge, shrink=None, use_radius=True, **kwargs):
        """
        format for pipe:
        [ CmdId, edge.id, x1, y1, z1, x2, y2, z2, radius ]
        """
        start, end = edge.source, edge.target
        crv = src.geom.MepCurve2d(start.geom, end.geom)
        r = cls.radius_from_edge(edge, use_radius)
        if shrink is not None:
            s1 = -r / 2 if edge.source.npred == 0 is False else 0.
            s2 = -r / 2 if edge.target.nsucs == 0 is False else 0.
            p1, p2 = crv.extend(s1, s2).points
        else:
            p1, p2 = crv.points

        cdef = [Cmds.Pipe.value, CmdType.Create.value, edge.id]
        return [cdef + [edge.id] + list(p1) + list(p2) + [r]]

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
    @staticmethod
    def _directs(va, tgt_cross, line1, line2, angle):
        eul = tb.axangle2euler(tgt_cross, angle)
        mat = tb.euler2mat(*eul)
        va2 = np.dot(mat, va.T).T

        a11 = geo.angle_between(va2[1], line1.direction)
        a12 = geo.angle_between(va2[1], line2.direction)
        a21 = geo.angle_between(va2[2], line1.direction)
        a22 = geo.angle_between(va2[2], line2.direction)

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
    def _create_geometry_rot(cls, line1, line2, family=None, use_angle=False, **kw):
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
        a11 = geo.angle_between(family_transformed1[1], line1.direction)
        a12 = geo.angle_between(family_transformed1[1], line2.direction)
        a21 = geo.angle_between(family_transformed1[2], line1.direction)
        a22 = geo.angle_between(family_transformed1[2], line2.direction)
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
    def place_family_point_angle(cls, node_in, node, node_up, family=None, **kwargs):
        """
        families default to placement at 0,0,0 with base connector at (-1,0,0)
        need to rotate the instance about an axis

        FamilyName, node.id, Location(3), direction(3), angle(1)

        """
        line1 = geo.Line(geo.Point(node.as_np), geo.Point(node_in.as_np))
        line2 = geo.Line(geo.Point(node.as_np), geo.Point(node_up.as_np))

        xforms = cls._create_geometry_rot(line1, line2, family=family, **kwargs)
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

        xforms = cls._create_geometry_rot(line1, line2, family=family, **kwargs)
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



    @classmethod
    def action(cls, cmd, *data, **kwargs):
        klass = cls.get_cls_map().get(cmd.__class__, None)
        if klass:
            return klass.create(cmd, *data, **kwargs)


class IStartegy(object):
    def __init__(self):
        self._actions = []

    def make_action(self, node, action):
        raise NotImplemented('concrete action not implemented')

    def on_success(self, node, action):
        raise NotImplemented('concrete action not implemented')

    def on_fail(self, node, action):
        """ base class return no further strategies
            superclasses will return
        """
        return None


class ICommand(object):
    """
    Granular control over generating actions


    """
    __def_in_node = '__creator'

    def __init__(self):
        self._commands = []
        self.strategy = None

    @property
    def strategies(self):
        return []

    def _action(self, graph_obj, strategy=None, **kwargs):
        """

        :param graph_obj:
        :param strategy:
        :param kwargs:
        :return:
        """
        if strategy is not None:
            if strategy in self.strategies:
                ix = self.strategies.index(strategy)
                self.strategy = self.strategies.pop(ix)()
            else:
                print(self.__class__.__name__, 'does not have ', strategy.__name__)
        elif len(self.strategies) > 0:
            self.strategy = self.strategies.pop(0)()

        if self.strategy is not None:
            graph_obj, cmds = self.strategy.make_action(graph_obj)
            return graph_obj, cmds
        return graph_obj, []

    def on_success(self, node, action):
        return self.strategy.on_success(node, action)

    def on_fail(self, graph_obj, action):
        """
        if there is a failure, retrieve next strategy and initialize
        :param graph_obj:
        :param action:
        :return:
        """
        new_strategy = self.strategy.on_fail(graph_obj, action)
        if new_strategy is None:
            return graph_obj, []
        self.strategy = new_strategy()
        return self.strategy.make_action(graph_obj)

    @classmethod
    def action(cls, graph_obj, **kwargs):
        """[create_successors, create_node, connect_predecessors]"""
        command_creator = graph_obj.get(cls.__def_in_node, None)
        if command_creator is None:
            command_creator = cls()
            graph_obj.write(cls.__def_in_node, command_creator)
        graph_obj, cmds = command_creator._action(graph_obj, **kwargs)
        command_creator._commands = cmds
        return graph_obj, cmds

    @classmethod
    def success(cls, graph_obj, action):
        """

        
        :param graph_obj:
        :param action:
        :return:

        ICommand.success(node, [[11, 50, 20]])
        """
        command_creator = graph_obj.get(cls.__def_in_node, None)
        if command_creator is None:
            print('missing __creator in ', graph_obj.id)
            return graph_obj
        graph_obj, cmds = command_creator.on_success(graph_obj, action)
        return graph_obj, cmds

    @classmethod
    def fail(cls, graph_obj, action):

        command_creator = graph_obj.get(cls.__def_in_node, None)
        if command_creator is None:
            print('missing __creator in ', graph_obj.id)
            return graph_obj
        graph_obj, cmds = command_creator.on_fail(graph_obj, action)
        return graph_obj, cmds


class IPipe(ICommand):
    class PipeBase(object):
        val = Cmds.Pipe.value

    class FromPoints(PipeBase):
        def make_action(self, edge, **kwargs):
            return CurvePlacement.from_points(edge, **kwargs)

        def on_success(self, edge, action):
            edge.write(built=True)
            return edge

    class ConnectorPoint(PipeBase):
        def make_action(self, edge, **kwargs):
            start, end = edge.source, edge.target
            r = CurvePlacement.radius_from_edge(edge, True)
            return [[Cmds.Pipe.value, edge.id, start.id] + list(end.geom) + [r]]

        def on_success(self, edge, action):
            edge.write(built=True, conn1=True)
            return edge

        def on_fail(self, edge,  action):
            return IPipe.FromPoints


class IElbow(ICommand):
    fam = 'Elbow - Generic'
    val = Cmds.Elbow.value
    # strategy = [0, 1]

    def __init__(self):
        super(ICommand, self).__init__()

    class ElbowBase(IStartegy):
        fam = 'Elbow - Generic'
        val = Cmds.Elbow.value

    class FromConnectors(ElbowBase):
        def make_action(self, node, **kwargs):
            src_edge = node.predecessors(ix=0, edges=True)
            tgt_edge = node.successors(ix=0, edges=True)

            # 1. place target Edge, 2. connect pred and succ with elbow
            tgt_edge, cmd1 = IPipe.action(tgt_edge)
            if not is_built(src_edge):
                return node, cmd1
            cmd2 = [[self.val] + Command.connectn(src_edge, tgt_edge)]
            return node, cmd1 + cmd2

        def on_success(self, node, action):
            if action[0] == Cmds.Pipe.value:
                tgt_edge = node.successors(ix=0, edges=True)
                tgt_edge.write(built=True, conn1=True)
            elif action[0] == self.val:
                node.write(built=True)
            return node, []

        def on_fail(self, node, action):
            return IElbow.FromGeometryPlacement

    class FromGeometryPlacement(ElbowBase):
        def make_action(self, node, **kwargs):
            n1 = node.predecessors(ix=0)
            n2 = node.successors(ix=0)
            cdef = [Cmds.FamilyOnPoint.value, n2.id, self.fam]
            return node, [cdef + Command.family_point_angle(n1, node, n2, family=self.fam, **kwargs)]

        def on_success(self, node, action):
            node.write(built=True, conn1=True)
            return node, []

    @property
    def strategies(self):
        return [self.FromConnectors,
                self.FromGeometryPlacement]


class ITee(ICommand):
    fam = 'Tee - Generic'
    val = Cmds.Tee.value

    def __init__(self):
        super(ICommand, self).__init__()

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

    class TeeBase(IStartegy):
        fam = 'Tee - Generic'
        val = Cmds.Tee.value

    class FromGeometryPlacement(TeeBase):
        def make_action(self, node, **kwargs):
            n1, n2, _ = ITee.tee_nodes(node)
            cdef = [Cmds.FamilyOnPoint.value, node.id, self.fam]
            return [cdef + Command.family_point_angle(n1, node, n2, family=self.fam, **kwargs)]

        def on_success(self, node, action):
            if action[0] == Cmds.Pipe.value:
                tgt_edge = node.successors(ix=0, edges=True)
                tgt_edge.write(built=True, conn1=True)
            elif action[0] == self.val:
                node.write(built=True)
            return node, []

        def on_fail(self, node, action):
            return ITee.FromConnectors

    class FromConnectors(TeeBase):
        def make_action(self, node, **kwargs):
            n1, n2, tee_n = ITee.tee_nodes(node)
            edgein, out1, out2 = ITee.tee_edges(node)

            cdef = [[Cmds.Tee.value, node.id] + Command.connectn(n1, n2, tee_n)]
            return

        def on_success(self, node, action):
            node.write(built=True, conn1=True)
            return node, []

    @property
    def strategies(self):
        return [self.FromGeometryPlacement ]


_cmd_cls = {'tee': ITee,
             # 'coupling': CouplingCmd,
            'pipe': IPipe,
            'elbow': IElbow}


def make_actions_for(node):
    command_creator = _cmd_cls.get(node.get('$create', None), None)
    if command_creator is not None:
        return command_creator.action(node)
    return node, []


# class Elbow(CommandFactory):
#     strategy = [0, 1]
#
#     @classmethod
#     def from_two_connectors(cls, node):
#         cmds = []
#
#         eprd = node.predecessors(ix=0, edges=True)
#         etgt = node.successors(ix=0, edges=True)
#
#         cdef = [Cmds.Elbow.value, CmdType.BuildFull.value, node.id]
#         return [cdef + Command.connectn(eprd, etgt)]
#
#     @classmethod
#     def from_location_geom(cls, node, **kwargs):
#         faml = 'Elbow - Generic'
#         n1 = node.predecessors(ix=0)
#         n2 = node.successors(ix=0)
#         cdef = [Cmds.FamilyOnPoint.value, CmdType.Create.value, n2.id, faml]
#         return [cdef + Command.family_point_angle(n1, node, n2, family=faml, **kwargs)]
#
#     @classmethod
#     def move_connect(self, node):
#         pass
#
#     @classmethod
#     def get_func_map(cls):
#         return {0: cls.from_two_connectors,
#                 1: cls.from_location_geom,
#                 2: cls.move_connect
#                 }


class Coupling(CommandFactory):
    strategy = [0]  # todo 2 - implement

    @classmethod
    def from_two_connectors(cls, node):
        eprd = node.predecessors(ix=0, edges=True)
        etgt = node.successors(ix=0, edges=True)
        cdef = [Cmds.Coupling.value, CmdType.BuildFull.value, node.id]
        return [cdef + Command.connectn(eprd, etgt)]

    @classmethod
    def from_location_geom(cls, node, **kwargs):
        return [[]]

    @classmethod
    def from_adjusted_pipes(cls):
        return [[]]

    @classmethod
    def get_func_map(cls):
        return \
            {
                0: cls.from_two_connectors,
                1: cls.from_location_geom,
                2: cls.from_adjusted_pipes,
            }


# class Tee(CommandFactory):
#     strategy = [1, 0]
#
#     @staticmethod
#     def tee_nodes(node):
#         # assert node.get('is_tee') is True
#         preds = node.predecessors(edges=True)
#         succs = node.successors(edges=True)
#         pred = preds[0] if node.npred == 1 else None
#
#         if node.nsucs == 2 and node.npred == 1:
#             suc_edge1, suc_edge2 = succs
#
#             if suc_edge2.get('tap_edge', None) is True:
#                 return pred.source, suc_edge2.target, suc_edge2
#
#             elif suc_edge1.get('tap_edge', None) is True:
#                 return pred.source, suc_edge1.target, suc_edge1
#
#             elif pred.get('tap_edge', None) is True:
#                 return suc_edge1.source, pred.target, pred
#
#         elif node.nsucs == 2 and node.npred == 0:
#             suc_edge1, suc_edge2 = succs
#             if suc_edge2.get('tap_edge', None) is True:
#                 return suc_edge1.target, suc_edge2.target, suc_edge2
#
#             elif suc_edge1.get('tap_edge', None) is True:
#                 return suc_edge2.target, suc_edge1.target, suc_edge1
#         return None, None, None
#
#     @staticmethod
#     def tee_edges(node):
#         preds = node.predecessors(edges=True)
#         succs = node.successors(edges=True)
#         pred = preds[0] if node.npred == 1 else None
#         if node.nsucs == 2 and node.npred == 1:
#             suc_edge1, suc_edge2 = succs
#
#             if suc_edge2.get('tap_edge', None) is True:
#                 return pred, suc_edge1, suc_edge2
#
#             elif suc_edge1.get('tap_edge', None) is True:
#                 return pred, suc_edge2, suc_edge1
#
#             elif pred.get('tap_edge', None) is True:
#                 return suc_edge1, suc_edge2, pred
#
#         elif node.nsucs == 2 and node.npred == 0:
#             suc_edge1, suc_edge2 = succs
#             if suc_edge2.get('tap_edge', None) is True:
#                 return suc_edge1, suc_edge1, suc_edge2
#
#             elif suc_edge1.get('tap_edge', None) is True:
#                 return suc_edge2, suc_edge2, suc_edge1
#         return None, None, None
#
#     @classmethod
#     def from_three_connectors(cls, node):
#         n1, n2, tee_n = cls.tee_nodes(node)
#         cdef = [Cmds.Tee.value, CmdType.BuildFull.value, node.id]
#         return [cdef + Command.connectn(n1, n2, tee_n)]
#
#     @classmethod
#     def from_location_geom(cls, node, **kwargs):
#         fam = 'Tee - Generic'
#         n1, n2, _ = cls.tee_nodes(node)
#         cdef = [Cmds.FamilyOnPoint.value, CmdType.Create.value, node.id, fam]
#         return [cdef + Command.family_point_angle(n1, node, n2, family=fam,  **kwargs)]
#
#     @classmethod
#     def from_adjusted_pipes(cls, node, **kwargs):
#         cmd1 = cls.from_location_geom(node, **kwargs)
#         # [create_successors, create_node, connect_predecessors]
#
#
#         return cmd1
#
#     @classmethod
#     def get_func_map(cls):
#         return {0: cls.from_three_connectors,
#                 1: cls.from_location_geom,
#                 2: cls.from_adjusted_pipes
#                 }

