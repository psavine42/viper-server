
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


class PipeCmd(Enum):
    Connectors = 0
    Points = 1
    ConnectorPoint = 2


class StandardFamily(object):
    name = None
    basis_x, basis_y, basis_z = None, None, None


class CommandFactory(object):
    @classmethod
    def get_func_map(cls):
        return {}

    @classmethod
    def create(cls, cmd, *data, **kwargs):
        fn = cls.get_func_map().get(cmd.value, None)
        if fn is not None:
            return fn(*data, **kwargs)


class CurvePlacement(CommandFactory):
    @classmethod
    def from_connectors(cls, edge, **kwargs):
        return [[Cmds.Pipe.value, edge.id, edge.source.id, edge.target.id, edge.get('radius', 1 / 12)]]

    @classmethod
    def from_points(cls, edge, shrink=None, use_radius=False, **kwargs):
        """
        format for pipe:
        [ CmdId, edge.id, x1, y1, z1, x2, y2, z2, radius ]
        """

        is_last = edge.target.nsucs == 0
        start, end = edge.source, edge.target
        crv = src.geom.MepCurve2d(start.geom, end.geom)
        if shrink is not None:
            if is_last is False:
                p1, p2 = crv.extend(shrink, shrink).points
            else:
                p1, p2 = crv.extend(shrink, 0.).points
        else:
            p1, p2 = crv.points

        vec = [Cmds.Pipe.value, edge.id] + list(p1) + list(p2)
        if use_radius is True and edge.get('radius') is not None:
            vec.append(edge.get('radius') * 2)
        else:
            vec.append(2 * 1 / 12)
        return [vec]

    @classmethod
    def from_point_connector(cls, edge, use_radius=False, **kwargs):
        start, end = edge.source, edge.target
        vec = [Cmds.Pipe.value, edge.id, start.id] + list(end.geom)
        if use_radius is True and edge.get('radius') is not None:
            vec.append(edge.get('radius') * 2)
        else:
            vec.append(1 / 12)
        return [vec]

    @classmethod
    def get_func_map(cls):
        return {0: cls.from_connectors,
                1: cls.from_points,
                2: cls.from_point_connector,
                }


class Command(CommandFactory):
    @classmethod
    def cmd_pipe(cls, cmd, *data, **kwargs):
        return CurvePlacement.create(cmd, *data, **kwargs)

    @classmethod
    def _cmd_indexed(cls, cmd_type, *elems, index=None):
        if index is None:
            index = [1] + [0] * (len(elems) - 1)
        return [[cmd_type.value] + list(map(lambda x: x.id, elems)) + index]

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
    def _cmd_id(cls, node, **kwargs):
        "generic command just getting the ID - used for DELETE"
        return [node.id]

    @classmethod
    def _cmd_id_index(cls, node, index, **kwargs):
        return [node.id, index]

    @classmethod
    def cmd_family_on_face(cls, item, index, family_index, **kwargs):
        """
        [ CmdId, edge_or_tee, index_of_connector ]
        """
        return [[Cmds.FamilyOnFace.value, item.id, index, family_index]]

    @classmethod
    def _cmd_connect(cls, src, tgt, **kwargs):
        return cls._cmd_id_index(src, 1) + cls._cmd_id_index(tgt, 0)

    # Interactive ------------------------------------
    @classmethod
    def cmd_move_pipe_end(cls, edge, end_index, xyz, **kwargs):
        return cls._cmd_id_index(edge, end_index) + xyz

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
    def place_family_point_angle(cls, node_in, node, node_up, family=None, use_angle=False, **kwargs):
        """
        families default to placement at 0,0,0 with base connector at (-1,0,0)
        need to rotate the instance about an axis

        FamilyName, node.id, Location(3), direction(3), angle(1)

        """
        node_geom = node.as_np
        line1 = geo.Line(geo.Point(node_geom), geo.Point(node_in.as_np))
        line2 = geo.Line(geo.Point(node_geom), geo.Point(node_up.as_np))
        own_angle = line1.angle_to(line2)

        famx, famy = cls.family_directions(family)
        famz = np.array([0, 0, 1])
        if use_angle is True:
            famy = np.array([np.sin(own_angle), np.cos(own_angle), 0  ])

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
        final_angle = 0
        tests = [a11, -a11, a12, -a12, a21, -a21, a22, -a22]
        best = 1e6
        while tests:
            t = tests.pop()
            res = cls._directs(family_transformed1, tgt_cross, line1, line2, t)
            if res < best:
                best, final_angle = res, t

        # angle on plane
        line_norm = geo.Line(geo.Point(node_geom), geo.Point(node_geom + tgt_cross))
        plane_norm = line_norm.dual
        own_angle = line1.projected_on(plane_norm).angle_to(line2.projected_on(plane_norm))

        # radius
        edge = gutil.edge_between(node_in, node)
        if edge is not None and edge.get('radius', None) is not None:
            radius = edge.get('radius', None)
        else:
            radius = 1/24

        tsf1 = axa[0].tolist() + [axa[1]]
        tsf2 = tgt_cross.tolist() + [final_angle]
        own_prop = [own_angle, radius]

        return [[Cmds.FamilyOnPoint.value, family, node.id, 2]
                + list(node_geom) + tsf1 + tsf2 + own_prop]

    @classmethod
    def get_func_map(cls):
        return {1: cls.cmd_pipe,
                2: cls._cmd_connect,
                3: cls._cmd_id,
                4: cls._cmd_connect,
                5: cls.cmd_tee_simple,
                6: cls.cmd_family_on_face,
                7: cls.cmd_tee_branch_out,
                8: cls._cmd_connect,
                9: None,
                10: None,
                12: cls.place_family_point_angle,
                11: cls.cmd_move_pipe_end,
                14: cls.place_family_point_angle,
        }



