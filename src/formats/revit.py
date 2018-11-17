
"""

Revit utilities
"""
import json
import os
from enum import Enum
import lib.geo as geo
import src.geom
_TOLERANCE = 128 # 1/128 inch


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
    FamilyOnPoint = 14
    PipeBetween = 15



class StandardFamily(Enum):
    Elbow = 1
    Tee = 2
    Tap = 3
    Union = 4
    Transition = 5



class Command(object):
    @classmethod
    def cmd_pipe(cls, edge, shrink=None, use_radius=False, **kwargs):
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
            vec.append(2 * 1/12)
        return [vec]

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
        return [node.id]

    @classmethod
    def cmd_family_on_face(cls, item, index, family_index, **kwargs):
        """
        [ CmdId, edge_or_tee, index_of_connector ]
        """
        return [item.id, index, family_index]

    @classmethod
    def _cmd_connect(cls, src, tgt, **kwargs):
        return [src.id, 1, tgt.id, 0]

    # Interactive ------------------------------------
    @classmethod
    def cmd_move_pipe_end(cls, edge, end, xyz, **kwargs):
        return [edge.id, end] + xyz

    @classmethod
    def place_family_point_angle(cls, node, **kwargs):
        """

        FamilyName, node.id, Location(3), direction(3), angle(1)

        """
        if node.npred > 0:
            pred = node.predecessors(edges=True)[0]
        else:
            pred = node.successors(edges=True)[0]
        t = pred.curve.line.direction
        base = geo.Line(geo.Point(0, 0, 0), geo.Point(-1, 0, 0))
        ldir = geo.Line(geo.Point(0, 0, 0), geo.Point(t))
        angle = ldir.angle_to(base)
        family_name = node.get("$create")
        return [family_name, node.id] + list(node.geom) + angle

    @classmethod
    def pipe_between(cls, edge, **kwargs):
        return [edge.id, edge.source.id, edge.target.id, edge.get('radius', 1/12)]

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
                11: cls.cmd_move_pipe_end,
                14: cls.place_family_point_angle,
                15: cls.pipe_between,
        }

    @classmethod
    def create(cls, cmd, *data, **kwargs):
        fn = cls.get_func_map().get(cmd.value, None)
        if fn is not None:
            return [[cmd.value] + fn(*data, **kwargs)]

