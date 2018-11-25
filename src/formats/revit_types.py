from enum import Enum


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

