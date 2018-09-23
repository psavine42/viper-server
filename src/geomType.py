"""
these definitions should be shared between client and server

"""

from enum import Enum


class FPSymbols(Enum):
    UpHead = 0
    DownHead = 1
    HorzHead = 2


class GeomType(Enum):
    NONE = 0
    ARC = 1
    CIRCLE = 2
    LINE = 3
    POLYLINE = 4
    SYMBOL = 5
    SOLID = 6
    FACE = 7
    MESH = 8


class SystemType(Enum):
    FireProtection = 0
    Plumbing = 1
    HotWater = 2
    ColdWater = 3
    HVAC = 4






