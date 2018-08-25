"""
convert inputs of various call formats to 'System' objects

"""
from viper import System, GeomType
from src.geom import MepCurve2d, FamilySymbol
from enum import Enum
from shapely.geometry import MultiLineString
from uuid import uuid4
from shapely.ops import linemerge
import numpy as np
from src.utils import round_tup

class SystemFactory:
    _ROUND = 6

    @classmethod
    def from_segs(cls, segments, sys=System, lr=None, **kwargs):
        segs = []
        for x in segments:
            p1, p2 = sorted([x[0],  x[1]])
            segs.append(MepCurve2d(p1, p2, layer=lr))
        return sys(segments=segs, **kwargs)

    @classmethod
    def from_request(cls, segments, **kwargs):
        segs, syms = [], []
        for x in segments:
            lr = x.get('layer', None)
            seg = cls.handle_segment(x['pts'], layer=lr)
            if seg:
                segs.append(seg)
        return System(segments=segs, **kwargs)

    @classmethod
    def handle_segment(cls, pts, **kwargs):
        x1, y1, z1, x2, y2, z2 = [round(p, cls._ROUND) for p in pts]
        if not (x1 == x2 and y1 == y2):
            p1, p2 = sorted([(x1, y1, 0), (x2, y2, 0)])
            return MepCurve2d(p1, p2, **kwargs)

    @classmethod
    def _get_child_points(cls, lr, dct):
        pts, lyr = [], []
        this_lr = dct.get('layer', None)
        if this_lr:
            lr.append(this_lr)
        for c in dct.get('children', []):
            pt = [round(p, cls._ROUND) for p in c.get('points', [])]
            _pts = [pt[x:x + 3] for x in range(0, len(pt), 3)]
            a, b = divmod(len(_pts), 3)
            pts += _pts[0:3*a]
            lr, _pts = cls._get_child_points(lr, c)
            pts += _pts
        return lr, pts

    @classmethod
    def handle_symbol_abstract(cls, dct):
        segs, syms = [], []
        lr, pts = cls._get_child_points([], dct)
        if pts:
            sym = FamilySymbol(*pts, layer=set(lr).pop(), type=GeomType(5), data=dct)
            syms.append(sym)
        return segs, syms

    @classmethod
    def handle_symbol_recursive(cls, dct):
        cnt = 0
        pts = np.zeros(3)
        lr = dct.get('layer', None)
        segs, syms = cls.to_segments(dct.get('children', []))
        for seg in segs:
            p1, p2 = seg.points_np()
            pts += p1[0] + p2[0]
            cnt += 2
        for sym in syms:
            p1 = sym.points_np()
            pts += p1[0]
            cnt += 1
        pts /= cnt
        sym = FamilySymbol(*pts.tolist(), type=GeomType(5), children=segs+syms, layer=lr)
        return [], [sym]

    @classmethod
    def to_segments(cls, segments):
        segs, syms = [], []
        for x in segments:
            lr = x.get('layer', None)
            pt = [round(p, cls._ROUND) for p in x.get('points', [])]
            gt = GeomType(x.get('geomType', 0))

            if gt in [GeomType['ARC'], GeomType['CIRCLE']]:
                syms.append(FamilySymbol(*pt, type=gt, layer=lr))

            elif gt == GeomType['LINE']:
                segs.append(cls.handle_segment(pt, layer=lr))

            elif gt == GeomType['POLYLINE']:
                opt = uuid4()
                xyzs = [pt[x:x + 3] for x in range(0, len(pt), 3)]
                for i in range(1, len(xyzs)):
                    segs.append(cls.handle_segment(xyzs[i-1] + xyzs[i], layer=lr, pl=opt))

            elif gt == GeomType['SYMBOL']:
                _sgs, _sms = cls.handle_symbol_recursive(x)
                segs += _sgs
                syms += _sms

        return segs, syms

    @classmethod
    def to_multi_line_string(cls, segments):
        segs, syms = cls.to_segments(segments)
        return linemerge(segs)

    @classmethod
    def from_serialized_geom(cls, segments, **kwargs):
        segs, syms = cls.to_segments(segments)
        return cls._to_system(segs, syms, **kwargs)

    @classmethod
    def _to_system(cls, segments, symbols, sys=System, **kwargs):
        return sys(segments=segments, symbols=symbols, **kwargs)

