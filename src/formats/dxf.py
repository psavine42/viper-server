import dxfgrabber
from dxfgrabber.dxfentities import Body, Line, Insert
import src.formats.sat as sat
import numpy as np


_DEFAULT_OPTIONS = {
    "grab_blocks": True,
    "assure_3d_coords": True,
    "resolve_text_styles": False,
}


class DxfAdapter(object):
    def __init__(self, fl):
        self._dxf = dxfgrabber.readfile(fl)
        self._reader = sat.SABReader()
        self._translate = 0
        self._scale = self._dxf.header.get('$DIMALTF', 1)
        self._rotate = self._dxf.header.get('$ANGDIR', 0)

    @property
    def scale(self):
        return self._scale

    def apply_xform(self, point_like):
        return self._translate + (point_like * self._scale)

    def undo_xform(self, point_like):
        (point_like - self._translate) / self._scale

    def _process_inserts(self, inserts=None):
        if inserts is None:
            fn = lambda x: isinstance(x, Insert)
        else:
            if isinstance(inserts, str):
                inserts = [inserts]
            fn = lambda x: isinstance(x, Insert) and x.name in inserts
        blocks = list(filter(fn, self._dxf.entities))

    def process(self, inserts=None):
        """"""
        # acis_ents = [self._reader.read_single(x.acis) for x in self.bodies]
        pipes = list(filter(lambda x: x.valid is True,
                       map(sat.SACCylinder,
                           map(lambda x: self._reader.read_single(x.acis),
                               self.bodies))))
        print(len(pipes))
        self._translate = -1 * self.apply_xform(pipes[0].line.numpy)
        return pipes



    # utils ------------------------------------
    @property
    def bodies(self):
        return list(filter(lambda x: isinstance(x, Body), self._dxf.entities))

    @property
    def lines(self):
        return list(filter(lambda x: isinstance(x, Line), self._dxf.entities))

    @property
    def inserts(self):
        return list(filter(lambda x: isinstance(x, Insert), self._dxf.entities))
