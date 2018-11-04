from _struct import Struct, unpack_from, pack
import numpy as np
import lib.geo as geo
from enum import Enum

# UNPACKERS
UINT8      = Struct('<B').unpack_from
UINT16     = Struct('<H').unpack_from
UINT16_2D  = Struct('<HH').unpack_from
SINT16     = Struct('<h').unpack_from
UINT32     = Struct('<L').unpack_from
SINT32     = Struct('<l').unpack_from
FLOAT32    = Struct('<f').unpack_from
FLOAT32_2D = Struct('<ff').unpack_from
FLOAT32_3D = Struct('<fff').unpack_from
RGBA       = Struct('<ffff').unpack_from
FLOAT64    = Struct('<d').unpack_from
FLOAT64_2D = Struct('<dd').unpack_from
FLOAT64_3D = Struct('<ddd').unpack_from
DATETIME   = Struct('<Q').unpack_from


def getEntityRef(data, offset):
    index, i = getSInt32(data, offset)
    return index, i


def getTagChar(data, index):
    return data[index], index + 1


def get_true(data, index):
    return True, index


def get_false(data, index):
    return False, index


def getTagOpen(data, index):
    return '{', index


def getTagClose(data, index):
    return '}', index


def get_tag_end(data, index):
    return '#', index


def getUInt16(data, offset):
    val, = UINT16(data, offset)
    return val, offset + 2


def getUInt16A(data, offset, size):
    val = unpack_from('<' +'H'*int(size), data, offset)
    val = list(val)
    return val, int(offset + 2 * size)


def getSInt16(data, offset):
    val, = SINT16(data, offset)
    return val, offset + 2


def getSInt16A(data, offset, size):
    val = unpack_from('<' +'h'*int(size), data, offset)
    val = list(val)
    return val, int(offset + 2 * size)


def getUInt32(data, offset):
    val, = UINT32(data, offset)
    return val, offset + 4


def getUInt32A(data, offset, size):
    val = unpack_from('<' +'L'*int(size), data, offset)
    val = list(val)
    return val, int(offset + 4 * size)


def getSInt32(data, offset):
    val, = SINT32(data, offset)
    return val, offset + 4


def getSInt32A(data, offset, size):
    val = unpack_from('<' + 'l'*int(size), data, offset)
    val = list(val)
    return val, int(offset + 4 * size)


def getFloat32(data, offset):
    val, = FLOAT32(data, offset)
    val = float(val)
    return val, offset + 4


def getFloat32A(data, offset, size):
    singles = unpack_from('<' + 'f'*int(size), data, offset)
    val = [float(s) for s in singles]
    return val, int(offset + 4 * size)


def getFloat32_2D(data, index):
    val = FLOAT32_2D(data, index)
    val = list(val)
    return val, int(index + 0x8)


def getFloat32_3D(data, index):
    val = FLOAT32_3D(data, index)
    val = list(val)
    return val, int(index + 0xC)


def getFloat64A(data, offset, size):
    val = unpack_from('<' + 'd'*int(size), data, offset)
    val = list(val)
    return val, int(offset + 8 * size)


def getFloat64_2D(data, index):
    val = FLOAT64_2D(data, index)
    val = list(val)
    return val, int(index + 0x10)


def getFloat64_3D(data, index):
    val = FLOAT64_3D(data, index)
    val = list(val)
    return val, int(index + 0x18)


def getUInt8(data, offset):
    val, = UINT8(data, offset)
    return val, offset + 1


def getBoolean(data, offset):
    val, = UINT8(data, offset)
    if val == 1:
        return True, offset + 1
    elif val == 0:
        return False, offset + 1
    raise ValueError(u"Expected either 0 or 1 but found %02X" %(val))


def getUInt8A(data, offset, size):
    end = int(offset + size)
    assert end <= len(data), "Trying to read UInt8 array beyond data end (%d, %X > %X)" %(size, end, len(data))
    val = unpack_from('<' +'B'*int(size), data, offset)
    val = list(val)
    return val, end


def __getStr(data, offset, end):
    txt = data[offset: end].decode('cp1252')
    txt = txt.encode('utf8').decode("utf8")
    return txt, end


def get_str1(data, offset):
    l, i = getUInt8(data, offset)
    return __getStr(data, i, l + i)


def getStr2(data, offset):
    l, i = getUInt16(data, offset)
    return __getStr(data, i, l + i)


def getStr4(data, offset):
    l, i = getUInt32(data, offset)
    return __getStr(data, i, l + i)


def get_float64(data, offset):
    val, = FLOAT64(data, offset)
    return val, offset + 8


# Logic -------------------------------------------------
class AcisObj(object):
    named_refs = {

    }
    def __init__(self, *args, index=None):
        self.index = index
        self._base_args = args
        self._children = []

    @classmethod
    def create(cls, *args, **kwargs):
        make = SABReader.classes.get(args[0][-1], None)
        if make is not None:
            return make(*args, **kwargs)
        else:
            return cls(*args, **kwargs)

    def refs(self, index=None, types=None):
        res = []
        for pos, (t, v) in enumerate(self._base_args):
            if t == 'TAG_ENTITY_REF' and v != -1:
                if index is not None and pos == index:
                    return pos, v
                # elif types is not None and
                else:
                    res.append((pos, v))
        return res

    def set_ref(self, pos, v):
        self._base_args[pos][1] = v

    @property
    def values(self):
        return [x[-1] for x in self._base_args]

    @property
    def name(self):
        return self._base_args[0][-1]

    def __repr__(self):
        st = '\n{}:{}:{}'.format(self.__class__.__name__,
                                 self.name, self.index)
        for i, ar in enumerate(self._base_args):
            st += '\n' + str(ar)
        st += '\n-----------'
        return st


class Face(AcisObj):
    def __init__(self, *args, **kwargs):
        AcisObj.__init__(self, *args, **kwargs)
        self._next       = None      # The next face
        self._loop       = None      # The first loop in the list
        self._parent     = None      # Face's owning shell
        self.unknown     = None      # ???
        self._surface    = None      # Face's underlying surface
        self.sense       = 'forward' # Flag defining face is reversed
        self.sides       = 'single'  # Flag defining face is single or double sided
        self.side        = None      # Flag defining face is single or double sided
        self.containment = False     # Flag defining face is containment of double-sided faces




class ACBody(AcisObj):
    def __init__(self, *args, **kwargs):
        AcisObj.__init__(self, *args, **kwargs)
        self.lump = None
        self.transform = None

    def children_of_type(self, ntype):
        if isinstance(ntype, str):
            fn = lambda x: x.name == ntype
        else:
            fn = lambda x: isinstance(x.name, ntype)
        return list(filter(fn, self._children))

    def build(self, ent_list):
        self._children = ent_list[1:]
        return self


class ACTransform(AcisObj):
    def __init__(self, *args, **kwargs):
        AcisObj.__init__(self, *args, **kwargs)
        xf = []
        for typ, v in self._base_args:
            if typ == 'TAG_VECTOR_3D':
                xf.append(v)
        self.affine = np.asarray(xf[0:3])
        self.translate = np.asarray(xf[-1])

        self.scale = self.values[-4]
        self.rotation = self.values[-3]
        self.reflection = self.values[-2]
        self.shear = self.values[-1]

    def apply(self, pt):
        return self.scale*(np.dot(self.affine, pt) + self.translate)

    @property
    def geom(self):
        return None


class ACEllipse(AcisObj):
    def __init__(self, *args, **kwargs):
        AcisObj.__init__(self, *args, **kwargs)
        v = self.values
        self.sub_type = v[1]
        self.center = v[5]
        self.normal = v[6]
        self.major = v[7]
        self.ratio = v[8]

    @property
    def geom(self):
        return np.array(self.center)


class ACPoint(AcisObj):
    def __init__(self, *args, **kwargs):
        AcisObj.__init__(self, *args, **kwargs)

    @property
    def geom(self):
        return np.array(self.values[-1])


class SACCylinder(object):
    """

    """
    def __init__(self, acbody: ACBody):
        self._data = acbody
        self._line = None
        self._radius = None
        self.reset()

    def _init_line(self):
        pts = self._data.children_of_type('point')
        xfm = self._data.children_of_type('transform')
        if len(pts) != 2 or len(xfm) != 1:
            return None
        xf = xfm[0]
        p1 = geo.Point(xf.apply(pts[0].geom))
        p2 = geo.Point(xf.apply(pts[1].geom))
        self._line = geo.Line(p1, p2)

    def _init_radius(self):
        els = self._data.children_of_type('ellipse')
        if len(els) != 2:
            return None
        mjr = els[0].major
        mag = np.sum(np.array(mjr) ** 2, axis=-1) ** 0.5
        self._radius = mag

    def reset(self):
        self._init_radius()
        self._init_line()

    @property
    def valid(self):
        if self.line is not None and self.radius is not None:
            return True
        return False

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, v):
        self._radius = v

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, v):
        self._line = v


# Reader --------------------------------------------------
class SABReader(object):
    """
    REader for the SAB ASIC format. will fill in classes as needed
        tag  name       dec-size
    """
    read = {
        2:  ['TAG_CHAR',        getTagChar  ],  # character (unsigned 8 bit)
        3:  ['TAG_SHORT',       getSInt16   ],  # 16Bit signed value
        4:  ['TAG_LONG',        getSInt32   ],  # 32Bit signed value
        5:  ['TAG_FLOAT',       getFloat32],    # 32Bit IEEE Float value
        6:  ['TAG_DOUBLE',      get_float64],   # 64Bit IEEE Float value
        7:  ['TAG_UTF8_U8',     get_str1],      # 8Bit length + UTF8-Char
        8:  ['TAG_UTF8_U16',    getStr2],       # 16Bit length + UTF8-Char     2
        9:  ['TAG_UTF8_U32_A',  getStr4],       # 32Bit length + UTF8-Char     4
        10: ['TAG_TRUE',        get_true],      # Logical true value           1
        11: ['TAG_FALSE',       get_false   ],  # Logical false value          1
        12: ['TAG_ENTITY_REF',  getEntityRef],  # Entity reference             1
        13: ['TAG_IDENT',       get_str1    ],  # Sub-Class-Name
        14: ['TAG_SUBIDENT',    get_str1    ],  # Base-Class-Name
        15: ['TAG_OPEN',        getTagOpen  ],  # '{' Opening block tag
        16: ['TAG_CLOSE',       getTagClose ],  # '}' Closing block tag
        17: ['TAG_TERMINATOR',  get_tag_end ],  # '#' sign
        18: ['TAG_UTF8_U32_B',  getStr4     ],  # 32Bit length + UTF8-Char
        19: ['TAG_POSITION',    getFloat64_3D], # 3D-Vector scaled (scaling will be done later because of text fil
        20: ['TAG_VECTOR_3D',   getFloat64_3D], # 3D-Vector normalized
        21: ['TAG_ENUM_VALUE',  getUInt32   ],  # value of an enumeration
        22: ['TAG_VECTOR_2D',   getFloat64_2D]  # U-V-Vector
    }
    classes = {
        'point':        ACPoint,
        'body':         ACBody,
        'transform':    ACTransform,
        'ellipse':      ACEllipse
    }

    @classmethod
    def tag_reader_fn(cls, tag):
        f = cls.read.get(tag, None)
        if f is None:
            return None, None
        else:
            return f

    @staticmethod
    def first_body(data):
        return len(data.split(b'body')[0]) - 2

    @staticmethod
    def build_body(ent_list):
        body = ent_list[0]
        return body.build(ent_list)

    @classmethod
    def readfile(cls, data, offset=None):
        if offset is None:
            offset = cls.first_body(data)
        index = 1
        ln = len(data)
        out, current_ent = [], []
        while offset < ln:
            tag = data[offset]
            if tag is None:
                return out
            offset += 1
            name, fn = cls.tag_reader_fn(tag)
            if name is None:
                return out
            res, offset = fn(data, offset)
            if res == '#':
                out.append(AcisObj.create(*current_ent, index=index))
                current_ent = []
                index += 1
            else:
                current_ent.append([name, res])

        return out

    @classmethod
    def read_single(cls, data, offset=None):
        ent_list = cls.readfile(data, offset)
        return cls.build_body(ent_list)




