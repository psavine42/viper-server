from src.geomType import GeomType
from .opers import *
from operator import eq


class EngineHeurFP(object):
    def __init__(self):

        is_source = IF('npred', eq, 0, var='source')
        one_pred = IF('npred', eq, 1)
        mutex(is_source, one_pred)

        is_endpt = IF('nsucs', eq, 0)
        one_succ = IF('nsucs', eq, 1)
        two_sucs = IF('nsucs', eq, 2)
        mutex(is_endpt, one_succ, two_sucs)

        no_sym = NOT(HAS('type'))
        has_sym = HAS('type')
        mutex(no_sym, has_sym)

        is_sym = AND(has_sym, IF('type', eq, GeomType.SYMBOL))
        is_arc = AND(has_sym, IF('type', eq, GeomType.ARC))
        is_circle = AND(has_sym, IF('type', eq, GeomType.CIRCLE))
        mutex(is_circle, is_arc, no_sym, is_sym)

        symbolic = OR(is_circle, is_arc, is_sym, var='symbol-like')

        is_split = AND(one_pred, two_sucs, var='split')
        vbranch = AND(is_split, symbolic, var='vbranch')
        hbranch = AND(is_split, NOT(symbolic), var='hbranch')
        mutex(vbranch, hbranch)

        isElbow = AND(one_pred, one_succ, NOT(symbolic), var='elbow')
        UpElbow = AND(one_pred, one_succ, symbolic, var='elbow+rise')
        mutex(isElbow, UpElbow)

        # isUpright = AND(one_succ, symbolic, var='vHead')
        isDrop = AND(is_endpt, symbolic, var='dHead')
        other_end = AND(is_endpt, NOT(symbolic), var='unlabeled_end')
        mutex(isDrop, other_end)

        rt = Mutex(is_source, other_end, UpElbow, isDrop, vbranch, hbranch, isElbow, var='solved')
        self.root = rt

    @property
    def final_labels(self):
        return ['IsRiser', 'dHead', 'elbow',  'unlabeled_end', 'elbow+rise',
                'vbranch', 'vHead', 'source', 'hbranch']












