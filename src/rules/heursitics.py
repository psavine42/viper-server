import networkx as nx
import numpy as np
from .base import Heuristic, edge_attr
from src.geomType import GeomType
from .engine import *
from .opers import *
from operator import eq



class EngineHeurFP(object):
    def __init__(self):

        is_source = IF('npred', eq, 0, var='source')
        one_pred = IF('npred', eq, 1)
        mutex(is_source, one_pred)

        is_endpnt = IF('nsucs', eq, 0)
        one_succ = IF('nsucs', eq, 1)
        two_sucs = IF('nsucs', eq, 2)
        mutex(is_endpnt, one_succ, two_sucs)

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

        # isUpright = AND(is_endpnt, symbolic, var='vHead')
        isDrop = AND(is_endpnt, symbolic, var='dHead')
        other_end = AND(is_endpnt, NOT(symbolic), var='unlabeled_end')
        mutex(isDrop, other_end)

        # riser = INSUCS(isDrop, lambda xs: xs.count(True) == 1, var='IsRiser')

        rt = Mutex(is_source, other_end, UpElbow,
                   isDrop, vbranch, hbranch, isElbow, var='solved')
        self.root = rt

    @property
    def final_labels(self):
        return ['IsRiser', 'dHead', 'elbow',  'unlabeled_end', 'elbow+rise',
                'vbranch', 'vHead', 'source', 'hbranch']

    def run(self, root):
        return


class HeuristicsFP(Heuristic):
    """"""
    def __init__(self, **kwargs):
        super(HeuristicsFP, self).__init__()
        self._heurstics = {
            'branch':self.branch,
            'split': self.split,
            'elbow': self.elbow,
            'dhead': self.end,
            'hhead': None,
            'vhead': self.vertical_head,
        }
        self._heurdict = {
            (1, 2): ['split', 'branch'],
            (1, 0): ['dhead'],
            (1, 1): ['vhead', 'elbow'],
        }
    @classmethod
    def end(cls, G=None, el=None, pred=None, sucs=None, seen=None):
        " Fireprotection drop "
        cond1 = len(pred) == 1
        cond2 = len(sucs) == 0
        cond3 = G.nodes[el].get('type', None) != GeomType.SYMBOL
        return all([cond1, cond2, cond3])

    @classmethod
    def elbow(cls, G=None, el=None, pred=None, sucs=None, seen=None):
        """ bend in a pipe - place elbow
             ^
             |
             +<-->

            place elbow
        """
        dr1 = edge_attr(G, (pred[0], el), 'direction')
        dr2 = edge_attr(G, (el, sucs[0]), 'direction')
        return not np.array_equal(dr1, dr2)

    @classmethod
    def vertical_head(cls, G=None, el=None, pred=None, sucs=None, seen=None):
        """
            symbol on a line

            <--o-->
        """
        dr1 = edge_attr(G, (pred[0], el), 'direction')
        dr2 = edge_attr(G, (el, sucs[0]), 'direction')
        return np.array_equal(dr1, dr2)

    @classmethod
    def split(cls, G=None, el=None, pred=None, sucs=None, seen=None):
        """ Split

            <--+-->
               ^
               |
            two lines are colinear, and one is orthagonal

        """
        dr1 = edge_attr(G, (el, sucs[0]), 'direction')
        dr2 = edge_attr(G, (el, sucs[1]), 'direction')
        return np.array_equal(dr1, -1 * dr2)

    @classmethod
    def branch(cls, G=None, el=None, pred=None, sucs=None, seen=None):
        """ Branch
               ^
               |
            -->+-->

            two lines are colinear, and one is orthagonal
        """
        dr = edge_attr(G, (pred[0], el), 'direction')
        dr1 = edge_attr(G, (el, sucs[0]), 'direction')
        dr2 = edge_attr(G, (el, sucs[1]), 'direction')
        return any([np.array_equal(dr, dr1), np.array_equal(dr, dr2)])

    def _symbol(self, G=None, el=None, pred=None, sucs=None, seen=None):
        """ Branch
               ^        <--+-->
               |           ^
            -->O-->        |


            branch with symbol - rise - annotate successors as raised
        """
        _symbol_actions = {
            'branch': self.branch,
            'split': self.split,
            'elbow': self.elbow,
            'end': self.end,
            'vhead': self.vertical_head,
        }
        data = G.nodes[el]
        label = _symbol_actions.get(data.get('label', ''), None)
        if not label:
            return None


# def _edgelen(ed, data):
    # crv =src.geom.MepCurve2d(*ed)
    # return crv.length


# props = {'edge.length': _edgelen }









