import networkx as nx
import numpy as np
from .base import Heuristic, edge_attr
from src.geomType import GeomType
from .engine import *
from .opers import *
from operator import eq


class EngineHeurFP(object):
    def __init__(self):
        is_symbol = HAS('symbol')
        is_source = IF('npred', eq, 0)

        is_endpnt = IF('nsucs', eq, 0)

        one_succ = IF('nsucs', eq, 1)
        one_pred = IF('npred', eq, 1)
        two_sucs = IF('nsucs', eq, 2)

        is_arc = IF('symbol', eq, GeomType.ARC)
        is_circle = IF('symbol', eq, GeomType.CIRCLE)

        is_split = AND.as_prop('split', one_pred, two_sucs)

        circle_or_symbol = OR.as_prop('symbol-like', is_symbol, is_circle, is_arc)

        vbranch = AND.as_prop('vbranch', is_split, circle_or_symbol)
        hbranch = AND.as_prop('hbranch', is_split, NOT(vbranch))

        isElbow = AND.as_prop('elbow', one_pred, one_succ, NOT(circle_or_symbol))
        isUpright = AND.as_prop('vHead', one_pred, one_succ, circle_or_symbol)
        isDrop = Property('dHead', AND(is_endpnt, circle_or_symbol))


        riser = INSUCS.as_prop('IsRiser', isDrop, lambda xs: xs.count(True) == 1)

        Source = Property('source', is_source)
        rt = OR.as_prop('solved', isUpright, Source, riser, isDrop, vbranch, hbranch, isElbow)
        self.root = rt

    @property
    def final_labels(self):
        return ['IsRiser', 'dHead', 'elbow', 'split', 'vbranch', 'vHead', 'source']

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









