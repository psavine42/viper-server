import math
from unittest import TestCase
from .seg_data import get_rendered
from src.structs import Cell
from src.structs.aotp import *
from src import Node
from src.structs.arithmetic import Interval, Constant, Var
import numpy as np
import operator as OP
from src.misc.utils import cells_to_nx
from src.misc import visualize, utils
from src.rules.heursitics import HeuristicFP
import pprint

"""
    Art of the Propagator

    Sussman et all

         +----(cell)--->+
        /                \
    (node)---(cell)----->+(prop)
      | \                /
      |  +----(cell)<---+
      |         |
      |         +(prop)+---------->(cell) 
      |         |
      |  +----(cell)--->+
      | /                \
    (node)---(cell)----->+(prop)
        \                /
         +----(cell)<---+


Computing information about the structure based on 
all levels of the graph (inside of node / among many)
with propagators
             
             
"""

_styles = {
            'prop':
                {'color': '#ff0000',
                 'size':200,
                 'keys': {'type', 'fn', 'cnt'}
                 },
            'cell':
                {'color': '#aad0ff',
                 'size': 100,
                 'keys': { 'var'}
                 },
            'cell+content':{
                'color': '#1e82fc',
                'size': 400,
                'keys': {'content', 'var'}
            },
            'cell+res': {
                'color': '#ba44ff',
                'size': 500,
                'keys': {'content', 'var'}
            },
            'cell+input': {
                'color': '#66ffc1',
                'size': 500,
                'keys': {'content', 'var'}
            }
        }


def similar_triangles1(s_ba, h_ba, s, h):
    ratio = Cell()
    divider(h_ba, s_ba, ratio)
    multiplier(s, ratio, h)


def similar_triangles2(s_ba, h_ba, s, h):
    ratio = Cell()
    divider(h_ba, s_ba, ratio)
    pproduct(s, ratio, h)


def similar_triangles3(s_ba, h_ba, s, h):
    ratio = Cell()
    pdivide(h_ba, s_ba, ratio)
    pproduct(s, ratio, h)


_keys = [('npred', 0), ('nsucs', 0), ('type', False), ('$result', None) ]
_classes = ['$source', '$end_unk', '$dnhead',
            '$Hsplit', '$Vsplit', '$elbows', '$uphead']



class TestCat(TestCase):

    def test_props1(self):
        # root = get_rendered()

        # make a cell
        cell = Cell('$type')

        # apply cells to a node

        # apply cells to a pattern of node

    def bldg_triangl(self, trifn):
        baro_h = Cell()
        baro_s = Cell()
        bldg_h = Cell()
        bldg_s = Cell()

        trifn(baro_s, baro_h, bldg_s, bldg_h)

        bldg_s.add_contents(Interval(54.9, 56.1))
        baro_s.add_contents(Interval(0.36, 0.37))
        baro_h.add_contents(Interval(0.3, 0.32))

        a1, a2 = bldg_h.contents()
        assert np.allclose(a1, 45.75)
        assert np.allclose(a2, 48.5189)

    def test_building1(self):
        self.bldg_triangl(similar_triangles1)

    def test_building2(self):
        self.bldg_triangl(similar_triangles2)

    def test_fn_prop_constr(self):
        c1 = Cell()
        c2 = Cell()
        c3 = Cell()
        adder(c1, c2, c3)
        c1.add_contents(Constant(5))
        c2.add_contents(Constant(2))
        assert c3.contents == Constant(7)

    def init_procedure(self):
        npred, nsucc, has_sym, result = \
            Cell(var='IN_npred'), Cell(var='IN_suc'),\
            Cell(var='IN_sym'), Cell(var='res')

        classes = ['$source', '$end_unk', '$dnhead',
                   '$Hsplit', '$Vsplit', '$elbows', '$uphead' ]

        c0, c1, c2 = discrete_cells([0, 1, 2])
        label_cells = discrete_cells(classes)

        no_pred, one_pred = p_for_classes(npred, [c0, c1])
        no_succ, one_succ, two_succ = p_for_classes(nsucc, [c0, c1, c2])

        not_sym = prop_fn_to(OP.not_, has_sym)

        c_end_unk = prop_fn_to(OP.and_, no_succ, not_sym)
        c_endhead = prop_fn_to(OP.and_, no_succ, has_sym)

        c_one_one = prop_fn_to(OP.and_, one_pred, one_succ)

        c_elbow_s = prop_fn_to(OP.and_, c_one_one, not_sym)
        c_up_head = prop_fn_to(OP.and_, c_one_one, has_sym)
        c_spliter = prop_fn_to(OP.and_, one_pred, two_succ)
        c_split_h = prop_fn_to(OP.and_, c_spliter, not_sym)
        c_split_v = prop_fn_to(OP.and_, c_spliter, has_sym)

        class_cells = [no_pred, c_end_unk, c_endhead, c_split_h,
                       c_split_v, c_elbow_s, c_up_head]
        linear_classifier(class_cells, label_cells, result)

        return npred, nsucc, has_sym, result

    def test_classifier(self):
        npred, nsucc, symb, result = self.init_procedure()
        npred.add_contents(0)
        # if i add content here, i get the final result
        assert result.contents() == '$source'

        # if i add content here, result is not affected
        nsucc.add_contents(1)
        assert result.contents() == '$source'

    def test_classifier2(self):
        npred, nsucc, symb, result = self.init_procedure()
        npred.add_contents(1)
        assert result.value != '$source'
        symb.add_contents(False)
        nsucc.add_contents(1)
        assert result.value != '$source'

    def test_not(self):
        cell = Cell()
        not_sym = prop_fn_to(OP.not_, cell)
        cell.add_contents(False)
        assert not_sym.value == True

    def show_procedure(self):
        npred, nsucc, symb, res = self.init_procedure()
        npred.add_contents(1)
        nsucc.add_contents(1)
        assert res.value == '$elbow'
        visualize.prop_plot(utils.cells_to_nx([nsucc, npred]), meta=_styles)

    def test_vbranch(self):
        npred, nsucc, symb, res = self.init_procedure()
        npred.add_contents(1)
        nsucc.add_contents(2)
        symb.add_contents(True)
        assert res.value == '$Vsplit'

    def test_hbranch(self):
        npred, nsucc, symb, res = self.init_procedure()
        npred.add_contents(1)
        nsucc.add_contents(2)
        symb.add_contents(False)
        assert res.value == '$Hsplit'

    def test_end_unk(self):
        npred, nsucc, symb, res = self.init_procedure()
        npred.add_contents(1)
        nsucc.add_contents(0)
        symb.add_contents(False)
        assert res.value == '$end_unk'

    def test_end_head(self):
        npred, nsucc, symb, res = self.init_procedure()
        npred.add_contents(1)
        nsucc.add_contents(0)
        symb.add_contents(True)
        assert res.value == '$dnhead'

    def test_up_head(self):
        npred, nsucc, symb, res = self.init_procedure()
        npred.add_contents(1)
        nsucc.add_contents(1)
        symb.add_contents(True)
        assert res.value == '$uphead'

    def test_eng(self):
        from src.rules.heursitics import HeuristicFP
        n1 = Node((1, 1, 0))
        n2 = Node((5, 1, 0))
        n3 = Node((5, 5, 0), type='symbol')
        nodes = [n1, n2, n3]
        n1.connect_to(n2)
        n2.connect_to(n3)

        heur = HeuristicFP(_classes, _keys)
        for n in nodes:
            heur.__call__(n)
        # print(n2)
        assert n2.get(heur.result_key) == '$elbows', n2.get(heur.result_key)
        assert n1.get(heur.result_key) == '$source', n1.get(heur.result_key)
        assert n3.get(heur.result_key) == '$dnhead', n3.get(heur.result_key)

    def test_condit(self):
        c1 = Cell(var='input')
        if_true = Cell('wooo')
        fn = lambda x: isinstance(x, str)
        outcell, f = conditional(c1, if_true, fn)
        c1.add_contents('aaak')
        assert outcell.value == 'wooo'

    def mini_system(self):
        n1 = Node((1, 1, 0))
        n2 = Node((5, 1, 0), type='symbol')
        n3 = Node((5, 5, 0), type='symbol')
        n4 = Node((10, 1, 0))

        nodes = [n1, n2, n3, n4]
        n1.connect_to(n2)
        n2.connect_to(n3)
        n2.connect_to(n4)
        return nodes

    def test_edge_dirs(self):
        nodes = self.mini_system()
        n1, n2, n3, n4 = nodes

        heur = HeuristicFP(_classes, _keys)
        for n in nodes:
            heur.__call__(n)

        def node_fn(*input_contents):
            cnt = 0
            for contents in input_contents:
                if contents == '$dnhead':
                    cnt += 1
                elif isinstance(contents, int):
                    cnt += contents
            return cnt

        heur.reducer(n1, '$result', '$num_heads', node_fn)
        assert n1.cells['$num_heads'].value == 1
        assert n2.cells['$num_heads'].value == 1

        n4.write('type', 'symbol')
        assert n1.cells['$num_heads'].value == 2
        assert n2.cells['$num_heads'].value == 2
        n3.write('type', False)
        assert n1.cells['$num_heads'].value == 1
        assert n2.cells['$num_heads'].value == 1

    def setUp(self):
        self._profile = False

    def test_edge_p(self):
        nodes = self.mini_system()
        heur = HeuristicFP(_classes, _keys)
        for node in nodes:
            heur.propagate_edges(node)

        n1, n2, n3, n4 = nodes
        e1_2 = n1.successors(edges=True)[0]
        e2_3 = n3.predecessors(edges=True)[0]
        e2_4 = n4.predecessors(edges=True)[0]

        c1d = e2_3.cells['angle_to'].value
        c2d = e2_4.cells['angle_to'].value

        # print(c1d, c2d)
        assert math.isclose(c1d.val, math.pi / 2), \
            'expected:{}, got:{} '.format(math.pi / 2, c1d)

        assert math.isclose(c2d.val, 0), \
            'expected:{}, got:{} '.format(0, c1d)


    def test_steward(self):
        """
        Test that structure of propogators is stable after:
            - added
            - added
        """
        from src.structs.propagator import Provenance, Steward
        nodes = self.mini_system()
        heur = HeuristicFP(_classes, _keys)
        for node in nodes:
            heur.__call__(node)

        n1, n2, n3, n4 = nodes
        deps = Provenance()

        stw = Steward(n3)
        res = stw.compute_sources(allow_roots=['$GT'])
        print(stw)
        for s in stw.deps:
            print(s)
        for cell in n3.cells.values():
            print(cell)
            print(cell._support._roots)
            print(cell._support._depth)



    def test_large(self):
        from spec.seg_data import load_segs
        segs = load_segs()

    def test_angles(self):
        import src.structs.aotp as A
        _profile = False
        d1 = np.array([1, 0, 0])
        d2 = np.array([0, 1, 0])
        _m = math.pi / 2
        assert math.isclose(A._angle(d1, d2).val, _m),  \
            'expected {} got {}'.format(_m, A._angle(d1, d2).val)

        c1 = Cell(var='cell1', profile=_profile)
        c2 = Cell(var='cell2', profile=_profile)
        ca = Cell(var='cell3', profile=_profile)
        p_angle(c1, c2, ca)

        c1.add_contents(d1)
        # set_trace([c1, c2, ca], c=True)
        assert ca.value is None, \
            'expected None, got {}'.format(ca.value)

        c2.add_contents(d2)

        assert math.isclose(ca.contents.val, _m), \
            'expected None, got {}'.format(_m, ca.contents.val)

    def test_copy(self):
        pass

    def test_simil(self):
        pass

    def test_(self):
        pass












