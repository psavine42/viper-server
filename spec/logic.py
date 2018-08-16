import unittest
import importlib
from src.rules import engine
from src.rules.engine import RuleEngine, KB
from src.rules.property import Property
from src.rules.graph import Node
import operator
from operator import eq
from src.geomType import GeomType
from src.rules.opers import *
from pprint import pprint
from src import visualize
from spec.build_spec import _segments
from src.factory import SystemFactory


def read_props(node, k):
    # print(node , ':', node.tmps)
    return node.get(k, None)


class TestLogic(unittest.TestCase):
    def setUp(self):
        is_symbol = HAS('symbol')
        is_end = IF('nsucs', eq, 0)
        is_circle = IF.as_prop('is_circle', 'symbol', eq, GeomType.CIRCLE)
        possible_end = OR.as_prop('pos_end', is_symbol, is_circle)

        isEnd = Property('IsDrop', AND(is_end, possible_end))
        """
        Now we want to say, node has one end in successors
        one_drop_in_sucs = 
        is_riser = AND(is_symbol, one_drop_in_sucs)
        """
        one_drop_in_sucs = INSUCS(isEnd, lambda xs: xs.count(True) == 1)

        is_riser = Property('IsRiser', one_drop_in_sucs)

        self.term = OR.as_prop('IsDone', is_riser, isEnd)

    def tearDown(self):
        self.term = None

    def test_prop1(self):
        cond = IF('nsucs', eq, 0)
        isEnd = Property('IsEnd', cond)

        node1 = Node(1)
        assert cond(node1) is True

        res1 = isEnd(node1)
        assert res1 is True
        assert node1.get('IsEnd') is True

        node2 = Node(2)
        node1.connect_to(node2)

        assert cond(node1) is False

    def test_and(self):
        is_symbol = HAS('symbol')
        is_end    = IF('nsucs', eq, 0)
        is_circle = IF('symbol', eq, GeomType.CIRCLE)

        is_drop_head = AND(is_end, is_circle)

        # setup Nodes
        n0 = Node(0)
        n1 = Node(1, symbol=GeomType.CIRCLE)
        n2 = Node(2, symbol=GeomType.CIRCLE)

        # graph
        n0.connect_to(n1)
        n1.connect_to(n2)

        assert is_drop_head(n1) is False
        assert is_drop_head(n2) is True
        assert is_symbol(n0) is False
        assert is_symbol(n1) is True

    def test_itm(self):
        n0 = Node(0, symbol=GeomType.CIRCLE)
        n1 = Node(1)
        n2 = Node(2, symbol=GeomType.CIRCLE)
        n0.connect_to(n1)
        n1.connect_to(n2)
        read_props(n2, 'IsDrop')

        assert self.term(n0) is True

        read_props(n2, 'IsDrop')
        print('\n')
        print(n0, n0.tmps)
        print(n1, n1.tmps)
        print(n2, n2.tmps)

        assert read_props(n2, 'IsDrop') is True
        assert read_props(n0, 'IsRiser') is True
        assert not read_props(n2, 'IsRiser')

    def test_eng(self):
        print('\n')
        ## pprint(self.term.pre_conditions())
        rl = RuleEngine(term_rule=self.term)
        pprint(rl._freq)


    def test_eng2(self):
        from src.rules.heursitics import EngineHeurFP
        rules = EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root)

        _root = (2, 1, 0)
        system = SystemFactory.from_segs(_segments, root=_root, lr='a')
        system = system.bake()
        root = engine.nx_to_nodes(system.G, system.root)

        # pprint(rl._freq)
        # nxg = engine.props_to_nx(rules.root)
        # visualize.simple_plot(nxg)

        root = Eng.yield_queue(root)
        nxg = Eng.plot(root, rules.final_labels)


