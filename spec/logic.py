from unittest import TestCase
import numpy as np
from operator import eq
from pprint import pprint
from shapely.geometry import LineString, Point

from spec.seg_data import *

from src.rules.opers import *

from src import visualize, SystemFactory, RenderNodeSystem

from src.rules import RuleEngine, KB, heursitics
from src.rules.property import Property

from src import process, viper
from src import propogate as gp

from src.geom import rebuild_mls, to_mls
import src.structs as gr
import src.render.render_propogators as rpg




def read_props(node, k):
    # print(node , ':', node.tmps)
    return node.get(k, None)


_root = (2, 1, 0)


def node_line(line, prev=None):
    for i in range(len(line)):
        n = Node(line[i])
        if prev is None:
            prev = n
        else:
            n.connect_to(prev)
            prev = n
    return prev


def node_tree(pts, prev=None):
    for i in range(len(pts)):
        pt = pts[i]
        if isinstance(pt[0], int):
            n = Node(pt)

            if prev is None:
                prev = n
            else:
                n.connect_to(prev)
                prev = n
        elif isinstance(pt[0], tuple):
            pt, rest = pts[i]
            n = Node(pt)
            if prev is None:
                prev = n
            else:
                n.connect_to(prev)
                node_tree(rest, n)
                prev = n
    return prev




class TestProp(TestCase):
    def get_sys(self):
        system = SystemFactory.from_segs(SEGMENTS, root=_root, lr='a')
        system = system.bake()
        return viper.nx_to_nodes(system)

    def test_dist_prop(self):
        root = self.get_sys()
        propagator = gp.DistanceFromSource()
        propagator(root)
        for n in root.__iter__():
            pred = n.predecessors()
            if len(pred) == 1:
                assert pred[0].get(propagator.var) + 1 == n.get(propagator.var)

    def test_order_prop(self):
        root = self.get_sys()
        propagator = gp.BuildOrder()
        propagator(root)
        order = set()
        cnt = 0
        for n in root.__iter__():
            print(n)
            cnt += 1
            order.add(n.get(propagator.var))
        assert len(order) == cnt

    def test_dist_to_end(self):
        root = self.get_sys()
        propagator = gp.DistanceFromEnd()
        propagator(root)
        for n in root.__iter__():
            if len(n.successors()) == 0:
                assert n.get(propagator.var) == 0

    def test_loop_neg(self):
        root = self.get_sys()
        propagator = gp.LoopDetector()
        propagator(root, data=[])
        for n in root.__iter__():
            assert n.get(propagator.var) is not True

    def test_loop_pos(self):

        connect_loop = [(8., 8., 0), (4., 4., 0)]
        SEGMENTS.append(connect_loop)
        system = SystemFactory.from_segs(SEGMENTS, root=_root, lr='a')
        system = system.bake()
        root = viper.nx_to_nodes(system)
        propagator = gp.LoopDetector()
        propagator(root, data=[])
        for n in root.__iter__():
            if n.geom in connect_loop:
                assert n.get(propagator.var) is True

    def test_edge_det(self):
        root = self.get_sys()
        propagator = gp.DirectionWriter()
        propagator(root)
        for n in root.__iter__():
            for e in n.successors(edges=True):
                print(e)

    def test_overlap_resolver(self):
        pass

    def test_remover_sm(self):
        system = SystemFactory.from_segs(
            SEGMENTS, sys=viper.System, root=_root, lr='a')
        system.bake()
        system.gplot(fwd=True, bkwd=False)

    def test_remover_cl(self):
        system = SystemFactory.from_segs(
            SEGMENTS_COL, sys=viper.System, root=_root, lr='a')
        system.aplot()

    def test_remover_lg(self):
        segs = load_segs()
        system = SystemFactory.from_serialized_geom(
            segs, sys=viper.System, root=(-246, 45, 0))
        system.bake()
        system.gplot(fwd=True, bkwd=False)

    def test_reverse(self):
        n1 = Node(1)
        n2 = Node(2)
        edge = n1.connect_to(n2)
        edge.reverse()
        assert edge.target == n1
        assert edge.source == n2

    def test_merge_self(self):
        n1 = [(1, 1), (1, 4.8), (1.2, 5), (1, 5.2), (1, 10)]
        prev = node_line(n1)
        gp.Cluster()(prev)
        for n in prev.__iter__(fwd=True, bkwd=True):
            print(n, *n.neighbors())

    def test_geom_sims(self):
        l2 = LineString([(1, 2), (1, 4), (4, 6), (4, 8)])
        l1 = LineString([(1, 3), (1, 4), (4, 6), (1, 4)])
        print(l1)
        ds = l1.union(l1)
        print(ds)

    def test_adder(self):
        n1 = [(1, 2), (1, 4), (4, 6), (4, 8)]
        prev = node_line(n1)

        ndd = Node((1, 3))
        pa = gp.PointAdder(ndd)
        pa(prev)
        for n in prev.__iter__(fwd=True, bkwd=True):
            print(n)
        G = viper.nodes_to_nx(prev)
        visualize.gplot(G)


    def test_point(self):
        point = Point(1, 8)
        l3 = [(1, 3), (1, 10), (10, 6)]

        r3 = to_mls(l3)
        print(r3)
        res = rebuild_mls(r3, point)
        print(res)
        tgt = to_mls([(1, 3), (1, 8), (1, 10), (10, 6)])
        assert res == tgt

    def test_254(self):
        segs = load_segs()

        segs, syms = SystemFactory.to_segments(segs)
        fsg = []
        fsm = []
        print(syms[0])
        mx = -260
        mn = -270
        for seg in segs:
            sg = list(seg.coords)
            if mn < sg[0][0] < mx or mn < sg[1][0] < mx:
                fsg.append(seg)

        for seg in syms:
            sg = list(seg.coords)
            if mn < sg[0][0] < mx:
                fsm.append(seg)
        print(fsm[0])
        system = viper.SystemV3(segments=fsg, symbols=fsm, root=(-246, 45, 0))
        system.aplot()



class TestRenderProp(TestCase):
    def test_riser_fn(self):
        root = self.test_translate()
        rcp = viper.System.recipe()
        rcp(root)

        rules = heursitics.EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root)
        Kb = KB(rules.root)
        root = Eng.alg2(root, Kb)

        renderer = RenderNodeSystem()

        root = renderer.render(root)

        print('nodes ', len(root))
        visualize.print_iter(root)
        meta = Eng.annotate_type(root, rules.final_labels)
        visualize.plot3d(root, meta)

    def test_translate(self):
        root = vertical_branch()

        end1 = gr.node_at(root, (8, 6, 0))
        root2 = vertical_branch()
        rpg.Translate()(root2, data=np.array([8, 8, 0]))

        end1.connect_to(root2)
        return root
        # visualize.plot3d(root2, {})



class TestLogic(TestCase):

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

        # assert self.term(n0) is True

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
        rl = RuleEngine(term_rule=self.term)
        pprint(rl._freq)

    def test_eng2(self):
        from src.rules.heursitics import EngineHeurFP
        rules = EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root)

        system = SystemFactory.from_segs(SEGMENTS, root=_root, lr='a')
        system = system.bake()
        root = viper.nx_to_nodes(system)

        root = Eng.yield_queue(root)
        nxg = Eng.plot(root, rules.final_labels)

    def test_compile_eng3(self):
        rules = heursitics.EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root)
        Kb = KB(rules.root)
        print(Kb.get_vars())

        print(Kb.agenda)

    def test_eng3(self):

        rules = heursitics.EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root, mx=400, debug=True, nlog=1)

        _root = (2, 1, 0)
        system = SystemFactory.from_segs(SEGMENTS, root=_root, lr='a')
        system = system.bake()
        root = viper.nx_to_nodes(system)

        Kb = KB(rules.root)
        print(Kb)
        root = Eng.alg2(root, Kb, )
        nxg = Eng.plot(root, rules.final_labels)

    def test_eng4(self):
        system = SystemFactory.from_serialized_geom(load_segs(),
                                                    sys=viper.System,
                                                    root=(-246, 45, 0))
        system = system.bake()
        root = viper.nx_to_nodes(system)
        print(root)
        rules = heursitics.EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root, mx=2500, debug=False, nlog=20)
        Kb = KB(rules.root)
        root = Eng.alg2(root, Kb)
        nxg = Eng.plot(root, rules.final_labels)

    def test_eng5(self):
        data = load_segs(fl='1535158393.0-revit-signal')
        system = SystemFactory.from_serialized_geom(
            data, sys=viper.System, root=(-246, 45, 0))
        system = system.bake()
        root = system.root
        print(root)
        rules = heursitics.EngineHeurFP()
        Eng = RuleEngine(term_rule=rules.root, mx=2500, debug=False, nlog=20)
        Kb = KB(rules.root)
        root = Eng.alg2(root, Kb)

        print('nodes ', len(root))

        renderer = RenderNodeSystem()
        meta = Eng.annotate_type(root, rules.final_labels)
        root = renderer.render(root)

        print('nodes ', len(root))
        visualize.plot3d(root, meta)

    def test_eng_full(self):
        """
        Test the engine as executed by server

        """
        import time
        start = time.time()
        data = load_segs(fl='1535158393.0-revit-signal')
        points = [[-246.0000000012448, 45.31190012691635, 0.0]]
        proc = process.SystemProcessorV3()
        ds = proc.process(data, points, system_type='FP')
        [print(k, len(v)) for k, v in ds.items()]
        visualize.dump_data(ds)
        for g in ds['geom']:
            x1, y1, z1, x2, y2, z2 = g
            res = [x1 == x2, y1 == y2, z1 == z2]
            assert not all(res)
        end = time.time()
        print('time {} secs'.format(end - start))

    def test_loadsyms(self):
        segs = load_segs()
        ds = [x for x in segs if x['children'] != []]
        system = SystemFactory.from_serialized_geom(ds, root=(-246, 45, 0))


