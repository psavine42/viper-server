import unittest
import os
from src.comms.smartbuild import GraphFile
import src.formats.revit as rvt
from src.structs import Node, Edge
import src.structs.node_utils as gutils

_base_path = '/home/psavine/source/viper/data/out/commands/'
# _path = '/home/psavine/source/viper/data/out/commands/graphtest.pkl'
_Cmd_pipe = rvt.Cmds.Pipe.value
_Pipe_CnPt = rvt.PipeCmd.ConnectorPoint.value
_Pipe_PtPt = rvt.PipeCmd.Points.value

_Cmd_tee = rvt.Cmds.Tee.value
_Cmd_elb = rvt.Cmds.Elbow.value


def msg(tgt, res):
    return 'expected {}, got {}'.format(tgt, res)


def _edge_status(edge):
    return edge.get('built'), edge.get('conn1'), edge.get('conn2')


class TestGraphFile(unittest.TestCase):
    _path = _base_path + 'graphtest.pkl'

    def setUp(self):
        self.s = GraphFile(file_path=self._path)
        self.s._reload()

    def test_load_success(self):
        assert os.path.exists(self._path) is True, 'missing file'
        num_ents = len(list(self.s.root.__iter__()))
        assert num_ents == 274, 'got {} ents'.format(num_ents)
        assert rvt.is_built(self.s.root) is False, msg(False, rvt.is_built(self.s.root))

    def test_first(self):
        target = 'Handshake'
        hs = self.s.on_first()
        assert hs == target, msg(target, hs)
        first = self.s.on_response('Ready')
        assert isinstance(first, str), msg('str', first)

    def _edge_stat(self, e, *args):
        tb, tc1, tc2 = args
        b, c1, c2 = _edge_status(e)
        tgts = [x is True for x in [tb, tc1, tc2]]
        actl = [x is True for x in [b, c1, c2]]
        assert tgts == actl, 'built ' + msg(tgts, actl)

    def _test_node_success(self, node: Node, ntype):
        mgr = rvt.make_actions_for(node)
        a1 = mgr.next_action()
        assert isinstance(a1, list), msg('list', a1)
        assert a1[0] == ntype, msg(ntype, a1[0])

        mgr.on_success(a1)
        assert rvt.is_built(node) is True, msg(True, rvt.is_built(node))

        for i in range(node.nsucs):
            a2 = mgr.next_action()
            assert a2[0] == _Cmd_pipe, msg(_Cmd_pipe, a2[0])
            assert a2[2] == _Pipe_CnPt, msg(_Pipe_CnPt, a2[2])
            mgr.on_success(a2)

        for e in node.successors(edges=True):
            self._edge_stat(e, True, True, not True)

    def _test_node_fail(self, node: Node, ntype):
        mgr = rvt.make_actions_for(node)
        a1 = mgr.next_action()
        assert isinstance(a1, list), msg('list', a1)
        assert a1[0] == ntype, msg(ntype, a1[0])

        mgr.on_fail(a1)
        assert rvt.is_built(node) is False, msg(False, rvt.is_built(node))

        for i in range(node.nsucs):
            a2 = mgr.next_action()
            assert a2[0] == _Cmd_pipe, msg(_Cmd_pipe, a2[0])
            assert a2[2] == _Pipe_PtPt, msg(_Pipe_PtPt, a2[2])
            mgr.on_success(a2)

        for e in node.successors(edges=True):
            b, c1, c2 = _edge_status(e)
            assert b is True, msg(True, b)
            assert c1 is not True, msg(False, c1)
            assert c2 is not True, msg(False, c2)

    def test_root_success(self):
        node = self.s.root
        self._test_node_success(node, _Cmd_tee)

    def test_root_fail(self):
        node = self.s.root
        self._test_node_fail(node, _Cmd_tee)

    def test_run(self):
        h = self.s.on_first()
        resp = self.s.on_response('Ready')
        cnt = 0
        while len(self.s) > 0 and resp not in ['END', 'DONE']:
            print(cnt, self.s.current_node.id, resp[8:12], self.s.current_node.get('$create'))
            cnt += 1
            resp = self.s.on_response('SUCCESS')
            if resp in ['END', 'DONE']:
                print('')
                break

        assert cnt > 274

    def test_run_strict(self):
        h = self.s.on_first()
        resp = self.s.on_response('Ready')
        prev_node = self.s.current_node

        cnt_op = 0
        cnt_nd = 0
        while len(self.s) > 0 and resp not in ['END', 'DONE']:
            print(cnt_nd, cnt_op,
                  self.s.cmd_mgr.gobj.id,
                  resp[8:10],
                  self.s.cmd_mgr.gobj.get('$create'))
            cnt_op += 1
            resp = self.s.on_response('SUCCESS')

            if prev_node.id != self.s.current_node.id:
                cnt_nd += 1
                # print(prev_node.id)
                assert prev_node.get('built', None) is True, 'ndoe not bult'
                for e in prev_node.successors(edges=True):
                    self._edge_stat(e, True, True, None)
                for e in prev_node.predecessors(edges=True):
                    self._edge_stat(e, True, True, True)
                prev_node = self.s.cmd_mgr.gobj
        print(cnt_nd)
        assert cnt_nd == 274

    def test_elbow_1(self):
        node_id = 9674342137
        node = gutils.node_with_id(self.s.root, node_id)
        print(node.get('$create'))
        assert node is not None
        mgr = rvt.make_actions_for(node)
        print(mgr._state)
        a = mgr.next_action()
        print(a[0:3], '\n', mgr._state)
        assert isinstance(a, list), msg('list', a)
        assert a[0] == _Cmd_pipe, msg(_Cmd_pipe, a[0])

        mgr.on_success(a)
        a = mgr.next_action()
        print(a[0:3], '\n', mgr._state)

        assert a[0] == _Cmd_elb, msg(_Cmd_elb, a[0])

        mgr.on_success(a)
        print(mgr._state)
        a = mgr.next_action()
        print(a, '\n', mgr._state)

        assert a is None, msg(None, a)

        e = node.successors(edges=True, ix=0)
        self._edge_stat(e, True, True, not True)

    def test_elbow_fail(self):
        node_id = 9674342137
        node = gutils.node_with_id(self.s.root, node_id)
        # print(node.get('$create'))
        assert node is not None
        mgr = rvt.make_actions_for(node)
        a = mgr.next_action()
        # print(a)

        assert a[0] == _Cmd_pipe, msg(_Cmd_pipe, a[0])
        mgr.on_success(a)
        a = mgr.next_action()
        # print(a)
        assert a[0] == _Cmd_elb, msg(_Cmd_elb, a[0])
        mgr.on_fail(a)
        a = mgr.next_action()
        self._edge_stat(node.successors(edges=True, ix=0), True, True, not True)

    def _single_node_success(self, node):
        mgr = rvt.make_actions_for(node)
        a = mgr.next_action()
        while a is not None:
            print(a)
            assert isinstance(a, list), msg('list', a)
            mgr.on_success(a)
            a = mgr.next_action()
            print(mgr._state)

        for e in node.successors(edges=True):
            self._edge_stat(e, True, True, None)
        for e in node.predecessors(edges=True):
            self._edge_stat(e, None, None, True)

    def test_tee1(self):
        node_id = 390499279
        node = gutils.node_with_id(self.s.root, node_id)
        assert node is not None
        self._test_node_success(node, _Cmd_tee)

    def test_tee2(self):
        node_id = 5566475637
        node = gutils.node_with_id(self.s.root, node_id)
        self._single_node_success(node)

    def _test_scenario(self, node, results):
        mgr = rvt.make_actions_for(node)
        a = -1
        while results and a is not None:
            print('\n')
            a = mgr.next_action()
            r = results.pop(0)
            print(a, r)
            if r == 1:
                mgr.on_success(a)
            elif r == 0:
                mgr.on_fail(a, 'x')
            print(mgr._state)

    def test_elbow_f(self):
        node_id = 9674342137
        node = gutils.node_with_id(self.s.root, node_id)

        res = [1, 0, 1, 1, 1]
        self._test_scenario(node, res)

class TestGraphFull(TestGraphFile):
    _path = _base_path + 'graphtest_full.pkl'



