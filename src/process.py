
import math, time, json
from src.factory import SystemFactory
from src.rules.engine import RuleEngine, KB
from src.render.render import RenderSystem, RenderNodeSystem
from src.rules import heursitics
from src.geom import MepCurve2d
from src import walking
import viper

_sys_dict = {
    'FP': heursitics.EngineHeurFP
}


class SystemProcessor(object):
    def __init__(self):
        self._shrink = -0.01
        self._z = 10.0

    def get_system(self, ky):
        return _sys_dict.get(ky)()

    def process(self, data, points, system_type='FP'):
        # tmp_arg = {'shrink': 0.5, 'base_z': 0}
        # build graph
        system = SystemFactory.from_serialized_geom(
            data, sys=viper.SystemV2, root=points[0])

        # bake attributes
        system.bake()

        # compute info about graph
        heuristic = self.get_system(system_type)
        system = heuristic(system)

        # based on labels, finallize graph (adding new info)
        system = RenderSystem()(system)

        # bake the build order
        system.bake_attributes(system.root, full=False)

        # create build instructions
        geom, inds = self.finalize(system.G, system.root)
        out_data = {'geom': geom, 'indicies': inds}
        return out_data

    def _prepare(self, start, end):
        crv = MepCurve2d(start, end)
        p1, p2 = crv.extend(self._shrink, self._shrink).points
        if math.isnan(p1[0]):
            print(start, end, p1, p2)
        vec = list(p1) + list(p2)
        vec[2] += self._z
        vec[5] += self._z
        return vec

    def finalize(self, G, root):
        """
            Create List of

        :param G:
        :return:
            geom = [
                [10, 5, 0,   10, 0, 0] # [0, [0 , 1]]
                [10, 0, 0,   5, 0, 0]  # [1, [0 , 1]]
                [5,  0, 0,   5, 5, 0]  # [2, [0 , 1]]
            ]
            inds = [
                [[0, 1] , [1, 0] ]
                [[1, 1] , [2, 0] ]
            ]
        """
        geom, inds = [], []

        def add_to_res(p1, p2, preds, sucs, seen):
            line_ix = G[p1][p2].get('order')
            geom.append(self._prepare(p1, p2))
            if sucs:
                sub_inds = [line_ix, 1]
                for p in sucs:
                    sub_inds.extend([G[p2][p].get('order'), 0])
                inds.append(sub_inds)

        walking.walk_edges(G, root, add_to_res)
        return geom, inds


class SystemProcessorV2(object):
    def __init__(self):
        self._shrink = -0.01
        self._z = 10.0

    def get_system(self, ky):
        return _sys_dict.get(ky)()

    def process(self, data, points, system_type='FP'):
        # build graph
        system = SystemFactory.from_serialized_geom(
            data, sys=viper.SystemV2, root=points[0])

        # bake attributes
        system.bake()

        # compute info about graph
        heuristic = self.get_system(system_type)
        system = heuristic(system)

        # based on labels, finallize graph (adding new info)
        system = RenderSystem()(system)

        # bake the build order
        system.bake_attributes(system.root, full=False)

        # create build instructions
        geom, inds = self.finalize(system.G, system.root)
        out_data = {'geom': geom, 'indicies': inds}
        return out_data

    def _prepare(self, start, end):
        crv = MepCurve2d(start, end)
        p1, p2 = crv.extend(self._shrink, self._shrink).points
        if math.isnan(p1[0]):
            print(start, end, p1, p2)
        vec = list(p1) + list(p2)
        vec[2] += self._z
        vec[5] += self._z
        return vec

    def finalize(self, G, root):
        """
            Create List of

        :param G:
        :return:
            geom = [
                [10, 5, 0,   10, 0, 0] # [0, [0 , 1]]
                [10, 0, 0,   5, 0, 0]  # [1, [0 , 1]]
                [5,  0, 0,   5, 5, 0]  # [2, [0 , 1]]
            ]
            inds = [
                [[0, 1] , [1, 0] ]
                [[1, 1] , [2, 0] ]
            ]
        """
        geom, inds = [], []

        def add_to_res(p1, p2, preds, sucs, seen):
            line_ix = G[p1][p2].get('order')
            geom.append(self._prepare(p1, p2))
            if sucs:
                sub_inds = [line_ix, 1]
                for p in sucs:
                    sub_inds.extend([G[p2][p].get('order'), 0])
                inds.append(sub_inds)

        walking.walk_edges(G, root, add_to_res)
        return geom, inds


class SystemProcessorV3(object):
    def __init__(self):
        self._shrink = -0.01
        self._z = 10.0

    def get_system(self, ky):
        return _sys_dict.get(ky)()

    def _log_inputs(self, *args):
        st = str(round(time.time(), 0))
        data = json.dumps(*args)
        with open('./data/{}-revit-signal.json'.format(st), 'w') as F:
            F.write(data)

    def process(self, data, points=[(-246, 45, 0)], system_type='FP', **kwargs):
        """

        :param data:
        :param points:
        :param system_type:
        :param kwargs: arguments to render the system
        :return:
        """
        # build graph
        self._log_inputs(data, points)
        print(points)
        system = SystemFactory.from_serialized_geom(
            data, sys=viper.SystemV3, root=tuple(points[0]))

        # bake attributes
        system.bake()
        root = system.root

        # compute info about graph
        heuristic = self.get_system(system_type)
        Eng = RuleEngine(term_rule=heuristic.root,
                         mx=1e6, debug=False, nlog=20)
        Kb = KB(heuristic.root)
        root = Eng.alg2(root, Kb)

        # based on labels, finallize graph (adding new info)
        root = RenderNodeSystem()(root)

        # create build instructions
        geom, inds, syms = self.finalize(root)
        return {'geom': geom, 'indicies': inds, 'symbols': syms}

    def _prepare(self, start, end):
        crv = MepCurve2d(start, end)
        p1, p2 = crv.extend(self._shrink, self._shrink).points
        if math.isnan(p1[0]):
            print(start, end, p1, p2)
        vec = list(p1) + list(p2)
        vec[2] += self._z
        vec[5] += self._z
        return vec

    def finalize(self, root):
        """
            Create List of geometry to be created in revit

        :param root: node
        :return:
            geom = [
                [x1, y1, z1, x2, y2, z2]

                [10, 5, 0,   10, 0, 0] # [0, [0 , 1]]
                [10, 0, 0,   5, 0, 0]  # [1, [0 , 1]]
                [5,  0, 0,   5, 5, 0]  # [2, [0 , 1]]
            ]
            inds = [
                [ [ mepcurve_index, end_index ] x number of points to connect]

                [[0, 1] , [1, 0] ]
                [[1, 1] , [2, 0] ]
            ]
            syms = [
                [ symbol_type, mepcurve_index, end_index ]
            ]
        """
        geom, inds, syms = [], [], []
        for node1 in root.__iter__():
            for node2 in node1.successors():

                suc = node1.edge_to(node2)
                line_ix = suc.get('order')
                geom.append(self._prepare(node1.geom, node2.geom))
                if len(node2.successors()) > 0:
                    sub_inds = [line_ix, 1]
                    for p in node2.successors(edges=True):
                          sub_inds.extend([p.get('order'), 0])
                    inds.append(sub_inds)

                elif node2.get('type', None) is not None:
                    # [ symbol_type, mepcurve_index, end_index]
                    syms.append([1, line_ix, 1])

        return geom, inds, syms

