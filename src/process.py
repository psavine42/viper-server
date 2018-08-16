
import math
from src.factory import SystemFactory
from src.render import RenderSystem
from src.rules.heursitics import HeuristicsFP
from src.geom import MepCurve2d
from src import walking


_sys_dict = {
    'FP':HeuristicsFP
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
        system = SystemFactory.from_request(data)

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