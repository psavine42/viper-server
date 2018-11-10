
import math, time, json
from src import RenderNodeSystem, SystemFactory, RuleEngine, KB
from src.rules import heursitics
from src.geom import MepCurve2d
from enum import Enum

from src import viper

_sys_dict = {
    'FP': heursitics.EngineHeurFP
}


class Symbolic(object):
    pass


class SystemRequest(object):
    def __init__(self, **kwargs):
        self._base_z = kwargs.get('base_z', 0)
        self._dHead = kwargs.get('dHead_z', 0)
        self._vHead = kwargs.get('vHead_z', 0)
        self._slope = kwargs.get('slope', 0)

    def __repr__(self):
        st = ''


"""
COMMANDS: 

CreateElbow
Connet + ConnectTo(connector1, connector2) -> pipe
Elbow + ConnectTo(connector1, connector2) -> elbow
CreateTee()
CreateTap(pipe, connector)
          ix     ix       
          
 pipe   conn1   conn2    
[ix    [ 0 ,    1     ] ]
[                       ]
[                       ]


"""


class Recipe(object):
    def trigger(self, node):
        return node

    def apply(self, edge):
        pass


class Pipe(Recipe):
    def apply(self, edge):
        line_ix = edge.get('order')
        next_sucs = edge.target.successors(edges=True)
        is_last = len(next_sucs) == 0
        sub_inds = [line_ix, 1]
        new_inds = []
        return sub_inds + new_inds


class Tap(Recipe):
    def apply(self, edge):
        """  """
        line_ix = edge.get('order')
        return geom, line_ix, syms




def finalize(root, prepare_fn, dtol=0.1):
    """
        Create List of geometry to be created in revit

    :param root: node
    :return:
        ENDPOINTS OF PIPES
        these are build first
        geom = [
            [x1, y1, z1, x2, y2, z2]

            [10, 5, 0,   10, 0, 0] # [0, [0 , 1]]
            [10, 0, 0,   5, 0, 0]  # [1, [0 , 1]]
            [5,  0, 0,   5, 5, 0]  # [2, [0 , 1]]
        ]

        INDICES OF CONNECTORS TO CONNECT
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
            suc_edge = node1.edge_to(node2)
            line_ix = suc_edge.get('order')
            next_sucs = node2.successors(edges=True)
            is_last = len(next_sucs) == 0

            # [ x1, y1, z1, x2, y2, z2 ]

            geom.append(prepare_fn(node1, node2, last=is_last))
            symbol_type = node2.get('$create', None)
            if not is_last:

                # [ mepcurve_index, end_index ]
                sub_inds = [line_ix, 1]
                new_inds = []
                has_similar = False
                for p_edge in next_sucs:
                    """
                    this is a hack for revit. 
                    if there are more than two connectors, then they have to be connected in order:
                        [conn1, conn2, branch], 

                    where conn1 and conn2 are on pipes with same direction, 
                    and branch has a different direction

                    """
                    if p_edge.similar_direction(suc_edge) is True:
                        has_similar = True
                        new_inds.insert(0, 0)
                        new_inds.insert(0, p_edge.get('order'))
                    else:
                        new_inds += [p_edge.get('order'), 0]

                if has_similar is True:
                    inds.append(sub_inds + new_inds)
                else:
                    inds.append(new_inds + sub_inds)

            if symbol_type is not None:

                # [ symbol_type, mepcurve_index, end_index]
                syms.append([symbol_type, line_ix, 1])
    return geom, inds, syms


class SystemProcessorV3(object):
    def __init__(self):
        self._shrink = -0.05
        self._z = 10.0

    def get_system(self, ky):
        return _sys_dict.get(ky)

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
        self._log_inputs(data, points)

        # build graph,  bake attributes
        system = SystemFactory.from_serialized_geom(data, sys=viper.SystemV3, root=tuple(points[0]))
        system.bake()

        syst_meta = SystemRequest(type=system_type, **kwargs)

        heuristic = self.get_system(system_type)()
        rl_engine = RuleEngine(term_rule=heuristic.root, mx=1e6, debug=False)
        fact_base = KB(heuristic.root)
        root_node = rl_engine.alg2(system.root, fact_base)

        # based on labels, finallize graph (adding new info)
        sysrender = RenderNodeSystem()
        root_node = sysrender.render(root_node)
        meta_data = rl_engine.annotate_type(root_node, heuristic.final_labels)

        # create build instructions
        geom, inds, syms = self.finalize(root_node)
        return {'geom': geom, 'indicies': inds, 'symbols': syms}

    def _prepare(self, start, end, last=False):
        """
        this is a revit hack. if the points of two curves are the same, the fitting
        becomes all messed up.

        if a node is the last one do not shrink that end as there are no more fittings
        :param start:
        :param end:
        :param last:
        :return:
        """
        crv = MepCurve2d(start.geom, end.geom)
        if last is False:
            p1, p2 = crv.extend(self._shrink, self._shrink).points
        else:
            p1, p2 = crv.extend(self._shrink, 0).points

        if math.isnan(p1[0]):
            print(start, end, p1, p2)
        vec = list(p1) + list(p2)
        return vec

    def finalize(self, root):
        return finalize(root, self._prepare)

