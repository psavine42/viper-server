from collections import defaultdict as ddict

from .misc import visualize
from .geomType import GeomType
from .geom import to_Nd

import src.propogate as gp
from src.misc.utils import *

from shapely.geometry import Point
from shapely.ops import linemerge, nearest_points


class System(object):
    _ROUND = 0

    def __init__(self, segments=[], symbols=[], root=None, **kwargs):
        self._node_dict = dict()
        self._root = None
        self._symbols = {round_tup(s.points[0], self._ROUND): s for s in symbols}
        self._keep_layers = []
        if segments:
            self._build(segments, root)

    def _filter_keep(self, lines, pt):
        if pt is None:
            pt = Point(0, 0, 0)
        elif isinstance(pt, tuple):
            pt = Point(*list(pt))

        root, dist = None, 1e10
        for crv in lines:
            if pt.distance(crv) < dist:
                root, dist = crv, pt.distance(crv)

        self._keep_layers.append(root.layer)
        fls = []
        for l in lines:
            if l.layer == root.layer or l.type == GeomType.SYMBOL:
                fls.append(l)
        p1, p2 = [Point(x) for x in root.points]
        root_pt = tuple(list(p1.coords if pt.distance(p1) < pt.distance(p2) else p2.coords)[0])
        return fls, root_pt

    def _get_intersects(self, mep_curves, pt=None, syms=None):
        mls_ob = linemerge(mep_curves)
        root, _ = nearest_points(mls_ob, Point(pt))
        root = to_Nd(root, 3)

        merged = list(mls_ob)
        print('building', len(merged))

        dists = ddict(dict)
        acomps = []
        for i in range(len(merged)):
            ndict = dict()
            cords = list(merged[i].coords)
            for k in range(len(cords)):
                pnt = cords[k]
                if pnt not in ndict:
                    # todo something better with symbols
                    if round_tup(pnt, self._ROUND) in self._symbols:
                        ds = self._symbols[round_tup(pnt, self._ROUND)]
                        if 'children' in ds._opts:
                            ds._opts.pop('children')

                        ndict[pnt] = Node(pnt, **ds.to_dict)
                    else:
                        ndict[pnt] = Node(pnt)
                if k > 0:
                    ndict[cords[k - 1]].connect_to(ndict[pnt])
            acomps.append(ndict)
            for j in range(i+1, len(merged)):
                dist = merged[i].distance(merged[j])
                dists[i][j] = dist
                dists[j][i] = dist

        print('indexes created')
        acomps = self._build_network(acomps, merged, dists)
        # todo really?
        for comp in acomps:
            if comp is not None:
                for k, v in comp.items():
                    if v.geom == root:
                        self._root = v
                        break

    @staticmethod
    def _build_network(comps, merged, dists):
        q = [0]

        while q and len(dists) != 1 and len(comps) != 1:
            i = q.pop()
            mk, mv = min(dists[i].items(), key=lambda x: x[1])

            this_comp = comps[i]
            othr_comp = comps[mk]

            # compute nearest point
            pi, po = [to_Nd(x) for x in nearest_points(merged[i], merged[mk])]
            in_this = pi in this_comp
            in_othr = po in othr_comp

            if in_this and in_othr:
                this_comp[pi].connect_to(othr_comp[po])

            elif in_this and not in_othr:
                if pi != po:
                    new_node = Node(po)
                    pa = gp.PointAdder(new_node)
                    pa(list(othr_comp.values())[0])
                    this_comp[pi].connect_to(new_node)
                else:
                    pa = gp.PointAdder(this_comp[pi])
                    pa(list(othr_comp.values())[0])

            elif not in_this and in_othr:
                if pi != po:
                    new_node = Node(pi)
                    pa = gp.PointAdder(new_node)
                    pa((list(this_comp.values())[0]))
                    othr_comp[po].connect_to(new_node)
                else:
                    pa = gp.PointAdder(othr_comp[po])
                    pa(list(this_comp.values())[0])

            else:
                print('ERR')

            for k, nd in othr_comp.items():   # move nodes to current comp
                this_comp[k] = nd

            merged[i] = merged[i].union(merged[mk]) # update geometry
            merged[mk] = None                       # remove previous
            comps[mk] = None                        # remove previous

            for k, v in dists[mk].items():
                if k in dists[i]:
                    min_dist = min([dists[i][k], v])
                    dists[i][k] = min_dist
                dists[k].pop(mk)
            dists.pop(mk)

            if len(dists) == 1:
                break

            nextk, bestm = None, 1e10
            for k, v in dists.items():
                mk, mv = min(v.items(), key=lambda x: x[1])
                if mv < bestm:
                    nextk = k
                    bestm = mv
                if mv == 0:
                    break
            q.append(nextk)
        return comps

    def _build(self, segments=[], root=None):
        segments, root_pt = self._filter_keep(segments, root)
        print(root_pt)
        self._get_intersects(segments, pt=root_pt)

    def gplot(self, **kwargs):
        G = nodes_to_nx(self._root, **kwargs)
        visualize.gplot(G)

    def aplot(self, **kwargs):
        G = sys_to_nx(self)
        for s in self._symbols.keys():
            G.add_node(s, type='symbol')
        visualize.gplot(G, **kwargs)

    @staticmethod
    def recipe():
        return gp.Chain(
            gp.EdgeRouter(),        # clean up
            gp.EdgeDirector(),      # compute direction
            gp.GraphTrim(),         #
            gp.DistanceFromSource(),
            gp.DistanceFromEnd(),   #
            gp.DirectionWriter(),   #
            gp.Cluster()            # remove suspicious clusters
        )

    @property
    def root(self):
        return self._root

    def bake(self):
        print('init bake', self._root)
        self.recipe()(self._root)
        print('base baked')
        return self


