from collections import defaultdict as ddict
from collections import Counter

from .misc import visualize
from .geomType import GeomType
from .geom import to_Nd

import numpy as np
import src.propogate as gp
from src.misc.utils import *

from shapely.geometry import Point
from shapely.ops import linemerge, nearest_points
from typing import List
import torch
from src.geom import MEPSolidLine
from scipy import spatial


class SolidSystem(object):
    _ROUND = 0
    """
    Solid System has solids, so first must infer lines from solids 
    """
    def __init__(self, solids=[], symbols=[], root=None, **kwargs):
        self._root = None
        self._symbols = {round_tup(s.points[0], self._ROUND): s for s in symbols}
        self._keep_layers = []
        self._solid_centers = []
        self._face_centers = None
        self._index_starts = []
        self._index_layers = []
        self._tree = None
        if solids:
            self.res = self._build2(solids)

    def _prebuld_indexes(self, solids: List[MEPSolidLine], mode='np') -> None:
        # _face_centers = []
        self._index_starts = [0]  # faces[0] starts at index 0
        _num_solids = len(solids)

        for s in solids:
            self._index_starts.append(self._index_starts[-1] + len(s.children))
            self._index_layers.append(s.layer)
            self._solid_centers.append(s.centroid)
        if mode == 'np':
            self._index_layers = np.asarray(self._index_layers)
            self._solid_centers = np.asarray(self._solid_centers)
            self._tree = spatial.KDTree(self._solid_centers)
            # self._face_centers = np.asarray(_face_centers)
        else:
            self._solid_centers = torch.Tensor(self._solid_centers)
            # self._face_centers = torch.Tensor(_face_centers)
        print('layers :{} \ncentroids: {} \nstarts: {} '.format(
            len(self._index_layers), len(self._solid_centers), len(self._index_starts)))

    def _build(self, solids: List[MEPSolidLine]):
        """
            returns connectivity matrix to turn into line graph

            [index_of_solid : [ face_to_connect,  index_of_other_face ] ]

            :param solids:
            :return:
        """
        self._prebuld_indexes(solids)
        results = []
        _num_solids = len(solids)
        _indexes = set(range(0, self._face_centers.size(0)))

        for i in range(_num_solids):

            start_of_this = self._index_starts[i]
            end_of_this = self._index_starts[i+1]
            n_of_this = end_of_this - start_of_this
            range_this = set(range(start_of_this, end_of_this))

            tensr_this = torch.LongTensor(list(_indexes.difference(range_this)))
            best_dist_solid = 1e9
            best_indx_solid = [None, None]

            # todo implement as tensors?
            # todo dynamic hopping ceonnections cannot be on adjacent faces of a geom ??

            for j in range_this:

                min_dist = torch.index_select(self._face_centers, 0, tensr_this) - self._face_centers[j]
                min_dist = torch.sum(min_dist ** 2, dim=1) ** 0.5
                min_dist, nearest_ix = min_dist.topk(1, largest=False)
                min_dist, nearest_ix = min_dist.item(), nearest_ix.item()

                if min_dist < best_dist_solid:
                    best_dist_solid = min_dist
                    re_near = nearest_ix + n_of_this if nearest_ix > j else nearest_ix
                    best_indx_solid = [j, re_near]
            results.append(best_indx_solid)
        return results

    def _build2(self, solids: List[MEPSolidLine]):
        """
        returns connectivity matrix to turn into line graph

        [index_of_solid : [ face_to_connect,  index_of_other_face ] ]

        :param solids:
        :return:
        """
        results = []
        self._prebuld_indexes(solids)
        print(Counter(self._index_layers.tolist()))
        for _layer in set(self._index_layers):

            ixs_on_layer = np.where(self._index_layers == _layer)[0]
            if len(ixs_on_layer) < 3:
                continue

            ndict = {}
            todo_local = list(range(len(ixs_on_layer)))
            # lyr_centrs = self._solid_centers[ixs_on_layer]

            for i in todo_local:
                this_gix = ixs_on_layer[i]
                this_xyz = self._solid_centers[this_gix]
                this_cord = tuple(this_xyz.tolist())
                # dist_mat = spatial.distance.cdist(lyr_centrs, [this_xyz]).flatten()
                # othr_lix = np.argpartition(dist_mat, 2, axis=-1)[1]
                # othr_gix = ixs_on_layer[othr_lix]

                # othr_cord = tuple(self._solid_centers[othr_gix].tolist())
                ndict[this_cord] = Node(this_cord, item=solids[this_gix])

                dist_mat, ixs = self._tree.query([this_xyz], k=3)
                # print(ixs)

                for othr_gix in ixs[0].tolist()[1:]:
                    othr_cord = tuple(self._solid_centers[othr_gix ].tolist())
                    if othr_cord not in ndict:
                        ndict[othr_cord] = Node(othr_cord, item=solids[othr_gix])
                    ndict[this_cord].connect_to(ndict[othr_cord])

                # ndict[this_cord] = Node(this_cord, item=solids[this_gix])

            results.append(ndict)
        return results


class System(object):
    _ROUND = 0

    def __init__(self, segments=[], symbols=[], root=None, **kwargs):
        self._root = None
        self._symbols = {round_tup(s.points[0], self._ROUND): s for s in symbols}
        self._keep_layers = []
        if segments:
            self._build(segments, root)
        return

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

        merged = list(mls_ob)   # List of MultiLineStrings merged by shapely
        dists = ddict(dict)     # distances of [NodeSystem to NodeSystem]
        acomps = []             # holds NodeSystem equivalent of LineString

        for i in range(len(merged)):
            ndict = dict()      # holds {(x,y,z) : Node(x,y, z) }
            cords = list(merged[i].coords)

            # iterate through each point on line making nodes along the way
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
                    # connect k_th point in line to k_th + 1
                    ndict[cords[k - 1]].connect_to(ndict[pnt])

            acomps.append(ndict)
            # create index of distances between
            for j in range(i+1, len(merged)):
                dist = merged[i].distance(merged[j])
                dists[i][j] = dist
                dists[j][i] = dist

        print('indexes created ... ')
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


