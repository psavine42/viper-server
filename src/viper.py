from collections import defaultdict as ddict
from collections import Counter

from .misc import visualize
from .geomType import GeomType
from .geom import to_Nd

import tqdm, time
import numpy as np
import src.propogate as gp
import src.misc.utils as utils
from src.misc.utils import *

from shapely.geometry import Point
from shapely.ops import linemerge, nearest_points
import src.geom
from src.geom import MEPSolidLine, MeshSolid
from typing import List
import trimesh
import torch
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
        self._kdtree = None
        self._pttree = None
        self.adj = ddict(set)
        self.facets = ddict(set)
        if solids:
            self._build2(solids)

    def _prebuld_indexes(self, solids, mode='np'):
        _num_solids = len(solids)
        self._index_starts = [0]  # faces[0] starts at index 0

        for s in solids:
            # s.process()
            self._index_starts.append(self._index_starts[-1] + len(s.children))
            self._index_layers.append(s.layer)
            self._solid_centers.append(s.centroid)

        self._index_layers = np.asarray(self._index_layers)
        self._solid_centers = np.asarray(self._solid_centers)
        self._kdtree = spatial.KDTree(self._solid_centers)
        print('layers :{} \ncentroids: {} \nstarts: {} '.format(
            len(self._index_layers), len(self._solid_centers), len(self._index_starts)))
        return solids

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
        self.res = results
        return results

    def _build3(self, solids, n=7):
        self.res = ddict(set)
        self._prebuld_indexes(solids)
        print(n)
        for _layer in set(self._index_layers):

            ixs_on_layer = np.where(self._index_layers == _layer)[0]
            if len(ixs_on_layer) < n:
                continue

            layer_tree = spatial.KDTree(self._solid_centers[ixs_on_layer])
            meshes = [solids[i] for i in ixs_on_layer.tolist()]

            for i, solid in enumerate(meshes):

                this_gix = ixs_on_layer[i]
                this_xyz = self._solid_centers[this_gix]
                dist_mat, ixs = layer_tree.query([this_xyz], k=n)

                for ix in ixs[0].tolist():
                    other_gix = ixs_on_layer[ix]
                    if this_gix in self.res[other_gix] or ix == i:
                        continue
                    other = spatial.distance.cdist(solid.points, meshes[ix].points)
                    ixr, mins = np.where(other < 0.1)
                    if len(ixr) > 1:
                        self.res[other_gix].add(this_gix)
                        self.res[this_gix].add(other_gix)
        return self.res

    def _build_no_layer(self, solids, n=7):
        self.res = ddict(set)
        meshes = [self.prebox(x) for x in solids]
        _tree = spatial.KDTree([m.centroid for m in meshes if m])
        _treep = spatial.KDTree(np.concatenate([m.vertices for m in meshes if m]))

        for this_gix, solid in tqdm.tqdm(enumerate(meshes)):
            if solid is None:
                continue

            this_xyz = solid.centroid
            dist_mat, ixs = _tree.query([this_xyz], k=n)

            for other_gix in ixs[0].tolist():

                if this_gix in self.res[other_gix] \
                        or other_gix == this_gix \
                        or meshes[other_gix] is None:
                    continue
                other = spatial.distance.cdist(solid.points, meshes[other_gix].points)
                ixr, mins = np.where(other < 0.1)
                if len(ixr) > 1:
                    self.res[other_gix].add(this_gix)
                    self.res[this_gix].add(other_gix)
        return meshes

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
            for i in todo_local:
                this_gix = ixs_on_layer[i]
                this_xyz = self._solid_centers[this_gix]
                this_cord = tuple(this_xyz.tolist())

                ndict[this_cord] = Node(this_cord, item=solids[this_gix])

                dist_mat, ixs = self._kdtree.query([this_xyz], k=3)

                for othr_gix in ixs[0].tolist()[1:]:
                    othr_cord = tuple(self._solid_centers[othr_gix].tolist())
                    if othr_cord not in ndict:
                        ndict[othr_cord] = Node(othr_cord, item=solids[othr_gix])
                    ndict[this_cord].connect_to(ndict[othr_cord])

            results.append(ndict)

        self.res = results
        return results

    @classmethod
    def should_preprocess(cls, solid):
        return solid.is_watertight is False

    @classmethod
    def prebox(cls, solids):
        for s in solids:
            if cls.should_preprocess(s) is True:
                bx = s.as_box()
                if bx is not None:
                    yield bx

    @staticmethod
    def is_irregular(box_mesh, tolerance):
        mn = np.min(box_mesh.edges_unique_length)
        mx = np.max(box_mesh.edges_unique_length)
        irreg = mx / mn < tolerance

        return irreg

    @staticmethod
    def score(scores):
        ixs = []
        cnt = 0
        while np.any(scores) and cnt < 2:
            sc0 = np.argmin(scores[:, 1])
            sc1 = np.argmin(scores[:, 2])
            sc2 = np.argmin(scores[:, 3])
            rank = [sc0, sc1, sc2]
            ix = Counter(rank).most_common()[0][0]
            ixs.append(int(scores[ix][0]))
            scores = np.delete(scores, ix, 0)
            cnt += 1
        return ixs

    def intial_filter(self, solids, irreg_tol=8.0):
        return list(
            filter(lambda x: self.is_irregular(x, irreg_tol),
                    self.prebox(solids))
            )

    @staticmethod
    def nearest_facets__x(this, others):
        """
        nearest facets using all points of this_box and other_box facets

        :param this:
        :param others:
        :return:
        """
        arm = []
        for other in others:

            # [ n_verts(8), n_verts(8) ]
            vdist = spatial.distance.cdist(this.vertices, other.vertices)

            # [ n_facet(6), n_point(4) ]
            face_ix = np.asarray([np.unique(x) for x in this.faces[this.facets].reshape(6, -1)])

            # [ n_facet(6), n_point(4), dists(8) ]
            vtof = vdist[face_ix]

            # [ total_dists(6), ]  <- [ n_facet(6), dists(4) ]
            mins = np.sum(np.min(vtof, axis=-1), axis=-1)
            arm.append(int(np.argmin(mins)))
        return arm

    @staticmethod
    def nearest_facets(this, others):
        """
        nearest facets using all points of this_box and other_box facets

        :param this:
        :param others:
        :return:
        """
        arm = []
        for other in others:
            r, dists, tid = trimesh.proximity.closest_point(other, this.facets_centroids)
            armi = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            arm.append(armi[0])
        return arm

    @staticmethod
    def nearest_facets_o(this, others):
        """ neareste facets using centroids - """
        arm = []
        for other in others:
            dists = spatial.distance.cdist(this.facets_centroids, other.facets_centroids)
            armi = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
            arm.append(armi[0])
        return arm

    def predict_neighs(self, ix, this_box, k):
        Nvert = 8
        start_ix = ix * Nvert
        ixs_this_np = np.arange(start_ix, start_ix + len(this_box.vertices))

        # [ nvert, n_topk ]
        dists, ixs = self._kdtree.query(this_box.vertices, k=k * Nvert)
        mask = np.asarray([[True if j not in ixs_this_np else False for j in i] for i in ixs])

        ixs_other = ixs // Nvert
        best_count = Counter()

        # [ num_facets, n_points ] x 2
        best_dists, best_index = np.zeros((6, 4)), np.zeros((6, 4))

        for fct_ix, facet in enumerate(this_box.facets):
            facet_vert_ix = np.unique(this_box.faces[facet])
            smask = mask[facet_vert_ix]
            vixs_other = ixs_other[facet_vert_ix]

            dist_other = dists[facet_vert_ix]
            best_dists[fct_ix] = np.array([dist_other[i, smask[i, :]][0] for i in range(4)])
            best_index[fct_ix] = np.array([vixs_other[i, smask[i, :]][0] for i in range(4)])
            best_count += Counter(vixs_other[smask])

        scores, keys = [], []
        candidates = np.unique(best_index)
        mx = best_count.most_common()[0][-1]

        for ii, (k, v) in enumerate(best_count.most_common()):
            if k in candidates:
                keys.append(k)
                cc = best_dists[np.where(best_index == k)]
                score = [k, (1. - v / mx), cc.mean(), cc.min()]
                scores.append(score)

        return self.score(np.asarray(scores))

    def build_no_layer2(self, solids, num_neigh=2, irreg_tol=8.0):
        c0 = time.time()
        print('filtering: ', len(solids))
        boxes = self.intial_filter(solids, irreg_tol)
        c1 = time.time()
        print('filtered: ', len(boxes), c1 - c0)

        # make a tree
        self._kdtree = spatial.KDTree(np.concatenate([m.vertices for m in boxes if m]))
        print('tree built ... ', time.time() - c1)

        for i, solid in tqdm.tqdm(enumerate(boxes)):
            indices = self.predict_neighs(i, solid,  num_neigh)

            # update adjacency
            self.adj[i].update(indices)
            for ix in indices:
                self.adj[ix].add(i)

            # compute nearest facets
            self.facets[i].update(self.nearest_facets_o(
                solid, [boxes[x] for x in indices])
             )

        return boxes


def dist_point_to_facets(pt, imesh_obb):
    vert_face = imesh_obb.vertices[imesh_obb.faces[imesh_obb.facets]]
    fct_cntrs = vert_face.reshape(imesh_obb.facets.shape[0], -1, 3).mean(axis=1)
    closet_ix = np.argmin(spatial.distance.cdist([pt], fct_cntrs).squeeze())
    return closet_ix, fct_cntrs[closet_ix]


class SolidSystemLF(SolidSystem):
    def __init__(self, **kwargs):
        SolidSystem.__init__(self, **kwargs)
        self._rtree = utils.make_index()
        self._input = dict()
        self.adj = ddict(set)

    def build_no_layer2(self, compounds, num_neigh=2, irreg_tol=8.0):
        """
        :param compounds: list of compound objects
        :param num_neigh:
        :param irreg_tol:
        :return:
        """

        # self._input[cmp.id] = cmp
        for i, cmp in enumerate(compounds):
            mesh_data = next(cmp.children_of_type(GeomType['MESH']))
            self._rtree.insert(i, src.geom.to_bbox(mesh_data), obj=i)

        for i, cmp in enumerate(compounds):
            obj = next(cmp.children_of_type(GeomType['MESH']))
            bbx = src.geom.to_bbox(obj)

            for other in self._rtree.intersection(bbx, objects=True):
                self.adj[i].add(other.object)
                self.adj[other.object].add(i)
        return compounds



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
                # point exists in both components
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


