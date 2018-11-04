from collections import defaultdict as ddict
from collections import Counter

from .misc import visualize
from .geomType import GeomType
from .geom import to_Nd
import lib.geo

import tqdm, time
import numpy as np
import src.propogate as gp
import src.misc.utils as utils
from src.misc.utils import *

from shapely.geometry import Point
from shapely.ops import linemerge, nearest_points
import shapely.geometry as sg
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

    def build_no_layer2(self, solids, num_neigh=2, irreg_tol=8.0, filt=True):
        c0 = time.time()
        if filt is True:
            print('filtering: ', len(solids))
            solids = self.intial_filter(solids, irreg_tol)
        c1 = time.time()
        print('filtered: ', len(solids), c1 - c0)

        # make a tree
        self._kdtree = spatial.KDTree(np.concatenate([m.vertices for m in solids if m]))
        print('tree built ... ', time.time() - c1)

        for i, solid in tqdm.tqdm(enumerate(solids)):
            indices = self.predict_neighs(i, solid,  num_neigh)

            # update adjacency
            for ix in indices:
                self.adj[ix].add(i)
                self.adj[i].add(ix)

            # compute nearest facets
            self.facets[i].update(self.nearest_facets_o(
                solid, [solids[x] for x in indices])
             )

        return solids

from scipy.spatial.distance import cdist
class IndexSystem(object):
    def __init__(self):
        self._inputs = None
        self._spheres = []
        self._bbxes = []
        self._start_ixs = None
        self.adj = None
        self.conn = ddict(set)
        self.G = ddict(dict)
        self._intersects = []
        # self.inters = []

    @classmethod
    def rebuild(cls, other):
        new = IndexSystem()
        new._inputs = other._inputs
        new._spheres = other._spheres
        new._bbxes = other._bbxes
        new._start_ixs = other._start_ixs
        new.adj = other.adj
        new.conn = other.conn
        new.G = other.G
        new._intersects = [] # other._intersects
        return new

    def __len__(self):
        return len(self._inputs)

    @property
    def spheres(self):
        return self._spheres

    @property
    def inputs(self):
        return self._inputs

    @property
    def sphere_centers(self):
        return self._spheres[:, :3]

    def sphere_indexes_for(self, ix):
        if ix == 0:
            return 0, self._start_ixs[ix]
        else:
            return self._start_ixs[ix - 1], self._start_ixs[ix]

    def spheres_for(self, ix):
        """ sphere corresponding to input_ix """
        start, end = self.sphere_indexes_for(ix)
        return self._spheres[start:end]

    def sphere_to_input(self, sphere_ix):
        """ input corresponding to sphere_ix """
        return np.where(self._start_ixs > sphere_ix)[0][0]

    def add_to_index(self, ix, tsphs, ois, osphs):
        for tsph, oi, osph in zip(tsphs, ois, osphs):
            if tsph not in self.G[ix]:
                self.G[ix][tsph] = ddict(set)
            self.G[ix][tsph][oi].add(osph)

    def remove_candidate(self, ix, tsph, candidate):
        if tsph in self.G[ix] and candidate in self.G[ix][tsph]:
            self.G[ix][tsph].pop(candidate)

    def resolve_3(self, this_ix, sphere_end, it, io, so, debug=False):
        """ ray from this.sphere_origin, to other1.sphere_inters
          does it intersect other2 ?

        determine the case if a ray starts from this.nearest_point(that)
         if it intersects others, before that, it should not be used
         (that is in between this and other)

         assumes sphere_end has one best solution, ie it is a real connector
         to some other
          """
        nexts = []
        best_d = 1e10
        this_sphere = self.spheres_for(this_ix)[sphere_end][:3]

        # candidates are spheres with index of sphere_end in their arr
        candidates = io[np.where(it == sphere_end)]
        other_sphr = so[np.where(it == sphere_end)]
        if len(np.unique(candidates)) <= 1:
            return it, io, so

        if debug is True:
            print('candidates:', candidates)
        for cand_ix in candidates:

            # local_inds can represent several candidates which are not cand_ix
            local_inds = np.argwhere(candidates != cand_ix).squeeze(axis=-1)
            cand_sphrs = other_sphr[local_inds]
            origin_pts = np.tile(this_sphere, (cand_sphrs.shape[0], 1))
            target_pts = self.sphere_centers[cand_sphrs]

            #
            inters = self._inputs[cand_ix].convex_hull.ray.intersects_location(
                origin_pts,
                (target_pts - origin_pts) / cdist(origin_pts, target_pts)
            )
            if debug is True:
                print('ray from {} to {}, passes thru {} ?-- {} '.format(
                    this_ix, self.sphere_to_input(cand_sphrs), cand_ix, inters[0]
                ))
            # print(inters[1])
            if len(inters[1]) > 0:
                # remove candidates with an intersection through another
                # dist is distance from origin_sphere to each intersection
                dist = np.argmin(cdist([this_sphere], inters[0]), axis=-1)
                print(cand_ix, cdist([this_sphere], inters[0]),  dist)
                nexts += candidates[local_inds][inters[1]].tolist()

        for n in set(nexts):
            # do removals
            keep = np.argwhere(io != n).squeeze(axis=-1)
            it, so, io = it[keep], so[keep], io[keep]

        return it, io, so

    def resolve_2(self, this_ix, sphere_end, it, io, so, debug=False):
        score, best = [1e10], None
        this_sphere = self.spheres_for(this_ix)[sphere_end][:3]
        use_ixs = np.argwhere(it == sphere_end).squeeze(axis=-1)
        candidates = io[use_ixs]
        for cand_ix in candidates:
            """
            closest     : (m,3) float, closest point on triangles for each point
            distance    : (m,)  float, distance
            triangle_id : (m,)  int, index of closest triangle for each point
            
            candidate surface vs this_sphere
            """
            inters = self._inputs[cand_ix].convex_hull.nearest.on_surface([this_sphere])
            this_dist = np.round(inters[1][0], 3)
            if this_dist < score:
                best, score = [cand_ix], this_dist
            elif this_dist == score:
                best.append(cand_ix)

        remove = set(candidates).difference(best)
        if remove:
            keep = np.concatenate([np.argwhere(io != x).squeeze(axis=-1) for x in remove])
            it, so, io = it[keep], so[keep], io[keep]
        return it, io, so

    class AdjResult(object):
        def __init__(self, parent, ix, it, io, so, done=False):
            self.parent = parent
            self.input_index = ix
            self.sphere_ixs = it
            self.neighs = io
            self.so = so
            self.is_done = done
            # self._others_connect

        # ------------------------------------
        @staticmethod
        def _elim_eq(unique, kvs):
            return [k for k, v in kvs.items() if v == unique]

        def parrallels(self):
            """
            check for objects that appeaer to touch all spheres
            only applies to objects with more than 3 spheres ...
            """
            if len(self) < 3:
                return []
            unique = set(self.sphere_ixs)
            res = [k for k, v in self.neigh_to_sphere.items() if v == unique]
            return res

        def elim_not_unique(self):
            if len(self) < 2:
                return []
            st = np.unique(np.stack([self.sphere_ixs, self.neighs]), axis=1)
            unique = set(st[0])
            kvs = self._as_graph(st[1], st[0])
            res = [k for k, v in kvs.items() if v == unique]
            return res

        def elim_by_mesh_fn(self, fn):
            for n in np.unique(self.neighs):
                res = fn(self.base, self.parent.inputsp[n])

        def eliminate_ixs(self, remove_ixs):
            all_ixs = set(range(0, len(self.sphere_ixs)))
            keep = np.array(list(all_ixs.difference(remove_ixs)))
            return self.__class__(
                self.parent, self.input_index, self.sphere_ixs[keep],
                self.neighs[keep], self.so[keep], done=self.is_done,
            )

        def eliminate_neigh(self, neigh):
            remove_this = np.argwhere(self.neighs == neigh).squeeze(-1)
            return self.eliminate_ixs(remove_this)

        def __repr__(self):
            st = str(self.input_index) + '\n'
            st += str(np.stack([self.sphere_ixs, self.neighs]))
            return st

        @staticmethod
        def _as_graph(kys, vls):
            res = ddict(set)
            for sph, neigh in zip(kys, vls):
                res[sph].add(neigh)
            return res

        @property
        def sphere_to_neigh(self):
            return self._as_graph(self.sphere_ixs, self.neighs)

        @property
        def neigh_to_sphere(self):
            return self._as_graph(self.neighs, self.sphere_ixs)

        @property
        def unique_neigh(self):
            return np.unique(self.neighs)

        @property
        def base(self):
            return self.parent._inputs[self.input_index]

        def __len__(self):
            return self.parent.spheres_for(self.input_index).shape[0]

        def apply_heuristics(self, hs):
            res = []
            for h in hs:
                res += h(self)
            return res

        @property
        def only_connection(self):
            g = self.neigh_to_sphere
            if len(g) == 1:
                return list(g.keys)[0]
            return None

        @property
        def can_use_obb(self):
            base = self.base
            cvv = base.as_mesh().convex_hull.volume
            bvv = base.as_obb.volume
            if (bvv - cvv) / cvv > 0.01:
                return False
            return True

        def nodes(self, other):
            """

            :param other:
            :return:
            """
            # Node()
            return

    def prune2(self, index1, index2):
        self.inters[index1] = self.inters[index1].eliminate_neigh(index2)
        self.inters[index2] = self.inters[index2].eliminate_neigh(index1)

    @property
    def inters(self):
        return self._intersects

    def create_connectivity1(self):
        # reset
        self._intersects = [None] * len(self._bbxes)
        for this_ix, bbx in enumerate(self._bbxes):
            it, io, so = self.intersections_for(this_ix)
            self._intersects[this_ix] = self.AdjResult(self, this_ix, it, io, so)

    def apply_heuristics(self, heurs, commit=False):
        res = {str(h): 0 for h in heurs}
        out = []
        # for h in heurs:
        #     prs = h(adv)
        #     res[str(h)] += len(prs)
        #     out += prs
        #     for p in prs:
        #         self.prune2(i, p)


    def create_connectivity(self):
        self._intersects = [None] * len(self._bbxes) # clear this for now
        before, after = [], []
        for this_ix, bbx in enumerate(self._bbxes):
            it, io, so = self.intersections_for(this_ix)
            before.append(len(np.unique(io)))
            start, end = self.sphere_indexes_for(this_ix)
            if start == end - 1:
                it, io, so = self.resolve_2(this_ix, 0, it, io, so)
            else:
                it, io, so = self.resolve_2(this_ix, 0, it, io, so)
                it, io, so = self.resolve_2(this_ix, end - start - 1, it, io, so)
            self._intersects.append(self.AdjResult(self, this_ix, it, io, so))
            after.append(len(np.unique(io)))
        print('before {}, after {}'.format(sum(before), sum(after)))

    def intersections_for(self, ix):
        """
        ix : (int) input index

        how to store this
        this:others

        sphere_this : sphere_others

        this : this_end : other : other_end


        Returns:
        ---------
        index of this sphere np.array(num_candidates)
        index of other input np.array(num_candidates)
        index of other sphere np.array(num_candidates)

        """
        start, end = self.sphere_indexes_for(ix)
        this_ixr = self.adj[start:end]

        sphere_ix, inters_ix = np.where(this_ixr == True)
        inters_inputs = np.array([self.sphere_to_input(x) for x in inters_ix])

        # filter intersections for input object's spheres
        not_self_inters = np.where(inters_inputs != ix)
        other_input_ix = inters_inputs[not_self_inters]
        other_spher_ix = inters_ix[not_self_inters]
        own_spheres_ix = sphere_ix[not_self_inters]

        return own_spheres_ix, other_input_ix, other_spher_ix

    def cand_dist(self, in_ix, candidate_ixs):
        if len(self.conn[in_ix]) > 2:
            return
        best, dist = None, 1e10

        for other in candidate_ixs:
            if other in self.conn[in_ix]:
                continue

            d = self._inputs[in_ix].as_obb.skeleton_distance(
                self._inputs[other].as_obb
            )
            if d < dist:
                best, dist = other, d

        if best is not None:
            self.conn[in_ix].add(best)
            self.conn[best].add(in_ix)
            return best

    def ix_by_end_point_sets(self, in_ix, this_ix, other_ix):
        sphere_to_others = ddict(set)
        others_to_sphere = ddict(set)
        for i, o in enumerate(other_ix):
            sphere_to_others[this_ix[i]].add(o)
            others_to_sphere[o].add(this_ix[i])

        n_spheres = self.spheres_for(in_ix).shape[0]
        if n_spheres > 1:
            IndexSystem.cand_dist(self, in_ix, sphere_to_others[0])
            IndexSystem.cand_dist(self, in_ix, sphere_to_others[n_spheres - 1])

        elif n_spheres == 1:
            IndexSystem.cand_dist(self, in_ix, sphere_to_others[0])
            IndexSystem.cand_dist(self, in_ix, sphere_to_others[0])
        else:
            print('no spheres! ',  in_ix)

    def __call__(self, solids, **kwargs):
        n_spheres = np.zeros(len(solids))
        self._inputs = solids
        for i, s in enumerate(solids):
            bbx = s.as_obb
            spheres = bbx.as_spheres()
            n_spheres[i] = spheres.shape[0]
            self._bbxes.append(bbx)
            self._spheres.append(spheres)
        self._spheres = np.concatenate(self._spheres)
        self._start_ixs = np.cumsum(n_spheres).astype(int)

        print('spheres shape: ', self._spheres.shape)
        self.adj = src.geom.sphere_intersections(self._spheres, self._spheres)

        return self.adj


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
        self._as_lines = []
        self.adj = ddict(set)

    @staticmethod
    def asline(smp):
        xg = []
        for e in smp.points[smp.edges_unique]:
            p1, p2 = e.tolist()
            xg.append(sg.LineString([p1, p2]))
        return sg.MultiLineString(xg)

    def build_no_layer2(self, compounds, **kwargs):
        """
        :param compounds: list of compound objects
        :param num_neigh:
        :param irreg_tol:
        :return:
        """
        self._input = compounds
        self._as_lines = [self.asline(x) for x in compounds]
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

    def line_distance(self, ix):
        this_l = self._as_lines[ix]
        dists = np.asarray([[this_l.distance(self._as_lines[i]), i]
                            for i in self.adj[ix] if i != ix])
        return np.sort(dists, axis=1)[0:2, -1].astype(int).tolist()



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


