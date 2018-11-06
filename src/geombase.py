import numpy as np
from scipy import spatial
import itertools
import trimesh
import math
from collections import Counter


def primary_axis(box_mesh, facets_ix):
    """
    assume that this is a line based solid.

    :param box_mesh: MeshSolid
    :param facets_ix: np.array of facet indices

    Returns
    ---------
    np.array([2, 3]) or None.
        points at facets
    """
    num_idx = len(facets_ix)
    if num_idx == 2:
        f1, f2 = box_mesh.facets_normal[facets_ix]
        if np.allclose(f1 * -1, f2):
            return box_mesh.facets_centroids[facets_ix]
        else:
            # make center to each
            return None
    elif num_idx == 1:
        # best direct option is to go center to known, then get opposite centroid
        f1 = -1 * box_mesh.facets_normal[facets_ix]
        other_ix = np.argmin(spatial.distance.cdist(box_mesh.facets_normal, f1))
        return box_mesh.facets_centroids[np.asarray([other_ix, facets_ix[0]])]

    elif num_idx > 2:
        for comb in itertools.combinations(facets_ix.tolist(), r=2):
            res = primary_axis(box_mesh, np.asarray(comb))
            if res is not None:
                return res
    return None


def sphere_intersections(sources, targets):
    """

    spheres in format [x , y , z, r]

    sources: shape([n, 4])
    targets: shape([m, 4])

    batching of base code
    function Colliding(a: Circle, b: Circle) returns Boolean {
      Vector2 diff = a.center - b.center
      Float maxDistance = a.radius + b.radius
      return diff.magnitude <= maxDistance

    Returns:
    -----------
    shape([n, m]) of ndtype(bool)

    True if i_n intersects i_m
    """
    # [ n, 4 ] -> [ m, n, 4]
    m_sources = np.tile(sources, (targets.shape[0], 1, 1)).transpose((1, 0, 2))

    # [ m, 4 ] -> [ m, n, 4]
    m_targets = np.tile(targets, (sources.shape[0], 1, 1))

    # compute
    diff = m_sources[:, :, :3] - m_targets[:, :, :3]
    max_dists = m_sources[:, :, -1] + m_targets[:, :, -1]
    return np.sqrt(np.sum(diff ** 2, axis=-1)) <= max_dists


def to_bbox(obj):
    """ Takes an iTrimesh """
    mn = obj.vertices.min(axis=0).tolist()
    mx = obj.vertices.max(axis=0).tolist()
    return tuple(mn + mx)


# Closest by -------------------------------------
def closest_by_facet_area(mesh1, mesh2):
    """
    return facets with most similar area
    :param mesh1:
    :param mesh2:

    Return:
    -------------
    facet_msh1, facet_msh2
    """
    cv1 = mesh1.as_mesh().convex_hull.facets_area
    cv2 = mesh2.as_mesh().convex_hull.facets_area

    # [ m ] -> [ m, n ]
    m_sources = np.tile(cv1, (cv2.shape[0], 1))
    m_targets = np.tile(cv2, (cv1.shape[0], 1)).transpose((1, 0))
    diff = np.abs(m_targets - m_sources)
    armi = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
    return armi


def closest_by_corners(mesh1, mesh2):
    pass


def closest_by_proximityq(mesh1, mesh2):
    """
    others -> this

    closest     : (m,3) float, closest point on triangles for each point
    distance    : (m,)  float, distance
    triangle_id : (m,)  int, index of closest triangle for each point
    """
    closest, dists, tid = mesh1.nearest.on_surface(mesh2.vertices)
    rdist = np.round(dists, 3)
    print(np.round(closest, 3))
    print(rdist)
    print(tid)

    # armi = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    # figure out which of these make a face
    # print(mesh1.triangles )

    # print([i[0] for i in sorted(enumerate(-1*dists), key=lambda x:x[1])])
    return np.lexsort(( np.arange(0, dists.shape[0]), dists ))


def closest_by_obb_facet_centroids(mesh1, mesh2):
    """ pick 1 """
    dists = spatial.distance.cdist(mesh1.as_obb.facets_centers,
                                   mesh2.as_obb.facets_centers)
    armi = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    return armi, dists[armi]


def closest_by_tri_centroids(mesh1, mesh2):
    """ pick 1 """
    ds1 = mesh1.as_mesh().convex_hull
    ds2 = mesh2.as_mesh().convex_hull
    dists = spatial.distance.cdist(ds1.triangles_center,
                                   ds2.triangles_center)
    armi = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    print(dists[armi])

    print(ds1.facets)
    print(ds2.facets)

    return armi, dists[armi]


def by_tri_centroids(mesh1, meshes):
    """
    get nearest on surface points

    :param mesh1: current
    :param meshes:
    :return:
    """
    ds1 = mesh1.as_mesh().convex_hull
    ds2 = [x.as_mesh().convex_hull.vertices for x in meshes]
    lns = np.cumsum([len(x) for x in ds2])
    verts = np.concatenate(ds2)

    closest, dists, tid = mesh1.nearest.on_surface(verts)
    armi = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    print(dists[armi])

    print(ds1.facets)
    # print(ds2.facets)

    return closest, dists, tid


def by_tri_verts(mesh1, mes2, tol=0.001):
    """
    get nearest on surface points

    :param mesh1: current
    :param meshes:
    :return:
    """
    closest, dists, tid = mesh1.nearest.on_surface(mes2.vertices)
    keep_dist = np.argwhere(dists < tol)
    return closest[keep_dist], np.round(dists, 3)[keep_dist], tid[keep_dist]


# Temp Heuristics ------------------------------------------
def h_points_close_to_face(mesh1, mesh2, tol=0.001):
    """ this really is about how they connect if both connect"""
    cl, dist, tris = by_tri_verts(mesh1, mesh2, tol=tol)
    if len(dist) > 3:
        return cl, dist, tris
    return None


def heur_matching_facet_centroids(mesh1, mesh2, tol=1e-2):
    es = np.min(np.concatenate([mesh1.edges_unique_length,
                                mesh2.edges_unique_length]))
    inds, dists = closest_by_obb_facet_centroids(mesh1, mesh2)
    if dists * es < tol:

        return inds, dists
    return None


def connection_point_to_axis(this, other):
    """
    find revit's equivelant of connector.curve geometry
    vf

    :param this:
    :param other:
    :return:
    point: np.shape([3]), point on this.line
    dist:  float,         distances
    ix:    int            index of other.points which is closest
    """
    osk = other.as_obb.skeleton
    this_axes = this.as_obb.skeleton
    closest_res = osk.closest_points(this_axes.lines)
    while closest_res:
        min1 = np.argmin([x[1] for x in closest_res])
        point, d, ix = closest_res.pop(min1)
        if this.as_obb.contains([point.numpy])[0]:
            return point, d, ix


# classifier helpers -------------------------------------------
def is_cylinder(mesh_solid, tol=0.01):
    """
    difference between volume of box and cylinder should be
    factor of pi/4

    :param mesh_solid:
    :param tol:
    :return:
    """
    bbx_vol = mesh_solid.as_obb.volume
    cvx_vol = mesh_solid.convex_hull.volume
    dif_vol = np.abs(cvx_vol - bbx_vol * math.pi / 4) / cvx_vol
    if dif_vol < tol:
        return True
    return False


def as_cylinder(mesh_solid, tol=0.01):
    lines = mesh_solid.as_obb.skeleton.lines
    lens = [np.round(x.length, 2) for x in lines]
    cnt = Counter(lens)
    if len(cnt) < 3:
        # obb should have two sides roughly equal sides
        side = cnt.most_common()[0][0]
        h = cnt.most_common()[-1][0]
        cylinder_vol = h * math.pi * (side / 2) ** 2

        cvx_vol = mesh_solid.convex_hull.volume
        dif_vol = np.abs(cvx_vol - cylinder_vol) / cvx_vol
        if dif_vol < tol:
            return lines[lens.index(h)]

    return None


def is_cylinder_sides(mesh_solid, tol=0.01):
    """

    :param mesh_solid:
    :param tol:
    :return:
    """
    if as_cylinder(mesh_solid, tol=tol) is not None:
        return True
    return False


def is_rectangular(mesh_solid, tol=1e-3):
    bbx_vol = mesh_solid.as_obb.volume
    cvx_vol = mesh_solid.convex_hull.volume
    dif_vol = np.abs(cvx_vol - bbx_vol) / cvx_vol
    if dif_vol < tol:
        return True
    return False


# --------------------------------------------------
def remove_dangles_f(solid_mesh, facets_tol=2):
    """
    remove junky facets with small areas -
    these should be stored for latter use ...

    :param solid_mesh:
    :param facets_tol:
    :return:
    """
    solid_mesh.process()
    areas = solid_mesh.as_mesh().facets_area

    # take top facets
    counts, bins = np.histogram(areas, bins=facets_tol, range=(0, np.max(areas)))
    bin_ix = np.where(counts > 0)[0][0]

    valid_ix = np.where(solid_mesh.facets_area >= bins[bin_ix])[0]
    print(len(solid_mesh.facets_area), counts)

    valid_face = solid_mesh.facets[valid_ix]
    valid_face = np.concatenate([solid_mesh.faces[x] for x in valid_face])
    unf = np.unique(valid_face)
    verts = solid_mesh.vertices[unf]
    ixs = np.concatenate([np.where(unf == x)[0] for x in valid_face.reshape(-1)]).reshape(-1, 3)
    ns = solid_mesh.__class__(faces=ixs, vertices=verts, process=True, **solid_mesh.base_args)
    if len(ns.faces) > 1:
        return ns


def remove_dangles(solid_mesh, facets_tol=2, take_bin=0):
    """
    remove junky facets with small areas -
    these should be stored for latter use ...

    :param solid_mesh:
    :param facets_tol:
    :return:
    """
    solid_mesh.process()
    solid_mesh = solid_mesh.as_mesh()
    areas = solid_mesh.area_faces

    # bin all the faces - dangles will end up in lowest bins
    # bins: [0, 2, 0, 0, 3]
    counts, bins = np.histogram(areas, bins=facets_tol, range=(0, np.max(areas)))
    bin_ix = np.where(counts > 0)[0][0]

    valid_ix = np.where(areas >= bins[bin_ix])[0]
    # print(len(areas), counts)
    faces = solid_mesh.faces[valid_ix]
    unf = np.unique(faces)
    verts = solid_mesh.vertices[unf]

    ixs = np.concatenate([np.where(unf == x)[0] for x in faces.reshape(-1)]).reshape(-1, 3)
    # print(ixs)
    ns = solid_mesh.__class__(faces=ixs, vertices=verts, process=True, uid=solid_mesh.id)
    if len(ns.faces) > 1:
        return ns


def box_properties(box_mesh, axis_facet_ix):
    """
    at some point need to calculate info to create a line -

    start - end, as well as dimensions : l, w, h of box

    :param box_mesh:
    :param axis_ix:
    :return:
    """
    assert len(axis_facet_ix) == 2
    f1, f2 = box_mesh.facets_normal[axis_facet_ix]
    length = np.abs(f1 - f2)
    fct = box_mesh.facets[axis_facet_ix[0]]
    exmpl_verts = box_mesh.vertices[fct]
    box_dim_edge = np.unique(exmpl_verts[:, 0, :] - exmpl_verts[:, 1, :], axis=0)
    box_dims = (box_dim_edge ** 2).sum(axis=1) ** 0.5
    return {'length': length, 'wh': box_dims}

