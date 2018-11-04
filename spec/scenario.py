import unittest
import pickle, json
import os
import time
import numpy as np
import trimesh
from rectpack import newPacker
from src.geom import CadInterface
_base_path = '/home/psavine/source/viper/data/scenario/'


def to_structure(segs, use_syms=False):
    solids = []
    for x in segs:
        for j in CadInterface.factory(use_syms=use_syms, **x):
            if j:
                solids.append(j)
    return solids


def load_from_json(pth):
    with open(pth, 'r') as f:
        segs = json.load(f)
        f.close()
    return segs


def make_scenario(system, indicies, path=None, **kwargs):
    """

    :param system:
    :param indicies: to save in scenario
    :param path:
    :param kwargs:
    :return:
    """
    scene = [system.inputs[i] for i in indicies]
    print(os.path.dirname('.'))
    # processing
    pts = np.concatenate([x.vertices for x in scene])
    xform = trimesh.Trimesh(vertices=pts).bounding_box_oriented.primitive.transform
    xform = np.linalg.inv(xform)
    for mesh in scene:
        mesh = mesh.as_mesh()        # set base data
        mesh.apply_transform(xform)  # move to 0 coords
        mesh.reset()                 # clear rep cache

    new = system.__class__()
    new(scene)
    if path is None:
        path = str(time.time()).split('.')[0]

    with open(_base_path + path + '.pkl', 'wb') as f:
        pickle.dump(new, f)

    # if there are targets - write them to match this file


def load_scenarios(todo=None):
    exist = os.listdir(_base_path)
    if todo is None:
        todo = exist

    for p in exist:
        print(p, todo)
        if p not in todo:
            continue

        with open(_base_path + p, "rb") as f:
            res = pickle.load(f)
            f.close()
            yield res


def load_and_run(fn, todo=None, **kwargs):
    """
    layout https://github.com/secnot/rectpack
    newPacker([, mode][, bin_algo][, pack_algo][, sort_algo][, rotation])
    :param fn:
    :param todo:
    :param kwargs:
    :return:
    """

    for scene in load_scenarios(todo=todo):
        fn(scene, **kwargs)


class Scenario(object):
    def __init__(self, fn, **kwargs):
        self.fn = fn

    def __call__(self, system, **kwargs):
        self.fn(system, **kwargs)


