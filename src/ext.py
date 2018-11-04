import os
import platform
from distutils.spawn import find_executable
from trimesh.interfaces.generic import MeshScript
from multiprocessing.dummy import Pool as ThreadPool

_search_path = os.environ['PATH']
_vhacd_executable = None

for _name in ['vhacd', 'testVHACD']:
    _vhacd_executable = find_executable(_name, path=_search_path)
    if _vhacd_executable is not None:
        break

exists = _vhacd_executable is not None


def batch_convex_decomposition(meshes, **kwargs):
    if not exists:
        raise ValueError('No vhacd available!')
    _mesh_names = []
    argstring = ' --input $mesh_0 --output $mesh_post --log $script'

    # pass through extra arguments from the input dictionary
    for key, value in kwargs.items():
        argstring += ' --{} {}'.format(str(key),
                                       str(value))

    with MeshScript(meshes=meshes,
                    script='',
                    tmpfile_ext='obj') as vhacd:
        result = vhacd.run(_vhacd_executable + argstring)
    return result



def pdecompose(meshes, nthread=4):
    """
    decompose meshes in parralel
    # todo - feed them to scrip in batches ?
    :param meshes:
    :param nthread:
    :return:
    """
    pool = ThreadPool(nthread)
    results = pool.map(lambda x : x.convex_components, meshes)
    pool.close()
    pool.join()
    return results

