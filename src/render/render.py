from . import render_propogators as rp
from ..propogate import propagators
import numpy as np

_global = {
    'drop_head_z': 9,
    'vert_head_z': 11,
    'drop_head_offset': 0,
    'vert_head_offset': 0,
    'base_z': 10,
    'level_height': 13,

    'branch_offset': 0.5,    # if 0, remove
    'slope': 0,
    'system_type': 0,

}


class RenderNodeSystem(object):
    def __init__(self, **kwargs):
        """

        -give final spec for system
        -build should be in order.
        -should accomodate revit transactions

        :param G:
        """
        self.index = {}
        self._dict = kwargs
        self._shrink = kwargs.get('shrink', -0.25)
        self._base_z = self.get_arg('base_z')
        self._level_z = self.get_arg('level_height')

        self._branch_ofs = self.get_arg('branch_offset') # - self._base_z
        self._v_head_ofs = self.get_arg('vert_head_z') # - self._base_z
        self._d_head_ofs = self.get_arg('drop_head_z') # - self._base_z

    def get_arg(self, key):
        if key in self._dict:
            return self._dict.get(key, None)
        else:
            return _global.get(key, 0)

    def render(self, system_root):
        rp.Translate()(system_root, data=np.array([0., 0., self._base_z]))
        render = propagators.Chain(
            rp.RedrawPropogator('source', fn=rp.vertical, geom=self._level_z),
            rp.RedrawPropogator('elbow+rise', fn=rp.riser_fn, geom=self._branch_ofs),
            rp.RedrawPropogator('vbranch', fn=rp.vbranch, geom=self._branch_ofs),
            rp.RedrawPropogator('dHead', fn=rp.drop_fn, geom=self._d_head_ofs),
            rp.Annotator('$create', mapping={'dHead': 1, }),
            propagators.BuildOrder(),
            )

        render(system_root)
        #if  self.get_arg('slope' ) > 0:
        #    rp.AddSlope()(system_root, slope=self.get_arg('slope'))

        return system_root

    def __call__(self, system):
        return self.render(system)

    def __str__(self):
        st = '<{}>'.format(self.__class__.__name__)
        st += 'head {} , branch {} '.format(self._d_head_ofs, self._branch_ofs)
        return st


