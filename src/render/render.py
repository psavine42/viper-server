from src import walking
from src.geom import MepCurve2d, FamilySymbol, add_coord
from . import render_propogators as P
from ..propogate import propagators



_global = {
    'drop_head_z':9,
    'vert_head_z':11,
    'drop_head_offset':0,
    'vert_head_offset':0,
    'base_z':10,

    'branch_z': 11,    # if 0, remove
    'slope':0,
    'system_type': 0,

}


class RenderSystem(object):
    def __init__(self, **kwargs):
        """

        -give final spec for system
        -build should be in order.
        -should accomodate revit transactions

        :param G:
        """
        # self.G = DiG
        self.index = {}
        self._dict = kwargs
        self._shrink = kwargs.get('shrink', -0.25)
        self._z = self.get_arg('base_z')
        self._branch_ofs = self._z - self.get_arg('branch_z')
        self._vhead_ofs = self._z - self.get_arg('vert_head_z')
        self._dhead_ofs = self._z - self.get_arg('drop_head_z')

    def get_arg(self, key):
        if key in self._dict:
            return self._dict.get(key, None)
        else:
            return _global.get(key, 0)

    def render_drop(self, system, prev, head):
        """
          +--+
          |  |
        - +  |
        """
        top = add_coord(prev, z=self._branch_ofs)
        top2 = add_coord(head, z=self._branch_ofs)
        drop = add_coord(head, z=self._dhead_ofs)

        print(top, top2, drop)
        system.G.remove_node(head)
        system.G.add_edges_from([(prev, top), (top, top2), (top2, drop)])
        # return system

    def render_upright(self, system, el):
        top = add_coord(el, z=self._branch_ofs)
        system.G.add_node(top)
        system.G.add_edge(el, top)
        # return system

    def render_VBranch(self, system, branch_edge):
        def prop_fn(el, pred, sucs, seen):
            new = add_coord(el, z=self._branch_ofs)
            system.edit_node(el, new)

        start, end = branch_edge

        top = add_coord(start, z=self._branch_ofs)
        system.G.add_node(top)
        system.G.add_edge(start, top)
        system.G.add_edge(top, end)
        walking.walk_dfs_forward(system.G, end, prop_fn)
        # return system

    def render(self, system):
        G = system.G

        def render_fn(p1, p2, preds, sucs, seen):
            data1 = G.nodes[p1]
            data2 = G.nodes[p2]
            own_data = G.get_edge_data(p1, p2)
            label = data2.get('label', None)
            own_label = data2.get('label', None)
            if label == 'dhead':
                self.render_drop(system, p1, p2)
            elif own_label == 'vbranch':
                # self.render_VBranch(system, (p2, ))
                pass

        walking.walk_edges(system.G, system.root, render_fn)
        return system

    def __call__(self, system):
        return self.render(system)


class RenderNodeSystem(object):
    def __init__(self, **kwargs):
        """

        -give final spec for system
        -build should be in order.
        -should accomodate revit transactions

        :param G:
        """
        # self.G = DiG
        self.index = {}
        self._dict = kwargs
        self._shrink = kwargs.get('shrink', -0.25)
        self._base_z = self.get_arg('base_z')
        self._branch_ofs = self.get_arg('branch_z') - self._base_z
        self._v_head_ofs = self.get_arg('vert_head_z') - self._base_z
        self._d_head_ofs = self.get_arg('drop_head_z') - self._base_z

    def get_arg(self, key):
        if key in self._dict:
            return self._dict.get(key, None)
        else:
            return _global.get(key, 0)

    def render(self, system_root):
        render = propagators.Chain(
            P.RedrawPropogator('elbow+rise', fn=P.riser_fn, geom=self._branch_ofs),
            P.RedrawPropogator('vbranch', fn=P.vbranch, geom=self._branch_ofs),
            P.RedrawPropogator('dHead', fn=P.drop_fn, geom=self._d_head_ofs),
            propagators.BuildOrder(),
            )
        render(system_root)
        return system_root

    def __call__(self, system):
        return self.render(system)

    def __str__(self):
        st = '<{}>'.format(self.__class__.__name__)
        st += 'head {} , branch {} '.format(self._d_head_ofs, self._branch_ofs)
        return st


