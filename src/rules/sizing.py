"""

an approximation of sizing using some fn of fixtures
"""

from ..structs import Node, Cell


class FlowResolver:
    def __init__(self, source_annos, valid_sizes):
        self._sources = source_annos
        self._sizes = valid_sizes

    def compute_system_flow(self, root):
        """
        for each pipe, compute how many fixtures it supports
        then do some sizing function based on this - normalize
        to the valid sized - basically some mapping

        :param root:
        :return:
        """

        return root




def apply_cells(root, cell_type):
    pass




