import scipy.spatial.distance as distance
from scipy.spatial import kdtree
import numpy as np

"""
So many distances 
"""


def lines_distance(lines, r=2.):
    """
    2 x 3
    :param lines:
    :return:
    """
    tree = kdtree.KDTree(np.concatenate(lines))
    tree.query_pairs(r)
