from rtree import index
import time

def index_2d():
    p = index.Property()
    p.dimension = 2
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    st = './data/index{}'.format(time.time())
    idx3d = index.Index(st, properties=p)
    return idx3d
