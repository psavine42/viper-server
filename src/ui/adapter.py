from lib.meshcat.src.meshcat import geometry, Visualizer


class ViewNode:
    def __init__(self, node):
        pass


class ViewEdge:
    def __init__(self, edge):
        origin = edge.geom[0]
        gm = geometry.Cylinder(len(edge), radius=0.2)



class ViewAdapter(Visualizer):
    pass



