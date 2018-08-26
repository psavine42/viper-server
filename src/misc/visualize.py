import networkx as nx
import pylab
from matplotlib import pyplot
import lib.figures as F
from src.geomType import GeomType
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.mplot3d import Axes3D

_styles = {
    'tee': {'size': 100, 'color': 1},
    'branch': {'size': 50, 'color': 0.6},
    'elbow': {'size': 50, 'color': 0.3},
    'end': {'size': 150, 'color': 0.1},
    'source': {'size': 200, 'color': 0.8},
    GeomType.SYMBOL: {'size': 250, 'color':0.4 }
}


def plot(mls, ext=0.2):
    fig = pyplot.figure(1,  dpi=90)
    ax = fig.add_subplot(121)
    for a in list(mls):
        F.plot_line(ax, a, color=F.DARKGRAY)
        # F.plot_coords(ax, pt, color=F.RED)

    # for pt in sys.G.nodes:
    #    x, y = pt
    #     ax.plot(x, y, 'o', color=F.RED)

    ax.set_title('b) collection')
    pyplot.show()


def default_label_fn(node, data):
    return data.get('label', '') + ':' + str(node)


def pos_label_fn(node, data):
    sym = data.get('type', '')
    lbl = data.get('label', '')
    ord = data.get('order', '')
    return '{} {} : {} - {}'.format(sym, lbl, str(node), ord)


def all_props(node, data):
    return str(data)


def order_label(node, data):
    return data.get('order', 'x')


def get_keys(node, data, keys):
    st = ''
    if 'all' in keys:
        keys = data.keys()
    for k in keys:
        st += ' {}: {} '.format(k, data.get(k, ''))
    return st


def hcolor():
    _HEX = list('0123456789ABCDEF')
    return '#' + ''.join(_HEX[randint(0, len(_HEX)-1)] for _ in range(6))


def simple_plot(G, meta=_styles, label=True):
    pos = nx.spring_layout(G)
    labels = {}
    for (p, d) in G.nodes(data=True):
        labels[p] = p
    nx.draw(G, pos,
            labels=labels,
            with_labels=label,
            arrowsize=20,
            edge_cmap=pyplot.cm.Blues,
            font_size=10)
    pylab.show()


def dump_data(res_data):
    import time, json
    st = str(round(time.time(), 0))
    with open('./data/out/{}.json'.format(st), 'w') as F:
        F.write(json.dumps(res_data))


def print_iter(root):
    for node in root.__iter__():
        print(node)


def plot3d(root, meta=None):
    if meta:
        for k, v in meta.items():
            if isinstance(v['color'], float):
                v['color'] = hcolor()
    fig = plt.figure()
    fig.set_size_inches(24, 12)
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_zlim([-2, 2])
    ax.axis('off')
    for n in root.__iter__():
        t = n.get('type', None)
        if t:
            if meta:
                size = meta[t]['size']
                col = meta[t]['color']
            else:
                col = 'r'
            x, y, z = n.geom
            ax.plot([x], [y], [z], marker='o', color=col)

        for x in n.successors(edges=True):
            p1, p2 = x.geom

            x, y, z = zip(p1, p2)
            ax.plot(x, y, z, color='919191')
    plt.show()


def _plot(G, lbl_fn, edge_fn=None, meta=_styles, label=True):
    pos, colors, labels, sizes = {}, [], {}, []

    for p, d in G.nodes(data=True):
        pos[p] = list(p)[0:2]
        n_type = d.get('type', '')
        if n_type == '':
            n_type = d.get('label', '')

        colors.append(meta.get(n_type, {}).get('color', 0.45))
        sizes.append(meta.get(n_type, {}).get('size', 20))
        labels[p] = lbl_fn(p, d)

    edge_fn = edge_fn if edge_fn else {'order'}
    edge_labels = {}
    for k, nbrdict in G.adjacency():
        for to, d in nbrdict.items():
            edge_labels[(k, to)] = get_keys((k, to), d, edge_fn)

    nx.draw(G, pos,
            node_size=sizes,
            labels=labels,
            with_labels=label,
            arrowsize=20,
            edge_cmap=pyplot.cm.Blues,
            node_color=colors,
            font_size=10)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    pylab.show()


def gplot(G, **kwargs):
    _plot(G, default_label_fn, **kwargs)


def ord_plot(G, **kwargs):
    _plot(G, pos_label_fn, **kwargs)


class Plot(object):
    """
    mode:
    dim: 2d or 3d plot
    edge:
    node:
    """
    def __init__(self, **kwargs):
        self._mode = kwargs.get('mode', 2)
        self._dim = kwargs.get('dim', 2)
        self._edge_k = kwargs.get('edge', {})
        self._node_k = kwargs.get('node', {})

    def edge_labels(self, G):
        res = {}
        for k, nbrdict in G.adjacency():
            for to, d in nbrdict.items():
                st = ''
                for k in self._node_k:
                    st += ' {}: {} '.format(k, d.get(k, ''))
                res[(k, to)] = st
        return res

    def node_labels(self, G):
        res = {}
        for k, nbrdict in G.adjacency():
            for to, d in nbrdict.items():
                st = ''
                for k in self._node_k:
                    st += ' {}: {} '.format(k, d.get(k, ''))
                res[(k, to)] = st
        return res

    def plot(self, root):
        fig = plt.figure()
        fig.set_size_inches(24, 12)
        if self._dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlim([-2, 2])
            ax.axis('off')
            for n in root.__iter__():
                t = n.get('type', None)
                if t:
                    # size = meta[t]['size']
                    # col = meta[t]['color']
                    x, y, z = n.geom
                    ax.plot([x], [y], [z], marker='o', color='r')

                for x in n.successors(edges=True):
                    p1, p2 = x.geom

                    x, y, z = zip(p1, p2)
                    ax.plot(x, y, z, color='919191')
        if self._dim == 3:
            pass



class Plotter(object):
    def __init__(self):
        import numpy as np
        x = np.random.rand(15)
        y = np.random.rand(15)
        self.names = np.array(list("ABCDEFGHIJKLMNO"))

        norm = 0
        cmap = 0
        self.fig, self.ax = plt.subplots()
        self.sc = plt.scatter(x, y, c=c, s=100, cmap=cmap, norm=norm)

        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                      textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

    def update_annot(self, ind):

        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                               " ".join([self.names[n] for n in ind["ind"]]))
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        self.annot.get_bbox_patch().set_alpha(0.4)

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

