import networkx as nx
import pylab
from matplotlib import pyplot
import lib.figures as F
from src.geomType import GeomType
import matplotlib.pyplot as plt
from random import randint
from src.structs import Node, Cell
from mpl_toolkits.mplot3d import axes3d, Axes3D

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
        st += '\n {}: {} '.format(k, data.get(k, ''))
    return st


def hcolor():
    _HEX = list('0123456789ABCDEF')
    return '#' + ''.join(_HEX[randint(0, len(_HEX)-1)] for _ in range(6))


def simple_plot(G, kys, meta=_styles, label=True):
    pos = nx.spring_layout(G)

    colors, labels, sizes =  [], {}, []
    for (p, d) in G.nodes(data=True):
        labels[p] = get_keys(p, d, kys)
        n_type = d.get('type', None)
        colors.append(meta.get(n_type, {}).get('color', 0.45))
        sizes.append(meta.get(n_type, {}).get('size', 20))
    nx.draw(G, pos,
            labels=labels,
            with_labels=label,
            arrowsize=20,
            node_size=sizes,
            node_color=colors,
            edge_cmap=pyplot.cm.Blues,
            font_size=10)
    pylab.show()


def prop_plot(G,  meta=_styles, label=True, pos=None):
    posx = pos if pos is not None else nx.spring_layout(G)
    # print(pos)
    colors, labels, sizes =  [], {}, []
    for (p, d) in G.nodes(data=True):
        n_type = d.get('type', None)

        if n_type == 'cell' :
            if d.get('var', None) is not None and  'IN_'  in d.get('var', ''):
                n_type = 'cell+input'

            elif d.get('var', '') == 'res':
                n_type = 'cell+res'

            elif d.get('content', None) is not None:
                n_type = 'cell+content'

        tkys = meta.get(n_type, {})['keys']
        labels[p] = get_keys(p, d, tkys)
        colors.append(meta.get(n_type, {}).get('color', 0.45))
        sizes.append(meta.get(n_type, {}).get('size', 20) )

    nx.draw(G, posx,
            labels=labels,
            with_labels=label,
            arrowsize=20,
            node_size=sizes,
            node_color=colors,
            edge_cmap=pyplot.cm.Blues,
            font_size=11)
    pylab.show()


def dump_data(res_data):
    import time, json
    st = str(round(time.time(), 0))
    with open('./data/out/{}.json'.format(st), 'w') as F:
        F.write(json.dumps(res_data))


def print_iter(root):
    for node in root.__iter__():
        print(node)


def _3darr(arr, colors=None):
    fig = plt.figure()
    fig.set_size_inches(24, 12)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], marker='o', color=colors)
    plt.show()


def plot3d(root, meta=None):
    if meta:
        for k, v in meta.items():
            if isinstance(v['color'], float):
                v['color'] = hcolor()
    fig = plt.figure()
    fig.set_size_inches(24, 12)
    ax = fig.add_subplot(111, projection='3d')
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
        labels[p] = get_keys(p, d, lbl_fn)

    if edge_fn:
        edge_labels = {}
        for k, nbrdict in G.adjacency():
            for to, d in nbrdict.items():
                edge_labels[(k, to)] = get_keys((k, to), d, edge_fn)
    else:
        edge_labels = None

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
    _plot(G, {'label'}, **kwargs)


def ord_plot(G, **kwargs):
    _plot(G, pos_label_fn, **kwargs)


class Plot(object):
    """
    mode:
    dim: 2d or 3d plot
    edge:
    node:
    """
    def __init__(self, *data, **kwargs):
        self._mode = kwargs.get('mode', 2)
        self._dim = kwargs.get('dim', 2)
        self._pos_fn = kwargs.get('pos', None)
        self._edge_k = kwargs.get('edge', {})
        self._node_k = kwargs.get('node', {})

        self._node_type_k = kwargs.get('ntype', None)

        self._meta = kwargs.get('meta', {})
        self._font = kwargs.get('font', 10)
        self._arrow = kwargs.get('font', 20)
        self._plot_fn = None
        if data:
            self.plot(data)

    def _plot_(self, G, pos, sizes, labels, colors):
        nx.draw(G, pos,
                node_size=sizes,
                labels=labels,
                with_labels=True,
                arrowsize=self._arrow,
                edge_cmap=pyplot.cm.Blues,
                node_color=colors,
                font_size=self._font)

    def _label(self, n, d, label_fn):
        if callable(label_fn):
            pass
        elif type(label_fn) in [set, list]:
            pass

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

    def __pos_from_node_coord(self, x):
        return list(x)[0:2]

    def __pos_geom(self, x):
        x, y, z = x.geom
        return [x], [y], [z]

    def plot(self, g):
        self.__validate_meta()

        if isinstance(g, nx.DiGraph):
            if self._pos_fn is None:
                node = next(g.nbunch_iter())
                if isinstance(node, tuple) and len(node) in [2, 3]:
                    self._pos_fn = self.__pos_from_node_coord

            if self._dim == 3:
                self._plot_fn = None
            elif self._dim == 2:
                self._plot_2dG(g)

        elif isinstance(g, Node):

            if self._dim == 3:
                self.plot3d_nodes(g)
            elif self._dim == 2:
                pass

        # elif isinstance(g, Cell):

        pass

    def plot2d(self):
        pass

    def _get_meta_val(self, k):
        if k not in self._meta:
            v = {}
            v['color'] = hcolor()
            v['size'] = randint(30, 200)
            self._meta[k] = v
            return v['size'], v['color']
        else:
            v = self._meta[k]
            return v['size'], v['color']

    def __validate_meta(self):
        for k, v in self._meta.items():
            if 'color' not in v or isinstance(v['color'], float):
                v['color'] = hcolor()
            if 'size' not in v:
                v['size'] = randint(30, 200)

    def _plot_2dG(self, G):
        pos, colors, labels, sizes = {}, [], {}, []

        for p, d in G.nodes(data=True):
            pos[p] = self._pos_fn(p)
            n_type = self._node_type_k(d)
            size, color = self._get_meta_val(n_type)
            colors.append(size)
            sizes.append(color)
            labels[p] = get_keys(p, d, self._node_k)

        edge_labels = {}
        for k, nbrdict in G.adjacency():
            for to, d in nbrdict.items():
                edge_labels[(k, to)] = get_keys((k, to), d, self._edge_k)

        self._plot_(G, pos, sizes, labels, colors)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    def plot3d_nodes(self, root):

        fig = plt.figure()
        fig.set_size_inches(24, 12)
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        for n in root.__iter__():
            t = n.get('type', None)
            if t:
                size, col = self._get_meta_val(t)

                x, y, z = n.geom
                ax.plot([x], [y], [z], marker='o', color=col)

            for x in n.successors(edges=True):
                p1, p2 = x.geom

                x, y, z = zip(p1, p2)
                ax.plot(x, y, z, color='919191')
        plt.show()



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

