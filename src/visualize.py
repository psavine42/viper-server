import networkx as nx
import pylab
from matplotlib import pyplot
import lib.figures as F
from src.geomType import GeomType

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


def order_label(node, data):
    return data.get('order', 'x')


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


def _plot(G, lbl_fn, meta=_styles, label=True):
    pos, colors, labels, sizes = {}, [], {}, []
    for (p, d) in G.nodes(data=True):

        pos[p] = list(p)[0:2]
        n_type = d.get('type', '')
        if n_type == '':
            n_type = d.get('label', '')

        colors.append(meta.get(n_type, {}).get('color', 0.45))
        sizes.append(meta.get(n_type, {}).get('size', 20))
        labels[p] = lbl_fn(p, d)

    edge_labels = {}
    for k, nbrdict in G.adjacency():
        for to, d in nbrdict.items():
            edge_labels[(k, to)] = order_label((k, to), d)

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



