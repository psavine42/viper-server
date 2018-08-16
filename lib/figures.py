from math import sqrt
from shapely import affinity
import networkx as nx
from random import random, randint
import pylab
import matplotlib.pyplot as plt

GM = (sqrt(5)-1.0)/2.0
W = 8.0
H = W*GM
SIZE = (W, H)

BLUE = '#6699cc'
GRAY = '#999999'
LIGHTGRAY = '#c4c4c4'
DARKGRAY = '#333333'
YELLOW = '#ffcc33'
GREEN = '#339933'
RED = '#ff3333'
BLACK = '#000000'

COLOR_ISVALID = {
    True: BLUE,
    False: RED,
}


def plot_line(ax, ob, color=GRAY, zorder=1, linewidth=3, alpha=1):
    x, y = ob.xy
    ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=zorder, alpha=alpha)


def plot_coords(ax, ob, color=GRAY, zorder=1, alpha=1):
    x, y = ob.xy
    ax.plot(x, y, 'o', color=color, zorder=zorder, alpha=alpha)


def color_isvalid(ob, valid=BLUE, invalid=RED):
    if ob.is_valid:
        return valid
    else:
        return invalid


def color_issimple(ob, simple=BLUE, complex=YELLOW):
    if ob.is_simple:
        return simple
    else:
        return complex


def plot_line_isvalid(ax, ob, **kwargs):
    kwargs["color"] = color_isvalid(ob)
    plot_line(ax, ob, **kwargs)


def plot_line_issimple(ax, ob, **kwargs):
    kwargs["color"] = color_issimple(ob)
    plot_line(ax, ob, **kwargs)


def plot_bounds(ax, ob, zorder=1, alpha=1):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color=BLACK, zorder=zorder, alpha=alpha)


def add_origin(ax, geom, origin):
    x, y = xy = affinity.interpret_origin(geom, origin, 2)
    ax.plot(x, y, 'o', color=GRAY, zorder=1)
    ax.annotate(str(xy), xy=xy, ha='center',
                textcoords='offset points', xytext=(0, 8))


def set_limits(ax, x0,  y0, xN,  yN):
    ax.set_xlim(x0, xN)
    ax.set_xticks(range(x0, xN+1))
    ax.set_ylim(y0, yN)
    ax.set_yticks(range(y0, yN+1))
    ax.set_aspect("equal")


####################################################################
_styles = {'IfcWall': {'size': 200, 'col': 1},
           'IfcDoor': {'size': 50, 'col': 0.5},
           'IfcWindow': {'size': 50, 'col': 0.5},
           'IfcSlab': {'size': 30, 'col': 0.5},
           'point': {'size': 20, },
           'edge': {'size': 40, },
           'face': {'size': 80, }
          }

_args = {'font_size': 5}


def random_styles(types):
    return {typ: {'size': randint(10, 200), 'color': random()} for typ in types}


def draw_save(G, meta=None, label=True, dr='out', font=10):
    pos = nx.spring_layout(G)
    style_map = _styles if meta is None else meta

    colors, labels, sizes = [], {}, []
    for (p, d) in G.nodes(data=True):
        n_type = d.get('type', '')
        colors.append(style_map.get(n_type, {}).get('color', 0.45))
        sizes.append(style_map.get(n_type, {}).get('size', 20))
        labels[p] = n_type + ':' + str(p)

    nx.draw(G, pos, node_size=sizes, labels=labels, with_labels=label, arrowsize=40,
            edge_cmap=plt.cm.Blues, node_color=colors, font_size=font)

    pylab.show()





