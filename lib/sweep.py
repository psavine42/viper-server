
import os
import string
from random import randint
import networkx as nx


def _get_names(n):
    """Generator for [n] number of names of lines."""
    from itertools import product

    def f():
        m = 1
        while True:
            it = product(*tuple([string.ascii_uppercase]*m))
            for c in it:
                yield ''.join(c)
            m += 1

    for _, c in zip(range(n), f()):
        yield c


_MAX_RIGHT = 10


def generate(n):
    leftmost = _MAX_RIGHT
    lines = []
    names = []
    for name in _get_names(n):
        width = randint(1, _MAX_RIGHT)
        left = randint(0, _MAX_RIGHT - width)
        right = left + width
        lines.append((name, left, right))
        leftmost = min(leftmost, left)
        names.append(name)
    return lines, names


class SweepLine:
    """Class for generating random lines, and performing sweepline."""

    def __init__(self, lines, names):
        """Generate [n] random lines, and initialize internal structures."""

        # from graphing import Graph

        self.lines = lines
        self.remaining_events = []

        leftmost = _MAX_RIGHT

        for i, (name, left, right) in enumerate(self.lines):
            self.lines[i] = (name, left-leftmost, right-leftmost)

        for i, (name, left, right) in enumerate(self.lines):
            self.remaining_events.append((left, i))
            self.remaining_events.append((right, i))

        self.remaining_events.sort()

        self.active_line_segments = []
        self.sweep_line = None

        self.is_done = False
        self.idx = 0
        self.a_line = None

        self.overlap_graph = nx.Graph(names)
        # self.interval_graph = nx.Graph(names)

    def _rem(self):
        for x, n in self.remaining_events:
            yield x

    def next(self):
        """Generate next image, or if done, generate pdf."""
        if self.is_done:
            return

        self.idx += 1

        if self.sweep_line is not None:
            self.remaining_events = self.remaining_events[1:]

        if len(self.remaining_events) == 0:  # End of everything
            if self.sweep_line is None:
                self.is_done = True
                return
            else:
                self.sweep_line = None
                return

        self.sweep_line, self.a_line = self.remaining_events[0]

        if self.sweep_line == self.lines[self.a_line][1]:  # left
            current_n = self.lines[self.a_line][0]
            current_r = self.lines[self.a_line][2]

            for i in self.active_line_segments:
                n, _, r = self.lines[i]
                if r < current_r:
                    self.overlap_graph.add_edge(n, current_n)
                # self.interval_graph.add_edge(n, current_n)

            self.active_line_segments.append(self.a_line)

        elif self.sweep_line == self.lines[self.a_line][2]:  # right
            self.active_line_segments.remove(self.a_line)

    def run(self):
        while not self.is_done:
            self.next()
