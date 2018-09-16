from src.geomType import GeomType
from .opers import *
import operator as OP
from src.structs import aotp as A
from src.structs.aotp import *
from src import Cell, Node



class EngineHeurFP(object):
    def __init__(self):

        is_source = IF('npred', OP.eq, 0, var='source')
        one_pred = IF('npred', OP.eq, 1)
        mutex(is_source, one_pred)

        is_endpt = IF('nsucs', OP.eq, 0)
        one_succ = IF('nsucs', OP.eq, 1)
        two_sucs = IF('nsucs', OP.eq, 2)
        mutex(is_endpt, one_succ, two_sucs)

        no_sym = NOT(HAS('type'))
        has_sym = HAS('type')
        mutex(no_sym, has_sym)

        is_sym = AND(has_sym, IF('type', OP.eq, GeomType.SYMBOL))
        is_arc = AND(has_sym, IF('type', OP.eq, GeomType.ARC))
        is_circle = AND(has_sym, IF('type', OP.eq, GeomType.CIRCLE))
        mutex(is_circle, is_arc, no_sym, is_sym)

        symbolic = OR(is_circle, is_arc, is_sym, var='symbol-like')

        is_split = AND(one_pred, two_sucs, var='split')
        vbranch = AND(is_split, symbolic, var='vbranch')
        hbranch = AND(is_split, NOT(symbolic), var='hbranch')
        mutex(vbranch, hbranch)

        isElbow = AND(one_pred, one_succ, NOT(symbolic), var='elbow')
        UpElbow = AND(one_pred, one_succ, symbolic, var='elbow+rise')
        mutex(isElbow, UpElbow)

        # isUpright = AND(one_succ, symbolic, var='vHead')
        isDrop = AND(is_endpt, symbolic, var='dHead')
        other_end = AND(is_endpt, NOT(symbolic), var='unlabeled_end')
        mutex(isDrop, other_end)

        rt = Mutex(is_source, other_end, UpElbow, isDrop, vbranch, hbranch, isElbow, var='solved')
        self.root = rt

    @property
    def final_labels(self):
        return ['IsRiser', 'dHead', 'elbow',  'unlabeled_end', 'elbow+rise',
                'vbranch', 'vHead', 'source', 'hbranch']


def _hrs_supported_heads(*input_contents):
    """

    :param input_contents:
    :return:
    """
    cnt = 0
    for contents in input_contents:
        if contents == '$dnhead':
            cnt += 1
        elif isinstance(contents, int):
            cnt += contents
    return cnt


def _hrs_edge_branch(*input_contents):
    return



def NodeAndEdgeProp(node, node_fn=None, edge_fn=None, **kwargs):
    """
    add propogator given node

        -over node + node.edges(fwd ?) + node.edges(bkwd?)

    :param node:
    :param edge_fn:
    :param kwargs: fwd=
    :return:
    """
    use_edges = True if callable(edge_fn) else False
    for obj in node.neighors(edges=use_edges, **kwargs):
        pass




class HeuristicFP(object):
    result_key = '$result'
    sup_name = '$GT'

    def __init__(self, classes, keys):
        self._labels = classes
        self._label_cells = A.discrete_cells(self.labels, support=self.sup_name)
        self._counters = A.discrete_cells([0, 1, 2], support=self.sup_name)
        self._keys = keys
        self.T = Cell(True, var='TRUTH', support=self.sup_name)
        self.F = Cell(False, var='FALSE' , support=self.sup_name)

    @property
    def labels(self):
        return self._labels

    def _propagate_system_level(self, node):
        """

        for system seperate into logical branches

            branch = 1 or more features with only one 'path' to source.

            (s) start of 'path' must be a 'branch' edge

            cells to be generated for each of these conditions


                    +-+--o
            |      /  |
            +--s--+   o
            |     +------o

        :param node:
        :return:
        """
        pass

    @staticmethod
    def propagate_edges(node):
        """
        need to compute the following:
            - for each edge :
                - angle to source
                - order of edges (edges around a node are linked list?)
                -

        :param node:
        :return:
        """
        preds = node.predecessors(edges=True)
        if not preds:
            return

        pdir = preds[0].get_cell(var='in_direction')
        pdir.add_contents(preds[0].direction)

        for i, edge in enumerate(node.successors(edges=True)):
            edge_dir = edge.get_cell(var='direction')
            angle_to = edge.get_cell(var='angle_to')
            p_angle(edge_dir, pdir, angle_to)
            edge_dir.add_contents(edge.direction)

    @staticmethod
    def reducer(root, src_var, tgt_var, f_of_nodes):
        """
        lets say we have some nodes with '$dnHead' in '$result' cell
        and want to compute:
        for all nodes:
            count(node.successors, P'$dnHead' in '$result' )

        we construct a pattern operation with 'key'

        example:

        def node_fn(input_contents):
            cnt = 0
            for contents in input_contents:
                if contents == '$uphead':
                    cnt += 1
                elif isinstance(contents, int):
                    cnt += contents
            return cnt

        add_pattern(node, '$result', '$num_heads', node_fn)

        case 1:
            cell2, cell3 = cell1.neigh
            res <--(p)-- f_of_nodes(cell1.cells[src_var], cell2.cells[tgt_var], cell3.cells[tgt_var])
            cell1.cells[tgt_var] <--(p)-- res

        case 2:
            cell2.cells[tgt_var] <--(p)-- cell2.cells[src_var]
            cell2.cells[tgt_var] <--(p)-- cell2.cells[src_var]
            cell3.cells[tgt_var] <--(p)-- cell3.cells[src_var]

            cell1.cells[tgt_var] <--(p)-- f_of_nodes(cell2.cells[tgt_var], cell3.cells[tgt_var])

        :param root: (node)
        :param src_var: (str)
        :param tgt_var: (str)
        :param f_of_nodes: (callable)
        """
        for node in root.__iter__():
            c_output = node.get_cell(tgt_var)
            c_f_of_this = node.get_cell(src_var)

            # add source + targets of previous cells
            inputs = [c_f_of_this]
            inputs += [s.get_cell(tgt_var) for s in node.successors()]

            # create propagator
            Propagator(inputs, c_output, f_of_nodes)

    def _propagate_local(self, *cells):
        """

        :param cells: cells for a node
        """
        npred, nsucc, symbol, result = cells       # l0

        # l1
        no_pred, one_pred = p_for_classes(npred, self._counters[0:2])
        no_succ, one_succ, two_succ = p_for_classes(nsucc, self._counters)

        # on_sym = Cell(True)
        has_sym, fc = conditional(symbol, self.T, lambda xs: xs is not False, false=self.F)
        not_sym = prop_fn_to(OP.not_, has_sym)

        # l2 endpoints
        c_end_unk = prop_fn_to(OP.and_, no_succ, not_sym)
        c_endhead = prop_fn_to(OP.and_, no_succ, has_sym)

        # l2 one in one outs - l3
        c_one_one = prop_fn_to(OP.and_, one_pred, one_succ)
        c_elbow_s = prop_fn_to(OP.and_, c_one_one, not_sym)
        c_up_head = prop_fn_to(OP.and_, c_one_one, has_sym)

        # l2 splits - l3
        c_spliter = prop_fn_to(OP.and_, one_pred, two_succ)
        c_split_h = prop_fn_to(OP.and_, c_spliter, not_sym)
        c_split_v = prop_fn_to(OP.and_, c_spliter, has_sym)

        # l4
        class_cells = \
            [no_pred, c_end_unk, c_endhead, c_split_h, c_split_v, c_elbow_s, c_up_head]
        linear_classifier(class_cells, self._label_cells, result)

    def _create_cells(self, node):
        cells = []
        for k, v in self._keys:
            c = Cell(var=k, support=node)
            cells.append(c)
            node.add_cell(c)
        return cells

    def __call__(self, node):
        # todo - this is awkward- cell needs an event handler
        cells = self._create_cells(node)
        self._propagate_local(*cells)
        for k, v in self._keys:
            pd = node.get(k, v)
            node.write(k, pd)
        self.reducer(node, '$result', '$num_heads', _hrs_supported_heads)
        self.propagate_edges(node)

    def __str__(self):
        st = self.__class__.__name__ + str(*self.labels)
        return st


def organize_features(node):
    from src.structs.propagator import Steward
    st = Steward(node)
    st.compute_sources(['$GT'])
    for d in st.deps:
        pass




class Prediction(object):
    def __init__(self, new_location):
        self._possible = new_location

    def copy_cells_to(self, source, tgt):
        # tgt.steward

        return

    def fill_properties_down(self, cells):


        return

    def compute_slimilarity(self, nd1, nd2):
        for dep in nd1.deps:
            pass
        return nd1

    def __call__(self, new_node):

        best, bsim = None, -1e7
        for nd in new_node.__iter__():
            sim = self.compute_slimilarity(nd, new_node)
            if sim > bsim:
                bsim = sim
                best = nd.id

        best_node = new_node[best]
        self.copy_cells_to(best_node, new_node)
        self.fill_properties_down(new_node)
        



