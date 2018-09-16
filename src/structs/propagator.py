from .base import uid_fn


class Propagator(object):
    """
    The only thing we need to do to bless it as a propagator is to
     attach it as a neighbor to the cells whose contents affect it,
     and schedule it for the first time

    It is a nullary procedure which
        -takes from inputs
        -writes to the outputs

    from sussman et al:
    (define (propagator neighbors to-do)
        (for-each
            (lambda (cell)
                    (new-neighbor! cell to-do))
             (listify neighbors))
        (alert-propagators to-do))


    outputs <- fn( inputs )
    """
    __slots__ = ['_id', '_inputs', '_fn', '_output', '_cnt', '_profile']
    def __init__(self, inputs, output, fn, profile=False):
        """

        :param inputs: list of Cell
        :param output: Cell
        :param fn:     callable
        :param profile: optional argument for debugging
        """
        self._id = uid_fn()
        self._profile = profile
        self._inputs = inputs
        self._fn = fn
        self._output = output
        self._cnt = 0
        for input in inputs:
            input.new_neighbor(self)
            output.add_support(input)

    def _has_information(self):
        # todo - how does the partial information work ??
        return not any([x.is_empty for x in self._inputs])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            fn_eq = self._fn == other._fn
            out_eq = self._output == other._output
            #tr = True
            #for inp in self._inputs:
            #    inp in other._inputs
            return fn_eq and out_eq  # and
        return False

    def __call__(self):
        if self._profile is True:
            print(self, [(x.is_empty, str(x)) for x in self._inputs])

        # check that all cells have a value
        if self._has_information():
            self._cnt += 1

            # Apply function to input cells
            res = self._fn(*[x.value for x in self._inputs])
            if self._profile is True:
                print(self, res)

            # write result to output
            self._output.add_contents(res)

    def __str__(self):
        return 'Propogator: {}'.format(self._fn.__name__)

    # def __del__(self):
    #     for input in self._inputs:
    #         input.remove_neighbor(self)
    #         # self._output.remove_support(input)
    #     self._fn = None
        # self._output

    @property
    def arity(self):
        return len(self._inputs)

    @property
    def id(self):
        return self._id

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output


class CondPropagator(Propagator):
    def __init__(self, pred, output, if_true, if_false, fn, **kwargs):
        """

        predicate - one input cell (predicate)
        add to output from one of if-true or if-false cells


        todo - is 'predicate dispatch' of merge op part of arg classes
        or separate?


        :param pred:
        :param output:
        :param if_true:
        :param if_false:
        """
        super(CondPropagator, self).__init__([pred], output, fn, **kwargs)
        self._on_false = if_false
        self._on_true = if_true

    def __call__(self):
        _inp = self._inputs[0]
        if self._profile is True:
            print('1', self, self._output, self._inputs[0], _inp.is_empty)

        if not _inp.is_empty:
            res = self._fn(_inp.value)
            if self._profile is True:
                print(res)

            if res is True:

                self._output.add_contents(self._on_true.contents)
            else:
                self._output.add_contents(self._on_false.contents)

        if self._profile is True:
            print(self, self._output, self._on_true, self._on_false)

    @property
    def arity(self):
        return 3




class Steward(object):
    """
    in charge of maintaining structure of propogators when
    a node/edge is modified
    """
    def __init__(self, object):
        self._root = object
        self._keys = []

        self._deps = []     # mutable
        self._shared = []   # mutable
        for cell in object.cells.values():
            self.register(cell)

    def register(self, cell):
        return self._keys.append(cell.var)

    @property
    def deps(self):
        return self._deps

    @property
    def shared(self):
        return self._shared

    def _handle_dep(self, cell, allow=None):
        #if cell in self._root.cells:
        #     return True
        if cell not in self._deps and cell not in self._shared:
            if cell.support:
                roots = cell.support_roots
                if allow:
                    for a in allow:
                        if a in roots:
                            roots.remove(a)

                if self._root not in roots:
                    return False
                elif self._root in roots and len(roots) > 1:
                    self._shared.append(cell)
                    return True
                else:
                    self._deps.append(cell)
                    return True
            else:
                self._deps.append(cell)
                return True

    def compute_sources(self, allow_roots=[]):
        """
        walk the pgraph looking for cells that only have
        this steward's cells as dependencies

        if a cell has another dependant,
            - where in the pgraph does this need to connect to

        - Owned     - derives from this object
        - Derived   - derives only from this object

        - Shared    - derives from multiple owners
        - Not owned - owner is another object

        walk forward along the propagator call chain
        at each new cell, check its support property

        Args:

            allow_roots: these sources of provenance are ignored
        """
        self._deps = []
        self._shared = []
        q = []
        for k in self._keys:
            q += list(self._root.cells[k].neighbors)
        seen = set()
        while q:
            propogator = q.pop(0)
            if propogator.id not in seen:

                seen.add(propogator.id)
                out_cell = propogator.output
                _continue = self._handle_dep(out_cell, allow_roots)

                if _continue is True:
                    q.extend(out_cell.neighbors)

                for in_cell in propogator.inputs:
                    if in_cell.id not in seen:

                        # if cell in_cell.neighbors
                        _continue = self._handle_dep(in_cell, allow_roots)
                        if _continue is True:
                            q.extend(in_cell.neighbors)

        return self._deps

    def __getitem__(self, item):
        pass

    def __setitem__(self, key, value):
        pass

    def __str__(self):
        st = '{} for: {}  '.format(self.__class__.__name__ , self._root)
        st += ' Deps: {}, shared: {}'.format(len(self._deps), len(self._shared))
        return st


class Provenance(object):
    """ """
    def __init__(self, cell=None):
        self._deps =[]


    def call2(self, cell):
        q = []
        # for r in cell:
        q += list(cell.neighbors)
        seen = set()
        while q:
            el = q.pop(0)  # this is a propagator
            if el.id not in seen:
                seen.add(el.id)
                out = el.output
                self._deps.append(out)
                q.extend(out.neighbors)
                for n in el.inputs:

                    if n.id not in seen:
                        self._deps.append(n )
                        q.extend(n.neighbors)
        return self._deps

    def __call__(self, cell):
        q = [cell]
        # seen = set()
        while q:
            this_cell = q.pop()
            #if this_cell.id not in seen:
            self._deps.append(this_cell)
            for pred in this_cell.neighbors:
                q.append(pred)
        return self._deps


