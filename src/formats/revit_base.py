from src.structs import Node, Edge


def is_complete(graph_obj):
    if isinstance(graph_obj, Node) and graph_obj.get('built', None) is True:
        return True
    elif isinstance(graph_obj, Edge) \
            and graph_obj.get('built', None) is True \
            and graph_obj.get('conn1', None) is True \
            and graph_obj.get('conn2', None) is True:
        return True
    return False


class IStartegy(object):
    """

    q: list of
    """
    base_val = None

    def __init__(self, parent,  **kwargs):
        """ parent : ICommandManager for a node. """
        self._parent = parent
        self._built = False
        self._succeeded = False
        self._failed = False
        self._q = []
        self._pos = 0

    # convinience ------------------------------------------
    @property
    def is_done(self):
        return self._built is True and \
               (self._succeeded is True or self._failed is True)

    @property
    def parent(self):
        return self._parent

    @property
    def obj(self):
        return self._parent.gobj

    @property
    def is_built(self):
        return self._built

    @property
    def cmd_base(self):
        return [self.base_val, self.obj.id]

    @property
    def is_initializing(self):
        return self.is_built is True \
               and self._succeeded is False \
               and self._failed is False

    # public interface --------------------------------------
    def success(self, gobj, action):
        """
        IF the command in revit does not return a FAIL or EXCEPTION

        """
        yield None

    def fail(self, gobj, action, msg):
        """
        todo add a failure message argument to this interface
        """
        yield None

    def action(self, gobj, **kwargs):
        """
        generate the primary actions for this object
        these must be superclasses, or lists.
        Lists are 'raw' instructions, while superclass instructions
        are stored in the queue and can generate their own conditional
        instuctions.
        """
        yield None

    # interface --------------------------------------
    def __repr__(self):
        st = 'Strategy:' + self.__class__.__name__
        st += ' for {}: {} '.format(self.obj.__class__.__name__, self.obj.id)
        st += ': cur {} / {} commands'.format(self._pos, len(self))
        st += ' {}, {}, {}'.format(self.is_built, self._succeeded, self._failed)
        return st

    def __str__(self):
        return 'Strategy:<{}>Parent:<{}>GraphObject:<{}>'.format(
            self.__class__.__name__,
            self.parent.__class__.__name__,
            self.obj.id
        )

    def __len__(self):
        return len(self._q)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.parent.id == other.parent.id
        return False

    def __next__(self):
        return self.next_action()

    # plumbing -----------------------------------------------
    @staticmethod
    def _unwrap(objs):
        res = []
        if objs is None:
            return res
        for o in objs:
            if isinstance(o, list) and isinstance(o[0], int):
                res.append(o)
            elif isinstance(o, list):
                res += IStartegy._unwrap(o)
            elif o is None:
                continue
            else:
                res.append(o)
        return res

    def on_success(self, action):
        """
        if the action is the one self is expecting
            - remove the action from queue
            - generate any functions defined in on-success, and append to q
            - if there is nothing else to do, set success flag

        otherwise, send the action success signal down the tree
        """
        if self._q[self._pos] == action:
            self._q.pop(self._pos)
            self._q += self._unwrap(self.success(self.obj, action))
            if len(self._q) == 0:
                self._succeeded = True
                self._pos = 0
        elif not isinstance(self._q[self._pos], list):
            self._q[self._pos].on_success(action)
        else:
            print('index warning')

    def on_fail(self, *args):
        """
        if the action is the one self is expecting
            - overwrite the queue with what to do in case of failure.
            - set failed flags, and go back to start of queue

        otherwise, send the action success signal down the tree
        """
        if self._q[self._pos] == args[0]:
            self._failed = True
            self._q = self._unwrap(self.fail(self.obj, *args))
            self._pos = 0
        else:
            self._q[self._pos].on_fail(*args)

    def _build_actions(self,  **kwargs):
        """
        wrapper to make sure self.action is only called once
        """
        if self.is_built is False:
            self._built = True
            self._q = self._unwrap(self.action(self.obj, **kwargs))

    def add_commands(self, data):
        """
        If this manager exists, another once can write to it
        using add_commands( fn )
        commands will be appended to end of queue
        """
        self._q.append(data)

    def next_action(self):
        """
        Insread of popping things from the queue to return,
        we keep them in the queue until a success/fail flag comes back

        self._pos keeps track of which item in the queue will be used
        to continue iteration
        """
        if self.is_built is False:
            self._build_actions()
        ix = self._pos
        while ix < len(self._q):
            if isinstance(self._q[ix], list):
                res = self._q[ix]
                self._pos = ix
                return res

            res = self._q[ix].next_action()
            if res is not None:
                self._pos = ix
                return res
            else:
                self._q.pop(ix)
        return None


class ICommandManager(object):
    """
    Granular control over generating actions

    CommandManager is a wrapper for concrete instruction generation.
    Superclasses can define conditions or how specific families, types, etc are built

    THere is a one to one relationship of this to a node, so it keeps
    all the information +

    attrs: graph_obj (Node or Edge) this object operates on
    state: root node of IStrategy tree

    """
    __def_in_node = '__creator'

    def __init__(self,
                 graph_obj, **kwargs):
        self._graph_obj = graph_obj
        self._strategies = []
        self._state = None

    def next_action(self):
        return self._state.next_action()

    def add_action(self, strategy):
        self._state.add_commands(strategy)

    def _init_strategy(self, strategy=None, **kwargs):
        """
        setup the strategy object
        """
        rstrategy = None
        if strategy is not None:
            if strategy in self.strategies:
                ix = self.strategies.index(strategy)
                rstrategy = self.strategies.pop(ix)(self, **kwargs)
            else:
                rstrategy = strategy(self, **kwargs)
        elif len(self._strategies) > 0:
            rstrategy = self._strategies.pop(0)(self, **kwargs)
        self._state = rstrategy

    def on_success(self, action):
        """ communicate the succeeded action """
        self._state.on_success(action)

    def on_fail(self, action, message=None):
        """
        communicate failure down the chain
        :param action: list(RevitCommand)
        :return: None
        """
        self._state.on_fail(action, message)

    def __next__(self):
        return self.next_action()

    def __iter__(self):
        """ get all commands as if nothing goes wrong
            bkwd compat for static file / misc testing
        """
        action = self.next_action()
        while action is not None:
            assert isinstance(action, list), 'not a list'
            yield action
            self.on_success(action)
            action = self.next_action()

    def __len__(self):
        return 1 if not self._state.is_done else 0

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.gobj.id == other.obj.id
        return False

    def __repr__(self):
        st = 'Builder:' + self.__class__.__name__
        st += 'for {}: {}'.format(self.gobj.__class__.__name__, self.gobj.id)
        st += '\n'
        return st

    # -----------------------------------------------------
    @property
    def succeeded(self):
        return is_complete(self._graph_obj)

    @property
    def gobj(self):
        return self._graph_obj

    @property
    def strategies(self):
        return self._strategies

    @strategies.setter
    def strategies(self, data):
        self._strategies = data

    # -----------------------------------------------------
    @classmethod
    def action(cls, gobj, **kwargs):
        """
        retrieve the command generator from the node
        [create_successors, create_node, connect_predecessors]"""
        command_creator = gobj.get(cls.__def_in_node, None)
        if command_creator is None:
            command_creator = cls(gobj)
            gobj.write(cls.__def_in_node, command_creator)
        return command_creator

    @classmethod
    def on(cls, gobj, strategy=None, **kwargs):

        com_creator = gobj.get(cls.__def_in_node, None)
        if com_creator is None:
            com_creator = cls(gobj, strategy=strategy, **kwargs)
            gobj.write(cls.__def_in_node, com_creator)
        else:
            strat = strategy(com_creator, **kwargs)
            com_creator.add_action(strat)
        return com_creator

    @classmethod
    def remove(cls, graph_obj):
        command_creator = graph_obj.get(cls.__def_in_node, None)
        if command_creator is not None:
            graph_obj.write(cls.__def_in_node, None)
        return graph_obj

    @classmethod
    def success(cls, gobj, action):
        """
        :param graph_obj:
        :param action:
        :return:

        ICommand.success(node, [[11, 50, 20]])
        """
        command_creator = gobj.get(cls.__def_in_node, None)
        if command_creator is None:
            print('missing __creator in ', gobj.id)
            return gobj
        command_creator.on_success(action)

    @classmethod
    def fail(cls, gobj, action):
        """ ICommand.fail(node, [[11, 50, 20]]) """
        command_creator = gobj.get(cls.__def_in_node, None)
        if command_creator is None:
            print('missing __creator in ', gobj.id)
            return gobj
        command_creator.on_fail(action)

