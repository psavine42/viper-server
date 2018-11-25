import os
import json
import networkx as nx
from src.misc import utils
from src.structs import node_utils as gutil
from src.formats import revit
import src.formats.skansk as sks
# from src.ui import visual as V

CSuccess = 'SUCCESS'
CHandshake = 'Handshake'
CReady = 'Ready'
CCommand = 'COMMAND'


def stringify(data):
    if isinstance(data, str):
        encoded = data
    else:
        encoded = ','.join(list(map(str, data)))
    return encoded


class IZServer(object):
    def __init__(self, **kwargs):
        # commands are stored in a dict
        self.commands = {}
        self.current = None
        self.q = []

    def prepare_data(self, data):
        """
        save the command to recover latter if needed
        :param raw:
        :return:
        """
        encoded = 'COMMAND,' + stringify(data)
        self.commands[encoded] = data
        self.current = encoded
        return encoded

    def __getitem__(self, encoded):
        if encoded in self.commands:
            return self.commands[encoded]
        return None

    def __len__(self):
        return len(self.q)

    # ----------------------------------------
    def reset(self):
        self.commands = {}
        self.current = None
        self.q = []

    def add(self, command):
        self.q.append(command)
        return 'OK'

    def state(self):
        return self.q

    def on_first(self):
        raise NotImplemented('Not Implemented in base class')

    def on_finish(self, *args):
        raise NotImplemented('Not Implemented in base class')

    # ----------------------------------------
    def on_success(self, *args):
        return self.on_default(*args)

    def on_fail(self, *args):
        return self.on_default(*args)

    def on_default(self, *args):
        raise NotImplemented('Not Implemented in base class')

    # ----------------------------------------
    def on_response(self, response):
        if response.startswith('SUCCESS'):
            return self.on_success(response)
        elif response.startswith('FAIL') or response.startswith('EXCEPTION'):
            return self.on_fail(response)
        elif response.startswith('DONE'):
            return self.on_finish(response)
        elif response in ['Ready', 'Handshake']:
            return self.on_default(response)
        return 'DONE'


class Simple(IZServer):
    def __init__(self, **kwargs):
        super(Simple, self).__init__()
        self._fail = []
        self.reset()

    def reset(self):
        self._fail = []
        self.commands = {}
        self.current = None
        self.q = []

    def add(self, command):
        self.q.append(command)
        return 'OK'

    def state(self):
        return {'todo': self.q, 'failed': self._fail}

    def on_first(self):
        return 'Handshake'

    def on_finish(self, *args):
        return 'Done'

    def on_fail(self, response):
        raw_error = self[self.current]
        if raw_error is not None:
            self._fail.append([raw_error, response])
        return self.on_default(response)

    def on_default(self, response):
        data = self.prepare_data(self.q.pop(0))
        return data


class CommandFile(Simple):
    """ Static list of command to run from a file
        file is reloaded on each run
    """
    def __init__(self, file_path=None):
        Simple.__init__(self)
        self._path = file_path
        self._fail = []
        self.reset()

    def _reload(self):
        if self._path is not None and os.path.exists(self._path) is True:
            with open(self._path, 'r') as F:
                self.q = json.load(F)

    def on_first(self):
        if len(self._fail) > 0:
            print('num fails {}'.format(len(self._fail)))
        self.reset()
        self._reload()
        return 'Handshake'

    def on_success(self, *args):
        return self.on_default(*args)

    def on_default(self, *args):
        raw = self.q.pop(0)
        encoded = self.prepare_data(raw)
        return encoded


class GraphFile(CommandFile):
    """ File containing NXGraph
        generate build instructions adaptively

    """
    def __init__(self, file_path=None, **kwargs):
        CommandFile.__init__(self, file_path=file_path)
        self.root = None    # read-only root node of graph
        self.q = []         # this is a list of nodes.
        self.last_action = None  # Current action (list)
        self.cmd_mgr = None
        self.current_node = None
        # print('loaded from : ', self._path)

    def __len__(self):
        """ length = main q + what is in the queue """
        return len(self.q) + len(self.cmd_mgr)

    def reset(self):
        # CommandFile.reset(self)
        self.current_node = None
        self.cmd_mgr = None
        self.last_action = None
        # self._reload()

    def _reload(self):
        if self._path is not None and os.path.exists(self._path) is True:
            with open(self._path, 'rb') as F:
                nx_graph = nx.read_gpickle(F)
                root_node = utils.nxgraph_to_nodes(nx_graph)
                self.root = root_node
                self.q = [root_node]
                self.current_node = root_node
                self.cmd_mgr = revit.make_actions_for(self.current_node)

    # resolution ------------
    def on_fail(self, response):
        self.cmd_mgr.on_fail(self.last_action )
        return self.on_default(response)

    def on_success(self, response):
        """ set the command_mgr state to success """
        self.cmd_mgr.on_success(self.last_action )
        return self.on_default(response)

    def on_default(self, *args):
        next_action = self.cmd_mgr.next_action()
        while next_action is None:
            # the action manager for current node is done
            # add node successors to the queue
            for n in self.current_node.successors():
                self.q.append(n)

            if len(self.q) == 0:   # todo not sure if needed
                return 'DONE'

            # set the current node, create new cmd_manager
            self.current_node = self.q.pop(0)
            self.cmd_mgr = revit.make_actions_for(self.current_node)
            next_action = self.cmd_mgr.next_action()

        self.last_action = next_action
        return 'COMMAND,' + stringify(next_action)


klasses = {'nx': GraphFile,
           'simple': Simple,
           'file': CommandFile}

