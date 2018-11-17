import os
import json

from src.formats.revit import Cmds, Command
import  src.formats.skansk as sks


class Strategy(object):
    """ I have seen things. I will learn from them """
    command = None

    def __init__(self):
        self.Q = []

    def read_error(self, err_message):
        main_line = err_message.split('\n')[0]

    def resolve(self, command, err_message):
        self.Q.append([command, err_message])


class StrategyTee(Strategy):
    command = Cmds.Tee






class RevitBuildStateMachine(object):
    """
    Generates commands for build server
    """

    def __init__(self, root_node):
        self._root = root_node
        self._current = None
        self.q = []

    # list interface -------------
    def __len__(self):
        return len(self.q)

    def on_first(self):
        return 'Ready'

    def reset(self):
        self.q = []
        self.q.append(self._root)

    # resolution ------------
    def on_response(self, command, message):
        if message in [b'SUCCESS', b'READY']:
            if len(self.q) > 0:
                el = self.q.pop(0)
                # sks.MakeInstructions.


        return


class CommandFile(object):
    """ Static list of command to run """

    def __init__(self, file_path):
        self._path = file_path
        self._data = []

    def __len__(self):
        return len(self._data)

    def reset(self):
        if os.path.exists(self._path) is True:
            with open(self._path, 'r') as F:
                self._data = json.load(F)

    def on_response(self, command, message):
        return self._data.pop(0)

    def on_first(self):
        if os.path.exists(self._path) is True:
            with open(self._path, 'r') as F:
                self._data = json.load(F)
        return 'Ready'


