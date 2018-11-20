import zmq
import time
import os
import json
from . import smartbuild as sb



_DONE = 'DONE'


class ZBuilder(object):
    context = zmq.Context()

    def __init__(self, port="5556"):
        self._address = "tcp://*:{}".format(port)
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self._address)
        self._data = []

    def serialize(self, command):
        return 'COMMAND,' + ','.join(list(map(str, command)))

    def send_instructions(self, commands, lim=None):
        cnt = 0
        comm = ''
        while len(commands) > 0:
            response = self.socket.recv()
            response = response.decode()

            if response.startswith('SUCCESS') is False:
                print(cnt, comm, response)

            if response == _DONE:
                print(cnt, _DONE)
                break
            elif lim is not None and cnt > lim:
                print(cnt, _DONE)
                break
            else:
                comm = commands.on_response(response)
                comm = self.serialize(comm)
                self.socket.send_string(comm)
                # time.sleep(0.1)
                cnt += 1

        msg = self.socket.recv()
        print(cnt, msg)
        self.socket.send_string('END')

    # def load_commands(self, file_path):
    #     if os.path.exists(file_path) is True:
    #         with open(file_path, 'r') as F:
    #             data = json.load(F)
    #         return data

    # def run_file(self, path):
    #     halt = False
    #     while halt is not True:
    #         msg = self.socket.recv()
    #         print('--------------------------')
    #         if msg.decode() == 'Ready':
    #             print('--------------------------')
    #             self.socket.send_string('Handshake')
    #             cmd = self.load_commands(path)
    #             if cmd is None:
    #                 print('file is missing')
    #                 break
    #             self.send_instructions(cmd)

    def run(self, instruction):
        halt = False
        while halt is not True:
            msg = self.socket.recv()
            msg = msg.decode()
            print(msg + '---------------------------')
            if msg == 'Ready':
                print('Handshake')
                self.socket.send_string('Handshake')
                cmd = instruction.on_first()
                if cmd is None:
                    print('no instructions')
                    break
                self.send_instructions(instruction)
            else:
                print('Early Done')
                self.socket.send_string('DONE')


if __name__ == '__main__':
    path = '/home/psavine/source/viper/data/out/commands/comms.json'
    Instr = sb.CommandFile(path)
    server = ZBuilder()
    server.run(Instr)
