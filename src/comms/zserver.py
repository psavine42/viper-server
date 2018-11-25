import zmq
from . import smartbuild as sb
import importlib

_DONE = 'DONE'


class ZBuilder(object):
    context = zmq.Context()

    def __init__(self, port="5556"):
        self._address = "tcp://*:{}".format(port)
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self._address)
        self.builder = None

    def send_instructions(self, lim=None):
        cnt = 0
        comm = ''
        while len(self.builder) > 0:
            raw = self.socket.recv()
            response = raw.decode()

            if response.startswith('SUCCESS') is False:
                print(cnt, comm, response)
            else:
                print(cnt, comm, 'SUCCESS')

            #elif cnt % 20 == 0:
            #    print('iter {}'.format(cnt))

            if response == _DONE:
                print(cnt, _DONE)
                break
            elif lim is not None and cnt > lim:
                print(cnt, _DONE)
                break
            else:
                comm = self.builder.on_response(response)
                self.socket.send_string(comm)
                cnt += 1

        msg = self.socket.recv()
        self.socket.send_string('END')
        print('sent {} commands '.format(cnt))

    def receive_instructions(self):
        cnt = 0
        while True:
            cmd = self.socket.recv_json()
            print(cmd)
            if cmd == ['DONE']:
                self.socket.send_json(['SUCCESS'])
                break
            else:
                comm = self.builder.add(cmd)
                self.socket.send_json([comm])
                cnt += 1
        print('added {} commands '.format(cnt))

    def run(self, builder):
        print('server running at ... ' + self._address)
        self.builder = builder
        halt = False
        while halt is not True:
            raw = self.socket.recv()
            print('-----------')

            msg = raw.decode()

            print(raw, msg)
            if msg == 'Ready':
                # revit comms - all string
                cmd = self.builder.on_first()
                if cmd is None:
                    print('no instructions')
                    break
                self.socket.send_string(cmd)
                self.send_instructions()

            elif msg == 'Update':
                self.socket.send_json(['OK'])
                self.receive_instructions()

            elif msg == 'RESET':
                importlib.reload(sb)
                self.socket.send_json(['OK'])
                data, kwargs = self.socket.recv_json()
                print('reset to : ', data, kwargs)
                klass = sb.klasses.get(data, None)
                if klass is not None:
                    self.builder = klass(**kwargs)
                    self.socket.send_json(['OK'])
                else:
                    self.socket.send_json(['FAIL'])

            elif msg == 'State':
                state = self.builder.state()
                self.socket.send_json(state)
            else:
                print('Early Done')
                self.socket.send_string('NOOP')


class CommandController:
    context = zmq.Context()

    def __init__(self, zmq_url):
        self.socket = None
        self.zmq_url = zmq_url
        self.connect_zmq()
        print("Instruction updater started:")

    def connect_zmq(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.zmq_url)

    def request_web_url(self):
        self.socket.send(b"url")
        response = self.socket.recv().decode("utf-8")
        return response

    def wait(self):
        self.socket.send(b"wait")
        return self.socket.recv().decode("utf-8")

    def state(self):
        return self._command('State')

    def reset(self, cls, **kwargs):
        self._command('RESET')
        return self.send([cls, kwargs])

    def update(self, data):
        self._command('Update')
        for xs in data:
            self.send(xs)
        return self.send(['DONE'])

    def _command(self, cmd):
        self.socket.send_string(cmd)
        res = self.socket.recv_json()
        return res

    def send(self, command):
        self.socket.send_json(command)
        return self.socket.recv_json()



if __name__ == '__main__':
    path = '/home/psavine/source/viper/data/out/commands/comms.json'
    Instr = sb.CommandFile(path)
    server = ZBuilder()
    server.run(Instr)

