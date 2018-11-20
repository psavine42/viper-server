import tornado.ioloop
import tornado.web
from src import server
import socket
import argparse


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def run_tornado():
    try:
        app = server.make_app()
        port = get_free_tcp_port()
        print(port)
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()
    # signal : CTRL + BREAK on windows or CTRL + C on linux
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()


def run_zserver():
    from src.comms.smartbuild import CommandFile
    from src.comms.zserver import ZBuilder
    path = '/home/psavine/source/viper/data/out/commands/comms.json'

    instructions = CommandFile(path)
    server = ZBuilder()
    server.run(instructions)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-a', '--action', type=str)
    arg = p.parse_args()
    if arg.action == 'z':
        run_zserver()

    else:
        run_tornado()

