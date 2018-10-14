import tornado.ioloop
import tornado.web
from src import server
import socket


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


if __name__ == "__main__":
    try:
        app = server.make_app()
        port = get_free_tcp_port()
        print(port)
        app.listen(port)
        tornado.ioloop.IOLoop.current().start()
    # signal : CTRL + BREAK on windows or CTRL + C on linux
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
