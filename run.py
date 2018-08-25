import tornado.ioloop
import tornado.web
from src import server


if __name__ == "__main__":
    try:
        app = server.make_app()
        app.listen(8888)
        tornado.ioloop.IOLoop.current().start()
    # signal : CTRL + BREAK on windows or CTRL + C on linux
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()
