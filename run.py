import tornado.ioloop
import tornado.web
from src import server


if __name__ == "__main__":
    app = server.make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()