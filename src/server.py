"""
Tornado server for revit
"""
import tornado.ioloop
import tornado.web
import json
import importlib
from tornado.escape import json_decode
import src.process


class MainHandler(tornado.web.RequestHandler):
    def get(self, data):
        self.write("Hello, world")

    def post(self, *args, **kwargs):
        importlib.reload(src.process)

        dx = json_decode(self.request.body)

        points = json_decode(dx.get('points'))
        data = json_decode(dx.get('data'))

        print('recieved')

        proc = src.process.SystemProcessor()
        ds = proc.process(data, points, system_type='FP')
        self.write(json.dumps(ds))


def make_app():
    return tornado.web.Application([
        (r"/(.*)", MainHandler),
    ])





