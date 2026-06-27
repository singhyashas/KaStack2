from wsgiref.simple_server import make_server
from app.main import app

if __name__ == '__main__':
    port = 8010
    httpd = make_server('0.0.0.0', port, app)
    print('Serving on http://0.0.0.0:{0}'.format(port))
    httpd.serve_forever()
