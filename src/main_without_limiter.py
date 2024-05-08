from flask import Flask, jsonify, request
from api import setup_routes
from config import DevelopmentConfig
from monitor import Monitor
from limiter import Limiter

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
monitor = Monitor()
limiter = Limiter() 


def create_app():
    setup_routes(app, monitor, limiter)
    print("Debug mode:", app.config['DEBUG'])
    return app

if __name__ == '__main__':
    app = create_app()
    # app.run(host='0.0.0.0', port=5000)
    app.run(host=app.config['HOST'], port=app.config['PORT'])