from flask import Flask, jsonify, request
from api import setup_routes
from config import DevelopmentConfig
from models.drl_model import DRLModel
from monitor import Monitor
from limiter import Limiter
from decision_maker import DecisionMaker
import threading

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
monitor = Monitor()
limiter = Limiter(100)  # 每秒最多100个请求





def create_app():
    setup_routes(app, monitor, limiter)
    monitor.timed_predict()  # 启动定时预测任务
    print("Debug mode:", app.config['DEBUG'])
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host=app.config['HOST'], port=app.config['PORT'])