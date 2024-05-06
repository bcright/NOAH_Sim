from flask import Flask, jsonify, request
from api import setup_routes
from config import DevelopmentConfig
from models.drl_model import DRLModel
from monitor import Monitor
from limiter import Limiter
from decision_maker import DecisionMaker

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
monitor = Monitor()
limiter = Limiter(100)  # 每秒最多100个请求
model = DRLModel.load_model('best_model.pth', state_dim=2, action_dim=1)
decision_maker = DecisionMaker(model, limiter)


@app.route('/predict', methods=['GET'])
def predict():
    monitor.update_metrics()
    current_state = monitor.get_current_state()
    prediction, new_limit = decision_maker.make_decision_and_adjust_limiter(current_state)

    return jsonify({
        'Predicted Action': prediction.item(),
        'New Limit': new_limit
    })
    
def create_app():
    setup_routes(app, monitor, limiter)
    print("Debug mode:", app.config['DEBUG'])
    return app

if __name__ == '__main__':
    app = create_app()
    # app.run(host='0.0.0.0', port=5000)
    app.run(host=app.config['HOST'], port=app.config['PORT'])