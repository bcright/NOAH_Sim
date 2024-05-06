from flask import Flask, jsonify, request
from api import setup_routes
from config import DevelopmentConfig
from models.drl_model import DRLModel
from monitor import Monitor
from limiter import Limiter

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
monitor = Monitor()
limiter = Limiter(100)  # 每秒最多100个请求
model = DRLModel.load_model('best_model.pth', state_dim=2, action_dim=1)


@app.route('/predict', methods=['GET'])
def predict():
    monitor.update_metrics()
    current_state = monitor.get_current_state()
    prediction = model.predict(current_state)
    
    # 假设模型输出是新的请求限制值
    new_limit = int(prediction.item() * 100)  # 根据需要调整这个转换逻辑
    limiter.adjust_threshold(new_limit)
    
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