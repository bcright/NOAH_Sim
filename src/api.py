from flask import request, jsonify

def setup_routes(app, monitor, limiter):
    @app.route('/')
    def home():
        return "Welcome to the Noah Simulation API!"

    @app.route('/request', methods=['POST'])
    def handle_request():
        if not limiter.allow_request():
            monitor.log_request(success=False)
            return jsonify({"error": "Too many requests"}), 429

        # 业务逻辑
        success = process_request()
        monitor.log_request(success=True)
        return jsonify({"success": success})

    def process_request():
        # 实际的业务逻辑处理，这里简单返回True作为示例
        return True
    
