from flask import request, jsonify

def setup_routes(app):
    @app.route('/request', methods=['POST'])
    def handle_request():
        # 模拟请求处理逻辑
        return jsonify({"message": "Request processed successfully"})