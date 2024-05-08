import psutil
import threading
from flask import Flask
from config import DevelopmentConfig
from models.drl_model import DRLModel
from limiter import Limiter
from decision_maker import DecisionMaker

class Monitor:
    def __init__(self):
        self.data = {
            'cpu_usage': [],
            'request_count': 0,
            'successful_requests': 0
        }
        self.limiter = Limiter(100)  # 每秒最多100个请求
        self.model = DRLModel.load_model('best_model.pth', state_dim=2, action_dim=1)
        self.decision_maker = DecisionMaker(self.model, self.limiter)

    def collect_cpu_usage(self):
        """收集CPU使用率"""
        usage = psutil.cpu_percent(interval=1)
        self.data['cpu_usage'].append(usage)

    def log_request(self, success=True):
        """记录请求数据"""
        self.data['request_count'] += 1
        if success:
            self.data['successful_requests'] += 1

    def get_statistics(self):
        """返回收集的统计数据"""
        avg_cpu_usage = sum(self.data['cpu_usage']) / len(self.data['cpu_usage']) if self.data['cpu_usage'] else 0
        success_rate = (self.data['successful_requests'] / self.data['request_count']) if self.data['request_count'] else 0
        return {
            'average_cpu_usage': avg_cpu_usage,
            'request_success_rate': success_rate
        }

    def reset(self):
        """重置监视器数据"""
        self.data = {
            'cpu_usage': [],
            'request_count': 0,
            'successful_requests': 0
        }

    def update_metrics(self):
        """更新监控数据"""
        self.collect_cpu_usage()
        self.log_request()

    def get_current_state(self):
        """获取当前状态，用于决策"""
        return self.get_statistics()

    def timed_predict(self):
        """定时执行预测并打印结果"""
        self.update_metrics()
        current_state = self.get_current_state()
        prediction, new_limit = self.decision_maker.make_decision_and_adjust_limiter(current_state)
        
        print({
            'Predicted Action': prediction.item(),
            'New Limit': new_limit
        })
        # 重新启动定时器
        threading.Timer(0.02, self.timed_predict).start()