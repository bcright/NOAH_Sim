import time
class Limiter:
    def __init__(self, max_requests_per_second):
        self.max_requests_per_second = max_requests_per_second
        self.requests = []
        self.current_time = time.time()

    def allow_request(self):
        """决定是否允许当前请求"""
        current_time = time.time()
        # 移除旧的请求记录
        while self.requests and self.requests[0] < current_time - 1:
            self.requests.pop(0)

        if len(self.requests) < self.max_requests_per_second:
            self.requests.append(current_time)
            return True
        return False

    def adjust_threshold(self, new_threshold):
        """调整请求阈值"""
        self.max_requests_per_second = new_threshold