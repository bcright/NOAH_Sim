class Monitor:
    def __init__(self):
        self.data = {
            'cpu_usage': [],
            'request_count': 0,
            'successful_requests': 0
        }

    def collect_cpu_usage(self, usage):
        """收集CPU使用率"""
        self.data['cpu_usage'].append(usage)

    def log_request(self, successful=True):
        """记录请求数据"""
        self.data['request_count'] += 1
        if successful:
            self.data['successful_requests'] += 1
    # def collect_data(self, data_point):
    #     self.data.append(data_point)

    # def get_statistics(self):
    #     return sum(self.data) / len(self.data)
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