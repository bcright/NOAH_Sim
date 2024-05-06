class Collector:
    def __init__(self, monitor, limiter):
        self.monitor = monitor
        self.limiter = limiter

    def collect(self):
        data = self.monitor.get_statistics()
        return data