class Limiter:
    def __init__(self, threshold):
        self.threshold = threshold

    def check_limit(self, current_load):
        if current_load > self.threshold:
            return False
        return True