

class DecisionMaker:
    def __init__(self, model, limiter):
        self.model = model
        self.limiter = limiter

    def make_decision_and_adjust_limiter(self, data):
        decision = self.model.predict(data)
        new_limit = int(decision.item() * 100)  # 根据需要调整这个转换逻辑
        self.limiter.adjust_threshold(new_limit)
        return decision, new_limit