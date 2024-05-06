from src.models.drl_model import DRLModel

class DecisionMaker:
    def __init__(self):
        self.model = DRLModel()

    def make_decision(self, data):
        decision = self.model.predict(data)
        return decision