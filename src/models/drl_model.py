import torch
import torch.nn as nn

class DRLModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DRLModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 假设动作空间是[-1, 1]
        )

    def forward(self, state):
        return self.network(state)
    @staticmethod
    def load_model(model_path, state_dim, action_dim):
        model = DRLModel(state_dim, action_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict(self, input_data):
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
        with torch.no_grad():
            return self.forward(input_tensor)