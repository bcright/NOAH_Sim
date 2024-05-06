from environment.noah_env import NoahEnv
from models.drl_model import DRLModel
from torch.optim import Adam
import torch


def train_model(episodes, learning_rate=0.01):
    env = NoahEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = DRLModel(state_dim, action_dim)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_reward = -float('inf')

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # action_probs = model(state_tensor)
            # action = action_probs.max(1)[1].view(1, 1)
            # next_state, reward, done, _ = env.step(action.item())
            # total_reward += reward

            # loss = -torch.log(action_probs.gather(1, action)) * reward
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # state = next_state            
            state = torch.FloatTensor(state).unsqueeze(0)
            action = model(state)
            next_state, reward, done, _ = env.step(action.detach().numpy()[0])

            total_reward += reward

            # 假设一个简单的损失函数，实际应用中需要根据问题定义复杂的损失
            loss = -torch.log(action) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            
        if total_reward > best_reward:
            best_reward = total_reward
            # 保存模型的状态字典
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Episode {episode}, New best reward: {total_reward}, model saved.")
        # print(f"Episode {episode}, Total reward: {total_reward}")


if __name__ == '__main__':
    train_model(100)  # 训练100个回合