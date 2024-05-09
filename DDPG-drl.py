import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import pandas as pd

# 设置 TensorFlow 日志级别为 ERROR
tf.get_logger().setLevel('ERROR')



# 定义环境
class AdaptiveRateLimitEnv:
    def __init__(self, c_free, T, N):
        self.c_free = c_free
        # self.state = np.array([c_free, 0])  # 初始CPU使用率和QPS
        self.current_cpu = c_free
        self.current_qps = 0
        self.limit = 100  # 初始QPS限制
        # self.current_qps = 0
        self.T = T  # CPU使用率的阈值
        self.N = N  # 状态中包含的性能指标对数量
        self.memory_S = []  # 存储性能指标对

    def reset(self,qps):
        # self.state = np.array([self.c_free, 0])
        self.current_cpu = self.c_free
        self.current_qps = qps
        self.limit = 100
        self.memory_S.clear()  # 清空历史数据
        self.prepopulate_interactions()
        return self.update_state()  # 返回初始状态
    
    def prepopulate_interactions(self):
        # 与环境进行1000次交互
        for _ in range(10):  # 与环境进行1000次交互
            state, _, _ = self.step(0)  # 动作a固定为0
            self.limit = self.current_qps
        
            count_qps = self.current_qps
            if self.current_qps > self.limit:
                count_qps = self.limit
                
            # 基本的CPU使用率更新
            mean_cpu_usage = self.c_free + 0.2 * count_qps

            # 添加正态分布随机扰动，假设均值为0，标准差为3
            random_perturbation = np.random.normal(0, 3)
            self.current_cpu = min(mean_cpu_usage+random_perturbation, 100)  # 更新CPU使用率
        # print(self.current_qps)
        # print(self.current_cpu)
        self.memory_S.append((self.current_qps,self.current_cpu))
        
    # def collect_initial_metrics(self, first_interaction):
    #     c0, q0 = first_interaction
    #     self.memory_S.append((q0, c0))  # 收集第一次交互的性能指标对
    #     # 构建初始状态s0
    #     # self.update_state()
    #     # self.state = [(0, self.c_free)] * (self.N - 1) + [(q0, c0)]
    #     # self.state = np.array(self.state).flatten()


    def step(self, action):
        # 模拟环境响应
        if action > 0 :
            self.limit = self.current_qps * action  # 动作是一个比例，用于计算新的QPS限制阈值
        else: 
            self.limit = self.current_qps
        print(f"Limit: {self.limit}")
        print(f"Cpu: {self.current_cpu}")
        print(f"Action: {action}")
        
        
        count_qps = self.current_qps
        if self.current_qps > self.limit:
            count_qps = self.limit
            
                # 基本的CPU使用率更新
        mean_cpu_usage = self.c_free + 0.2 * count_qps

        # 添加正态分布随机扰动，假设均值为0，标准差为3
        random_perturbation = np.random.normal(0, 3)
        self.current_cpu = min(mean_cpu_usage+random_perturbation, 100)  # 更新CPU使用率
        # self.current_cpu = min(self.c_free + 0.1 * count_qps, 100)  # 更新CPU使用率
        
        # 记录当前的QPS和CPU使用率
        # self.memory_S.append((self.current_qps, self.current_cpu))
        self.collect_performance_metric(count_qps, self.current_cpu)
        
        # 更新状态
        state = self.update_state()
        
        # 计算奖励
        reward = self.calculate_reward(action)
        
        return state, reward, False  # 环境状态，奖励，是否结束
    
    
    def collect_performance_metric(self, qps, cpu_usage):
        # 如果qps或cpu_usage是Numpy数组类型，则转换为Python浮点数
        qps = float(qps) if isinstance(qps, np.ndarray) else qps
        cpu_usage = float(cpu_usage) if isinstance(cpu_usage, np.ndarray) else cpu_usage
        
        # 收集并添加性能指标对到列表S中
        self.memory_S.append((qps, cpu_usage))
        # print("collet" + )
        # 保持列表长度为最新的N个记录
        if len(self.memory_S) > self.N:
            self.memory_S.pop(0)
    
    def update_state(self):
        # 确保状态包含最新的N个指标对，不足则用(0, c_free)补充
        # print(self.memory_S)
        padded = [(0, self.c_free)] * (self.N - len(self.memory_S))
        # print(padded)
        state = padded + self.memory_S[-self.N:]
        return np.array(state).flatten()  # 将状态展平以便输入到网络中
    
    def calculate_reward(self, action):
        # 计算所有QPS的平均值
        average_qps = np.mean([qps for qps, _ in self.memory_S])

        # 奖励函数计算
        if self.current_cpu <= self.T:
            reward = average_qps * action
        else:
            # reward = max(-np.exp(10 *(self.current_cpu - self.T)),-1000)
            # 使用平方差进行惩罚，而非指数
            penalty = (self.current_cpu - self.T) ** 2
            reward = -min(penalty, 1000)  # 限制最大惩罚
            
        # # 限制奖励值的范围，防止极端奖励影响模型稳定性
        # reward = max(min(reward, 100), -100)
        
        return reward

# 定义Actor和Critic网络
def create_actor(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(4, activation='relu', input_shape=(state_size,)),
        layers.Dense(4, activation='relu'),
        layers.Dense(action_size, activation='sigmoid')  # 使用sigmoid激活函数来输出动作值
    ])
    # 添加一个微小的偏移量
    epsilon = 1e-6
    def custom_sigmoid(x):
        return tf.keras.activations.sigmoid(x) * (1 - 2 * epsilon) + epsilon

    model.layers[-1].activation = custom_sigmoid
    return model

def create_critic(state_size, action_size):
    state_input = layers.Input(shape=(state_size,))
    action_input = layers.Input(shape=(action_size,))
    concat = layers.Concatenate()([state_input, action_input])
    x = layers.Dense(4, activation='relu')(concat)
    x = layers.Dense(4, activation='relu')(x)
    x = layers.Dense(1)(x)  # 输出单一的Q值
    return tf.keras.Model([state_input, action_input], x)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # self.memory_S = []  # 初始化S列表
        self.replay_buffer_R = deque(maxlen=2000)  # 初始化R列表

        self.actor = create_actor(state_size, 1)  # 假设动作空间是1维
        self.critic = create_critic(state_size, 1)
        self.target_actor = create_actor(state_size, 1)
        self.target_critic = create_critic(state_size, 1)
        self.actor_optimizer = tf.keras.optimizers.Adam(0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(0.001)

        # 初始化目标网络权重
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.tau = 0.005

    def act(self, state,N):
        state = np.reshape(state, [-1, 2])
        action = self.actor.predict(state)[0]
        # print(action)
        return action
    
    # def collect_metrics(self, qt, ct):
    #     # 收集和管理性能指标对
    #     if len(self.memory_S) < self.state_size:
    #         self.memory_S.append((qt, ct))
    #     else:
    #         self.memory_S.pop(0)
    #         self.memory_S.append((qt, ct))

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        # 向回放缓冲区添加元素
        self.replay_buffer_R.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.replay_buffer_R) < 50:
            return
        minibatch = random.sample(self.replay_buffer_R, 5)
        for state, action, reward, next_state, done in minibatch:
            
            # state = np.array(state).reshape(1, -1)
            # next_state = np.array(next_state).reshape(1, -1)
            # action = np.array([action]).reshape(1, -1)
            # 更新Critic网络
            # print(next_state)
            next_state = next_state.reshape(-1,2)
            state = state.reshape(-1,2)
            # print("!")
            # print(next_state)
            # 预测下一个状态的动作和Q值
            target_action = self.target_actor.predict(next_state)
            future_q = self.target_critic.predict([next_state, target_action])[0]

            # 计算TD误差
            # print(state)
            # print(action)
            current_q = self.critic.predict([state[len(state)-1:len(state), :],action])[0]
            print(f"q {current_q}")
            td_error = reward + self.gamma * future_q * (1 - done) - current_q

            # 更新Critic网络
            with tf.GradientTape() as tape:
                tape.watch([tf.convert_to_tensor(state[len(state)-1:len(state), :], dtype=tf.float32), tf.convert_to_tensor(action, dtype=tf.float32)])
                q_values = self.critic([state[len(state)-1:len(state), :], action])
                loss = tf.keras.losses.MSE(td_error ** 2, q_values)
            critic_grads = tape.gradient(loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            # 更新Actor网络
            with tf.GradientTape() as tape:
                tape.watch(tf.convert_to_tensor(state, dtype=tf.float32))
                actions = self.actor(state)
                q_values = self.critic([state, actions])
                actor_loss = -tf.reduce_mean(q_values)  # 求最大化Q值，即最小化负Q值
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            # print(next_state)
            # print(next_state[len(next_state)-1])
            # target_action = self.target_actor.predict(next_state)
            # print(target_action)
            # y = reward + self.gamma * self.target_critic.predict([next_state[len(next_state)-1], target_action])[0] * (1 - done)
            # with tf.GradientTape() as tape:
            #     critic_value = self.critic([state, action])
            #     critic_loss = tf.keras.losses.MSE(y, critic_value)
            # critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            # self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            # # 更新Actor网络
            # with tf.GradientTape() as tape:
            #     actions = self.actor(state)
            #     critic_value = self.critic([state, actions])
            #     actor_loss = -tf.reduce_mean(critic_value)
            # actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            # self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            
            # 更新目标网络
            self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
            self.update_target(self.target_critic.variables, self.critic.variables, self.tau)

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

# 保存模型
def save_models(epoch):
    agent.actor.save(f'actor_{epoch}_1.h5')
    agent.critic.save(f'critic_{epoch}_1.h5')
    print(f"Models saved at epoch {epoch}")


# 训练循环
if __name__ == "__main__":
    
    env = AdaptiveRateLimitEnv(c_free=10,T = 60, N = 50)
    state_size = 2
    action_size = 1  # 动作空间大小
    agent = DDPGAgent(state_size, action_size)
    episodes = 5
    
    # 用列表收集记录
    records = []

    for e in range(episodes):
        qps = np.random.randint(350, 400)
        state = env.reset(qps)
        total_reward = 0
        for time in range(100):
            print(time)
            # print(state)
            action = agent.act(state,N=50)
            next_state, reward, done = env.step(action)
            print(f"Reward: {reward}")
            # next_state = env.collect_performance_metric(next_state[1], next_state[0])  # 收集性能指标对
            # print(next_state)
            agent.add_to_replay_buffer(state, action, reward, next_state, done)
            # agent.memory.append((state, action, reward, next_state, done))
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break

        # 更新记录
        records.append({
            'Episode': e + 1,
            'QPS': qps,
            'Total Reward': total_reward,
            'Current CPU': env.current_cpu,  
            'Limit': env.limit,              
            'Action': action
        })
        # if (e + 1) % 100 == 0:  # 每100个epoch保存一次模型
        if e == episodes-1:
            save_models(e + 1)
            records_df = pd.DataFrame(records)
            records_df.to_csv('training_data1.csv', index=False)
        print(f"Episode: {e+1}, QPS: {qps} Total reward: {total_reward}")