import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, context_dim, n_arms):
        super(SimpleNN, self).__init__()
        # 定义一个简单的两层神经网络
        self.fc1 = nn.Linear(context_dim, 64)  # 第一层全连接层，输入维度为context_dim，输出为64
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(64, n_arms)  # 第二层全连接层，输出维度为n_arms（对应每个臂的奖励）

    def forward(self, x):
        x = self.fc1(x)  # 通过第一层
        x = self.relu(x)  # 通过ReLU激活函数
        x = self.fc2(x)  # 通过第二层，输出每个臂的奖励估计
        return x

class ContextualBanditWithNN:
    def __init__(self, context_dim, n_arms):
        self.n_arms = n_arms  # 臂的数量
        self.model = SimpleNN(context_dim, n_arms)  # 初始化神经网络模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # 使用Adam优化器
        self.loss_fn = nn.MSELoss()  # 使用均方误差作为损失函数

    def select_arm(self, context):
        """根据当前上下文预测每个臂的奖励，并选择奖励最高的臂"""
        context_tensor = torch.tensor(context, dtype=torch.float32)
        predicted_rewards = self.model(context_tensor)  # 预测每个臂的奖励
        return torch.argmax(predicted_rewards).item()  # 选择预测奖励最高的臂

    def update(self, chosen_arm, reward, context):
        """根据观察到的奖励和上下文更新模型参数"""
        self.optimizer.zero_grad()  # 清除梯度
        context_tensor = torch.tensor(context, dtype=torch.float32)
        predicted_rewards = self.model(context_tensor)  # 预测每个臂的奖励
        # 创建一个目标奖励张量，只有选择的臂的奖励为实际奖励，其余为预测值
        target = predicted_rewards.detach().clone()
        target[chosen_arm] = reward  # 更新目标值为实际奖励
        # 计算损失
        loss = self.loss_fn(predicted_rewards, target)
        # 反向传播和优化
        loss.backward()
        self.optimizer.step()

    def run(self, contexts, rewards_distribution, n_steps):
        """运行Contextual Bandits算法"""
        total_reward = 0  # 累积奖励
        chosen_arms = []  # 记录每一步选择的臂

        for step in range(n_steps):
            context = contexts[step]  # 获取当前步骤的上下文
            chosen_arm = self.select_arm(context)  # 选择一个臂
            # 根据当前上下文和臂的奖励分布生成奖励
            reward = np.random.normal(rewards_distribution[chosen_arm].dot(context))
            self.update(chosen_arm, reward, context)  # 更新模型参数
            total_reward += reward  # 累积总奖励
            chosen_arms.append(chosen_arm)  # 记录选择的臂

        return total_reward, chosen_arms


if __name__ == "__main__":
    # 示例使用
    n_arms = 3  # 假设有3个臂
    context_dim = 5  # 每个上下文有5个特征
    n_steps = 1000  # 总共进行1000次选择

    # 随机生成模拟的上下文矩阵（1000步，每步都有一个5维的上下文）
    contexts = np.random.rand(n_steps, context_dim)
    # 随机生成每个臂的奖励分布（每个臂一个5维的权重向量）
    rewards_distribution = np.random.rand(n_arms, context_dim)

    # 创建Contextual Bandit算法实例
    algo = ContextualBanditWithNN(context_dim, n_arms)
    total_reward, chosen_arms = algo.run(contexts, rewards_distribution, n_steps)

    print("总奖励:", total_reward)
    print("每个臂被选择的次数:", np.bincount(chosen_arms))
