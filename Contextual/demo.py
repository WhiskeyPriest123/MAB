import numpy as np

class ContextualBandit:
    def __init__(self, n_arms, context_dim):
        self.n_arms = n_arms  # 臂的数量
        self.context_dim = context_dim  # 上下文特征的维度
        # 初始化每个臂的线性模型参数（每个臂对应一个向量）
        self.weights = np.zeros((n_arms, context_dim))

    def predict(self, context):
        """根据当前上下文预测每个臂的奖励"""
        # 计算每个臂的线性模型输出
        return np.dot(self.weights, context)

    def select_arm(self, context):
        """选择奖励最大的臂"""
        predicted_rewards = self.predict(context)  # 预测每个臂的奖励
        return np.argmax(predicted_rewards)  # 选择预测奖励最高的臂

    def update(self, chosen_arm, reward, context):
        """根据观察到的奖励和上下文更新模型参数"""
        # 学习率，可以随选择次数动态调整，这里简单使用常数
        learning_rate = 0.1
        # 计算当前上下文下的预测奖励
        prediction = np.dot(self.weights[chosen_arm], context)
        # 计算误差
        error = reward - prediction
        # 更新权重：w_new = w_old + learning_rate * error * context
        self.weights[chosen_arm] += learning_rate * error * context

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

# 示例使用
n_arms = 3  # 假设有3个臂
context_dim = 5  # 每个上下文有5个特征
n_steps = 1000  # 总共进行1000次选择

# 随机生成模拟的上下文矩阵（1000步，每步都有一个5维的上下文）
contexts = np.random.rand(n_steps, context_dim)
# 随机生成每个臂的奖励分布（每个臂一个5维的权重向量）
rewards_distribution = np.random.rand(n_arms, context_dim)

# 创建Contextual Bandit算法实例
algo = ContextualBandit(n_arms, context_dim)
total_reward, chosen_arms = algo.run(contexts, rewards_distribution, n_steps)

print("总奖励:", total_reward)
print("每个臂被选择的次数:", np.bincount(chosen_arms))
