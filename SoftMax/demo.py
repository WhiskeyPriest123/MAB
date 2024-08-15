import numpy as np

class SoftmaxAlgorithm:
    def __init__(self, n_arms, tau):
        self.n_arms = n_arms  # 臂的数量
        self.tau = tau  # 温度参数
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的平均奖励估计值

    def softmax(self, values):
        """根据奖励值计算每个臂的选择概率"""
        exp_values = np.exp(values / self.tau)
        return exp_values / np.sum(exp_values)

    def select_arm(self):
        """根据Softmax概率分布选择臂"""
        probabilities = self.softmax(self.values)
        return np.random.choice(np.arange(self.n_arms), p=probabilities)

    def update(self, chosen_arm, reward):
        """根据选择的臂和获得的奖励更新估计值"""
        self.counts[chosen_arm] += 1  # 增加选择计数
        n = self.counts[chosen_arm]  # 该臂被选择的总次数
        value = self.values[chosen_arm]  # 当前的平均奖励
        # 更新平均奖励估计值 (增量更新公式)
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def run(self, rewards_distribution, n_steps):
        """运行Softmax算法"""
        total_reward = 0  # 累积奖励
        chosen_arms = []  # 记录每一步选择的臂

        for _ in range(n_steps):
            chosen_arm = self.select_arm()  # 选择一个臂
            reward = np.random.normal(rewards_distribution[chosen_arm])  # 根据分布获得奖励
            self.update(chosen_arm, reward)  # 更新奖励估计值
            total_reward += reward  # 累积总奖励
            chosen_arms.append(chosen_arm)  # 记录选择的臂

        return total_reward, chosen_arms

# 示例使用
n_arms = 3  # 假设有3个老虎机臂
tau = 0.1  # 温度参数
n_steps = 1000  # 总共进行1000次选择
rewards_distribution = [0.2, 0.5, 0.8]  # 假设每个臂的真实平均奖励

# 创建Softmax算法实例
algo = SoftmaxAlgorithm(n_arms, tau)
total_reward, chosen_arms = algo.run(rewards_distribution, n_steps)

print("总奖励:", total_reward)
print("每个臂被选择的次数:", np.bincount(chosen_arms))
