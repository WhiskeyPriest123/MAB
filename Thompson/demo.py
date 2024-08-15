import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms  # 臂的数量
        self.successes = np.zeros(n_arms)  # 每个臂的成功次数（α参数）
        self.failures = np.zeros(n_arms)  # 每个臂的失败次数（β参数）

    def select_arm(self):
        """根据Beta分布为每个臂采样，选择采样值最大的臂"""
        sampled_theta = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            # 从Beta分布中采样
            sampled_theta[arm] = np.random.beta(self.successes[arm] + 1, self.failures[arm] + 1)
        return np.argmax(sampled_theta)  # 选择采样值最大的臂

    def update(self, chosen_arm, reward):
        """根据获得的奖励更新该臂的成功和失败次数"""
        if reward == 1:
            self.successes[chosen_arm] += 1  # 成功次数+1
        else:
            self.failures[chosen_arm] += 1  # 失败次数+1

    def run(self, rewards_distribution, n_steps):
        """运行汤普森采样算法"""
        total_reward = 0  # 累积奖励
        chosen_arms = []  # 记录每一步选择的臂

        for _ in range(n_steps):
            chosen_arm = self.select_arm()  # 选择一个臂
            reward = np.random.binomial(1, rewards_distribution[chosen_arm])  # 生成0或1的奖励
            self.update(chosen_arm, reward)  # 更新奖励分布参数
            total_reward += reward  # 累积总奖励
            chosen_arms.append(chosen_arm)  # 记录选择的臂

        return total_reward, chosen_arms

# 示例使用
n_arms = 3  # 假设有3个老虎机臂
n_steps = 1000  # 总共进行1000次选择
rewards_distribution = [0.2, 0.5, 0.8]  # 假设每个臂的真实成功概率

# 创建汤普森采样算法实例
algo = ThompsonSampling(n_arms)
total_reward, chosen_arms = algo.run(rewards_distribution, n_steps)

print("总奖励:", total_reward)
print("每个臂被选择的次数:", np.bincount(chosen_arms))



# 记录每台老虎机的中奖与否：在每次拉动某个老虎机（即选择某个臂）后，算法会记录下这次操作是否获得了奖励（即中奖或失败）。

# 使用Beta分布选择最优的老虎机：根据记录的中奖和失败次数，算法为每个老虎机（臂）构建一个对应的Beta分布。在每次选择时，算法从这些Beta分布中采样，并选择采样值最高的老虎机（即选择认为最有可能成功的臂）。

# 记录reward和中奖与否：在选择并拉动了某个老虎机后，算法会根据结果（reward）更新该老虎机的成功和失败计数。成功时增加成功次数，失败时增加失败次数。这些更新会在后续选择中影响Beta分布，从而动态调整每个老虎机的选择概率。