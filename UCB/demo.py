import numpy as np

class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms  # 臂的数量
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的平均奖励估计值

    def select_arm(self, t):
        """选择臂的策略：根据上置信界"""
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm  # 如果有未被选择过的臂，优先选择
            average_reward = self.values[arm]
            delta = np.sqrt((2 * np.log(t)) / self.counts[arm])
            ucb_values[arm] = average_reward + delta
        return np.argmax(ucb_values)  # 选择UCB值最大的臂，

    def update(self, chosen_arm, reward):
        """根据选择的臂和获得的奖励更新估计值"""
        self.counts[chosen_arm] += 1  # 增加选择计数
        n = self.counts[chosen_arm]  # 该臂被选择的总次数
        value = self.values[chosen_arm]  # 当前的平均奖励
        # 更新平均奖励估计值 (增量更新公式)
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def run(self, rewards_distribution, n_steps):
        """运行UCB算法"""
        total_reward = 0  # 累积奖励
        chosen_arms = []  # 记录每一步选择的臂

        for t in range(1, n_steps + 1):
            chosen_arm = self.select_arm(t)  # 选择一个臂
            reward = np.random.normal(rewards_distribution[chosen_arm])  # 根据分布获得奖励
            self.update(chosen_arm, reward)  # 更新奖励估计值
            total_reward += reward  # 累积总奖励
            chosen_arms.append(chosen_arm)  # 记录选择的臂

        return total_reward, chosen_arms

# 示例使用
n_arms = 3  # 假设有3个老虎机臂
n_steps = 1000  # 总共进行1000次选择
rewards_distribution = [0.2, 0.5, 0.8]  # 假设每个臂的真实平均奖励

# 创建UCB算法实例
algo = UCB(n_arms)
total_reward, chosen_arms = algo.run(rewards_distribution, n_steps)

print("总奖励:", total_reward)
print("每个臂被选择的次数:", algo.counts)


## UCB方法相当于将原文中的记录reward 变成 UCB reward