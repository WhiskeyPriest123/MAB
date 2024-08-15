import numpy as np
from tqdm import tqdm

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms  # 臂的数量
        self.epsilon = epsilon  # 探索的概率
        self.counts = np.zeros(n_arms)  # 每个臂被选择的次数
        self.values = np.zeros(n_arms)  # 每个臂的平均奖励估计值
        print(self.counts)
        print(self.values)

    def select_arm(self):
        """选择臂的策略：探索或利用"""
        if np.random.rand() < self.epsilon:
            # 探索：随机选择一个臂
            return np.random.randint(0, self.n_arms)
        else:
            # 利用：选择当前已知回报最高的臂
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """根据选择的臂和获得的奖励更新估计值"""
        self.counts[chosen_arm] += 1  # 增加选择计数
        n = self.counts[chosen_arm]  # 该臂被选择的总次数
        value = self.values[chosen_arm]  # 当前的平均奖励
        # 更新平均奖励估计值 (增量更新公式)
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def run(self, rewards_distribution, n_steps):
        """运行ε-贪婪算法"""
        total_reward = 0  # 累积奖励
        chosen_arms = []  # 记录每一步选择的臂

        for _ in tqdm(range(n_steps)):
            chosen_arm = self.select_arm()  # 选择一个臂
            reward = np.random.normal(rewards_distribution[chosen_arm])  # 根据分布获得奖励
            self.update(chosen_arm, reward)  # 更新奖励估计值
            total_reward += reward  # 累积总奖励
            chosen_arms.append(chosen_arm)  # 记录选择的臂

        return total_reward, chosen_arms



if __name__ == "__main__":
    # 示例使用
    n_arms = 3  # 假设有3个老虎机臂
    epsilon = 0.1  # 10%的探索概率
    n_steps = 1000  # 总共进行1000次选择
    rewards_distribution = [0.2, 0.5, 0.8]  # 假设每个臂的真实平均奖励

    # 创建ε-贪婪算法实例
    algo = EpsilonGreedy(n_arms, epsilon)
    total_reward, chosen_arms = algo.run(rewards_distribution, n_steps)

    print("总奖励:", total_reward)
    print("每个臂被选择的次数:", algo.counts)



## 探索（Exploration）：在探索阶段，算法会根据 ε（探索概率）的值决定是否随机选择一个臂。这个阶段的目的是尝试不同的选择，以获取更多的信息。

# 利用（Exploitation）：如果不进行探索，算法就会利用当前已有的信息，选择那些历史上表现最好的臂，即选择回报率最高的臂。

# 计算奖励（Reward Calculation）：无论是探索还是利用，一旦选择了一个臂，算法都会根据选择的臂的真实奖励分布（通常通过模拟）来计算一个即时奖励（reward）。

# 信息更新（Update Information）：最后，算法将当前获得的奖励整合到现有的信息中（即更新该臂的平均奖励估计值），以便在后续的选择中做出更好的决策。

