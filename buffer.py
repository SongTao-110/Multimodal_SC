import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        # 初始化经验回放池的大小、批量大小和计数器
        self.mem_size = max_size  # 经验池的最大容量
        self.batch_size = batch_size  # 每次采样的数据量
        self.mem_cnt = 0  # 存储经验的计数器

        # 初始化存储状态、动作、奖励、下一个状态和终止标志的数组
        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)  # 状态的存储池
        self.action_memory = np.zeros((self.mem_size,), dtype=np.int64)  # 动作的存储池（明确为整数类型）
        self.reward_memory = np.zeros((self.mem_size,), dtype=np.float32)  # 奖励的存储池
        self.next_state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)  # 下一个状态的存储池
        self.terminal_memory = np.zeros((self.mem_size,), dtype=np.bool_)  # 终止标志的存储池

    def store_transition(self, state, action, reward, state_, done):
        # 根据当前存储计数器位置存储新的经验
        mem_idx = self.mem_cnt % self.mem_size  # 计算存储索引

        # 将当前的状态、动作、奖励、下一个状态和终止标志存储在对应位置
        self.state_memory[mem_idx] = state  # 存储状态
        self.action_memory[mem_idx] = action  # 存储动作
        self.reward_memory[mem_idx] = reward  # 存储奖励
        self.next_state_memory[mem_idx] = state_  # 存储下一个状态
        self.terminal_memory[mem_idx] = done  # 存储终止标志

        # 更新经验池计数器
        self.mem_cnt += 1

    def sample_buffer(self):
        # 确定可用的存储数量，获取最小的经验池大小和计数器
        mem_len = min(self.mem_size, self.mem_cnt)

        # 随机选择批量大小的索引，形成批量数据
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        # 根据选择的索引，获取批量的状态、动作、奖励、下一个状态和终止标志
        states = self.state_memory[batch]  # 批量状态
        actions = self.action_memory[batch]  # 批量动作
        rewards = self.reward_memory[batch]  # 批量奖励
        states_ = self.next_state_memory[batch]  # 批量下一个状态
        terminals = self.terminal_memory[batch]  # 批量终止标志

        return states, actions, rewards, states_, terminals

    def ready(self):
        # 判断当前经验池中存储的数据是否足够进行采样
        return self.mem_cnt >= self.batch_size  # 确保有足够的数据进行采样
