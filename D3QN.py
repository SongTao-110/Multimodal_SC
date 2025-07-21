import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)


        self.value_stream = nn.Linear(fc2_dim, 1)


        self.advantage_stream = nn.Linear(fc2_dim, action_dim)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))


        value = self.value_stream(x)
        advantage = self.advantage_stream(x)


        Q = value + (advantage - advantage.mean(dim=-1, keepdim=True)) 
        return Q

    def predict(self, state, deterministic=True):
        state_tensor = T.tensor(state, dtype=T.float).to(device)

        if state_tensor.dim() == 1: 
            state_tensor = state_tensor.unsqueeze(0)
        elif state_tensor.dim() != 2 or state_tensor.shape[1] != self.state_dim:
            raise ValueError(f"Invalid state shape: {state_tensor.shape}. Expected (1, {self.state_dim}).")

 
        q_vals = self.forward(state_tensor)

        action = T.argmax(q_vals, dim=-1).item()

        return action, None  


    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)


    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class D3QN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.01, epsilon=1.0, eps_end=0.05, eps_dec=1e-4,
                 max_size=1000000, batch_size=64, update_steps=20):

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = [i for i in range(action_dim)]
        self.update_step_counter = 0
        self.update_steps = update_steps


        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params.data + (1 - tau) * q_target_params.data)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
       
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def choose_action(self, observation, isTrain=True):
        state = T.tensor(observation, dtype=T.float).to(device)

      
        if state.dim() == 1:  
            state = state.unsqueeze(0)
        elif state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"Invalid state shape: {state.shape}. Expected (1, {self.state_dim}).")

   
        q_vals = self.q_eval.forward(state)
        action = T.argmax(q_vals).item()

      
        if isTrain and np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if not self.memory.ready():
            return

  
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()

  
        states_tensor = T.tensor(states, dtype=T.float32).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float32).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float32).to(device)
        terminals_tensor = T.tensor(terminals, dtype=T.bool).to(device)

        with T.no_grad():

            q_next = self.q_target.forward(next_states_tensor)
            max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)


            q_target = rewards_tensor + self.gamma * T.gather(q_next, 1, max_actions.unsqueeze(1)).squeeze(1)


            q_target[terminals_tensor] = rewards_tensor[terminals_tensor]

        q = self.q_eval.forward(states_tensor)[T.arange(self.batch_size), actions_tensor]

        loss = F.mse_loss(q, q_target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()


        self.update_network_parameters()


        self.q_eval.optimizer.step()

        self.update_step_counter += 1
        if self.update_step_counter % self.update_steps == 0:
            self.update_network_parameters()

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + f'Q_eval/D3QN_q_eval_{episode}.pth')
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + f'Q_target/D3QN_Q_target_{episode}.pth')
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + f'Q_eval/D3QN_q_eval_{episode}.pth')
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + f'Q_target/D3QN_Q_target_{episode}.pth')
        print('Loading Q_target network successfully!')
        return self.q_eval
