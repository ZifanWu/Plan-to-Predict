import random
import numpy as np
from operator import itemgetter

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.init_state_buffer = []
        self.eps_count = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = [state, action, reward, next_state, done]
        self.position = (self.position + 1) % self.capacity
        # NOTE
        if done:
            self.eps_count = 0
        else:
            self.eps_count += 1
        if self.eps_count == 1: # s0
            self.init_state_buffer.append(state)

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * int(append_len))

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample_init_state(self, batch_size):
        idxes = np.random.randint(0, len(self.init_state_buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.init_state_buffer))
        init_states = np.stack(batch)
        return init_states

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_for_RL(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done, horizon = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, horizon

    def push_for_RL(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = [state, action, reward, next_state, done, 1]
        self.position = (self.position + 1) % self.capacity

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)


class ReorderReplayMemory:
    def __init__(self, args):
        replay_size = int(args.env_retain_epochs * args.epoch_length)
        self.buffer = ReplayMemory(replay_size)
        self.ret_buf = []
        self.buf_ptr = 0        
        self.args = args
        self.last_state = None # s_t
        self.last_action = None # a_t
        self.last_reward = None # \r_{t+1}
        self.last_pred_next_s, self.last_pred_r = None, None # \hat{s}_{t+1}, \hat{r}_{t+1}
        self.last_m_LLreward = None
        self.rew_buffer = []
        self.s_mean = None
        self.s_std = None
        self.r_mean = None
        self.r_std = None
        self.eps_count = 0

    def _update_state_meanvar(self, new_s):
        old_s_mean = self.s_mean
        self.s_mean = (len(self.buffer) * self.s_mean + new_s) / (len(self.buffer) + 1)
        new_s_var = (len(self.buffer) * (self.s_std**2 + (self.s_mean - old_s_mean)**2) + (self.s_mean - new_s)**2) / (len(self.buffer) + 1)
        self.s_std = np.sqrt(new_s_var)
    
    def _update_rew_meanvar(self, new_r):
        old_r_mean = self.r_mean
        self.r_mean = (len(self.buffer) * self.r_mean + new_r) / (len(self.buffer) + 1)
        new_r_var = (len(self.buffer) * (self.r_std**2 + (self.r_mean - old_r_mean)**2) + (self.r_mean - new_r)**2) / (len(self.buffer) + 1)
        self.r_std = np.sqrt(new_r_var)

    def push(self, state, action, reward, next_state, done, pred_next_s, pred_r, step, m_LLreward=None):
        state, action, reward, next_state = state.copy(), action.copy(), reward.copy(), next_state.copy()
        if done: # last step
            m_state, m_action, m_reward, m_next_state, ter_m_state, ter_m_action, ter_m_reward, ter_m_next_state = \
                self._reorder_transitions(state, action, reward, next_state, pred_next_s, pred_r, m_LLreward, done)
            self._update_state_meanvar(m_state)
            self._update_rew_meanvar(m_reward)
            self.buffer.push_for_RL(m_state, m_action, m_reward, m_next_state, False)
            self._update_state_meanvar(ter_m_state)
            self._update_rew_meanvar(ter_m_reward)
            self.buffer.push_for_RL(ter_m_state, ter_m_action, ter_m_reward, ter_m_next_state, True)
            self.buf_ptr = self.buf_ptr + 2 # % self.args.replay_size
            self.eps_count += 1

            self.last_done_idx = self.buf_ptr
            # self._compute_return()
            self.eps_count = 0
        elif self.eps_count == 0: # first step
            self.last_state, self.last_action, self.last_reward = state, action, np.array([reward])
            self.last_pred_next_s, self.last_pred_r = pred_next_s, pred_r
            self.last_m_LLreward = m_LLreward
            self.eps_count += 1
        else:
            m_state, m_action, m_reward, m_next_state = self._reorder_transitions(state, action, reward, next_state, pred_next_s, pred_r, m_LLreward, done)
            if len(self.buffer) == 0:
                self.s_mean = m_state
                self.s_std = np.zeros_like(m_state)
                self.r_mean = m_reward
                self.r_std = np.zeros_like(m_reward)
            self._update_state_meanvar(m_state)
            self._update_rew_meanvar(m_reward)
            self.buffer.push_for_RL(m_state, m_action, m_reward, m_next_state, False)
            self.buf_ptr = self.buf_ptr + 1 #  % self.args.replay_size
            self.eps_count += 1

    def _reorder_transitions(self, state, action, reward, next_state, pred_next_s, pred_r, m_LLreward, done):
        if done:
            m_state = np.concatenate((self.last_state, self.last_action), axis=0)
            m_action = np.concatenate((self.last_reward, state - self.last_state), axis=-1)
            m_action = m_action if len(m_action.shape)==1 else m_action[0]
            if self.args.sqrt_rew:
                m_reward = - np.sqrt(((self.last_pred_next_s - state)**2).sum()) - self.args.beta * np.sqrt(((self.last_pred_r - self.last_reward)**2).sum())
            else:
                if self.args.abs_rew:
                    m_reward = - (np.abs(self.last_pred_next_s - state)).mean() - self.args.beta * (np.abs(self.last_pred_r - self.last_reward)).mean()
                else:
                    m_reward = - ((self.last_pred_next_s - state)**2).mean() - self.args.beta * ((self.last_pred_r - self.last_reward)**2).mean()
            m_next_state = np.concatenate((state, action), axis=0)
            m_reward = self.last_m_LLreward if m_LLreward is not None else m_reward

            ter_m_state = m_next_state.copy()
            ter_m_action = np.concatenate((np.array([reward]), next_state - state), axis=-1)
            ter_m_action = ter_m_action if len(ter_m_action.shape)==1 else ter_m_action[0]
            if self.args.sqrt_rew:
                ter_m_reward = - np.sqrt(((pred_next_s - next_state)**2).sum()) - self.args.beta * np.sqrt(((pred_r - reward)**2).sum())
            else:
                if self.args.abs_rew:
                    ter_m_reward = - (np.abs(pred_next_s - next_state)).mean() - self.args.beta * (np.abs(pred_r - reward)).mean()
                else:
                    ter_m_reward = - ((pred_next_s - next_state)**2).mean() - self.args.beta * ((pred_r - reward)**2).mean()
            next_action = np.zeros_like(action)
            ter_m_next_state = np.concatenate((next_state, next_action), axis=0)
            ter_m_reward = m_LLreward if m_LLreward is not None else ter_m_reward
            return m_state, m_action, m_reward, m_next_state, \
                ter_m_state, ter_m_action, ter_m_reward, ter_m_next_state
        else:
            m_state = np.concatenate((self.last_state, self.last_action), axis=0)
            m_action = np.concatenate((self.last_reward, state - self.last_state), axis=-1)
            m_action = m_action if len(m_action.shape)==1 else m_action[0]
            if self.args.sqrt_rew:
                m_reward = - np.sqrt(((self.last_pred_next_s - state)**2).sum()) - self.args.beta * np.sqrt(((self.last_pred_r - self.last_reward)**2).sum())
            else:
                if self.args.abs_rew:
                    m_reward = - (np.abs(self.last_pred_next_s - state)).mean() - self.args.beta * (np.abs(self.last_pred_r - self.last_reward)).mean()
                else:
                    m_reward = - ((self.last_pred_next_s - state)**2).mean() - self.args.beta * ((self.last_pred_r - self.last_reward)**2).mean()
            m_next_state = np.concatenate((state, action), axis=0)

            self.last_state, self.last_action, self.last_reward = state, action, np.array([reward])
            self.last_pred_next_s, self.last_pred_r = pred_next_s, pred_r
            m_reward = self.last_m_LLreward if m_LLreward is not None else m_reward
            self.last_m_LLreward = m_LLreward
            return m_state, m_action, m_reward, m_next_state

    def _compute_return(self):
        st_idx = self.buf_ptr - self.eps_count
        for j in range(self.eps_count):
            if j <= self.eps_count - self.args.n_boots: # j: 0:17-2. n_boots=2: R + γR' + γ^2 Q(s'', pi(s''))
                self.buffer.buffer[st_idx + j][3] = self.buffer.buffer[st_idx + j + self.args.n_boots - 1][3] # s_{t+H}
                ret_j = self.buffer.buffer[st_idx + j][2]
                for k in range(1, self.args.n_boots):
                    ret_j += self.buffer.buffer[st_idx + j + k][2] * self.args.model_gamma**k
                self.buffer.buffer[st_idx + j][2] = ret_j # H_step return
                self.buffer.buffer[st_idx + j].append(self.args.n_boots)
            else:
                self.buffer.buffer[st_idx + j][3] = self.buffer.buffer[self.buf_ptr - 1][3]
                ret_j = self.buffer.buffer[st_idx + j][2]
                for k in range(1, self.eps_count - j):
                    ret_j += self.buffer.buffer[j + k][2] * self.args.model_gamma**k
                self.buffer.buffer[st_idx + j][2] = ret_j
                self.buffer.buffer[st_idx + j].append(self.eps_count - j)

    def sample(self, batch_size):
        # if batch_size > self.last_done_idx:
        #     batch_size = self.last_done_idx - 1
        state, action, reward, next_state, done, horizon = self.buffer.sample_for_RL(batch_size) #, self.last_done_idx - 1) # len: batch_size, 5(s,a,r,s',d).  shape(_dim)
        # normarlize states
        # if self.args.use_state_normalization:
        #     state = (state - self.s_mean) / (self.s_std + self.args.epsilon)
        # if self.args.scale_rew or self.args.NormRew:
        #     assert (self.args.scale_rew != self.args.NormRew)
        #     if self.args.scale_rew:
        #         reward = reward / (self.r_std + self.args.epsilon)
        #     else:
        #         reward = (reward - self.r_mean) / (self.r_std + self.args.epsilon)
        return state, action, reward, next_state, done, horizon

    def __len__(self):
        return len(self.buffer)
