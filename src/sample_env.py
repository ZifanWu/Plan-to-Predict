import gym
import numpy as np

class EnvSampler():
    def __init__(self, args, env, max_path_length=1000):
        self.env = env
        self.args = args

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.sum_reward = 0
        
        self.cur_s_for_RLmodel = None

    def sample(self, agent, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset()
            if self.args.env_name == 'Ant-v2':
                self.current_state = self.current_state[:27]
            elif self.args.env_name == 'Humanoid-v2' or 'HumanoidStandup-v2':
                self.current_state = self.current_state[:45]
            elif self.args.env_name == 'InvertedDoublePendulum-v2':
                if self.args.noisy:
                    self.current_state = np.random.normal(loc=self.current_state, scale=self.args.NoiseRatio, size=self.current_state.shape)
            self.cur_s_for_RLmodel = self.current_state.copy()

        cur_state = self.current_state
        action = agent.select_action(self.current_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)
        if self.args.env_name == 'Ant-v2':
            next_state = next_state[:27]
        elif self.args.env_name == 'Humanoid-v2' or 'HumanoidStandup-v2':
            next_state = next_state[:45]
        elif self.args.env_name == 'InvertedDoublePendulum-v2':
            if self.args.noisy:
                next_state = np.random.normal(loc=self.current_state, scale=self.args.NoiseRatio, size=next_state.shape)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state
            self.cur_s_for_RLmodel = next_state

        return cur_state, action, next_state, reward, terminal, info

    def get_ter_action(self, agent):
        action = agent.select_action(self.cur_s_for_RLmodel, eval=False)
        return action
