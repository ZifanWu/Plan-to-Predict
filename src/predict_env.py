from email import policy
import re
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sac.utils import soft_update, hard_update
from sac.model import QNetwork, QNetwork_for_Model, nu_zeta_Network
import copy
import torch.nn.functional as F
import itertools


class StandardScaler(object):
    def __init__(self):
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


class PredictEnv:
    def __init__(self, args, model, env_name, model_type):
        self.model = model # cat(s_mean, r_mean), cat(s_var, r_var) =model.act(cat(s,a), b_size)
        self.env_name = env_name
        self.model_type = model_type
        self.args = args
        self.device = torch.device('cuda')
        if args.train_model_by_RL:
            self.alpha = torch.Tensor([args.model_alpha]).to(self.device)
            Q = QNetwork_for_Model if args.deeper_Q_net else QNetwork
            self.critic = Q(args.s_dim + args.a_dim, args.s_dim + 1, args.hidden_size).to(self.device)
            self.critic_target = Q(args.s_dim + args.a_dim, args.s_dim + 1, args.hidden_size).to(self.device)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.model_q_lr)
            hard_update(self.critic_target, self.critic)
            self.q_mean = None
            self.a_mean = None
            self.train_count = 0
            if args.use_state_normalization:
                self.s_mean = None
                self.s_std = None
            if args.deter_model:
                self.actor = model.ensemble_model
                self.actor_target = copy.deepcopy(self.actor)
                self.policy_optim = optim.Adam(self.actor.parameters(), lr=args.RLmodel_lr)
            else:
                self.policy = model
                self.policy_optim = optim.Adam(self.policy.ensemble_model.parameters(), lr=args.RLmodel_lr)
                # alpha tuning
                self.target_entropy = torch.Tensor([np.log(args.s_dim + 1)]).to(self.device)
                self.log_alpha = torch.Tensor(np.log([args.model_alpha])).to(self.device) # torch.zeros(1, requires_grad=True).to(self.device)
                self.log_alpha.requires_grad = True
                self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)
            if args.DualDICE:
                self.nu_net = nu_zeta_Network(args.s_dim + args.a_dim, args.s_dim + 1, args.hidden_size).to(self.device)
                self.nu_optim = optim.Adam(self.nu_net.parameters(), lr=args.nulr)
                self.zeta_net = nu_zeta_Network(args.s_dim + args.a_dim, args.s_dim + 1, args.hidden_size).to(self.device)
                self.zeta_optim = optim.Adam(self.zeta_net.parameters(), lr=args.zetalr)
                conjugate_exponent = self.args.function_exponent / (self.args.function_exponent - 1)
                self._f_star = lambda x: torch.abs(x) ** conjugate_exponent / conjugate_exponent
                self.if_zeta_trained = False
            if args.rew_norm or args.rew_scaling:
                self.r_mean, self.r_std = 0, 1
            if args.LL_reward and args.EnsFusionRew:
                self.fusion_para = (1/args.num_networks) * torch.ones(args.num_networks).to(self.device)
                self.fusion_para.requires_grad = True
                self.fusion_para_optim = optim.Adam([self.fusion_para], lr=3e-4)
            if args.MPCmodel:
                self.m_r_predictor = QNetwork_for_Model(args.s_dim + args.a_dim, args.s_dim + 1, args.hidden_size).to(self.device) # input: (s^m, a^m); output: \hat{r}^m
                self.m_r_pred_optim = optim.Adam(self.m_r_predictor.parameters(), lr=5e-4)
                self.scaler = StandardScaler()

    def _termination_fn(self, env_name, obs, act, next_obs):
        if env_name == "Hopper-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            # print(done.shape)
            return done
        elif env_name == "Walker2d-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == 'HalfCheetah-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:,None]
            return done
        elif env_name == 'Ant-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            x = next_obs[:, 0]
            not_done = 	np.isfinite(next_obs).all(axis=-1) \
                        * (x >= 0.2) \
                        * (x <= 1.0)

            done = ~not_done
            done = done[:,None]
            return done
        elif env_name == 'InvertedDoublePendulum-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            sin1, cos1 = next_obs[:,1], next_obs[:,3]
            sin2, cos2 = next_obs[:,2], next_obs[:,4]
            theta_1 = np.arctan2(sin1, cos1)
            theta_2 = np.arctan2(sin2, cos2)
            y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

            done = y <= 1
            
            done = done[:,None]
            return done
        elif env_name == 'InvertedPendulum-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            notdone = np.isfinite(next_obs).all(axis=-1) \
                    * (np.abs(next_obs[:,1]) <= .2)
            done = ~notdone

            done = done[:,None]

            return done
        elif env_name == 'Humanoid-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            z = next_obs[:,0]
            done = (z < 1.0) + (z > 2.0)

            done = done[:,None]
            return done
        elif env_name == 'Swimmer-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            B = obs.shape[0]
            done = np.zeros((B, 1)).astype(bool)
            return done
        elif env_name == 'Reacher-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            B = obs.shape[0]
            done = np.zeros((B, 1)).astype(bool)
            return done
        elif env_name == 'Pusher-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            B = obs.shape[0]
            done = np.zeros((B, 1)).astype(bool)
            return done
        elif env_name == 'Striker-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            B = obs.shape[0]
            done = np.zeros((B, 1)).astype(bool)
            return done
        elif env_name == 'Thrower-v2':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            B = obs.shape[0]
            done = np.zeros((B, 1)).astype(bool)
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1) # (n_particles, s_dim+a_dim)
        if self.args.use_state_normalization and self.args.NorSInPred:
            if self.s_mean is not None:
                if self.args.NorSOnly:
                    S = self.args.s_dim
                    inputs[:, :S] = (inputs[:, :S] - self.s_mean[:, :S].repeat(inputs.shape[0], 1).cpu().numpy()) / self.s_std[:, :S].repeat(inputs.shape[0], 1).cpu().numpy()
                else:
                    inputs = (inputs - self.s_mean.repeat(inputs.shape[0], 1).cpu().numpy()) / self.s_std.repeat(inputs.shape[0], 1).cpu().numpy()
        if self.args.deter_model:
            ensemble_model_means = self.model.deter_predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)

        ensemble_model_means[:, :, 1:] += obs # predict (s'-s)
        if not self.args.deter_model:
            ensemble_model_stds = np.sqrt(ensemble_model_vars)

        # if deterministic:
        #     ensemble_samples = ensemble_model_means
        # else:
        if self.args.deter_model:
            noise = (torch.randn_like(torch.from_numpy(ensemble_model_means)) * self.args.DMoNoise).clamp(-self.args.ClipDMoNoise, self.args.ClipDMoNoise)
            ensemble_samples = ensemble_model_means + noise.numpy()
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        if not self.args.deter_model:
            model_stds = ensemble_model_stds[model_idxes, batch_idxes]
            # log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        # print(model_means.shape, model_means.shape) # (1, _dim)
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        if not self.args.deter_model:
            return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        # info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, None

    def step_for_RLpred(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
        inputs = np.concatenate((obs, act), axis=-1)
        if self.args.use_state_normalization and self.args.NorSInPred:
            if self.s_mean is not None:
                if self.args.NorSOnly:
                    S = self.args.s_dim
                    inputs[:, :S] = (inputs[:, :S] - self.s_mean[:, :S].repeat(inputs.shape[0], 1).cpu().numpy()) / self.s_std[:, :S].repeat(inputs.shape[0], 1).cpu().numpy()
                else:
                    inputs = (inputs - self.s_mean.repeat(inputs.shape[0], 1).cpu().numpy()) / self.s_std.repeat(inputs.shape[0], 1).cpu().numpy()
        if self.args.deter_model:
            ensemble_model_means = self.model.deter_predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        if not self.args.deter_model:
            ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if self.args.deter_model:
            noise = (torch.randn_like(torch.from_numpy(ensemble_model_means)) * self.args.DMoNoise).clamp(-self.args.ClipDMoNoise, self.args.ClipDMoNoise)
            ensemble_samples = ensemble_model_means + noise.numpy()
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size) 
        batch_idxes = np.arange(0, batch_size)
        samples = ensemble_samples[model_idxes, batch_idxes]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        if self.args.deter_model:
            return next_obs[0], rewards[0]
        else:
            en_mu_mean, en_sig_mean = ensemble_model_means.mean(0)[0], ensemble_model_stds.mean(0)[0] # (_dim,)
            if not self.args.MPCmodel:
                next_obs, rewards = next_obs[0], rewards[0]
            return next_obs, rewards, en_mu_mean, en_sig_mean, terminals

    def step_for_MPC(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
        inputs = np.concatenate((obs, act), axis=-1)
        if self.args.use_state_normalization and self.args.NorSInPred:
            if self.s_mean is not None:
                if self.args.NorSOnly:
                    S = self.args.s_dim
                    inputs[:, :S] = (inputs[:, :S] - self.s_mean[:, :S].repeat(inputs.shape[0], 1).cpu().numpy()) / self.s_std[:, :S].repeat(inputs.shape[0], 1).cpu().numpy()
                else:
                    inputs = (inputs - self.s_mean.repeat(inputs.shape[0], 1).cpu().numpy()) / self.s_std.repeat(inputs.shape[0], 1).cpu().numpy()
        if self.args.deter_model:
            ensemble_model_means = self.model.deter_predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, 1:] += obs
        if not self.args.deter_model:
            ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if self.args.deter_model:
            noise = (torch.randn_like(torch.from_numpy(ensemble_model_means)) * self.args.DMoNoise).clamp(-self.args.ClipDMoNoise, self.args.ClipDMoNoise)
            ensemble_samples = ensemble_model_means + noise.numpy()
        else:
            ensemble_samples = ensemble_model_means # + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size) 
        batch_idxes = np.arange(0, batch_size)
        samples = ensemble_samples[model_idxes, batch_idxes]
        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        if self.args.deter_model:
            return next_obs[0], rewards[0]
        else:
            en_mu_mean, en_sig_mean = ensemble_model_means.mean(0)[0], ensemble_model_stds.mean(0)[0] # (_dim,)
            return next_obs, rewards, en_mu_mean, en_sig_mean, terminals

    def train_m_r_predictor(self, s, a, r, ns):
        pred_ns, pred_r, _, _, _ = self.step_for_RLpred(s, a)
        label = ((pred_ns - ns)**2).mean(-1) + self.args.beta * ((pred_r - r)**2).mean(-1)
        label = torch.from_numpy(label).unsqueeze(-1).numpy()

        m_s = np.concatenate((s, a), axis=-1)
        m_a = np.concatenate((pred_r, pred_ns - s), axis=-1)
        input = np.concatenate((m_s, m_a), axis=-1) # (len(pool), dim)
        self.SLtrain(input, label) # (B, dim) (B, 1)

    def SLtrain(self, inputs, labels, batch_size=256, holdout_ratio=0.2, max_epochs_since_update=5):
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(1)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs) # (B, s+a+s+r_dim)
        train_inputs = self.scaler.transform(train_inputs) # (len(env_pool), _dim)
        holdout_inputs = self.scaler.transform(holdout_inputs)

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(self.device)
        # holdout_inputs = holdout_inputs[None, :, :].repeat([1, 1, 1])
        # holdout_labels = holdout_labels[None, :, :].repeat([1, 1, 1])
        for epoch in itertools.count():
            train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(1)])
            for start_pos in range(0, train_inputs.shape[0], batch_size):
                idx = train_idx[start_pos: start_pos + batch_size] # (n_ens, b_size)
                train_input = torch.from_numpy(train_inputs[idx]).float().to(self.device) # (n_ens, b_size, _dim)
                train_label = torch.from_numpy(train_labels[idx]).float().to(self.device)
                if self.args.deter_model:
                    mean = self.m_r_predictor(train_input, ret_log_var=True)
                    loss, _ = self.ensemble_model.deter_loss(mean, train_label)
                else:
                    mean = self.m_r_predictor(train_input.squeeze())
                    loss = torch.pow(mean - train_label, 2).mean()
                self.m_r_pred_optim.zero_grad()
                loss.backward()
                self.m_r_pred_optim.step()
            with torch.no_grad():
                if self.args.deter_model:
                    holdout_mean = self.ensemble_model(holdout_inputs, ret_log_var=True)
                    _, holdout_mse_losses = self.ensemble_model.deter_loss(holdout_mean, holdout_labels)
                else:
                    holdout_mean = self.m_r_predictor(holdout_inputs)
                    holdout_loss = torch.pow(holdout_mean - holdout_labels, 2).mean().unsqueeze(0) # (1,)
                holdout_loss = holdout_loss.detach().cpu().numpy()
                break_train = self._save_best(epoch, holdout_loss)
                if break_train:
                    # print('number of hat(r)^m training iteration: {}'.format(epoch))
                    break

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True
        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def MPCact(self, init_s, init_a, t, H, agent):
        # s = s.unsqueeze(0).repeat(self.args.n_trajs, 1, 1) # (n_trajs, B, _dim) 并行rollout
        min_accumu_error = 99999999
        optimal_m_a = None
        optimal_ter = None
        for j in range(self.args.n_trajs):
            accumu_error_j = 0
            for step in range(int(H-t)):
                if step == 0:
                    s, a = init_s, init_a
                pred_next_s, pred_r, _, _, terminals = self.step_for_RLpred(s, a)
                m_s, m_a = np.concatenate((s, a), axis=-1), np.concatenate((pred_r, pred_next_s - s), axis=-1)
                input = torch.Tensor(np.concatenate((m_s, m_a), axis=-1)).to(self.device) # (rollB, _dim)
                pred_error = self.m_r_predictor(input) # (rollB, 1)
                accumu_error_j += pred_error.mean()
                nonterm_mask = ~terminals.squeeze(-1)
                if step == 0:
                    j_first_m_a = (pred_next_s, pred_r)
                    j_first_ter = terminals
                if nonterm_mask.sum() == 0:
                    break
                s = pred_next_s
                a = agent.select_action(s)
            if accumu_error_j < min_accumu_error:
                min_accumu_error = accumu_error_j
                optimal_m_a = j_first_m_a
                optimal_ter = j_first_ter
        pred_s, pred_r = optimal_m_a
        return pred_s, pred_r, optimal_ter

    def ParaMPCact(self, init_s, init_a, t, H, agent):
        B = init_s.shape[0]
        PlanningHorizon = int(min(self.args.MPCHorizon, H-t)) if self.args.Limit_Horizon else int(H-t)
        if H > self.args.max_plan_times:
            if t % self.args.plan_every_n_steps != 0:
                PlanningHorizon = 1
        init_s = torch.from_numpy(init_s).unsqueeze(0).repeat(self.args.n_trajs, 1, 1).view(self.args.n_trajs*B, -1) # (n_trajs, B, _dim) 并行rollout
        init_a = torch.from_numpy(init_a).unsqueeze(0).repeat(self.args.n_trajs, 1, 1).view(self.args.n_trajs*B, -1)
        error_j = torch.zeros(self.args.n_trajs, PlanningHorizon)
        for step in range(PlanningHorizon):
            if step == 0:
                s, a = init_s.numpy(), init_a.numpy()
            with torch.no_grad():
                pred_next_s, pred_r, _, _, terminals = self.step_for_MPC(s, a) if self.args.DeterMoForMPC else self.step_for_RLpred(s, a)
                m_s, m_a = np.concatenate((s, a), axis=-1), np.concatenate((pred_r, pred_next_s - s), axis=-1)
                input = torch.Tensor(np.concatenate((m_s, m_a), axis=-1)).to(self.device) # (rollB*n_trajs, _dim)
                pred_error = self.m_r_predictor(input) # (rollB*n_trajs, 1)
                gamma = self.args.model_gamma if self.args.DiscountMPC else 1
                error_j[:, step] = gamma**step * pred_error.view(self.args.n_trajs, B, 1).mean(1).squeeze()
            nonterm_mask = ~terminals.squeeze(-1)
            if step == 0:
                j_first_preds = torch.from_numpy(pred_next_s).view(self.args.n_trajs, B, -1).numpy()
                j_first_predr = torch.from_numpy(pred_r).view(self.args.n_trajs, B, -1).numpy()
                j_first_ter = torch.from_numpy(terminals).view(self.args.n_trajs, B, -1).numpy()
            if nonterm_mask.sum() == 0:
                break
            s = pred_next_s
            a = agent.select_action(s, eval=True) if self.args.DeterPoForMPC else agent.select_action(s, eval=False)
        accumu_errors = error_j.sum(-1)
        best_traj_num = accumu_errors.argmin()
        optimal_preds = j_first_preds[best_traj_num]
        optimal_predr = j_first_predr[best_traj_num]
        optimal_ter = j_first_ter[best_traj_num]
        return optimal_preds, optimal_predr, optimal_ter

    def SAC_train(self, m_s, m_a, m_r, m_ns, m_m, horizon, step, epoch_step, agent, m_ini_s):
        '''Train the stochastic model with SAC'''
        B = m_s.shape[0]
        state_batch = torch.FloatTensor(m_s).to(self.device) # (B, _dim)
        next_state_batch = torch.FloatTensor(m_ns).to(self.device)
        action_batch = torch.FloatTensor(m_a).to(self.device)
        # reward_batch = torch.FloatTensor(m_r).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(m_m).to(self.device).unsqueeze(1) # (B, 1)
        H_batch = torch.from_numpy(horizon).to(self.device).unsqueeze(1) # (B, 1) # for n-step bootstrapping

        with torch.no_grad():
            # FIXME Update next state (s', pi_{old}(s')) -> (s', pi(s'))
            updated_act = agent.policy.sample(next_state_batch[:, :self.args.s_dim])[2] # mean action of pi( |s')
            next_state_batch[:, self.args.s_dim:] = updated_act
            unnor_s_batch = copy.deepcopy(state_batch)
            unnor_ns_batch = copy.deepcopy(next_state_batch)
            if self.args.use_state_normalization:
                if self.args.NorSOnly:
                    state_batch[:, :self.args.s_dim] = (state_batch[:, :self.args.s_dim] - self.s_mean[:, :self.args.s_dim].repeat(B, 1)) / self.s_std[:, :self.args.s_dim].repeat(B, 1)
                    next_state_batch[:, :self.args.s_dim] = (next_state_batch[:, :self.args.s_dim] - self.s_mean[:, :self.args.s_dim].repeat(B, 1)) / self.s_std[:, :self.args.s_dim].repeat(B, 1)
                else:
                    state_batch = (state_batch - self.s_mean.repeat(B, 1)) / self.s_std.repeat(B, 1)
                    next_state_batch = (next_state_batch - self.s_mean.repeat(B, 1)) / self.s_std.repeat(B, 1)

            # FIXME Update reward 
            inputs = state_batch.cpu().numpy() if self.args.NorSInPred else unnor_s_batch.cpu().numpy()
            en_mus, en_vars = self.model.predict(inputs) # (n_ens, B, s_dim+1)
            E = en_mus.shape[0]
            mean_of_en_mus, std_of_en_mus = en_mus.mean(0), en_mus.std(0)
            if self.args.RewUp: # Update reward r^m_{old}=D(\hat{s'}, s') + beta*D(\hat{r}, r)
                if self.args.LL_reward:
                    if self.args.EnsFusionRew:
                        # h = lambda x: 
                        fus_para = nn.functional.softmax(self.fusion_para, dim=0).unsqueeze(-1).unsqueeze(-1) # .repeat(1, B, self.args.s_dim + 1) # (n_ens, B, s_dim+1)
                        ens_probs = torch.distributions.normal.Normal(torch.from_numpy(en_mus).to(self.device), torch.from_numpy(np.sqrt(en_vars)).to(self.device)).log_prob(action_batch.unsqueeze(0).repeat(E, 1, 1)) # (n_ens, B, s_dim+1)
                        fus_prob = (fus_para * torch.exp(ens_probs)).sum(0) # (B, s_dim+1)
                        en_mu_mean, en_var_mean = torch.from_numpy(mean_of_en_mus).to(self.device), torch.from_numpy(std_of_en_mus).to(self.device) # (B, s_dim+1)
                        mean_prob = torch.exp(torch.distributions.normal.Normal(en_mu_mean, torch.sqrt(en_var_mean)).log_prob(action_batch))
                        reward_batch = torch.log(fus_prob / mean_prob).mean().detach()
                        # train the fusion para
                        fus_para_loss = torch.log(fus_prob).mean()
                        self.fusion_para_optim.zero_grad()
                        fus_para_loss.backward()
                        self.fusion_para_optim.step()
                    else:
                        en_mu_mean, en_var_mean = torch.from_numpy(mean_of_en_mus).to(self.device), torch.from_numpy(std_of_en_mus).to(self.device) # (B, s_dim+1)
                        reward_batch = torch.distributions.normal.Normal(en_mu_mean, torch.sqrt(en_var_mean)).log_prob(action_batch).mean()
                else:
                    pred_next_s, pred_r = mean_of_en_mus[:, 1:], mean_of_en_mus[:, :1] # (B, s_dim) (B, 1)
                    pred_next_s, pred_r = torch.from_numpy(pred_next_s).to(self.device), torch.from_numpy(pred_r).to(self.device)
                    # NOTE!! pred_next_s = s'-s, 
                    pred_next_s += unnor_s_batch[:, :self.args.s_dim]
                    if self.args.use_state_normalization and self.args.NorSWhenComputeRew:
                        pred_next_s = (pred_next_s - self.s_mean.repeat(B, 1)[:, :self.args.s_dim]) / self.s_std.repeat(B, 1)[:, :self.args.s_dim]
                        label1 = next_state_batch
                    else:
                        label1 = unnor_ns_batch
                    if self.args.RSsepa:
                        if self.args.abs_rew:
                            reward_batch = - (torch.abs(pred_next_s - label1[:, :self.args.s_dim])).mean(-1) - self.args.beta * (torch.abs(pred_r - action_batch[:, :1])).mean(-1)
                        else:
                            reward_batch = - ((pred_next_s - label1[:, :self.args.s_dim])**2).mean(-1) - self.args.beta * ((pred_r - action_batch[:, :1])**2).mean(-1)
                    else:
                        label = torch.cat((label1[:, :self.args.s_dim], action_batch[:, :1]), dim=-1) # cat(s', r)
                        output = torch.cat((pred_next_s, pred_r), dim=-1)
                        reward_batch = - ((label - output)**2).mean(-1) # (B,)
            else:
                reward_batch = torch.FloatTensor(m_r).to(self.device)
            reward_batch = reward_batch.unsqueeze(-1) # (B, 1)
            if self.args.rew_scaling:
                reward_batch = reward_batch / (self.r_std + self.args.epsilon)
            elif self.args.rew_norm:
                reward_batch = (reward_batch - self.r_mean) / (self.r_std + self.args.epsilon)
            if self.args.STDPenRew:
                uncertainty_penalty = self.args.PenWeight * torch.from_numpy(std_of_en_mus).mean(-1, keepdim=True).to(self.device)
                reward_batch = reward_batch - uncertainty_penalty

            # FIXME Compute target y=r+gam*q'
            if self.args.use_state_normalization: # and self.args.NorSInPred: # NOTE
                next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch) # (n_ens, B, _dim)
            else:
                next_state_action, next_state_log_pi, _, _ = self.policy.sample(unnor_ns_batch)
            next_state_action, next_state_log_pi = next_state_action.mean(0), next_state_log_pi.mean(0)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi

            # Compute importance ratio
            if self.args.DualDICE and (epoch_step > self.args.WarmUpDualDICE_epo):
                # compute zeta(s,a) for every sample in the batch
                m_s_ratio = self.zeta_net(state_batch, action_batch) # (B, 1)
                if self.args.trgt_DICE:
                    trgt_ratio = self.zeta_net(next_state_batch, next_state_action)
                else:
                    trgt_ratio = torch.ones_like(min_qf_next_target).to(self.device)
            else:
                m_s_ratio = torch.ones_like(min_qf_next_target).to(self.device)
                trgt_ratio = torch.ones_like(min_qf_next_target).to(self.device)
            next_q_value = m_s_ratio * reward_batch.unsqueeze(-1) + mask_batch * self.args.model_gamma**H_batch * (trgt_ratio * min_qf_next_target)

        # value loss
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = ((m_s_ratio * qf1 - next_q_value)**2).mean() # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = ((m_s_ratio * qf2 - next_q_value)**2).mean() # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # policy loss
        if self.args.use_state_normalization: # and self.args.NorSInPred: # NOTE
            pi, log_pi, mean, std = self.policy.sample(state_batch) # (n_ens, B, s_dim+1)
        else:
            pi, log_pi, mean, std = self.policy.sample(unnor_s_batch)
        log_pi = log_pi.mean(0).mean(-1, keepdim=True) # (B, 1)
        qf1_pi, qf2_pi = self.critic(state_batch, pi.mean(0))
        if self.args.OutDICEratio:
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
        else:
            min_qf_pi = m_s_ratio * torch.min(qf1_pi, qf2_pi) # (B, 1)
        if self.args.simpleBC:
            BC_loss = ((mean - action_batch.unsqueeze(0).repeat(self.policy.network_size, 1, 1))**2).mean()
        else:
            BC_loss, _ = self.policy.ensemble_model.loss(mean, torch.log(std**2), action_batch.unsqueeze(0).repeat(self.policy.network_size, 1, 1))
        BC_loss = BC_loss / self.a_mean if self.args.norm_BCloss else BC_loss
        if not self.args.fully_imitate:
            if self.args.OutDICEratio:
                RL_loss = (m_s_ratio * ((self.alpha * log_pi) - min_qf_pi)).mean()
            else:
                RL_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            if self.args.NormRL:
                if self.args.AscLmda:
                    lmda = self.args.LmdaMn + (epoch_step / self.args.AnnealLmda) * (self.args.LmdaMx - self.args.LmdaMn)
                else:
                    lmda = self.args.lamb
                lmda = lmda / min_qf_pi.detach().abs().mean() if self.args.NorQOnly else lmda / self.q_mean.detach()
                RL_loss = lmda * RL_loss
                if self.args.ClipRL:
                    RL_loss = RL_loss.clamp(self.args.MinRLThres)
            policy_loss = BC_loss + RL_loss
        else:
            policy_loss = BC_loss

        # update policy and value
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        (qf1_loss+qf2_loss).backward()
        self.critic_optim.step()

        if self.args.AutoAlpha:
            self.alpha_optim.zero_grad()
            alpha_loss = - (self.log_alpha * (log_pi.mean().detach() + self.target_entropy)).mean()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if self.train_count % 25 == 0:
            if self.args.STDPenRew:
                print('BCloss: {}, RLloss: {}, Uncer: {}, Reward: {}'.format(BC_loss.item(), RL_loss.item(), uncertainty_penalty.mean().item(), reward_batch.mean().item()))
            else:
                print('BCloss: {}, RLloss: {}, Alpha: {}'.format(BC_loss.item(), RL_loss.item(), self.alpha.item()))

        # if step % self.args.target_update_interval == 0:
        soft_update(self.critic_target, self.critic, self.args.tau)
        self.train_count += 1
        if self.args.STDPenRew:
            return BC_loss.detach().item(), RL_loss.detach().item(), uncertainty_penalty.mean().item()
        return BC_loss.detach().item(), RL_loss.detach().item()

    def TD3_train(self, m_s, m_a, m_r, m_ns, m_m, horizon, step, epoch_step, agent):
        '''Train the determinstic model with TD3'''
        state_batch = torch.FloatTensor(m_s).to(self.device) # (B, _dim)
        next_state_batch = torch.FloatTensor(m_ns).to(self.device)
        action_batch = torch.FloatTensor(m_a).to(self.device)
        reward_batch = torch.FloatTensor(m_r).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(m_m).to(self.device).unsqueeze(1) # (B, 1)
        H_batch = torch.from_numpy(horizon).to(self.device).unsqueeze(1) # (B, 1) # for n-step bootstrapping

        # NOTE derive updated next state (s', pi_{old}(s')) -> (s', pi(s'))
        updated_act = agent.policy.sample(next_state_batch[:, :self.args.s_dim])[2] # mean action of pi( |s')
        next_state_batch[:, self.args.s_dim:] = updated_act.detach()

        with torch.no_grad():
			# Select action according to policy and add clipped noise
            noise = (torch.randn_like(action_batch) * self.args.DMoNoise).clamp(-self.args.ClipDMoNoise, self.args.ClipDMoNoise)
            next_action = self.actor_target(next_state_batch.unsqueeze(0).repeat(self.args.num_networks, 1, 1))
            next_action = next_action.mean(0) + noise
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + mask_batch * self.args.model_gamma**H_batch * target_Q
		# Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        # Compute critic loss
        critic_loss = nn.functional.mse_loss(current_Q1, target_Q) + nn.functional.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Delayed policy updates
        if self.train_count % 2 == 0:
            # Compute actor loss
            pi = self.actor(state_batch.unsqueeze(0).repeat(self.args.num_networks, 1, 1)).mean(0)
            Q1 = self.critic(state_batch, pi)[0]

            if self.args.AscLmda:
                lmda = self.args.LmdaMn + (epoch_step / self.args.AnnealLmda) * (self.args.LmdaMx - self.args.LmdaMn)
            else:
                lmda = self.args.lamb
            lmda = lmda / Q1.abs().mean().detach()
            self.RLloss = -lmda * Q1.mean()
            self.BCloss = nn.functional.mse_loss(pi, action_batch) 
            actor_loss = self.RLloss + self.BCloss

            # Optimize the actor 
            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()

            # Update the frozen target models
            soft_update(self.critic_target, self.critic, self.args.tau)
            soft_update(self.actor_target, self.actor, self.args.tau)

        self.train_count += 1
        if self.train_count % 20 == 0:
            print('RLloss: {}, BCloss: {}'.format(self.RLloss.item(), self.BCloss.item()))
        return self.RLloss.item(), self.BCloss.item()
        