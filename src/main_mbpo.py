from ast import arg
from curses import curs_set
import time
import gym
import torch
import numpy as np
from itertools import count
import logging
import os
from torch import fix_
import wandb
import socket
from pathlib import Path
import setproctitle
import torch.nn.functional as F

from sac.replay_memory import ReplayMemory, ReorderReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
# from arguments import readParser


def train(args, env_sampler, predict_env, agent, env_pool, model_pool, env_pool_for_RLtrainmodel=None):
    total_step = 0
    reward_sum = 0
    rollout_length = args.rollout_min_length
    test_returns = []
    fix_rollout_length = False
    STOPTRAIN = False
    if args.train_model_by_RL:
        exploration_before_start(args, env_sampler, env_pool, agent, predict_env, env_pool_for_RLtrainmodel)
        if args.pretrain_model_on_explstep:# SL warm up
            train_predict_model(args, env_pool, predict_env)
    else:
        exploration_before_start(args, env_sampler, env_pool, agent)
    epoch = 0

    for _ in range(args.num_epoch):
        sta = time.time()
        epo_len = args.epoch_length
        train_policy_steps = 0
        for i in range(epo_len): # i: 1->1000
            if i > 0 and (i+1) % args.model_train_freq == 0 and args.real_ratio < 1.0:
                if not STOPTRAIN:
                    train_predict_model(args, env_pool, predict_env)

                if int(total_step/epo_len) > args.aver_interval + 1:
                    if args.early_stopping and test_returns[-1] > args.trgt_ret and sum(test_returns[-args.aver_interval:]) > args.aver_interval * args.trgt_ret:
                        fix_rollout_length = True
                    if args.TrainEARLYSTOP and sum(test_returns[-args.TrSTOP_Aver_Interval:]) > args.TrSTOP_Aver_Interval * args.TrSTOP_TrgtRet:
                        STOPTRAIN = True
                if not fix_rollout_length:
                    new_rollout_length = set_rollout_length(args, int(total_step/epo_len))
                    if rollout_length != new_rollout_length:
                        rollout_length = new_rollout_length
                        model_pool = resize_model_pool(args, rollout_length, model_pool)
            if i > 0 and (i+1) % args.model_train_freq == 0 and args.real_ratio < 1.0:
                if not STOPTRAIN:
                    # model data collection
                    if args.MPCmodel:
                        MPCrollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)
                    else:
                        rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)

            # true env data collection
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
            env_pool.push(cur_state, action, reward, next_state, done)

            # train the policy
            if not STOPTRAIN:
                if len(env_pool) > args.min_pool_size:
                    train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, i, env_pool, model_pool, agent)
            total_step += 1
            if total_step % epo_len == 0 or total_step == 1: # logging, no training
                '''
                avg_reward_len = min(len(env_sampler.path_rewards), 5)
                avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
                logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
                print(total_step, env_sampler.path_rewards[-1], avg_reward)
                '''
                env_sampler.current_state = None
                sum_reward = 0
                done = False
                eval_step = 0
                while not done: # or eval_step <= 1000:
                    cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True)
                    sum_reward += reward
                    eval_step += 1
                logging.info("Step Reward: " + str(total_step) + " " + str(sum_reward))
                print('EX: {}, step: {}, test_return: {}, seed: {}, FIX_ROLL_LEN: {}, STOP_TRAIN: {}, rollout_len: {}, cuda_num: {}'.format(args.experiment_name, total_step, sum_reward, args.seed, fix_rollout_length, STOPTRAIN, rollout_length, args.cuda_num))
                if args.use_wandb:
                    wandb.log({"test_return": sum_reward}, step=total_step)
                    run_dir = str(wandb.run.dir) + '/' + args.experiment_name
                    if not os.path.exists(run_dir):
                        os.makedirs(run_dir)
                    test_returns.append(sum_reward)
                    np.save(run_dir + '/test_return', test_returns)
                else:
                    pass
        epoch += 1
        end = time.time()
        print('This epoch spends {} seconds!!'.format(end - sta))
        if STOPTRAIN:
            print("Early Stop Training")

def RLtrain_predict_model(args, epoch_step, env_pool_for_RLtrainmodel, predict_env, agent, env_pool):
    # for RLloss and BCloss normalization
    q_means = []
    a_means = []
    s_batches = [] # (n_samples, B, _dim)
    rews = []
    if not args.deter_model:
        if args.NormRL or args.norm_BCloss:
            for _ in range(1):
                with torch.no_grad():
                    m_s, m_a, _, m_ns, _, _ = env_pool_for_RLtrainmodel.sample(len(env_pool_for_RLtrainmodel)) # FIXME
                    # m_s, m_a, _, m_ns, _, _ = env_pool_for_RLtrainmodel.sample(2)
                    state_batch = torch.FloatTensor(m_s).to('cuda')
                    action_batch = torch.FloatTensor(m_a).to('cuda')
                    if args.use_state_normalization:
                        s_batches.append(state_batch)
                    qf1, qf2 = predict_env.critic(state_batch, action_batch)
                    pi, log_pi, mean, std = predict_env.policy.sample(state_batch)
                    log_pi = log_pi.mean(0).mean(-1, keepdim=True) # NOTE
                    q_means.append(torch.abs(log_pi*predict_env.alpha - torch.min(qf1, qf2)).mean())
                    if args.simpleBC:
                        a_means.append(torch.abs((pi - action_batch.repeat(args.num_networks, 1, 1))**2 + torch.log(std**2)).mean())
                    else:
                        a_means.append(predict_env.policy.ensemble_model.loss(mean, torch.log(std**2), action_batch.unsqueeze(0).repeat(args.num_networks, 1, 1))[0])
                    if args.rew_scaling or args.rew_norm:
                        inputs = np.concatenate((m_s[:, :args.s_dim], m_s[:, args.s_dim:]), axis=-1)
                        en_mu_mean = predict_env.model.predict(inputs)[0].mean(0) # (B, s_dim+1)
                        label = np.concatenate((m_ns[:, :args.s_dim], m_a[:, :1]), axis=-1) # cat(s', r)
                        # pred_s, pred_r = en_mu_mean[:, 1:], en_mu_mean[:, :1] # (B, s_dim) (B, 1)
                        if args.use_state_normalization and args.NorSWhenComputeRew:
                            B = args.batch_size
                            pred_next_s = (pred_next_s - predict_env.s_mean.repeat(B, 1)) / predict_env.s_std.repeat(B, 1)
                        rews.append(-((label - en_mu_mean)**2).mean(-1)) # (B,)
            predict_env.q_mean = torch.stack(q_means).mean()
            predict_env.a_mean = torch.stack(a_means).mean()
            if args.use_state_normalization:
                s_batches = torch.stack(s_batches).view(-1, args.s_dim+args.a_dim) # (n_samples*B, _dim)
                predict_env.s_mean, predict_env.s_std = s_batches.mean(0, keepdim=True), s_batches.std(0, keepdim=True)
            if args.rew_scaling or args.rew_norm:
                rews = np.stack(rews)
                predict_env.r_mean, predict_env.r_std = rews.mean(), rews.std()

    if args.DualDICE:
        if epoch_step == args.WarmUpDualDICE_epo and not predict_env.if_zeta_trained:
            K = int(args.FirstDualDICE_step / args.n_zeta_train)
            predict_env.if_zeta_trained = True
        else:
            K = 0
        for _ in range(K):
            train_dual_dice(args, env_pool_for_RLtrainmodel, predict_env, agent, env_pool)
    m_ini_s = None

    if args.Asc_n_MoTr:
        n_MoTr = args.min_n_MoTr + (epoch_step / args.Anneal_n_MoTr) * (args.max_n_MoTr - args.min_n_MoTr)
    else:
        n_MoTr = args.num_modeltrain_repeat
    if args.train_RLmodel_on_full_pool:
        m_state_batch, m_action_batch, m_reward_batch, m_next_state_batch, done_batch = env_pool_for_RLtrainmodel.sample(int(len(env_pool_for_RLtrainmodel)))
        train_idx = np.random.permutation(m_state_batch.shape[0])
        for start_pos in range(0, m_state_batch.shape[0], args.RLmodel_batch_size): # (0, B, 2B, ...)
            idx = train_idx[start_pos: start_pos + args.RLmodel_batch_size]
            if args.use_permutation:
                m_state, m_action, m_reward, m_next_state, done = m_state_batch[idx], m_action_batch[idx], m_reward_batch[idx], m_next_state_batch[idx], done_batch[idx]
            else:
                m_state, m_action, m_reward, m_next_state, done = m_state_batch[start_pos: start_pos + args.RLmodel_batch_size], \
                    m_action_batch[start_pos: start_pos + args.RLmodel_batch_size], m_reward_batch[start_pos: start_pos + args.RLmodel_batch_size], \
                    m_next_state_batch[start_pos: start_pos + args.RLmodel_batch_size], done_batch[start_pos: start_pos + args.RLmodel_batch_size]
            # print(m_state.shape) (b_size, _dim)
            done = (~done).astype(int)
            predict_env.RL_train(m_state, m_action, m_reward, m_next_state, done, start_pos)
    else:
        BClosses, RLlosses, uncer_pens = [], [], []
        for i in range(int(n_MoTr)):
            train_dual_dice(args, env_pool_for_RLtrainmodel, predict_env, agent, env_pool)
            m_state, m_action, m_reward, m_next_state, done, horizon = env_pool_for_RLtrainmodel.sample(int(args.RLmodel_batch_size))
            done = (~done).astype(int)
            if args.deter_model: # TD3
                BCloss, RLloss = predict_env.TD3_train(m_state, m_action, m_reward, m_next_state, done, horizon, i, epoch_step, agent)
            else: # SAC
                if args.STDPenRew:
                    BCloss, RLloss, uncer_pen = predict_env.SAC_train(m_state, m_action, m_reward, m_next_state, done, horizon, i, epoch_step, agent, m_ini_s=m_ini_s)
                else:
                    BCloss, RLloss = predict_env.SAC_train(m_state, m_action, m_reward, m_next_state, done, horizon, i, epoch_step, agent, m_ini_s=m_ini_s)
            BClosses.append(BCloss)
            RLlosses.append(RLloss)
            if args.STDPenRew:
                uncer_pens.append(uncer_pen)
        if args.STDPenRew:
            return np.array(BClosses).mean(), np.array(RLlosses).mean(), np.array(uncer_pens).mean()
        return np.array(BClosses).mean(), np.array(RLlosses).mean()


def train_dual_dice(args, env_pool_for_RLtrainmodel, predict_env, agent, env_pool):
    for _ in range(args.n_zeta_train):
        with torch.no_grad():
            init_s = env_pool.sample_init_state(args.DICE_batch_size)
            init_a = agent.policy.sample(torch.Tensor(init_s).to(predict_env.device))[2].cpu().numpy()
            init_m_s = np.concatenate((init_s, init_a), axis=-1)
            init_s_batch = torch.FloatTensor(init_m_s).to(predict_env.device) # (B, _dim)
            init_a_batch = predict_env.policy.sample(init_s_batch)[0].mean(0)

            m_s, m_a, _, m_ns, _, _ = env_pool_for_RLtrainmodel.sample(args.DICE_batch_size)
            state_batch = torch.FloatTensor(m_s).to(predict_env.device) # (B, _dim)
            next_state_batch = torch.FloatTensor(m_ns).to(predict_env.device)
            action_batch = torch.FloatTensor(m_a).to(predict_env.device)
            updated_act = agent.policy.sample(next_state_batch[:, :args.s_dim])[2] # mean action of pi( |s')
            next_state_batch[:, args.s_dim:] = updated_act
            next_state_action = predict_env.policy.sample(next_state_batch)[0].mean(0) # (n_ens, B, _dim)
        if args.use_state_normalization:
            B = state_batch.shape[0]
            state_batch = (state_batch - predict_env.s_mean.repeat(B, 1)) / predict_env.s_std.repeat(B, 1)
            next_state_batch = (next_state_batch - predict_env.s_mean.repeat(B, 1)) / predict_env.s_std.repeat(B, 1)
            init_s_batch = (init_s_batch - predict_env.s_mean.repeat(B, 1)) / predict_env.s_std.repeat(B, 1)

        nu, next_nu, init_nu = predict_env.nu_net(state_batch, action_batch), predict_env.nu_net(next_state_batch, next_state_action), predict_env.nu_net(init_s_batch, init_a_batch)
        zeta = predict_env.zeta_net(state_batch, action_batch)
        # print(nu.shape, next_nu.shape, init_nu.shape, zeta.shape)
        delta_nu = nu - args.model_gamma * next_nu
        nuloss = delta_nu * zeta.detach() - predict_env._f_star(zeta.detach()) - (1 - args.model_gamma) * init_nu
        zetaloss = -(delta_nu.detach() * zeta - predict_env._f_star(zeta) - (1 - args.model_gamma) * init_nu.detach())

        predict_env.nu_optim.zero_grad()
        nuloss.mean().backward()
        predict_env.nu_optim.step()
        predict_env.zeta_optim.zero_grad()
        zetaloss.mean().backward()
        predict_env.zeta_optim.step()


def exploration_before_start(args, env_sampler, env_pool, agent, predict_env=None, env_pool_for_RLtrainmodel=None):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        # env_pool.push(cur_state, action, reward, next_state, done)
        if args.train_model_by_RL: # reorder the buffer for RL-based model learning
            if not args.MPCmodel:
                if args.deter_model:
                    pred_next_s, pred_r = predict_env.step_for_RLpred(cur_state, action)
                else:
                    pred_next_s, pred_r, en_mu_mean, en_sig_mean, _ = predict_env.step_for_RLpred(cur_state, action)
                if i == 0:
                    step = 0
                elif i < args.init_exploration_steps - 1:
                    step = 1
                else:
                    step = args.epoch_length
                if args.LL_reward:
                    assert args.deter_model == False
                    m_action = np.concatenate((np.array([reward]), next_state - cur_state), axis=-1)
                    m_LLreward = torch.distributions.normal.Normal(torch.Tensor(en_mu_mean), torch.Tensor(en_sig_mean)).log_prob(torch.Tensor(m_action)).detach().cpu().numpy().mean()
                else:
                    m_LLreward = None
                env_pool_for_RLtrainmodel.push(cur_state, action, reward, next_state, done, pred_next_s, pred_r, step, m_LLreward=m_LLreward)
            env_pool.push(cur_state, action, reward, next_state, done)
        else:
            env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)

import time
def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool)) # (len(pool), _dim)
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
    st = time.time()
    if args.MPCmodel:
        predict_env.train_m_r_predictor(state, action, reward, next_state)
    en = time.time()
    # print('Train_hat(r)^m_Cost', en-st) # n_trajs: 10, len:6 时，16s


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = int(args.model_retain_epochs * model_steps_per_epoch)

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def MPCrollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, _, _, _, _ = env_pool.sample_all_batch(args.rollout_batch_size) # (rollB, _dim)
    st = time.time()
    for t in range(rollout_length):
        action = agent.select_action(state)
        next_states, rewards, terminals = predict_env.ParaMPCact(state, action, t, rollout_length, agent)
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]
    en = time.time()
    # print('MPCrolloutCost', en-st) # n_trajs: 10, len:6 时，16s


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # Get a batch of actions
        action = agent.select_action(state)
        # print(state.shape, action.shape) # (rollout_batch_size, _dim)
        next_states, rewards, terminals, info = predict_env.step(state, action) # (n_batch, b_size, _dim)
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]
    # print('rolloutCost', en-st) # len=6: 4s


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat

from gym.spaces import Box
from env import register_mbpo_environments

class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


def main(args=None):
    from arguments import readParser
    if args is None:
        args = readParser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    torch.set_num_threads(args.n_training_threads)
    args.seed = torch.randint(0, 10000, (1,)).item()
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / args.env_name / args.experiment_name

    # Initial environment
    # register_mbpo_environments()
    env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    args.s_dim = env.observation_space.shape[0]
    args.a_dim = env.action_space.shape[0]
    print(args.s_dim, args.a_dim)
    if args.env_name == 'Ant-v2':
        args.s_dim = int(27)
    elif args.env_name == 'Humanoid-v2':
        args.s_dim = int(45)
    # args.beta = 5
    # print(env.observation_space.shape, np.prod(env.observation_space.shape))
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    if args.use_wandb:
        run = wandb.init(config=args,
                         project='RL-based model learning for model-based RL',
                         entity=args.user_name,
                         notes=socket.gethostname(),
                         name=str(args.experiment_name) +
                              "_{}_".format(args.cuda_num) + str(args.seed),
                         group=args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    setproctitle.setproctitle(
        str(args.env_name) + "-" + str(args.seed) + "@" + str(args.user_name))

    # Intial agent
    agent = SAC(args.s_dim, env.action_space, args)

    # Initial ensemble model
    state_size = args.s_dim # np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    env_model = EnsembleDynamicsModel(args, args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                        use_decay=args.use_decay)
    # Predict environments
    predict_env = PredictEnv(args, env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool_for_RLtrainmodel = ReorderReplayMemory(args) if args.train_model_by_RL else None
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(args, env)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool, env_pool_for_RLtrainmodel)
    
    if args.use_wandb:
        run.finish()


if __name__ == '__main__':
    main()
