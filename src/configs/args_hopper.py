import argparse
from curses.ascii import FF
from re import T
from tkinter.tix import Tree
from tokenize import Triple


def get_hopper_args(parser):
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)') # Walker2d, Ant, HalfCheetah, Humanoid, 
    parser.add_argument('--init_exploration_steps', type=int, default=5000, metavar='A',
                        help='exploration steps initially')
    parser.add_argument('--n_trajs', default=4)
    parser.add_argument('--DiscountMPC', default=False)
    parser.add_argument('--model_gamma', default=0.99)
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M') # default 100000
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch') # default 1000
    parser.add_argument('--rollout_min_epoch', type=int, default=1, metavar='A',
                        help='rollout min epoch')
    parser.add_argument('--rollout_max_epoch', type=int, default=80, metavar='A',
                        help='rollout max epoch')
    parser.add_argument('--rollout_min_length', type=int, default=2, metavar='A',
                        help='rollout min length')
    parser.add_argument('--rollout_max_length', type=int, default=8, metavar='A',
                        help='rollout max length')
    parser.add_argument('--Limit_Horizon', default=False)
    parser.add_argument('--MPCHorizon', default=8)
    parser.add_argument('--plan_every_n_steps', default=int(2))
    parser.add_argument('--early_stopping', default=True)
    parser.add_argument('--aver_interval', type=int, default=12)
    parser.add_argument('--trgt_ret', default=2500)
    parser.add_argument('--TrainEARLYSTOP', default=True)
    parser.add_argument('--TrSTOP_Aver_Interval', type=int, default=1)
    parser.add_argument('--TrSTOP_TrgtRet', default=3500)
    parser.add_argument('--DeterMoForMPC', default=False)
    parser.add_argument('--DeterPoForMPC', default=False)

    parser.add_argument('--NorQOnly', default=True)
    parser.add_argument('--use_state_normalization', default=True)
    parser.add_argument('--NorSOnly', default=False) # FIXME
    parser.add_argument('--NorSInPred', default=False) # FIXME
    parser.add_argument('--STDPenRew', default=True)
    parser.add_argument('--PenWeight', default=0)
    parser.add_argument('--RewUp', default=False)
    parser.add_argument('--NorSWhenComputeRew', default=False)
    parser.add_argument('--LL_reward', default=False, help='LikeLihood')
    parser.add_argument('--EnsFusionRew', default=False)
    parser.add_argument('--RSsepa', default=True)

    parser.add_argument('--abs_rew', default=False)
    parser.add_argument('--rew_scaling', default=True)
    parser.add_argument('--rew_norm', default=False)

    parser.add_argument('--DualDICE', default=True)
    parser.add_argument('--OutDICEratio', default=True) # NOTE
    parser.add_argument('--trgt_DICE', default=True)
    parser.add_argument('--WarmUpDualDICE_epo', default=5)
    parser.add_argument('--FirstDualDICE_step', default=int(1e5))
    parser.add_argument('--DICE_batch_size', default=1024)
    parser.add_argument('--n_zeta_train', default=1)
    parser.add_argument('--nulr', default=1e-4) # default 1e-4
    parser.add_argument('--zetalr', default=1e-4) # default 1e-3
    parser.add_argument('--function_exponent', default=1.5)

    parser.add_argument('--deter_model', default=False)
    parser.add_argument('--DMoNoise', default=0.2)
    parser.add_argument('--ClipDMoNoise', default=0.5)
    parser.add_argument('--deeper_Q_net', default=False)

    parser.add_argument('--ClipRL', default=False)
    parser.add_argument('--MinRLThres', default=2)
    parser.add_argument('--n_SLepochs', default=20)
    parser.add_argument('--pretrain_model_on_explstep', default=False)
    parser.add_argument('--XavierMo', default=False)
    parser.add_argument('--ReluMo', default=False)
    parser.add_argument('--ReduceRLMoLayer', default=False)
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--roll_RLmodel_freq', default=250)
    parser.add_argument('--RLmodel_train_freq', default=250)
    parser.add_argument('--model_retain_epochs', default=0.250, metavar='A', help='retain epochs')
    parser.add_argument('--env_retain_epochs', default=100)
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')

    parser.add_argument('--num_networks', type=int, default=7, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=5, metavar='E',
                        help='elite size (default: 5)')

    parser.add_argument('--num_modeltrain_repeat', default=50) # time to train model per step
    parser.add_argument('--Asc_n_MoTr', default=False, help='if or not to increase the number of model training during training')
    parser.add_argument('--min_n_MoTr', default=200)
    parser.add_argument('--max_n_MoTr', default=20)
    parser.add_argument('--Anneal_n_MoTr', default=10, help='number of epochs min->max would cost')
    parser.add_argument('--lamb', default=2.5)
    parser.add_argument('--AscLmda', default=False)
    parser.add_argument('--LmdaMn', default=0.5)
    parser.add_argument('--LmdaMx', default=1.5)
    parser.add_argument('--AnnealLmda', default=20)

    parser.add_argument('--RLmodel_lr', default=3e-4) # keep consistent with the policy
    parser.add_argument('--model_q_lr', default=3e-4)
    parser.add_argument('--RLmodel_batch_size', default=256) # 256 keep consistent with the policy training batch size
    parser.add_argument('--beta', default=1) # model RLtraining reward: (\hat{s}-s)^2 + beta*(\hat{r}-r)^2
    parser.add_argument('--model_alpha', default=0.2)
    parser.add_argument('--AutoAlpha', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--epsilon', default=1e-3)

    parser.add_argument('--RLembededtrain', default=False)
    parser.add_argument('--fully_imitate', default=False)
    parser.add_argument('--train_RLmodel_on_full_pool', default=False)
    parser.add_argument('--use_permutation', default=False)

    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    
    
    parser.add_argument('--expo_r', default=False)
    parser.add_argument('--n_boots', default=int(1)) # > 0 only!!
    parser.add_argument('--scale_rew', default=False) # NOTE
    parser.add_argument('--NormRew', default=False) # NOTE 
    parser.add_argument('--sqrt_rew', default=False)
    parser.add_argument('--simpleBC', default=False)
    parser.add_argument('--norm_BCloss', default=False)
    parser.add_argument('--lambda_out', default=False)
    parser.add_argument('--NormRL', default=True)

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')

    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')


    parser.add_argument('--num_epoch', type=int, default=100, metavar='A',
                        help='total number of epochs')
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size') # default 1000
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy') # default 256

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    return parser.parse_args()
