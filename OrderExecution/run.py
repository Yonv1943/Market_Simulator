import sys
import gym
from elegantrl.run import train_agent, train_agent_multiprocessing
from elegantrl.config import Config, get_gym_env_args, build_env
from elegantrl.agent import AgentPPO


def train_ppo_a2c_for_order_execution_vec_env():
    from elegantrl.envs.OrderExecutionEnv import OrderExecutionVecEnv

    # num_envs = 2 ** 9
    total = 2 ** (9 + 12)  # todo
    if GPU_ID == 1:
        num_envs = 2 ** 7
    elif GPU_ID == 6:
        num_envs = 2 ** 9
    elif GPU_ID == 7:
        num_envs = 2 ** 11
    else:
        assert GPU_ID == 0
        num_envs = 2 ** 13

    gamma = 0.998
    n_stack = 8

    agent_class = AgentPPO
    env_class = OrderExecutionVecEnv
    env_args = {'env_name': 'OrderExecutionVecEnv-v0',
                'num_envs': num_envs,
                'max_step': 5000,
                'state_dim': 48 * n_stack,
                'action_dim': 2,
                'if_discrete': False,

                'beg_date': '2022-08-09',
                'end_date': '2022-09-09',
                'if_random': False}
    # get_gym_env_args(env=OrderExecutionVecEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = total // num_envs

    args.batch_size = args.horizon_len * num_envs // 16
    args.repeat_times = 8  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.02

    args.eval_times = num_envs * 5
    args.eval_per_step = int(8e2)
    args.eval_env_class = env_class
    # args.eval_env_args = env_args
    args.eval_env_args = {'env_name': 'OrderExecutionVecEnv-v0',
                          'num_envs': num_envs,
                          'max_step': 5000,
                          'state_dim': 48 * n_stack,
                          'action_dim': 2,
                          'if_discrete': False,

                          'beg_date': '2022-09-10',
                          'end_date': '2022-09-16',
                          'if_random': False}

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 1
    train_agent_multiprocessing(args)
    # train_agent(args)
    """
    0.0 < 1.0 < 1.5 < 2.0
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  9.48e+03     423 |   22.63    5.7   4739     0 |   -2.83   1.89   0.00  -0.02
    0  9.48e+03     423 |   22.63
    0  3.32e+04     769 |   26.11    5.6   4739     0 |   -2.68   1.43   0.01  -0.10 
    0  3.32e+04     769 |   26.11  
    0  5.69e+04    1147 |   26.66    5.0   4739     0 |   -2.55   1.13   0.03  -0.14
    0  5.69e+04    1147 |   26.66   
    0  8.06e+04    1591 |   26.81    5.8   4739     0 |   -2.51   0.57   0.00  -0.17
    0  8.06e+04    1591 |   26.81  
    0  1.04e+05    1977 |   26.71    5.5   4739     0 |   -2.35   0.93   0.00  -0.24
    """


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    # DRL_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    # ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    train_ppo_a2c_for_order_execution_vec_env()
