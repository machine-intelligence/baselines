#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import DownsampleWrapper, RenderWrapper
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, FcPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    num_timesteps //= 4

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            if policy == 'cnn':
                env = RenderWrapper(env, 400, 600)
                env = DownsampleWrapper(env, 4)
            #env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    logger.configure()
    if policy == 'fc':
        policy_fn = FcPolicy
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, nsteps=5, nstack=1, total_timesteps=num_timesteps, lrschedule=lrschedule, max_episode_length=195)
    env.close()


def main():
    # TODO: make this a clf
    policy = 'cnn'
    train('CartPole-v0', num_timesteps=int(8e6), seed=0, policy=policy,
          lrschedule='linear', num_cpu=8)


if __name__ == '__main__':
    main()
