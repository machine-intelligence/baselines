#!/usr/bin/env python
import os, logging, gym
import numpy as np
from gym import spaces
import tensorflow as tf
from keras.models import Model, load_model
import keras.backend.tensorflow_backend as KTF
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import DownsampleWrapper, RenderWrapper, FrameStack
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, FcPolicy

NUM_THREADS = 5

def get_session(gpu_fraction=0.1):
    '''Force tensorflow not to take up the whole GPU on every thread.'''

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=NUM_THREADS))


class EncodeWrapper(gym.ObservationWrapper):
    """Load pre-trained environment model and use it to encode each observation."""

    def __init__(self, env):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        KTF.set_session(get_session())
        self.model = load_model('autoencoder4.h5')

        self.encoder_model = Model(self.model.input, self.model.get_layer('bottleneck').output, name='encoder')

        # take all actions as a batch, skip model batch dim
        shape = (self.action_space.n * self.encoder_model.output.shape.as_list()[1],)
        self.observation_space = spaces.Box(low=-3e38, high=3e38, shape=shape)

    def _observation(self, obs):
        # Put framestack at the end
        obs = np.moveaxis(obs, -1, 0)
        # Preprocess
        obs = 1 - obs[..., :96, :144, :] / 255.
        return np.concatenate(self.encoder_model.predict(
            [np.stack((obs,) * self.action_space.n),
             np.identity(self.action_space.n)]))

def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    num_timesteps //= 4

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = RenderWrapper(env, 400, 600)
            env = DownsampleWrapper(env, 4)
            env = FrameStack(env, 4)
            env = EncodeWrapper(env)
            #env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    if policy == 'fc':
        policy_fn = FcPolicy
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, nsteps=5, nstack=4, total_timesteps=num_timesteps, lrschedule=lrschedule, max_episode_length=195)
    env.close()


def main():
    # TODO: make this a clf
    policy = 'fc'

    train('CartPole-v0', num_timesteps=int(8e6), seed=0, policy=policy,
          lrschedule='linear', num_cpu=NUM_THREADS)


if __name__ == '__main__':
    main()
