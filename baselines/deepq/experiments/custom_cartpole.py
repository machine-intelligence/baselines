import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from keras.models import load_model
import baselines.common.tf_util as U
from gym import spaces
from skimage.transform import resize


from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
from baselines.common.atari_wrappers import FrameStack


class RenderWrapper(gym.ObservationWrapper):
    def __init__(self, env, w, h):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(w, h, 3))

    def _observation(self, obs):
        return self.env.render(mode='rgb_array')


class DownsampleWrapper(gym.ObservationWrapper):
    """Resize image, grayscale"""

    def __init__(self, env, scale):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.scale = scale
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(
            old_shape[0] // scale, old_shape[1] // scale, 1))

    def _observation(self, obs):
        return np.uint8(
            resize(np.mean(obs, axis=-1), (obs.shape[0] // self.scale, obs.shape[1] // self.scale), mode='edge'))


class EncodeWrapper(gym.ObservationWrapper):
    """Load pre-trained environment model and use it to encode each observation."""
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.model = load_model('autoencoder.h5')

        self.encoder_model = Model(model.input, model.get_layer('bottleneck').output, name='encoder')


        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(
                # take all actions as a batch, skip model batch dim
                self.action_space.n, encoder_model.output.shape.as_list()[1:], k))


    def _observation(self, obs):
        obs = 1 - obs[..., :96, :144, :] / 255.
        return encoder_model.predict(
            [np.stack([ob, ob]),
             np.arange(self.action_space.n)])

# TODO: make this a commandline arg
FROM_PIXELS = True
def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    logger.session().__enter__()
    if FROM_PIXELS:
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=False
        )

    with U.make_session(8):
        # Create the environment
        env = gym.make("CartPole-v0")
        env = RenderWrapper(env, 400, 600)
        env = DownsampleWrapper(env, 4)
        env = FrameStack(env, 4)
        # env = EncodeWrapper(env)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=2e6 if FROM_PIXELS else 1e4,
                                     initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
                logger.log('Mean ep reward {}'.format(np.mean(episode_rewards[-101:-1])))
                break
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
