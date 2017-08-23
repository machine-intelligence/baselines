import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from keras.models import load_model, Model
import baselines.common.tf_util as U
from gym import spaces
from collections import deque


from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
from baselines.common.atari_wrappers import FrameStack, DownsampleWrapper, RenderWrapper

class EncodeWrapper(gym.ObservationWrapper):
    """Load pre-trained environment model and use it to encode each observation."""
    def __init__(self, env):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.model = load_model('autoencoder.h5')

        self.encoder_model = Model(self.model.input, self.model.get_layer('bottleneck').output, name='encoder')

        # take all actions as a batch, skip model batch dim
        shape = (self.action_space.n  * self.encoder_model.output.shape.as_list()[1],)
        self.observation_space = spaces.Box(low=-3e38, high=3e38, shape=shape)


    def _observation(self, obs):
        obs = 1 - obs[..., :96, :144, :] / 255.
        obs = np.moveaxis(obs, -1, 0)[..., None]
        return np.concatenate(self.encoder_model.predict(
            [np.stack([obs, obs]),
             np.arange(self.action_space.n)]))

# TODO: make this a commandline arg
FROM_PIXELS = False
def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    logger.configure()
    if FROM_PIXELS:
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[64],
            dueling=False
        )

    with U.make_session(8):
        # Create the environment
        env = gym.make("CartPole-v0")
        if FROM_PIXELS:
            env = RenderWrapper(env, 400, 600)
            env = DownsampleWrapper(env, 4)
            env = FrameStack(env, 4)
            #env = EncodeWrapper(env)
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
        exploration = LinearSchedule(schedule_timesteps=2e4 if FROM_PIXELS else 1e4,
                                     initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = deque([0.], maxlen=100)
        episode_loss = deque([0.], maxlen=100)
        num_eps = 0
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
                episode_loss.append(0)
                num_eps += 1

            is_solved = t > 100 and np.mean(episode_rewards) >= 195
            if is_solved:
                # Show off the result
                env.render()
                logger.log('Mean ep reward {}'.format(np.mean(episode_rewards)))
                break
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    batch_loss = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    episode_loss[-1] += np.mean(batch_loss)

                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and num_eps % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("loss", np.mean(episode_loss))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
