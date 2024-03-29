import logging.config
import random

import gym
from gym import spaces
from tabulate import tabulate


class TwoInARowEnv(gym.Env):
    """A Naive OpenAI Gym environment for basic testing of RL agents.

    Two observations are required to predict the optimal action. Goal is to
    take action 1 when last two observations are the same, and take action 0
    otherwise.

    Observation Space
        2 possible observations: 0 or 1

    Action Space
    2 possible actions: 0 or 1

    Reward function
        if last two observations are both 0 or both 1, then reward is
            * +1 for taking action 1
            * -1 for taking action 0
        else
            * -1 for taking action 1
            * +1 for taking action 0
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps_per_episode=100):
        """
        Parameters
        ----------
        max_steps_per_episode : int, default=100
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the game does not end otherwise.

        Attributes
        ----------
        curr_episode : int
            Current episode as a count.
        obs_episode_memory : list<int>
            History of observations in episode.
        action_episode_memory : list<int>
            History of actions taken in episode.
        curr_step : int
            Current timestep in episode, as a count.
        action_space : gym.spaces.Discrete
            Action space.
        observation_space : gym.spaces.Discrete
            Observation space.
        """
        self.max_steps_per_episode = max_steps_per_episode
        self.__version__ = "0.0.2"
        logging.info("TwoInARow - Version {}".format(self.__version__))
        self.curr_episode = -1  # Set to -1 b/c reset() adds 1 to episode
        self.obs_episode_memory = []
        self.action_episode_memory = []
        self.curr_step = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)

    def step(self, action):
        """The agent takes a step in the environment.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : list
                A list of ones or zeros which together represent the state of
                the environment.
            reward : float
                Amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                Whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : dict
                Diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        done = self.curr_step >= self.max_steps_per_episode
        if done:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self.action_episode_memory[self.curr_episode].append(action)
        self._take_action(action)
        # Recompute done since action may have modified it
        done = self.curr_step >= self.max_steps_per_episode
        reward = self._get_reward()
        ob = self._get_obs()
        self.obs_episode_memory[self.curr_episode].append(ob)
        # Perform resets that happen after each timestep
        self._step_reset()
        return ob, reward, done, {}

    def reset(self):
        """Reset the state of the environment and returns an initial obs..

        Returns
        -------
        object
            The initial observation of the space.
        """
        self.curr_step = 0
        self.curr_episode += 1
        self.action_episode_memory.append([])
        initial_obs = self._get_obs()
        self.obs_episode_memory.append([initial_obs])
        return initial_obs

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def q_values(self, model):
        """Returns a string representation of the Q values for each state.

        This assumes a deterministic policy.

        Parameters
        ----------
        model : object
            Model trying to learn q-values of the env. Must have a `predict`
            method.

        Returns
        -------
        str
            String representation of Q values.

        TODOs
        -----
        If policy is stochastic, we could add a `sample` boolean parameter
        which would call `model.predict()` multiple times and average the
        returned values.
        """
        inputs = [[[0], [0]], [[1], [1]], [[0], [1]], [[1], [0]]]
        preds = [model.predict(inp) for inp in inputs]
        data = []
        for i, inp in enumerate(inputs):
            inp_str = ','.join([str(_inp[0]) for _inp in inp])
            data.append([inp_str]+list(preds[i][0]))
        s = tabulate(data, headers=['Obs. Seq', 'Action 0', 'Action 1'])
        s = '\n' + s + '\n'
        return s

    def _take_action(self, action):
        """How to change the environment when taking an action.

        Parameters
        ----------
        action : int
            Action.

        Returns
        -------
        None
        """
        if action not in [0, 1]:
            raise ValueError('Invalid action ', action)

    def _get_reward(self):
        """Obtain the reward for the current state of the environment.

        Returns
        -------
        float
            Reward.
        """
        action = self.action_episode_memory[self.curr_episode][-1]
        if len(self.obs_episode_memory[self.curr_episode]) < 2:
            if action == 0:
                r = 1
            else:
                r = -1
        else:
            last_n_obs = self.obs_episode_memory[self.curr_episode][-2:]
            if last_n_obs[0] == last_n_obs[1]:
                if action == 1:
                    r = 1
                else:
                    r = -1
            else:
                if action == 0:
                    r = 1
                else:
                    r = -1
        return r

    def _get_obs(self):
        """Obtain the observation for the current state of the environment.

        Returns
        -------
        list
            Observation.
        """
        return [random.choice([0, 1])]

    def _step_reset(self):
        """Performs resets that happen after each timestep.

        Returns
        -------
        None
        """
        pass
