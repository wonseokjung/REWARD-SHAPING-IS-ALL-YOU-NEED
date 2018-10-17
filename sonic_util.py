"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np

from stable_baselines.common.atari_wrappers import WarpFrame, FrameStack
#import gym_remote.client as grc
import retro
#env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
#from retro_contest.local import make

def make_env(game=None, state=None, stack=True, scale_rew=True, allowbacktrace=False):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock')
    #env = retro.make(game=game, state=state)
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

    env = SonicDiscretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if allowbacktrace:
        env = AllowBacktracking(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.level_pred = env.level_pred

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

#

    def step(self, action): # pylint: disable=E0202

        obs, rew, done, info = self.env.step(action)

        #self.env.render()

        self._cur_x += rew
        print(self._cur_x)
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
