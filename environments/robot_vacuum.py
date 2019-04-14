import sys

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FMFM",
        "FFFM",
        "MFFG"
    ],
    "8x8": [
        "SFOFFOFF",
        "FFOFFOFO",
        "FFFFOFFF",
        "FFFFFFFO",
        "OOFFOFFF",
        "OFFFFFFO",
        "FFFOFFFF",
        "FFFOFFFG"
    ],
    "15x15": [
        "SFFOFFFFFFFOFFF",
        "FFFOFFFFFFFOFOO",
        "FFFOFFFFFFFOFOO",
        "FFFFFOFFFOFFFFF",
        "FFFFFFFFFFFFFOO",
        "OOOFFOFFFOFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFOO",
        "OOOFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "OOFFFFFFFFFFFOO",
        "OOFFFFFFFFFFFOO",
        "FFFFOFFFFFOFFFF",
        "FFFFOFFFFFOFFFF",
        "FFFFOFFFFFOFFFG",
    ]
}

# Adapted from https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# This is a modified environment where rewards are given for every step in addition to on finding a goal.
# This reward shaping makes the problem easier for the learner to solve.
class RewardingRobotVacuumEnv(discrete.DiscreteEnv):
    """
    You are a robot vacuum trying to get back to its charging station.
    There are no negative reward absorbing states, ie no holes/traps/mines that make you start over.
    There are only obstacles.
    There is also a cat that may push you off course sometimes.
    The field is described using a grid like the following

        SFFF
        FOFO
        FFFO
        OFFG

    S : starting point
    F : floor
    O : obstacle
    G : goal

    The episode ends when you reach the goal.
    You receive a reward of 1 if you reach the goal and a small negative reward for each step.
    This problem does not assume a time limit representing battery failure as that would change
    the nature of the problem. It is something to be explored outside of MDPs.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="15x15", rewarding=True, step_reward=-0.1, cat_exists=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.step_reward = step_reward
        self.rewarding = rewarding
        self.cat_exists = cat_exists

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter == b'G':
                        li.append((1.0, s, 0, True))
                    else:
                        if cat_exists:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) == b'G'
                                rew = float(newletter == b'G')
                                if self.rewarding:
                                    if newletter == b'F':
                                        rew = self.step_reward
                                    elif newletter == b'O':
                                        #can't go to obstacle, stay in same state
                                        newstate = to_s(row,col)
                                        newletter = desc[row, col]
                                if b == a:
                                    li.append((0.6, newstate, rew, done))
                                else:
                                    li.append((0.2, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) == b'G'
                            rew = float(newletter == b'G')
                            if self.rewarding:
                                if newletter == b'F':
                                    rew = self.step_reward
                                elif newletter == b'O':
                                    #can't go to obstacle, stay in same state
                                    newstate = to_s(row,col)
                                    newletter = desc[row, col]
                            li.append((1.0, newstate, rew, done))

        super(RewardingRobotVacuumEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def colors(self):
        return {
            b'S': 'gold',
            b'F': 'brown',
            b'O': 'black',
            b'G': 'blue',
        }

    def directions(self):
        return {
            3: '⬆',
            2: '➡',
            1: '⬇',
            0: '⬅'
        }

    def new_instance(self):
        return RewardingRobotVacuumEnv(desc=self.desc, rewarding=self.rewarding, step_reward=self.step_reward,
                                      cat_exists=self.cat_exists)
