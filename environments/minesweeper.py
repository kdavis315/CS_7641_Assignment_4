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
        "SFFFFFFF",
        "FFFMFFFF",
        "FMFFFFMF",
        "FFFFFFFM",
        "FFFFFMFF",
        "MFMFMFFF",
        "FFFMFFFF",
        "FFFFFMFG"
    ],
    "20x20": [
        "SFMMFFFFFFMFFFFMMFFF",
        "FFFFFFMFFFFFFFFMFFMM",
        "FFMFFFFFFFFFFFFFMFFM",
        "FFMFMFFFFMFFFFFFFFFF",
        "FFMMFFFFFFFFFFFFFFFF",
        "FFFMFFFFFFFFMFFFFFFF",
        "FFFFFMFMMFMFMFFFFMFF",
        "MFFFFFFFFFMFFMFFFFFF",
        "FFMFFFFFFFFMFFFFFMFF",
        "FMFFFFFFFFFFFFFMFFFM",
        "FFFFFFFFFMFFFFFFFFFF",
        "FFFFFFMFFFMFFFFFFFFF",
        "FFFFFFFMFFFFMFFFFFFF",
        "MFFFFFFFFFFFFFFFMFFF",
        "FFFFFFFFFFFFFFFMMFMF",
        "MFFFFFFFMFFFFMFFFFFM",
        "FFMFFFFFMFFFMFFFFFMF",
        "FFMFMFFFFFFFFFFFFFFF",
        "FFMMMMFFFFFFFFFFFFFF",
        "MFFFFFFFFFFFFFFFMFFG"
    ]
}


# Adapted from https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# And inspired by Minesweeper
# This is a modified environment where rewards are given for every step in addition to on finding a goal.
# This reward shaping makes the problem easier for the learner to solve.
class RewardingMinesweeperEnv(discrete.DiscreteEnv):
    """
    You have to cross a minefield.
    If you step on a mine, kaboom! Start over.
    Also, the ground is pretty unstable and you may slip or trip and end up going a different direction
    than intended.
    The field is described using a grid like the following

        SFFF
        FMFM
        FFFM
        MFFG

    S : starting point, safe
    F : field, safe
    M : mine, kaboom!
    G : goal, safely made it across!

    The episode ends when you reach the goal or step on a mine.
    You receive a reward of 1 if you reach the goal, -1 for stepping on a mine, and a small negative reward otherwise.
    The mine and step rewards are configurable when creating an instance of the problem.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", rewarding=True, step_reward=-0.1, mine_reward=-1, is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.step_reward = step_reward
        self.mine_reward = mine_reward
        self.rewarding = rewarding
        self.is_slippery = is_slippery

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
                    if letter in b'GM':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GM'
                                rew = float(newletter == b'G')
                                if self.rewarding:
                                    if newletter == b'F':
                                        rew = self.step_reward
                                    elif newletter == b'M':
                                        rew = self.mine_reward
                                li.append((1.0 / 3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GM'
                            rew = float(newletter == b'G')
                            if self.rewarding:
                                if newletter == b'F':
                                    rew = self.step_reward
                                elif newletter == b'M':
                                    rew = self.mine_reward
                            li.append((1.0, newstate, rew, done))

        super(RewardingMinesweeperEnv, self).__init__(nS, nA, P, isd)

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
            b'F': 'palegreen',
            b'M': 'red',
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
        return RewardingMinesweeperEnv(desc=self.desc, rewarding=self.rewarding, step_reward=self.step_reward,
                                      mine_reward=self.mine_reward, is_slippery=self.is_slippery)
