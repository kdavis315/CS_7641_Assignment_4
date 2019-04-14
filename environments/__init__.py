import gym
from gym.envs.registration import register

from .minesweeper import *
from .minesweeper_clues import *
from .robot_vacuum import *

__all__ = ['RewardingMinesweeperEnv', 'RewardingMinesweeperCluesEnv', 'RewardingRobotVacuumEnv']

register(
    id='RewardingMinesweeper-v0',
    entry_point='environments:RewardingMinesweeperEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='RewardingMinesweeper8x8-v0',
    entry_point='environments:RewardingMinesweeperEnv',
    kwargs={'map_name': '8x8'}
)
register(
    id='RewardingMinesweeper20x20-v0',
    entry_point='environments:RewardingMinesweeperEnv',
    kwargs={'map_name': '20x20'}
)

register(
    id='RewardingMinesweeperNoRewards20x20-v0',
    entry_point='environments:RewardingMinesweeperEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='RewardingMinesweeperNoRewards8x8-v0',
    entry_point='environments:RewardingMinesweeperEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='RewardingMinesweeperClues-v0',
    entry_point='environments:RewardingMinesweeperCluesEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='RewardingMinesweeperClues8x8-v0',
    entry_point='environments:RewardingMinesweeperCluesEnv',
    kwargs={'map_name': '8x8'}
)
register(
    id='RewardingMinesweeperClues20x20-v0',
    entry_point='environments:RewardingMinesweeperCluesEnv',
    kwargs={'map_name': '20x20'}
)

register(
    id='RewardingMinesweeperCluesNoRewards20x20-v0',
    entry_point='environments:RewardingMinesweeperCluesEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='RewardingMinesweeperCluesNoRewards8x8-v0',
    entry_point='environments:RewardingMinesweeperCluesEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='RewardingRobotVacuum-v0',
    entry_point='environments:RewardingRobotVacuumEnv',
)

register(
    id='RewardingRobotVacuumNoReward-v0',
    entry_point='environments:RewardingRobotVacuumEnv',
    kwargs={'rewarding': False}
)


def get_rewarding_minesweeper_environment():
    return gym.make('RewardingMinesweeper8x8-v0')

def get_large_rewarding_minesweeper_environment():
    return gym.make('RewardingMinesweeper20x20-v0')


def get_minesweeper_environment():
    return gym.make('Minesweeper-v0')


def get_rewarding_no_reward_minesweeper_environment():
    return gym.make('RewardingMinesweeperNoRewards8x8-v0')


def get_large_rewarding_no_reward_minesweeper_environment():
    return gym.make('RewardingMinesweeperNoRewards20x20-v0')


def get_rewarding_minesweeper_clues_environment():
    return gym.make('RewardingMinesweeperClues8x8-v0')

def get_large_rewarding_minesweeper_clues_environment():
    return gym.make('RewardingMinesweeperClues20x20-v0')


def get_minesweeper_clues_environment():
    return gym.make('MinesweeperClues-v0')


def get_rewarding_no_reward_minesweeper_clues_environment():
    return gym.make('RewardingMinesweeperCluesNoRewards8x8-v0')


def get_large_rewarding_no_reward_minesweeper_clues_environment():
    return gym.make('RewardingMinesweeperCluesNoRewards20x20-v0')

def get_robot_vacuum_environment():
    return gym.make('RewardingRobotVacuum-v0')

def get_robot_vacuum_no_reward_environment():
    return gym.make('RewardingRobotVacuumNoReward-v0')


