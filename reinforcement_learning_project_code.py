import numpy as np
import argparse
import math
from datetime import datetime as dt


class project:
    def __init__(self):
        self.environment = {}
        self.states = []
        self.velocity = {}
        self.actions = {}
        self.rewards = {}
        self.initialize()
        self.policy = np.zeros((len(self.states), len(self.actions['actions'])))
        self.Q = np.zeros(self.policy.shape)

    def initialize(self):
        self.create_environment_1()
        self.set_states()
        self.set_velocity_measures()
        self.set_actions()
        self.set_rewards()

    def create_environment_1(self):
        grid = np.array([
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
            ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["X", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "#", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
        ])
        obstacles = [11, 12, 13, 14, 15, 16, 17, 18, 19, 50]
        source = 90
        goal = 9
        checkpoint = 64
        self.environment['grid'] = grid
        self.environment['obstacles'] = obstacles
        self.environment['source'] = source
        self.environment['goal'] = goal
        self.environment['checkpoint'] = checkpoint

    def set_states(self):
        size_of_environment = self.environment['grid'].shape
        for i in range(size_of_environment[0] * size_of_environment[1]):
            self.states.append((i // size_of_environment[0], i % size_of_environment[1]))

    def set_velocity_measures(self):
        self.velocity['MAX'] = 3
        self.velocity['MIN'] = 0
        self.velocity['V'] = 0

    def set_actions(self):
        self.actions['_RIGHT'] = 0; self.actions['RIGHT'] = 1; self.actions['RIGHT_'] = 2
        self.actions['_UP'] = 3; self.actions['UP'] = 4; self.actions['UP_'] = 5
        self.actions['_LEFT'] = 6; self.actions['LEFT'] = 7; self.actions['LEFT_'] = 8
        self.actions['actions'] = range(9)

    def set_rewards(self):
        self.rewards['crash'] = -10
        self.rewards['checkpoint'] = 0
        self.rewards['win'] = 1000
        self.rewards['step'] = -1


def main():
    test = project()


if __name__ == "__main__":
    main()
