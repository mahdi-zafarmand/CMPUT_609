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
        self.checkpoint = False

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
        start = 90
        goal = 9
        checkpoint = 64
        self.environment['grid'] = grid
        self.environment['size'] = grid.shape[0]
        self.environment['obstacles'] = obstacles
        self.environment['start'] = start
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

    def reset(self):
        self.environment['grid'] = np.array([
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
        self.velocity['V'] = 0
        self.velocity['V_MAX'] = 3

    def make_greedy(self, s, epsilon):
        ACTIONS = self.actions['actions']
        self.policy[s, :] = [epsilon / (len(ACTIONS) - 1.)] * len(ACTIONS)
        best_a = np.argmax(self.Q[s, :])
        self.policy[s, best_a] = 1 - epsilon
        assert np.isclose(np.sum(self.policy[s, :]), 1)

    def choose_actions(self, s, epsilon):
        self.make_greedy(s, epsilon)
        return np.random.choice(self.actions['actions'], p=self.policy[s, :])

    def choose_sigma(self, mode):
        if mode == 'SARSA':
            return 1
        elif mode == 'TreeBackUp':
            return 0
        elif mode == 'Qsigma':
            return (int) (np.random.random() < 0.5)
        else:
            print('ERROR, incorrcet sigma mode!')
            return None

    def move(self, s, a, beta):
        if np.random.random() < 1-beta:
            if a in [0, 3, 6] and self.velocity['V'] > self.velocity['V_MIN']:
                self.velocity['V'] -= 1
            elif a in [2, 5, 8] and self.velocity['V'] < self.velocity['V_MIN']:
                self.velocity['V'] += 1
        
        WIDTH = self.environment['size']
        r_border = range(WIDTH-1, WIDTH**2, WIDTH)
        l_border = range(0, WIDTH**2, WIDTH)
        t_border = range(WIDTH)

        units = range(self.velocity['V'])
        self.checkpoint = False

        if a < (len(self.actions['actions']) / 3):
            return move_right(s, units, r_border)
        elif a < 2 * (len(self.actions['actions']) / 3):
            return move_up(s, units, t_border)
        elif a < len(self.actions['actions']):
            return move_left(s, units, l_border)
        else:
            print('ERROR, incorrcet action!')
            return None

    def move_right(self, s, units, r_border):
        for i in units:
            self.environment['grid'][self.environment['states'][s + i]] = '>'
            if (s + i) in r_border or (s + i + 1) in self.environment['obstacles']:
                self.reset()
                return self.environment['start'], self.rewards['crash']
            elif (s + i + 1) == self.environment['checkpoint']:
                self.checkpoint = self.velocity['V_MAX'] != 5
                self.velocity['V_MAX'] = 5
            elif (s + i + 1) == self.environment['goal']:
                self.environment['grid'][self.environment['states'][s + i + 1]] = 'O'
                return s + i + 1, self.rewards['win']

        self.environment['grid'][self.environment['states'][s + self.velocity['V']]] = 'O'
        if self.checkpoint:
            return (s + self.velocity['V'], self.rewards['checkpoint']) 
        else:
            return (s + self.velocity['V'], self.rewards['step'])

    def move_up(self, s, units, t_border):
        WIDTH = self.environment['size']
        for i in units:
            self.environment['grid'][self.environment['states'][s - i * WIDTH]] = '|'
            if (s - i * WIDTH) in t_border or (s - (i + 1) * WIDTH) in self.environment['obstacles']:
                self.reset()
                return self.environment['start'], self.rewards['crash']
            elif (s + i + 1) == self.environment['checkpoint']:
                self.checkpoint = self.velocity['V_MAX'] != 5
                self.velocity['V_MAX'] = 5
            elif (s - (i + 1) * WIDTH) == self.environment['goal']:
                self.environment['grid'][self.environment['states'][s - (i + 1) * WIDTH]] = 'O'
                return s - (i + 1) * WIDTH, self.rewards['win']

        self.environment['grid'][self.environment['states'][s + self.velocity['V']]] = 'O'
        if self.checkpoint:
            return (s - self.velocity['V'] * WIDTH, self.rewards['checkpoint']) 
        else:
            return (s - self.velocity['V'] * WIDTH, self.rewards['step'])

    def move_left(self, s, units, l_border):
        for i in units:
            self.environment['grid'][self.environment['states'][s - i]] = '<'
            if (s - i) in l_border or (s - i - 1) in self.environment['obstacles']:
                self.reset()
                return self.environment['start'], self.rewards['crash']
            elif (s - i - 1) == self.environment['checkpoint']:
                self.checkpoint = self.velocity['V_MAX'] != 5
                self.velocity['V_MAX'] = 5
            elif (s - i - 1) == self.environment['goal']:
                self.environment['grid'][self.environment['states'][s - i - 1]] = 'O'
                return s - i - 1, self.rewards['win']

        self.environment['grid'][self.environment['states'][s - self.velocity['V']]] = 'O'
        if self.checkpoint:
            return (s - self.velocity['V'], self.rewards['checkpoint']) 
        else:
            return (s - self.velocity['V'], self.rewards['step'])


def main():
    test = project()
    test.choose_actions(1, 0.2)

if __name__ == "__main__":
    main()
