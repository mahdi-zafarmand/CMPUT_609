import numpy as np
import argparse
import math
from datetime import datetime as dt


class World:
    def __init__(self):
        self.environment = {}
        self.states = []
        self.velocity = {}
        self.actions = {}
        self.rewards = {}
        self.initialize()
        self.P = np.zeros((len(self.states), len(self.actions['actions'])))
        self.Q = np.zeros(self.P.shape)

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
        self.P[s, :] = [epsilon / (len(ACTIONS) - 1.)] * len(ACTIONS)
        best_a = np.argmax(self.Q[s, :])
        self.P[s, best_a] = 1 - epsilon
        assert np.isclose(np.sum(self.P[s, :]), 1)

    def choose_action(self, s, epsilon):
        self.make_greedy(s, epsilon)
        return np.random.choice(self.actions['actions'], p=self.P[s, :])

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
            return self.move_right(s, units, r_border)
        elif a < 2 * (len(self.actions['actions']) / 3):
            return self.move_up(s, units, t_border)
        elif a < len(self.actions['actions']):
            return self.move_left(s, units, l_border)
        else:
            print('ERROR, incorrcet action!')
            return None

    def move_right(self, s, units, r_border):
        for i in units:
            self.environment['grid'][self.states[s + i]] = '>'
            if (s + i) in r_border or (s + i + 1) in self.environment['obstacles']:
                self.reset()
                return self.environment['start'], self.rewards['crash']
            elif (s + i + 1) == self.environment['checkpoint']:
                self.checkpoint = self.velocity['V_MAX'] != 5
                self.velocity['V_MAX'] = 5
            elif (s + i + 1) == self.environment['goal']:
                self.environment['grid'][self.states[s + i + 1]] = 'O'
                return s + i + 1, self.rewards['win']

        self.environment['grid'][self.states[s + self.velocity['V']]] = 'O'
        if self.checkpoint:
            return (s + self.velocity['V'], self.rewards['checkpoint']) 
        else:
            return (s + self.velocity['V'], self.rewards['step'])

    def move_up(self, s, units, t_border):
        WIDTH = self.environment['size']
        for i in units:
            self.environment['grid'][self.states[s - i * WIDTH]] = '|'
            if (s - i * WIDTH) in t_border or (s - (i + 1) * WIDTH) in self.environment['obstacles']:
                self.reset()
                return self.environment['start'], self.rewards['crash']
            elif (s + i + 1) == self.environment['checkpoint']:
                self.checkpoint = self.velocity['V_MAX'] != 5
                self.velocity['V_MAX'] = 5
            elif (s - (i + 1) * WIDTH) == self.environment['goal']:
                self.environment['grid'][self.states[s - (i + 1) * WIDTH]] = 'O'
                return s - (i + 1) * WIDTH, self.rewards['win']

        self.environment['grid'][self.states[s + self.velocity['V']]] = 'O'
        if self.checkpoint:
            return (s - self.velocity['V'] * WIDTH, self.rewards['checkpoint']) 
        else:
            return (s - self.velocity['V'] * WIDTH, self.rewards['step'])

    def move_left(self, s, units, l_border):
        for i in units:
            self.environment['grid'][self.states[s - i]] = '<'
            if (s - i) in l_border or (s - i - 1) in self.environment['obstacles']:
                self.reset()
                return self.environment['start'], self.rewards['crash']
            elif (s - i - 1) == self.environment['checkpoint']:
                self.checkpoint = self.velocity['V_MAX'] != 5
                self.velocity['V_MAX'] = 5
            elif (s - i - 1) == self.environment['goal']:
                self.environment['grid'][self.states[s - i - 1]] = 'O'
                return s - i - 1, self.rewards['win']

        self.environment['grid'][self.states[s - self.velocity['V']]] = 'O'
        if self.checkpoint:
            return (s - self.velocity['V'], self.rewards['checkpoint']) 
        else:
            return (s - self.velocity['V'], self.rewards['step'])


def set_args():
    args = dict()
    args['mode'] = 'SARSA'
    args['n_episodes'] = 1000
    args['n_steps'] = 5
    args['gamma'] = 0.99
    args['alpha'] = 0.1
    args['epsilon'] = 0.1
    args['beta'] = 1
    args['n_experiments'] = 10
    return args
    

def setup_experiment(world):
    P = np.zeros((len(world.states), len(world.actions['actions'])))
    Q = np.zeros(P.shape)
    n_steps = []
    rewards = []
    return P, Q, n_steps, rewards


def initialize_episode_args():
    episode_terms = {}
    episode_terms['steps'] = 0
    episode_terms['reward'] = 0
    episode_terms['states'] = []
    episode_terms['actions'] = []
    episode_terms['q'] = []
    episode_terms['p'] = []
    episode_terms['sigmas'] = [1]
    episode_terms['targets'] = []
    return episode_terms


def initialize_episode(world, episode_terms, args):
    episode_terms['states'].append(world.environment['start'])
    action = world.choose_action(world.environment['start'], args['epsilon'])
    episode_terms['actions'].append(action)
    episode_terms['q'].append(world.Q[world.environment['start'], action])
    episode_terms['p'].append(world.P[world.environment['start'], action])
    return -1, np.inf, episode_terms


def main():
    world_instance = World()
    args = set_args()
    average_steps = []
    average_reward = []
    
    for _ in range(args['n_experiments']):
        P, Q, n_steps, rewards = setup_experiment(world_instance)
        start = dt.now()

        for ep in range(args['n_episodes']):
            print('\nEpisode: ' + str(_ + 1) + '/' + str(args['n_episodes']) + " ...")
            world_instance.reset()
            episode_terms = initialize_episode_args()
            t, T, episode_terms = initialize_episode(world_instance, episode_terms, args)

            while True:
                t += 1
                assert len(episode_terms['actions']) == len(episode_terms['sigmas']) == len(episode_terms['p']) == len(episode_terms['q'])
                if t < T:
                    s_next, r = world_instance.move(episode_terms['states'][t], episode_terms['actions'][t], args['beta'])
                    episode_terms['states'].append(s_next)
                    episode_terms['steps'] += 1
                    episode_terms['reward'] += r
                    if s_next == world_instance.environment['goal']:
                        T = t + 1
                        episode_terms['targets'].append(r - episode_terms['q'][t])
                    else:
                        a_next = world_instance.choose_action(episode_terms['states'][t+ 1], args['epsilon'])
                        episode_terms['actions'].append(s_next)
                        sigma = world_instance.choose_sigma(args['mode'])
                        episode_terms['sigmas'].append(sigma)
                        episode_terms['q'].append(world_instance.Q[s_next, a_next])
                        target = r + sigma * args['gamma'] * episode_terms['q'][t + 1]
                        target += (1 - sigma) * args['gamma'] * np.dot(world_instance.P[s_next, :], world_instance.Q[s_next, :])
                        target += target - episode_terms['q'][t]
                        episode_terms['targets'].append(target)
                        episode_terms['p'].append(world_instance.P[s_next, a_next])
                tau = t - args['n_steps'] + 1
                if tau >= 0:
                    E = 1
                    G = episode_terms['q'][tau]
                    for k in range(tau, min(t, T - 1)):
                        G += E * episode_terms['targets'][k]
                        E *= args['gamma'] * ((1-episode_terms['sigmas']) * episode_terms['p'][k + 1] + episode_terms['sigmas'][k + 1])
                    error = G - world_instance.Q[episode_terms['states'][tau], episode_terms['actions'][tau]]
                    world_instance.Q[episode_terms['states'][tau], episode_terms['actions'][tau]] += args['alpha'] * error
                    world_instance.make_greedy(episode_terms['states'][tau], args['epsilon'])
                if tau == T - 1:
                    break
            
            print(world_instance.environment['grid'])
            n_steps.append(episode_terms['steps'])
            rewards.append(episode_terms['reward'])
        
        average_steps.append(np.average(n_steps))
        print('average number of steps = ' + str(average_steps[-1]))

    print("\nsteps: " + str(average_steps))
    print("steps avg: " + str(np.average(average_steps)))
    print("rewards: " + str(average_reward))
    print("rewards avg: " + str(np.average(average_reward)))


if __name__ == "__main__":
    main()
